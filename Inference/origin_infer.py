"""Qwen3-0.6B 推理示例：PyTorch 原生加载 vs Transformers 加载（NPU）"""

import argparse
import json

import torch
import torch_npu  # noqa: F401
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

MODEL_PATH = "/data/models/Qwen3-0.6B"
DEVICE = "npu:0"
DEFAULT_PROMPT = "你好，请用一句话介绍一下你自己。"


def _build_chat_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in inputs.items()}


def infer_with_pytorch(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 128) -> str:
    """方案一：config + safetensors 手动加载，手写自回归推理"""
    torch.npu.set_device(DEVICE)

    with open(f"{MODEL_PATH}/config.json", encoding="utf-8") as f:
        config = Qwen3Config.from_dict(json.load(f))

    model = Qwen3ForCausalLM(config)
    model.load_state_dict(load_file(f"{MODEL_PATH}/model.safetensors"), strict=True)

    dtype = getattr(config, "dtype", torch.bfloat16)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    model = model.to(device=DEVICE, dtype=dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = _build_chat_inputs(tokenizer, prompt)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=DEVICE)],
                    dim=-1,
                )
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)


def infer_with_transformers(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 128) -> str:
    """方案二：Transformers from_pretrained + generate"""
    torch.npu.set_device(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", trust_remote_code=True
    ).to(DEVICE).eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B NPU 推理")
    parser.add_argument("--method", choices=["pytorch", "transformers", "both"], default="both")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.method in ("pytorch", "both"):
        print("[方案一] PyTorch 原生加载")
        print(infer_with_pytorch(args.prompt, args.max_new_tokens), end="\n\n" if args.method == "both" else "\n")

    if args.method in ("transformers", "both"):
        print("[方案二] Transformers 加载")
        print(infer_with_transformers(args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
