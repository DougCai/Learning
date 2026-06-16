# 原生推理关键 API

对应 `infer.py` 方案一中的典型用法：

```python
model = Qwen3ForCausalLM(config)
model.load_state_dict(load_file(f"{MODEL_PATH}/model.safetensors"), strict=True)
model = model.to(device=DEVICE, dtype=dtype).eval()

with torch.no_grad():
    outputs = model(input_ids=generated, attention_mask=attention_mask)
```

## `load_file`（safetensors）

- **作用**：从 `.safetensors` 读取权重，返回 `{参数名: Tensor}` 字典。
- **为何用它**：比传统 `.bin` 更安全（不执行 pickle）、加载较快。
- **示例**：`load_file("/data/models/Qwen3-0.6B/model.safetensors")`

## `load_state_dict`

- **作用**：把字典里的张量按名称写入模型的 `weight` / `bias`。
- **`strict=True`**：键名与形状必须完全一致，否则报错（避免“以为加载成功、实际未对齐”）。
- **流程位置**：先 `Qwen3ForCausalLM(config)` 建空壳（随机参数）→ `load_state_dict` 覆盖为预训练权重。
- **对比**：`from_pretrained()` 在内部完成同样的事。

## `model.eval()`

- **作用**：切换到**推理模式**（与 `model.train()` 相对）。
- **主要影响**：
  - **Dropout**：关闭，不再随机丢弃神经元。
  - **BatchNorm**：使用训练阶段累计的 running 统计量，而非当前 batch。
- **推理必用**：保证输出稳定、可复现。

## `torch.no_grad()`

- **作用**：关闭自动求导，不构建计算图。
- **效果**：不计算梯度、省显存、通常更快。
- **推理必用**：只需前向得到 `logits`，不需要 `backward()`。

## `eval()` vs `no_grad()`

| | `eval()` | `no_grad()` |
|---|----------|-------------|
| 改什么 | 层的前向**行为**（Dropout 等） | 是否追踪**梯度** |
| 推理 | 需要 | 需要 |

两者职责不同，推理时通常一起使用。

## Transformers vs PyTorch 加载

**PyTorch（方案一）** — 手动分步：

```python
config = Qwen3Config.from_dict(json.load(f))
model = Qwen3ForCausalLM(config)
model.load_state_dict(load_file(...), strict=True)
model = model.to(device=DEVICE, dtype=dtype).eval()
```

**Transformers（方案二）** — 一行加载：

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype="auto", trust_remote_code=True
).to(DEVICE).eval()
```

| | PyTorch | Transformers |
|---|---------|--------------|
| 读 config | 手动 `json` + `Qwen3Config` | `from_pretrained` 内部自动 |
| 建模型 | 显式 `Qwen3ForCausalLM(config)` | 按 config 自动选类 |
| 读权重 | `load_file` + `load_state_dict` | 内部读 `model.safetensors` |
| dtype | 手动从 config 解析 | `dtype="auto"` 自动选 |
| 上设备 | `.to(device=DEVICE, dtype=dtype)` | `.to(DEVICE)` |

两者最终都是：建好模型 → 载入权重 → 上 NPU → `eval()`；Transformers 把中间步骤封装进 `from_pretrained`。

## 手写循环 vs `model.generate()`

**PyTorch（方案一）** — 手写自回归：

```python
with torch.no_grad():
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        # attention_mask 末尾 +1；遇 eos 则 break
```

**Transformers（方案二）** — 一行生成：

```python
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
```

| | 手写循环 | `model.generate()` |
|---|----------|-------------------|
| 循环 | 显式 `for` | 内部封装 |
| 选 token | `argmax` 贪心 | `do_sample=False`，等价贪心 |
| 序列更新 | 手动 `torch.cat` | 内部处理 |
| attention_mask | 每步手动追加 | 内部处理 |
| 停止条件 | 手动判断 `eos_token_id` | 内部处理 |
| KV Cache | 无，每步 forward 全序列 | 内部可启用，更高效 |

两者都在 `no_grad()` 下做自回归生成；方案一便于理解流程，方案二生产常用、功能更全。
