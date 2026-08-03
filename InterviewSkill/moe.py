import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # gate
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        # up
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        # down
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_expert, top_k, num_shared_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_expert, bias=False)
        self.top_k = top_k
        self.experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(num_expert)])
        self.num_shared_experts = num_shared_experts
        self.shared_experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(num_shared_experts)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size * seq_len, d_model)
        
        # 1. gate (linear)
        # N, d -> N, num_expert
        gate_logits = self.gate(x)

        # 2. topk logits / indices
        topk_logits, topk_indices = torch.topk(gate_logits, self.top_k)

        # 3. weight
        topk_weights = torch.softmax(topk_logits, dim=-1)

        # 4. moe
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            expert_indices = topk_indices[:, slot]
            expert_weights = topk_weights[:, slot:slot+1]
            for e in range(len(self.experts)):
                mask = (expert_indices == e)
                if mask.any():
                    out[mask] += expert_weights[mask] * self.experts[e](x[mask])
        
        # 5. shared expert
        for e in range(self.num_shared_experts):
            out = out + self.shared_experts[e](x)
        
        return out.reshape(batch_size, seq_len, d_model)


moe = MoE(d_model=8, d_ff=32, num_expert=4, top_k=2, num_shared_experts=1)
x = torch.randn(2, 5, 8)
out = moe(x)
print("x.shape: ", x.shape, ", out.shape: ", out.shape)