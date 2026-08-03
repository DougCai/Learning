import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads=None):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads or num_q_heads
        self.head_dim = d_model // num_q_heads
        self.q_per_kv = num_q_heads // self.num_kv_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.head_dim * self.num_kv_heads, bias=False)
        self.w_v = nn.Linear(d_model, self.head_dim * self.num_kv_heads, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def repeat_kv(self, x):
        if self.q_per_kv == 1:
            return x
        B, H, S, D = x.shape
        x = x[:, :, None].expand(B, H, self.q_per_kv, S, D)
        return x.reshape(B, H * self.q_per_kv, S, D)

    def forward(self, x, causal=True):
        B, S, _ = x.shape

        # Q, K, V
        Q = self.w_q(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # repeat kv (GQA, MQA)
        K, V = self.repeat_kv(K), self.repeat_kv(V)

        # score
        # BHSD @ BHDS -> BHSS
        score = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.triu(torch.ones(S, S, dtype=bool), diagonal=1)
            score = score.masked_fill(mask, float("-inf"))

        # attn
        # BHSS @ BHSD -> BHSD
        attn = torch.softmax(score, dim=-1) @ V

        # merge heads: BHSD -> [B, S, num_q_heads * head_dim] = [B, S, d_model]
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        # output projection
        return self.w_o(attn)


mha = Attention(d_model=32, num_q_heads=4)
gqa = Attention(d_model=32, num_q_heads=4, num_kv_heads=2)
mqa = Attention(d_model=32, num_q_heads=4, num_kv_heads=1)

x = torch.randn(2, 3, 32)
print("[MHA] x.shape: ", mha(x, True).shape)
print("[GQA] x.shape: ", gqa(x, True).shape)
print("[MQA] x.shape: ", mqa(x, True).shape)