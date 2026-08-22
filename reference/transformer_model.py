"""A compact decoder-only Transformer for teaching.

Adapted from MiniMind's model/model_minimind.py. The production-oriented
features (GQA, KV caching, Flash Attention, YaRN, MoE, and Hugging Face
wrappers) are deliberately omitted so the core data path stays visible.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 4
    intermediate_size: int = 352
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even so RoPE can rotate pairs")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight).to(dtype=x.dtype)


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
) -> tuple[Tensor, Tensor]:
    """Return cos/sin tables for adjacent pairs in each attention head."""
    pair_ids = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = theta ** (-pair_ids / head_dim)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate x shaped [batch, heads, tokens, head_dim] by adjacent pairs."""
    batch, heads, tokens, head_dim = x.shape
    pairs = x.float().reshape(batch, heads, tokens, head_dim // 2, 2)
    x_even, x_odd = pairs.unbind(dim=-1)
    cos = cos[:tokens].view(1, 1, tokens, head_dim // 2)
    sin = sin[:tokens].view(1, 1, tokens, head_dim // 2)
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        cos, sin = precompute_rope(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def split_heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        q = apply_rope(self.split_heads(self.q_proj(x)), self.rope_cos, self.rope_sin)
        k = apply_rope(self.split_heads(self.k_proj(x)), self.rope_cos, self.rope_sin)
        v = self.split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(tokens, tokens, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = weights @ v

        context = context.transpose(1, 2).contiguous().view(batch, tokens, width)
        return self.out_proj(context)


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TransformerModel(nn.Module):
    """Decoder-only language model returning next-token logits."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence is longer than max_seq_len")

        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.final_norm(x))


if __name__ == "__main__":
    torch.manual_seed(7)
    config = TransformerConfig(
        vocab_size=32,
        hidden_size=16,
        num_heads=4,
        num_layers=2,
        intermediate_size=32,
        max_seq_len=8,
    )
    model = TransformerModel(config)
    token_ids = torch.tensor([[1, 5, 9, 2]])
    logits = model(token_ids)
    print("input:", tuple(token_ids.shape))
    print("logits:", tuple(logits.shape))  # (1, 4, 32)
    print("parameters:", sum(p.numel() for p in model.parameters()))
