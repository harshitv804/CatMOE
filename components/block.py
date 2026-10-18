import torch
from liger_kernel.transformers import LigerRMSNorm
from torch import nn

from .config import ModelConfig
from .gqa_attn import GroupedQueryAttention
from .moe import SparseMOE


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(norm(x)), then x + MoE(norm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        # Every field each submodule consumes, in one place.
        self.attn = GroupedQueryAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            bias=cfg.bias,
            rope_theta=cfg.rope_theta,
            context_len=cfg.context_len,
        )

        self.moe = SparseMOE(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            bias=cfg.bias,
            num_shared_experts=cfg.num_shared_experts,
            shared_d_ff=cfg.shared_d_ff,
        )

        self.attn_norm = LigerRMSNorm(cfg.d_model, eps=1e-6)
        self.moe_norm = LigerRMSNorm(cfg.d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, use_cache: bool):
        x = x + self.attn(self.attn_norm(x), use_cache=use_cache)

        moe_out, aux_loss, z_loss = self.moe(self.moe_norm(x))
        x = x + moe_out

        return x, aux_loss, z_loss

    def reset_cache(self) -> None:
        self.attn.reset_cache()
