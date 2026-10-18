import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerRMSNorm
from torch import nn

from .rope import RotaryEmbedding


def causal_mask(
    query_len: int,
    key_len: int,
    key_start: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """True = attend. Shape: (query_len, key_len)."""
    q_pos = torch.arange(
        key_start + key_len - query_len,
        key_start + key_len,
        device=device,
    ).unsqueeze(1)

    k_pos = torch.arange(
        key_start,
        key_start + key_len,
        device=device,
    ).unsqueeze(0)

    return k_pos <= q_pos


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None,
        bias: bool,
        rope_theta: float,
        context_len: int,
    ) -> None:
        super().__init__()

        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads

        if d_model <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("d_model, n_heads, and n_kv_heads must be positive")

        if d_model % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, and "
                "n_heads must be divisible by n_kv_heads"
            )

        if (d_model // n_heads) % 2 != 0:
            raise ValueError("head_dim must be even to use RoPE")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)

        self.gate_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)

        # QK-norm (applied before RoPE)
        self.q_norm = LigerRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = LigerRMSNorm(self.head_dim, eps=1e-6)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            context_len=context_len,
            theta_base=rope_theta,
        )

        # KV cache (inference only; never saved in state_dict)
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self) -> None:
        """Clear the KV cache. Call when starting a new sequence."""
        self.cache_k = None
        self.cache_v = None

    def _split_heads(
        self,
        t: torch.Tensor,
        n_heads: int,
    ) -> torch.Tensor:
        b, s, _ = t.shape
        return t.view(b, s, n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Positions of new tokens
        past_len = (
            self.cache_k.shape[2] if use_cache and self.cache_k is not None else 0
        )

        position_ids = torch.arange(
            past_len,
            past_len + seq_len,
            device=x.device,
        )

        # 2. Project to Q / K / V
        q = self.q_norm(self._split_heads(self.q_proj(x), self.n_heads))
        k = self.k_norm(self._split_heads(self.k_proj(x), self.n_kv_heads))
        v = self._split_heads(self.v_proj(x), self.n_kv_heads)

        gate = torch.sigmoid(self._split_heads(self.gate_proj(x), self.n_heads))

        # 3. Apply RoPE
        q = self.rope(q, position_ids=position_ids)
        k = self.rope(k, position_ids=position_ids)

        # 4. Update KV cache
        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)

            self.cache_k = k
            self.cache_v = v

        # 5. Causal attention mask
        key_len = k.shape[2]

        mask = causal_mask(
            query_len=seq_len,
            key_len=key_len,
            key_start=past_len,
            device=x.device,
        )

        # 6. Grouped-query causal attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            enable_gqa=True,
        )

        out = out * gate

        # 7. Merge heads and project output
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        return self.o_proj(out)
