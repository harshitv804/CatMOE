from types import SimpleNamespace

import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerSwiGLUMLP
from torch import nn


class _SwiGLU(nn.Module):
    """Plain-PyTorch SwiGLU: the reference op Liger's kernel replaces."""

    def __init__(self, d_model: int, d_ff: int, bias: bool) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FeedForward(nn.Module):
    """SwiGLU feed-forward network, plain PyTorch.

    Projections sit under ``self.mlp`` to match ``LigerFeedForward``'s keys, so
    checkpoints move between the two implementations with no key remapping.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool,
    ) -> None:
        super().__init__()

        self.mlp = _SwiGLU(d_model, d_ff, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LigerFeedForward(nn.Module):
    """SwiGLU feed-forward network using Liger Kernel."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()

        config = SimpleNamespace(
            hidden_size=d_model,
            intermediate_size=d_ff,
            hidden_act="silu",
        )

        self.mlp = LigerSwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
