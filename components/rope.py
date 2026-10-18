import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        context_len: int,
        theta_base: float,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(context_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]

        # Non-persistent buffers move with the module but are not saved
        # in its state_dict.
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x shape: (batch, heads, seq_len, head_dim)
        batch_size, _, seq_len, _ = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)

        position_ids = position_ids.to(
            device=self.cos_cached.device,
            dtype=torch.long,
        )

        if position_ids.ndim == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError("1D position_ids must have shape (seq_len,)")
            cos = self.cos_cached[position_ids][None, None, :, :]
            sin = self.sin_cached[position_ids][None, None, :, :]
        elif position_ids.ndim == 2:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    "2D position_ids must have shape (batch_size, seq_len)"
                )
            cos = self.cos_cached[position_ids][:, None, :, :]
            sin = self.sin_cached[position_ids][:, None, :, :]
        else:
            raise ValueError("position_ids must be 1D or 2D")

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        half_dim = x.shape[-1] // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]

        return torch.cat(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin),
            dim=-1,
        )
