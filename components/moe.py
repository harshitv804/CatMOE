import torch
import torch.nn.functional as F
from torch import nn

from .ffn import LigerFeedForward


class TopKRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        bias_update_rate: float,
    ) -> None:
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError("Invalid top_k")

        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate

        self.gate = nn.Linear(d_model, num_experts, bias=False)

        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

    def compute_aux_loss(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        # Fraction of token assignments per expert
        counts = torch.bincount(
            indices.reshape(-1),
            minlength=self.num_experts,
        ).to(device=scores.device, dtype=scores.dtype)

        assignment_fraction = (counts / indices.numel()).detach()

        # Mean sigmoid score per expert
        mean_score = scores.mean(dim=(0, 1))

        # Load-balancing loss
        return self.num_experts * torch.sum(assignment_fraction * mean_score)

    def compute_z_loss(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Router z-loss in FP32
        log_partition = torch.logsumexp(
            logits,
            dim=-1,
        )

        return log_partition.square().mean()

    @torch.no_grad()
    def update_expert_bias(self, indices: torch.Tensor) -> None:
        # indices: (B, T, K)
        counts = torch.bincount(
            indices.reshape(-1),
            minlength=self.num_experts,
        ).to(
            device=self.expert_bias.device,
            dtype=self.expert_bias.dtype,
        )

        if counts.sum() == 0:
            return

        target_count = counts.sum() / self.num_experts

        self.expert_bias.add_(self.bias_update_rate * torch.sign(target_count - counts))

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        logits = self.gate(x).to(torch.float32)  # (B, T, E)

        # 2. Convert logits to sigmoid scores
        scores = torch.sigmoid(logits)

        # 3. Add Aux-Free bias before Top-K selection
        selection_scores = scores + self.expert_bias

        _, indices = torch.topk(selection_scores, k=self.top_k, dim=-1)  # (B, T, K)

        # 4. Normalize weights across the selected experts (noise-free logits)
        selected_scores = scores.gather(dim=-1, index=indices)
        weights = F.softmax(selected_scores, dim=-1)  # (B, T, K)

        # No auxiliary losses needed at inference
        aux_loss = logits.new_zeros(())
        z_loss = logits.new_zeros(())

        if self.training:
            # 3. Auxiliary load-balancing loss / Router z-loss
            aux_loss = self.compute_aux_loss(scores, indices)
            z_loss = self.compute_z_loss(logits)

            self.update_expert_bias(indices)

        return weights, indices, aux_loss, z_loss


class SparseMOE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        bias: bool,
        num_shared_experts: int,
        shared_d_ff: int | None,
        bias_update_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        if num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0")

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            bias_update_rate=bias_update_rate,
        )

        self.experts = nn.ModuleList(
            [
                LigerFeedForward(d_model=d_model, d_ff=d_ff, bias=bias)
                for _ in range(num_experts)
            ]
        )

        # Shared experts: always active for every token, no routing.
        # Empty ModuleList when num_shared_experts == 0.
        shared_hidden = d_ff if shared_d_ff is None else shared_d_ff
        self.shared_experts = nn.ModuleList(
            [
                LigerFeedForward(d_model=d_model, d_ff=shared_hidden, bias=bias)
                for _ in range(num_shared_experts)
            ]
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        # 1. Routing weights, expert IDs, and losses
        weights, indices, aux_loss, z_loss = self.router(x)

        # 2. Flatten batch and sequence dimensions
        flat_x = x.reshape(-1, D)
        flat_weights = weights.reshape(-1, self.top_k)
        flat_indices = indices.reshape(-1, self.top_k)

        # 3. Output buffer
        output = torch.zeros_like(flat_x)

        # 4. Dispatch tokens to their selected experts
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (flat_indices == expert_id).nonzero(as_tuple=True)

            if token_idx.numel() == 0:
                continue

            expert_output = expert(flat_x.index_select(0, token_idx))
            expert_weights = flat_weights[token_idx, slot_idx].unsqueeze(-1)

            # FIX: cast to the buffer dtype. Under autocast the expert output
            # is bf16/fp16 while `output` is fp32, and index_add rejects that.
            output.index_add_(
                0,
                token_idx,
                (expert_output * expert_weights).to(output.dtype),
            )

        output = output.reshape(B, T, D)

        # 5. Add the always-on shared experts
        for shared in self.shared_experts:
            output = output + shared(x)

        return output, aux_loss, z_loss
