import math

import torch
from liger_kernel.transformers import LigerRMSNorm
from torch import nn

from .block import TransformerBlock
from .config import ModelConfig


def init_model_weights(model, router_std=0.01):
    # Infer model dimensions
    d_model = model.tok_emb.embedding_dim
    n_layers = len(model.blocks)

    # Standard deviation for ordinary projections
    init_std = 1.0 / math.sqrt(d_model)

    # Depth-scaled residual branch outputs
    residual_std = 1.0 / math.sqrt(2 * n_layers * d_model)

    with torch.no_grad():
        # Token embedding
        nn.init.normal_(model.tok_emb.weight, mean=0.0, std=init_std)

        for block in model.blocks:
            attn = block.attn

            # Attention input projections
            for name in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                layer = getattr(attn, name, None)
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=init_std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

            # Attention output projection
            nn.init.normal_(attn.o_proj.weight, std=residual_std)
            if attn.o_proj.bias is not None:
                nn.init.zeros_(attn.o_proj.bias)

            # Router: small initial logits, approximately uniform routing
            router = block.moe.router.gate
            nn.init.normal_(router.weight, mean=0.0, std=router_std)

            # Routed and shared expert MLPs
            all_experts = list(block.moe.experts) + list(block.moe.shared_experts)

            for expert in all_experts:
                mlp = expert.mlp

                # SwiGLU input projections
                for name in ("gate_proj", "up_proj"):
                    layer = getattr(mlp, name)
                    nn.init.normal_(layer.weight, std=init_std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

                # SwiGLU output projection
                layer = mlp.down_proj
                nn.init.normal_(layer.weight, std=residual_std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            # RMSNorm scale
            for norm in (block.attn_norm, block.moe_norm):
                nn.init.ones_(norm.weight)

        nn.init.ones_(model.final_norm.weight)

        # Untied output head
        nn.init.normal_(model.lm_head.weight, std=init_std)
        if model.lm_head.bias is not None:
            nn.init.zeros_(model.lm_head.bias)

    return model


class MoETransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.top_k = cfg.top_k
        self.num_experts = cfg.num_experts

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LigerRMSNorm(cfg.d_model, eps=1e-6)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def reset_cache(self) -> None:
        for block in self.blocks:
            block.reset_cache()

    def num_parameters(self, active_only: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())

        if not active_only:
            return total

        routed_expert_params = sum(
            p.numel() for block in self.blocks for p in block.moe.experts.parameters()
        )

        active_fraction = self.top_k / self.num_experts
        active_routed_params = round(routed_expert_params * active_fraction)

        non_routed_params = total - routed_expert_params

        return non_routed_params + active_routed_params

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.tok_emb(input_ids)

        aux_total = x.new_zeros((), dtype=torch.float32)
        z_total = x.new_zeros((), dtype=torch.float32)

        for block in self.blocks:
            x, aux, z = block(x, use_cache=use_cache)
            aux_total = aux_total + aux
            z_total = z_total + z

        n = len(self.blocks)
        aux_loss = aux_total / n  # mean over layers
        z_loss = z_total / n

        logits = self.lm_head(self.final_norm(x))
        return logits, aux_loss, z_loss
