from components import MoETransformerLM
from components import ModelConfig
import torch

cfg = ModelConfig()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MoETransformerLM(cfg).to(device)

print(model)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


total = count_params(model)

routed = 0
shared = 0
active_routed = 0

for block in model.blocks:
    moe = block.moe

    experts = list(moe.experts)
    routed += sum(count_params(expert) for expert in experts)

    shared += count_params(moe.shared_experts)

    top_k = moe.router.top_k
    num_experts = len(experts)

    # Exact for equal-sized experts; estimates for unequal sizes.
    expert_params = [count_params(expert) for expert in experts]
    active_routed += (
        sum(sorted(expert_params, reverse=True)[:top_k])
        if top_k == num_experts
        else sum(expert_params) * top_k / num_experts
    )

other = total - routed - shared
active_est = other + shared + active_routed

print(f"Total:          {total / 1e6:.3f}M")
print(f"Routed:         {routed / 1e6:.3f}M")
print(f"Shared:         {shared / 1e6:.3f}M")
print(f"Other:          {other / 1e6:.3f}M")
print(f"Active routed:  {active_routed / 1e6:.3f}M")
print(f"Active est.:    {active_est / 1e6:.3f}M")
print(f"Active ratio:   {100 * active_est / total:.2f}%")
