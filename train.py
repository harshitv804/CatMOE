from __future__ import annotations

import math
from pathlib import Path

import torch
from datasets import load_dataset
from liger_kernel.transformers import LigerCrossEntropyLoss
from torch.optim import AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from components import (
    ModelConfig,
    MoETransformerLM,
    TrainingConfig,
    init_model_weights,
)
from components.config import tokenizer
from packed_dataloader import make_packed_dataloader


def CosineScheduler_WarmUp(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup to the base LR, then cosine decay. num_cycles=0.5 -> one half-cosine."""
    if not 0 <= num_warmup_steps <= num_training_steps:
        raise ValueError("Require 0 <= num_warmup_steps <= num_training_steps")

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(2.0 * math.pi * num_cycles * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def count_steps(loader) -> int:
    """Batches per epoch. The loader has no ``__len__``, so this tokenizes one pass."""
    return sum(1 for _ in loader)


def perplexity(loss: float) -> float:
    """exp(loss); guards the OverflowError math.exp raises above ~88."""
    return math.exp(loss) if loss < 88.0 else float("inf")


def build_optimizers(
    model: MoETransformerLM, tcfg: TrainingConfig
) -> tuple[Muon, AdamW]:
    """Muon for 2D hidden matmuls; AdamW for ``tok_emb``, ``router``, and 1D params."""
    muon_params: list[torch.Tensor] = []
    adamw_params: list[torch.Tensor] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_matmul = (
            p.ndim == 2 and "tok_emb" not in name and "router" not in name
        )
        (muon_params if is_hidden_matmul else adamw_params).append(p)

    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(
        f"[optim] Muon : {len(muon_params):>4d} tensors, {n_muon / 1e6:8.1f}M params\n"
        f"[optim] AdamW: {len(adamw_params):>4d} tensors, {n_adamw / 1e6:8.1f}M params"
    )

    muon = Muon(muon_params, lr=tcfg.lr_muon, weight_decay=tcfg.weight_decay)
    adamw = AdamW(
        adamw_params,
        lr=tcfg.lr_adamw,
        betas=tcfg.betas,
        weight_decay=tcfg.weight_decay,
    )
    return muon, adamw


def train_step(
    model: MoETransformerLM,
    muon: Muon,
    adamw: AdamW,
    schedulers: tuple[LambdaLR, ...],
    input_ids: torch.Tensor,
    tcfg: TrainingConfig,
    ce: LigerCrossEntropyLoss,
) -> dict[str, float]:
    """One forward/backward/step. Returns scalar logs."""
    autocast_dtype = torch.bfloat16 if tcfg.use_bf16 else torch.float32

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        logits, aux_loss, z_loss = model(input_ids, use_cache=False)

        # Causal shift, HF-style. Cross-entropy in fp32 for stability.
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = input_ids[:, 1:].reshape(-1)
        lm_loss = ce(shift_logits.float(), shift_labels)

        # aux_loss / z_loss are already averaged over layers by model.forward.
        total_loss = lm_loss + tcfg.aux_coef * aux_loss + tcfg.z_coef * z_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
    muon.step()
    adamw.step()
    # One scheduler per optimizer (LambdaLR binds to a single optimizer, and
    # torch's ChainedScheduler rejects schedulers from different optimizers).
    # Must run after optimizer.step(), or the first schedule value is skipped.
    for sched in schedulers:
        sched.step()
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)

    return {
        "lm_loss": lm_loss.detach().item(),
        "aux_loss": aux_loss.detach().item(),
        "z_loss": z_loss.detach().item(),
        "total_loss": total_loss.detach().item(),
        "lr_muon": muon.param_groups[0]["lr"],
        "lr_adamw": adamw.param_groups[0]["lr"],
    }


def save_checkpoint(
    path: Path,
    model: MoETransformerLM,
    muon: Muon,
    adamw: AdamW,
    step: int,
    epoch: int,
    row: int,
) -> None:
    """Atomically write model + optimizer state (tmp file, then rename).

    Stores only the ``(step, epoch, row)`` cursor; the LR schedule is a pure
    function of the step. Keys match the plain-PyTorch model, so this same
    checkpoint also loads with ``strict=True`` there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "row": row,
            "model": model.state_dict(),
            "muon": muon.state_dict(),
            "adamw": adamw.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: MoETransformerLM,
    muon: Muon,
    adamw: AdamW,
) -> tuple[int, int, int]:
    """Restore model/optimizer state; return the ``(step, epoch, row)`` cursor."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    # torch casts every optimizer state tensor to its parameter's device/dtype,
    # so CPU-loaded state lands correctly on CUDA.
    muon.load_state_dict(ckpt["muon"])
    adamw.load_state_dict(ckpt["adamw"])
    return int(ckpt["step"]), int(ckpt["epoch"]), int(ckpt["row"])


def evaluate(
    model: MoETransformerLM,
    val_loader,
    tcfg: TrainingConfig,
    ce: LigerCrossEntropyLoss,
    device: torch.device,
) -> float | None:
    """Token-weighted mean CE over the val split, or None if it yields no blocks.

    ``model.eval()`` matters: ``TopKRouter.forward`` mutates ``expert_bias`` in
    training mode. Eval also zeroes aux/z, so only the LM loss is meaningful.
    """
    autocast_dtype = torch.bfloat16 if tcfg.use_bf16 else torch.float32
    total_loss = 0.0
    total_tokens = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for input_ids, _labels, _attn_mask, _data_index in val_loader:
                input_ids = input_ids.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    # use_cache=False keeps the KV cache untouched, so positions
                    # restart at 0 for every batch and no reset is needed.
                    logits, _aux_loss, _z_loss = model(input_ids, use_cache=False)

                    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                    shift_labels = input_ids[:, 1:].reshape(-1)
                    lm_loss = ce(shift_logits.float(), shift_labels)

                # Weight by token count so a short final batch can't skew the mean.
                n_tokens = shift_labels.numel()
                total_loss += lm_loss.item() * n_tokens
                total_tokens += n_tokens
    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        return None
    return total_loss / total_tokens


def main() -> None:

    torch.set_default_dtype(torch.float32)
    assert torch.cuda.is_available(), (
        "training requires CUDA (Liger Kernels are CUDA-only)"
    )
    device = torch.device("cuda")

    mcfg = ModelConfig()
    tcfg = TrainingConfig()
    ce = LigerCrossEntropyLoss()

    assert tcfg.block_size <= mcfg.context_len, "block_size must be <= context_len"

    model = MoETransformerLM(mcfg).to(device)
    init_model_weights(model)
    model.train()
    muon, adamw = build_optimizers(model, tcfg)

    # ---- data ----
    full_dataset = load_dataset("parquet", data_files=tcfg.data_path, split="train")
    all_indices = list(range(len(full_dataset)))

    val_indices = all_indices[: tcfg.val_size]
    train_indices = all_indices[tcfg.val_size :]

    val_dataset = full_dataset.select(val_indices)
    train_dataset = full_dataset.select(train_indices)

    def make_train_loader(start_row_index: int = 0):
        return make_packed_dataloader(
            train_dataset,
            tokenizer,
            text_column=tcfg.text_column,
            block_size=tcfg.block_size,
            batch_size=tcfg.batch_size,
            append_eos=True,
            start_row_index=start_row_index,
        )

    val_loader = make_packed_dataloader(
        val_dataset,
        tokenizer,
        text_column=tcfg.text_column,
        block_size=tcfg.block_size,
        batch_size=tcfg.batch_size,
        append_eos=True,
    )

    # ---- resume ----
    # Restore before the schedulers exist: LambdaLR reads "initial_lr" from the
    # param groups at construction and the checkpoint already carries it.
    if tcfg.resume_from is not None:
        resume_step, resume_epoch, resume_row = 0, 0, 0
        resume_step, resume_epoch, resume_row = load_checkpoint(
            tcfg.resume_from, model, muon, adamw
        )
        print(
            f"[ckpt] resumed step={resume_step} epoch={resume_epoch} "
            f"row={resume_row} from {tcfg.resume_from}"
        )

    # ---- LR schedule ----
    # The loader streams and has no __len__, so the epoch length is unknown up
    # front. Resolve the horizon: max_steps wins (smoke test), then an explicit
    # num_training_steps (skips the pass), else count exactly. Always measured
    # from row 0, so a resume does not shrink the horizon.
    if tcfg.max_steps is not None:
        num_training_steps = tcfg.max_steps
        print(f"[sched] horizon = max_steps = {num_training_steps}")
    elif tcfg.num_training_steps is not None:
        num_training_steps = tcfg.num_training_steps
        print(f"[sched] horizon = num_training_steps = {num_training_steps}")
    else:
        steps_per_epoch = count_steps(make_train_loader(0))
        num_training_steps = tcfg.num_epochs * steps_per_epoch
        print(
            f"[sched] counted {steps_per_epoch} steps/epoch; horizon = "
            f"{tcfg.num_epochs} epoch(s) x {steps_per_epoch} = "
            f"{num_training_steps} (set TrainingConfig.num_training_steps to "
            f"skip this pass)"
        )

    # CosineScheduler_WarmUp requires 0 <= warmup <= horizon.
    num_warmup_steps = min(tcfg.num_warmup_steps, num_training_steps)
    if num_warmup_steps != tcfg.num_warmup_steps:
        print(f"[sched] warmup clamped {tcfg.num_warmup_steps} -> {num_warmup_steps}")

    # Both schedulers share the same curve; each multiplies its own optimizer's
    # initial LR (muon 0.02, adamw 3e-4), so the peaks stay independent.
    schedulers: tuple[LambdaLR, ...] = tuple(
        CosineScheduler_WarmUp(opt, num_warmup_steps, num_training_steps)
        for opt in (muon, adamw)
    )

    # The schedule is a pure function of the step, so a resume rebuilds it by
    # replaying rather than storing scheduler state.
    if resume_step:
        for _ in range(resume_step):
            for sched in schedulers:
                sched.step()
        print(
            f"[sched] fast-forwarded {resume_step} steps -> "
            f"lr muon={muon.param_groups[0]['lr']:.3e} "
            f"adamw={adamw.param_groups[0]['lr']:.3e}"
        )

    print(
        f"[sched] {num_warmup_steps} warmup / {num_training_steps} total steps; "
        f"peak lr muon={tcfg.lr_muon} adamw={tcfg.lr_adamw}"
    )

    ckpt_path = Path(tcfg.checkpoint_dir) / "ckpt.pt"

    step = resume_step
    for epoch in range(resume_epoch, tcfg.num_epochs):
        # The resumed epoch continues from the saved row; later epochs start over.
        start_row = resume_row if epoch == resume_epoch else 0
        for input_ids, _labels, _attn_mask, data_index in make_train_loader(start_row):
            input_ids = input_ids.to(device, non_blocking=True)
            logs = train_step(model, muon, adamw, schedulers, input_ids, tcfg, ce)

            if step % tcfg.log_every == 0:
                print()
                print(
                    f"[🔥train] epoch: {epoch} | step: {step} | "
                    f"lm_loss: {logs['lm_loss']:.4f} | "
                    f"ppl: {perplexity(logs['lm_loss']):.2f} | "
                    f"aux_loss: {logs['aux_loss']:.4f} | "
                    f"z_loss: {logs['z_loss']:.4f} | "
                    f"total_loss: {logs['total_loss']:.4f} | "
                    f"lr_muon: {logs['lr_muon']:.3e} | "
                    f"lr_adamw: {logs['lr_adamw']:.3e}"
                )

                val_loss = evaluate(model, val_loader, tcfg, ce, device)
                if val_loss is not None:
                    print(
                        f"[🧪val] step: {step} | "
                        f"val_lm_loss: {val_loss:.4f} | "
                        f"val_ppl: {perplexity(val_loss):.2f}"
                    )

            step += 1
            if tcfg.checkpoint_every is not None and step % tcfg.checkpoint_every == 0:
                # data_index[0] is the source row of the batch's first block.
                # Resuming there replays that batch's rows but never skips any.
                save_checkpoint(
                    ckpt_path, model, muon, adamw, step, epoch, int(data_index[0])
                )
                print(f"[ckpt] saved step={step} -> {ckpt_path}")

            if tcfg.max_steps is not None and step >= tcfg.max_steps:
                print(f"reached max_steps={tcfg.max_steps}, stopping")
                return


if __name__ == "__main__":
    main()
