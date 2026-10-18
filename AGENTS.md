# AGENTS.md

A from-scratch MoE transformer (GQA attention, RoPE, sparse MoE, LM head) in the
`components/` package, plus a streaming packed-token DataLoader in
`packed_dataloader.py`. No training script, README, tests, CI, lint, or
typecheck config exists.

## Interpreter and environment

Use the in-repo uv venv: `.venv/bin/python` (Python 3.12.15). Installed:
`torch 2.14.1+cu132`, `liger-kernel 0.8.4`, `transformers 5.19.0`,
`datasets 5.1.0`, `pyarrow 26.0.0`. CUDA is available. There is no `pip`:

```bash
uv pip --python .venv/bin/python install <pkg>
```

**The model requires CUDA.** It uses Liger Kernel ops (`LigerRMSNorm`,
`LigerSwiGLUMLP`), which are Triton/CUDA-only; a CPU forward dies inside Triton
with `Pointer argument cannot be accessed from Triton (cpu tensor?)`. The
`device = "cuda" if torch.cuda.is_available() else "cpu"` line repeated in
`test.py` / `test.ipynb` will crash if it ever falls back to CPU.

## Imports and cwd

Run everything with the repo root as cwd. There is no installable package and no
root `__init__.py`, so `from components import ...` fails from any other cwd:

```bash
.venv/bin/python -c "from components import MoETransformerLM, ModelConfig"
```

`components/__init__.py` re-exports `MoETransformerLM`, `ModelConfig`,
`SparseMOE`, `TopKRouter`, `RotaryEmbedding`, `GroupedQueryAttention`,
`FeedForward`, `TransformerBlock`. Submodules also use relative imports.

`components/config.py` calls `AutoTokenizer.from_pretrained("mistral-s4-tokenizer")`
at import time, so importing `ModelConfig` reads a tokenizer from disk via a
relative path — another reason cwd must be the repo root.

## Verification

No test suite; verify with ad-hoc `.venv/bin/python -c` / scripts from the repo
root. `test.py` prints a parameter breakdown; `test.ipynb` is the only usage
example (gitignored, and partly stale — it still references a nonexistent
`my-gemma-tokenizer`). `git` 2.56.0 is installed.

## Architecture facts that are easy to get wrong

- `ModelConfig` (dataclass, `components/config.py`) is the single source of
  truth; submodules receive explicit scalars, never the config object.
- Model API is not HF-shaped. `model(input_ids, use_cache)` returns
  `(logits, aux_loss, z_loss)`. `use_cache` is a **required positional arg with
  no default**. There is no `labels` argument and no internal loss; compute
  cross-entropy externally (shift logits/labels by one inside the loss).
- `aux_loss` and `z_loss` from `model.forward` are already averaged over layers;
  don't divide again. In `eval()` both are zero tensors; in `train()` they are
  routed from `TopKRouter`.
- `num_parameters(active_only=True)` estimates active routed params as
  `top_k / num_experts` of total routed-expert params (top_k is exact only when
  `top_k == num_experts`).
- RoPE tables are precomputed for exactly `context_len` positions. Exceeding it
  is **not** a clean `IndexError` on CUDA — the out-of-range gather triggers a
  device-side assert that poisons the CUDA context. Keep every forward's total
  length (prompt + cache) `<= context_len`.
- KV cache (`cache_k`/`cache_v`) is non-persistent: absent from `state_dict()`.
  `use_cache=True` appends and derives positions from cache length, so call
  `model.reset_cache()` between sequences or positions accumulate. The router's
  `expert_bias` buffer **is** persistent (it is in `state_dict()`).

## MoE routing (`components/moe.py`)

- `TopKRouter` uses **sigmoid** scores, not a softmax over all experts. Selection
  uses `scores + expert_bias` (a learnable per-expert bias, "aux-free" balancing);
  the returned weights are `softmax` over only the selected experts, computed from
  the bias-free `scores`.
- The bias is updated in place under `torch.no_grad()` in training via
  `update_expert_bias` (default `bias_update_rate=1e-3`), so training mutates
  buffer state and is not purely a function of weights.
- `aux_loss = num_experts * sum(assignment_fraction * mean_score)`;
  `z_loss` is the mean squared `logsumexp` of router logits in fp32. Both are
  skipped when `not self.training`.
- Dispatch is a Python `for` loop over experts using
  `index_select` / `index_add_` (not a grouped GEMM), so it is slow and
  memory-heavy with 64 experts per block. The output buffer is fp32 and expert
  outputs are cast to it (the `FIX` comment) — required under autocast; keep it.
- Shared experts are always-on (unrouted) and added after routing; empty
  `ModuleList` when `num_shared_experts == 0`.

## Attention (`components/gqa_attn.py`)

- GQA via `F.scaled_dot_product_attention(..., enable_gqa=True)`; requires
  `n_heads % n_kv_heads == 0` and even `head_dim`.
- QK-norm (LigerRMSNorm) is applied per head **before** RoPE. There is also an
  output sigmoid gate (`gate_proj`) applied to attention output.

## packed_dataloader.py

`make_packed_dataloader(dataset, tokenizer, text_column, *, block_size=512, ...)`
accepts an HF Arrow `Dataset` or a `pyarrow.Table` and yields batches of
`(input_ids, labels, attention_mask, data_index)`. The first three are
`[batch, block_size]` with `labels == input_ids`; `data_index` is `[batch]` and
holds each block's source row index (repeated for multi-block rows).

- `append_eos` defaults to `True`: each block is `block_size - 1` row tokens plus
  the tokenizer's EOS in the last slot. Requires `block_size >= 2` and a set
  `tokenizer.eos_token_id` (both raise otherwise). `append_eos=False` restores
  plain `block_size` chunking.
- `num_workers` must be `0` (raises otherwise) because the row cursor is
  sequential and resumable via `start_row_index` (inclusive, so it reprocesses
  that row).
- Rows shorter than one chunk emit nothing; row tails are discarded. Blank and
  whitespace-only rows are skipped before tokenization. There is no padding, so
  every sample is a full block and the attention mask is all ones. Blocks never
  span source rows.

## Data and tokenizer

- `mistral-s4-tokenizer/` is a locally saved fast tokenizer (Mistral Small 4,
  vocab 131072, EOS id 2). It is gitignored.
- `multilingual_mixed.parquet` (gitignored) has one `text` column; load it with
  `load_dataset("parquet", data_files="multilingual_mixed.parquet", split="train")`.
