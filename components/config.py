from dataclasses import dataclass

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistral-s4-tokenizer", use_fast=True)


@dataclass
class ModelConfig:
    """Single source of truth. Every field below is consumed somewhere."""

    # Vocabulary / sequence
    vocab_size: int = tokenizer.vocab_size
    context_len: int = 1024  # RoPE tables cover exactly this many positions

    # Transformer
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int | None = 4  # GQA: None -> n_heads; n_heads % n_kv_heads == 0
    d_ff: int = 256  # hidden size of each routed expert
    bias: bool = False
    rope_theta: float = 10_000.0

    # Mixture of experts
    num_experts: int = 64
    top_k: int = 8
    num_shared_experts: int = 2  # 0 = no shared experts
    shared_d_ff: int | None = None  # per shared expert; None -> d_ff


@dataclass
class TrainingConfig:
    """Single source of truth for the training loop. Every field below is consumed somewhere."""

    # Data
    data_path: str = "F:\\IndicCorpV2_Mixed"
    text_column: str = "text"
    val_size: int = 10000  # first VAL_SIZE rows are held out for validation

    # Batching / schedule
    block_size: int = (
        1024  # must be <= ModelConfig.context_len (RoPE covers exactly context_len)
    )
    batch_size: int = 6
    num_epochs: int = 1
    log_every: int = 100

    # Precision / loss weights
    use_bf16: bool = True  # autocast dtype; master weights stay fp32
    aux_coef: float = 0.01  # weight on the MoE load-balancing loss
    z_coef: float = 0.001  # weight on the router z-loss
    max_grad_norm: float = 1.0

    # Optimizer - placeholder hyperparameters, tune to taste.
    lr_muon: float = 1e-4  # 0.02
    lr_adamw: float = 1e-5  # 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)

    use_lr_scheduler: bool = False
    # LR schedule (cosine with linear warmup, shared by both optimizers)
    num_warmup_steps: int = 5_000
    # None -> count the exact number of steps with one tokenize-only pass over
    # the loader (it streams and has no __len__), then multiply by num_epochs.
    # Set an int to skip that pass.
    num_training_steps: int | None = 100_000

    # Checkpointing (one rolling file: ~3.7 GB with the current model config)
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int | None = 1000  # None disables writing checkpoints
    resume_from: str | None = (
        "checkpoints/ckpt_22000.pt"  # path to a checkpoint to resume from
    )
