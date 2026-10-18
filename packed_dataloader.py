"""Tokenize Arrow-backed text rows on demand for causal-LM pretraining.

Rows are read sequentially, blank rows are skipped, and each row is split into
non-overlapping full blocks. Incomplete row tails are discarded. When
``append_eos`` is enabled, each block carries one fewer row token so the
tokenizer's EOS token can take the final slot of every emitted block.
"""

from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral
from typing import Any

import pyarrow as pa
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, IterableDataset


def _normalize_token_rows(tokenized: Any, expected_rows: int) -> list[list[int]]:
    """Normalize common tokenizer output containers to list-of-lists."""
    if isinstance(tokenized, torch.Tensor) or hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    else:
        tokenized = list(tokenized)

    # Accommodate tokenizers that return a flat list for a one-row batch.
    if expected_rows == 1 and tokenized and isinstance(tokenized[0], Integral):
        tokenized = [tokenized]

    if len(tokenized) != expected_rows:
        raise ValueError(
            "Tokenizer returned a different number of rows than it received."
        )
    return [[int(token_id) for token_id in row] for row in tokenized]


class _ArrowTextBlocks(IterableDataset):
    """Stream full token blocks from an Arrow-backed text column."""

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        text_column: str,
        block_size: int,
        tokenization_batch_size: int,
        add_special_tokens: bool,
        append_eos: bool,
        start_row_index: int,
    ) -> None:
        super().__init__()
        if isinstance(dataset, HFDataset):
            if text_column not in dataset.column_names:
                raise ValueError(
                    f"Column {text_column!r} was not found. Available columns: "
                    f"{dataset.column_names}."
                )
            # Keep only the requested column and don't alter the caller's format.
            self.source = dataset.select_columns([text_column]).with_format(None)
            self._is_hf_dataset = True
        elif isinstance(dataset, pa.Table):
            if text_column not in dataset.column_names:
                raise ValueError(
                    f"Column {text_column!r} was not found. Available columns: "
                    f"{dataset.column_names}."
                )
            self.source = dataset.select([text_column])
            self._is_hf_dataset = False
        else:
            raise TypeError(
                "dataset must be a Hugging Face datasets.Dataset or a "
                f"pyarrow.Table; got {type(dataset).__name__}."
            )

        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        # One block slot is reserved for the trailing EOS, so the row content
        # carried by each block is one token shorter when append_eos is on.
        self.content_size = block_size - 1
        self.tokenization_batch_size = tokenization_batch_size
        self.add_special_tokens = add_special_tokens
        self.append_eos = append_eos
        self.start_row_index = start_row_index
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if append_eos and self.eos_token_id is None:
            raise ValueError("append_eos=True, but tokenizer.eos_token_id is not set.")
        if append_eos and block_size < 2:
            raise ValueError(
                "append_eos=True requires block_size >= 2 so each block can hold "
                f"at least one content token plus EOS; got block_size={block_size}."
            )

    def _read_text_batch(self, start: int, end: int) -> list[Any]:
        if self._is_hf_dataset:
            return self.source[start:end][self.text_column]
        return (
            self.source.slice(start, end - start).column(self.text_column).to_pylist()
        )

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for batch_start in range(
            self.start_row_index, len(self.source), self.tokenization_batch_size
        ):
            batch_end = min(
                batch_start + self.tokenization_batch_size, len(self.source)
            )
            raw_texts = self._read_text_batch(batch_start, batch_end)

            texts: list[str] = []
            row_indices: list[int] = []
            for offset, text in enumerate(raw_texts):
                if text is None:
                    continue
                if not isinstance(text, str):
                    raise TypeError(
                        f"Expected strings in column {self.text_column!r}; row "
                        f"{batch_start + offset} has type {type(text).__name__}."
                    )
                if text.strip():
                    texts.append(text)
                    row_indices.append(batch_start + offset)

            # Filtering happens before tokenization, so empty/whitespace rows
            # never reach the tokenizer.
            if not texts:
                continue

            encoded = self.tokenizer(
                texts,
                add_special_tokens=self.add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            tokenized_rows = _normalize_token_rows(
                encoded["input_ids"], expected_rows=len(texts)
            )

            # When append_eos is on, each block is (block_size - 1) row tokens
            # followed by EOS, so every emitted sample ends with EOS. Otherwise
            # rows are split into plain block_size chunks with no added EOS.
            chunk_size = self.content_size if self.append_eos else self.block_size
            eos_suffix = [int(self.eos_token_id)] if self.append_eos else []

            for row_index, ids in zip(row_indices, tokenized_rows):
                if not ids:
                    continue

                complete_length = len(ids) // chunk_size * chunk_size
                for offset in range(0, complete_length, chunk_size):
                    tokens = ids[offset : offset + chunk_size] + eos_suffix
                    input_ids = torch.tensor(tokens, dtype=torch.long)
                    labels = input_ids.clone()
                    attention_mask = torch.ones_like(input_ids)
                    data_index = torch.tensor(row_index, dtype=torch.long)
                    yield input_ids, labels, attention_mask, data_index


def make_packed_dataloader(
    dataset: Any,
    tokenizer: Any,
    text_column: str,
    *,
    block_size: int = 512,
    batch_size: int = 8,
    tokenization_batch_size: int = 256,
    add_special_tokens: bool = True,
    append_eos: bool = True,
    start_row_index: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a sequential, on-the-fly DataLoader of packed token blocks.

    Each source row is tokenized independently. Complete non-overlapping
    chunks are yielded; rows shorter than a chunk and leftover tokens at the end
    of longer rows are discarded. Blank and whitespace-only rows are skipped
    before tokenization. No blocks are created by combining tokens from
    different source rows.

    With ``append_eos=True`` (the default), a chunk is ``block_size - 1`` row
    tokens and the block's final position is always the tokenizer's EOS token,
    so every emitted sample ends with EOS. With ``append_eos=False``, no EOS is
    added and chunks are the full ``block_size`` tokens.

    The returned batch is a tuple in this order:
    ``(input_ids, labels, attention_mask, data_index)``. The first three tensors
    have shape ``[batch_size, block_size]`` except the final batch when
    ``drop_last=False``. ``data_index`` has shape ``[batch_size]`` and contains
    the source row index for each block (repeated if a row yields multiple
    blocks). Since every yielded sample is a full block, the attention mask is
    all ones and empty DataLoader batches are not produced.

    ``start_row_index`` is inclusive. Resuming from a saved row index therefore
    reprocesses that row, which may repeat some blocks but avoids starting from
    the beginning. For this sequential row cursor, use ``num_workers=0``.

    Args:
        dataset: A Hugging Face Arrow ``Dataset`` (including one formatted with
            ``format="arrow"``) or a ``pyarrow.Table``.
        tokenizer: A Hugging Face tokenizer accepting a batch of strings.
        text_column: Name of the source text column.
        block_size: Number of tokens in each emitted block. When ``append_eos``
            is true this must be at least 2, since one slot holds the EOS token.
        batch_size: Number of emitted token blocks per training batch.
        tokenization_batch_size: Number of source rows tokenized together at a
            time. This limits temporary preprocessing memory.
        add_special_tokens: Whether to use the tokenizer's configured special
            token behavior for each source row.
        append_eos: If true (default), each ``block_size`` block is filled with
            ``block_size - 1`` row tokens and the tokenizer's EOS token in the
            last position. Rows with fewer than ``block_size - 1`` tokens emit
            no blocks. If false, rows are split into plain ``block_size`` chunks
            with no added EOS.
        start_row_index: Source row to begin from, inclusive. Defaults to 0.
        num_workers: Must be 0 to preserve sequential, resumable row indices.
        pin_memory: Whether DataLoader should pin returned tensors.
        drop_last: Whether to drop the final batch if it contains fewer than
            ``batch_size`` token blocks.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if append_eos and block_size < 2:
        raise ValueError(
            "append_eos=True requires block_size >= 2 so each block can hold "
            "at least one content token plus EOS."
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(tokenization_batch_size, int) or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be a positive integer.")
    if not isinstance(start_row_index, int) or isinstance(start_row_index, bool):
        raise TypeError("start_row_index must be an integer.")
    if start_row_index < 0:
        raise ValueError("start_row_index must be non-negative.")
    if num_workers != 0:
        raise ValueError(
            "num_workers must be 0 when using sequential data_index resume."
        )

    block_dataset = _ArrowTextBlocks(
        dataset=dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        block_size=block_size,
        tokenization_batch_size=tokenization_batch_size,
        add_special_tokens=add_special_tokens,
        append_eos=append_eos,
        start_row_index=start_row_index,
    )
    if start_row_index > len(block_dataset.source):
        raise ValueError(
            f"start_row_index ({start_row_index}) exceeds dataset length "
            f"({len(block_dataset.source)})."
        )
    return DataLoader(
        block_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = ["make_packed_dataloader"]
