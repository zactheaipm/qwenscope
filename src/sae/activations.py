"""Streaming activation extraction for SAE training.

Activations are streamed through the SAE in batches during training —
we never store the full 200M tokens of activations to disk.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch

from src.model.hooks import ActivationCache

logger = logging.getLogger(__name__)


class ActivationStream:
    """Streams activations from the model through a callback.

    Designed for SAE training: extracts activations at one layer,
    streams them in batches to the SAE trainer, never stores the full set.

    When the dataset produces ``document_ids`` (from sequence packing),
    tokens near document boundaries are excluded from the activation
    stream to mitigate cross-document state leakage — particularly
    important for DeltaNet layers where recurrent state carries
    information across document boundaries.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer: int,
        dataset_iter: Iterator[dict[str, torch.Tensor]],
        batch_size: int = 32,
        max_seq_length: int = 2048,
        device: str = "cuda",
        boundary_margin: int = 2,
    ) -> None:
        """Initialize the activation stream.

        Args:
            model: The HuggingFace model.
            tokenizer: The tokenizer.
            layer: Layer index to extract activations from.
            dataset_iter: Iterator yielding tokenized batches with 'input_ids'
                and 'attention_mask' keys, and optionally 'document_ids'.
            batch_size: Model inference batch size.
            max_seq_length: Maximum sequence length for tokenization.
            device: Device for model inference.
            boundary_margin: Number of tokens to exclude on each side of a
                document boundary within packed sequences.  Tokens within this
                margin are the most contaminated by cross-document attention
                leakage and DeltaNet recurrent state bleed.  Set to 0 to
                disable boundary masking.  Default 2 removes the EOS separator
                and the 1 token on each side, which is the minimum to avoid
                the strongest contamination artifacts.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.dataset_iter = dataset_iter
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        self.boundary_margin = boundary_margin
        self._tokens_processed = 0
        self._tokens_excluded_boundary = 0

    @property
    def tokens_processed(self) -> int:
        """Total tokens streamed so far."""
        return self._tokens_processed

    @property
    def tokens_excluded_boundary(self) -> int:
        """Total tokens excluded due to document boundary proximity."""
        return self._tokens_excluded_boundary

    def _build_boundary_mask(self, document_ids: torch.Tensor) -> torch.Tensor:
        """Build a boolean mask that excludes tokens near document boundaries.

        A document boundary occurs where ``document_ids[i] != document_ids[i+1]``
        (transition between documents in a packed sequence).  Tokens within
        ``boundary_margin`` positions of any boundary are masked out.

        Args:
            document_ids: (B, S) tensor of per-token document indices.
                Values of -1 indicate padding (already handled by attention_mask).

        Returns:
            (B, S) boolean tensor where True = keep, False = exclude.
        """
        B, S = document_ids.shape
        if self.boundary_margin <= 0:
            return torch.ones(B, S, dtype=torch.bool, device=document_ids.device)

        # Detect boundary positions: where adjacent tokens have different doc IDs
        # (excluding padding-to-padding transitions where both are -1)
        shifted = torch.cat([document_ids[:, :1], document_ids[:, :-1]], dim=1)
        is_boundary = (document_ids != shifted) & (document_ids >= 0) & (shifted >= 0)
        # is_boundary[b, i] is True at the first token of each new document

        # Expand boundaries by margin on each side
        boundary_zone = is_boundary.clone()
        for offset in range(1, self.boundary_margin + 1):
            # Shift left (tokens BEFORE the boundary)
            if offset < S:
                padded = torch.cat([
                    is_boundary[:, offset:],
                    torch.zeros(B, offset, dtype=torch.bool, device=is_boundary.device),
                ], dim=1)
                boundary_zone |= padded
            # Shift right (tokens AFTER the boundary)
            if offset < S:
                padded = torch.cat([
                    torch.zeros(B, offset, dtype=torch.bool, device=is_boundary.device),
                    is_boundary[:, :-offset],
                ], dim=1)
                boundary_zone |= padded

        return ~boundary_zone

    def stream(self) -> Iterator[torch.Tensor]:
        """Yield activation batches of shape (batch_size * seq_len, hidden_dim).

        Flattens batch and sequence dimensions for SAE training.
        Each yielded tensor is a batch of activation vectors ready for
        the SAE's forward pass.

        When ``document_ids`` are present in the batch (from sequence packing),
        tokens near document boundaries are excluded to mitigate cross-document
        contamination in DeltaNet recurrent state.
        """
        cache = ActivationCache(self.model, layers=[self.layer])

        # Register hooks once for the entire stream, not per-batch.
        # With 200M tokens and batch_size=4096 this avoids ~48K unnecessary
        # hook register/remove cycles.
        with cache.active():
            for batch in self.dataset_iter:
                input_ids = batch["input_ids"].to(self.device)  # (B, S)
                attention_mask = batch["attention_mask"].to(self.device)  # (B, S)

                with torch.no_grad():
                    self.model(input_ids=input_ids, attention_mask=attention_mask)

                    acts = cache.get(self.layer)  # (B, S, D)

                    # Start with attention mask (excludes padding)
                    mask = attention_mask.bool()  # (B, S)

                    # If document_ids present, also exclude boundary tokens
                    if "document_ids" in batch and self.boundary_margin > 0:
                        doc_ids = batch["document_ids"].to(self.device)
                        boundary_keep = self._build_boundary_mask(doc_ids)
                        n_boundary_excluded = int(
                            (mask & ~boundary_keep).sum().item()
                        )
                        self._tokens_excluded_boundary += n_boundary_excluded
                        mask = mask & boundary_keep

                    # Mask and flatten: (B, S, D) → (N, D)
                    mask_3d = mask.unsqueeze(-1)  # (B, S, 1)
                    acts_masked = acts[mask_3d.expand_as(acts)].view(-1, acts.shape[-1])

                    self._tokens_processed += int(mask.sum().item())
                    cache.clear()

                yield acts_masked

    def stream_tokens(self, max_tokens: int) -> Iterator[torch.Tensor]:
        """Stream activations up to a maximum token count.

        Args:
            max_tokens: Stop streaming after this many tokens.

        Yields:
            Activation batches of shape (N, hidden_dim).
        """
        for acts in self.stream():
            yield acts
            if self._tokens_processed >= max_tokens:
                if self._tokens_excluded_boundary > 0:
                    pct = 100 * self._tokens_excluded_boundary / (
                        self._tokens_processed + self._tokens_excluded_boundary
                    )
                    logger.info(
                        "Boundary masking: excluded %d tokens (%.1f%%) near "
                        "document boundaries in packed sequences",
                        self._tokens_excluded_boundary,
                        pct,
                    )
                logger.info(
                    "Reached token limit: %d / %d",
                    self._tokens_processed,
                    max_tokens,
                )
                break
