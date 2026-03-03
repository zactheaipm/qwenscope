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
    ) -> None:
        """Initialize the activation stream.

        Args:
            model: The HuggingFace model.
            tokenizer: The tokenizer.
            layer: Layer index to extract activations from.
            dataset_iter: Iterator yielding tokenized batches with 'input_ids'
                and 'attention_mask' keys.
            batch_size: Model inference batch size.
            max_seq_length: Maximum sequence length for tokenization.
            device: Device for model inference.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.dataset_iter = dataset_iter
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        self._tokens_processed = 0

    @property
    def tokens_processed(self) -> int:
        """Total tokens streamed so far."""
        return self._tokens_processed

    def stream(self) -> Iterator[torch.Tensor]:
        """Yield activation batches of shape (batch_size * seq_len, hidden_dim).

        Flattens batch and sequence dimensions for SAE training.
        Each yielded tensor is a batch of activation vectors ready for
        the SAE's forward pass.
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

                    # Mask out padding positions before flattening
                    # attention_mask: (B, S) → (B, S, 1)
                    mask = attention_mask.unsqueeze(-1).bool()  # (B, S, 1)
                    acts_masked = acts[mask.expand_as(acts)].view(-1, acts.shape[-1])  # (N, D)

                    self._tokens_processed += attention_mask.sum().item()
                    cache.clear()

                # acts_masked is already detached (the cache hook calls .detach()),
                # so requires_grad is False regardless of where this yield sits.
                # The yield placement here is not critical for gradient tracking.
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
                logger.info(
                    "Reached token limit: %d / %d",
                    self._tokens_processed,
                    max_tokens,
                )
                break
