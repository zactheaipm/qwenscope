"""TopK Sparse Autoencoder architecture implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder.

    Architecture:
        encoder: (hidden_dim, dict_size) — projects activations to feature space
        decoder: (dict_size, hidden_dim) — reconstructs from sparse features

    The encoder output is sparsified by keeping only the top-k activations.
    The decoder is initialized as the transpose of the encoder (per Gao et al.).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        dict_size: int = 16384,
        k: int = 64,
    ) -> None:
        """Initialize the TopK SAE.

        Args:
            hidden_dim: Input activation dimension (residual stream width).
            dict_size: Dictionary size (number of features).
            k: Number of top features to keep active.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dict_size = dict_size
        self.k = k

        # Encoder: projects from activation space to feature space
        self.encoder = nn.Linear(hidden_dim, dict_size, bias=True)
        # Decoder: reconstructs from sparse features back to activation space.
        # No bias — the pre_bias already models the activation-space mean; a
        # separate decoder bias would create two redundant learned constants that
        # compete to absorb the same signal and make feature magnitudes uninterpretable.
        self.decoder = nn.Linear(dict_size, hidden_dim, bias=False)

        # Initialize decoder as transpose of encoder (per Gao et al.)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        # Normalize decoder columns to unit norm immediately after init so that
        # feature magnitudes in the encoder are comparable across dictionary entries.
        with torch.no_grad():
            norms = self.decoder.weight.data.norm(dim=0, keepdim=True)
            self.decoder.weight.data /= norms.clamp(min=1e-8)

        # Pre-encoder bias (subtracted before encoding, added back after decoding)
        self.pre_bias = nn.Parameter(torch.zeros(hidden_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature space.

        Args:
            x: Activations of shape (..., hidden_dim).

        Returns:
            Sparse features of shape (..., dict_size) with only k nonzero values.
        """
        sparse_features, _ = self._encode_with_indices(x)
        return sparse_features

    def _encode_with_indices(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode activations and return both sparse features and topk indices.

        Args:
            x: Activations of shape (..., hidden_dim).

        Returns:
            Tuple of (sparse_features, topk_indices):
                sparse_features: (..., dict_size) with only k nonzero values.
                topk_indices: (..., k) indices of the selected features.
        """
        # Subtract pre-bias
        x_centered = x - self.pre_bias  # (..., hidden_dim)

        # Project to feature space
        latents = self.encoder(x_centered)  # (..., dict_size)

        # TopK sparsification: keep only top-k values, zero the rest.
        # Clamp to non-negative: negative feature activations are theoretically
        # unsound for monosemanticity (a "feature" firing negatively means it fires
        # against its direction). ReLU after selection is the standard fix.
        topk_values, topk_indices = torch.topk(latents, self.k, dim=-1)
        topk_values = topk_values.clamp(min=0)

        # Create sparse output
        sparse_features = torch.zeros_like(latents)
        sparse_features.scatter_(-1, topk_indices, topk_values)

        return sparse_features, topk_indices

    def normalize_decoder(self) -> None:
        """Project decoder columns back to unit norm.

        Call after each optimizer step. Ensures feature directions in the
        encoder remain interpretable and comparable across dictionary entries.
        """
        with torch.no_grad():
            norms = self.decoder.weight.data.norm(dim=0, keepdim=True)
            self.decoder.weight.data /= norms.clamp(min=1e-8)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space.

        Args:
            features: Sparse features of shape (..., dict_size).

        Returns:
            Reconstruction of shape (..., hidden_dim).
        """
        # Project back to activation space and add pre-bias
        return self.decoder(features) + self.pre_bias  # (..., hidden_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → sparsify → decode.

        Args:
            x: Activations of shape (..., hidden_dim).

        Returns:
            Tuple of:
                reconstruction: (..., hidden_dim)
                features: (..., dict_size) — sparse
                loss: scalar reconstruction loss (MSE)
        """
        features = self.encode(x)  # (..., dict_size)
        reconstruction = self.decode(features)  # (..., hidden_dim)

        # Reconstruction loss: MSE
        loss = nn.functional.mse_loss(reconstruction, x)

        return reconstruction, features, loss

    def save(self, path: Path) -> None:
        """Save the SAE to disk.

        Saves weights in safetensors format and config as JSON.

        Args:
            path: Directory to save to. Created if it doesn't exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights (ensure contiguous for safetensors)
        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, path / "weights.safetensors")

        # Save config
        config = {
            "hidden_dim": self.hidden_dim,
            "dict_size": self.dict_size,
            "k": self.k,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved SAE to %s", path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> TopKSAE:
        """Load a saved SAE from disk.

        Args:
            path: Directory containing weights.safetensors and config.json.
            device: Device to load weights to.

        Returns:
            Loaded TopKSAE instance.
        """
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Create model
        sae = cls(
            hidden_dim=config["hidden_dim"],
            dict_size=config["dict_size"],
            k=config["k"],
        )

        # Load weights to CPU first, then move to target device
        state_dict = load_file(path / "weights.safetensors", device="cpu")
        sae.load_state_dict(state_dict)
        sae = sae.to(device)

        logger.info("Loaded SAE from %s (hidden_dim=%d, dict_size=%d, k=%d)", path, sae.hidden_dim, sae.dict_size, sae.k)
        return sae
