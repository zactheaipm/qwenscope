"""Probe-guided feature selection for steering.

Trains a linear probe on SAE feature activations to predict behavioral
scores, then uses the probe weight vector as a steering direction.

This handles polysemanticity naturally: instead of selecting individual
features (which may encode multiple concepts), the probe finds the
linear combination of features that best predicts the target behavior.
The probe weight vector IS the steering direction in feature space.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.data.contrastive import BehavioralTrait, ContrastivePair
from src.model.hooks import ActivationCache
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


class LinearProbe:
    """Linear probe from SAE features to behavioral score.

    Fits weights w such that score ≈ w @ features + b, using
    ridge regression (L2-regularized least squares).

    The weight vector w gives the steering direction: features with
    positive w increase the behavioral score, negative w decrease it.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize the probe.

        Args:
            alpha: L2 regularization strength. Higher = more regularization,
                sparser effective weight vector.
        """
        self.alpha = alpha
        self.weights: torch.Tensor | None = None  # (dict_size,)
        self.bias: float = 0.0
        self.r_squared: float = 0.0

    def fit(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        """Fit the linear probe using ridge regression.

        Args:
            features: SAE feature activations, shape (n_samples, dict_size).
            scores: Behavioral scores, shape (n_samples,).
        """
        n, d = features.shape
        assert scores.shape == (n,), f"Expected ({n},), got {scores.shape}"

        # Center scores
        score_mean = scores.mean()
        scores_centered = scores - score_mean

        # Ridge regression closed form: w = (X^T X + αI)^{-1} X^T y
        X = features.float()
        y = scores_centered.float()

        XtX = X.T @ X  # (d, d)
        XtX.diagonal().add_(self.alpha)  # Add regularization
        Xty = X.T @ y  # (d,)

        # Solve via Cholesky (more stable than inverse for large d)
        try:
            L = torch.linalg.cholesky(XtX)
            self.weights = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:
            # Fallback to lstsq if Cholesky fails
            logger.warning("Cholesky failed, falling back to lstsq")
            self.weights = torch.linalg.lstsq(XtX, Xty).solution

        self.bias = float(score_mean.item())

        # Compute R² for diagnostics
        y_pred = X @ self.weights
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = (y ** 2).sum()  # centered
        self.r_squared = float(1 - ss_res / (ss_tot + 1e-8))

        logger.info(
            "Probe fit: R²=%.4f, weight norm=%.4f, "
            "top weight=%.4f, alpha=%.1f",
            self.r_squared,
            self.weights.norm().item(),
            self.weights.abs().max().item(),
            self.alpha,
        )

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict behavioral scores.

        Args:
            features: SAE feature activations, shape (n_samples, dict_size).

        Returns:
            Predicted scores, shape (n_samples,).
        """
        return features.float() @ self.weights + self.bias

    def get_steering_direction(
        self,
        top_k: int = 20,
    ) -> tuple[list[int], torch.Tensor]:
        """Get top-k features and their probe-derived multipliers for steering.

        The probe weight vector naturally gives per-feature importance.
        Positive weights → features that increase the behavioral score.
        Negative weights → features that decrease it.

        For steering, we want to amplify positive-weight features and
        ablate negative-weight features (same logic as TAS-based steering).

        Args:
            top_k: Number of features to include.

        Returns:
            Tuple of (feature_indices, multipliers):
                feature_indices: List of top-k feature indices by |weight|.
                multipliers: Tensor of per-feature multipliers.
                    Positive-weight features get multiplier > 0 (amplify).
                    Negative-weight features get multiplier = 0 (ablate).
        """
        assert self.weights is not None, "Must call fit() first"

        abs_weights = self.weights.abs()
        k = min(top_k, abs_weights.shape[0])
        _, topk_indices = torch.topk(abs_weights, k)
        feature_indices = topk_indices.tolist()

        # Build per-feature multipliers
        # Positive weight → amplify (multiplier scales with weight magnitude)
        # Negative weight → ablate (multiplier = 0)
        multipliers = torch.zeros(k)
        for i, idx in enumerate(feature_indices):
            w = self.weights[idx].item()
            if w > 0:
                # Scale multiplier by relative weight magnitude
                multipliers[i] = 1.0  # will be scaled by steering multiplier
            else:
                multipliers[i] = 0.0  # ablate

        return feature_indices, multipliers


class ProbeGuidedExtractor:
    """Extracts SAE features from model forward passes for probe training.

    Collects (features, score) pairs from contrastive data where the
    "score" is derived from the contrastive label (HIGH=1, LOW=0).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae_dict: dict[str, TopKSAE],
        layer_map: dict[str, int],
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sae_dict = sae_dict
        self.layer_map = layer_map
        self.device = device
        self._sae_dtypes = {
            sae_id: next(sae.parameters()).dtype
            for sae_id, sae in sae_dict.items()
        }

    def _tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def extract_features_from_pairs(
        self,
        pairs: list[ContrastivePair],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Extract SAE features from contrastive pairs.

        For each pair, runs both HIGH and LOW through the model and collects
        last-token SAE features. Labels: HIGH=1.0, LOW=0.0.

        Args:
            pairs: Contrastive pairs.

        Returns:
            Dict: sae_id → (features, labels)
                features: (2 * n_pairs, dict_size)
                labels: (2 * n_pairs,) — 1.0 for HIGH, 0.0 for LOW
        """
        layers = list(set(self.layer_map.values()))
        cache = ActivationCache(self.model, layers=layers)

        accum: dict[str, list[torch.Tensor]] = {
            sae_id: [] for sae_id in self.sae_dict
        }
        labels: list[float] = []

        with torch.no_grad():
            for i, pair in enumerate(pairs):
                if (i + 1) % 10 == 0:
                    logger.info("Probe extraction: %d / %d pairs", i + 1, len(pairs))

                for version, label in [
                    (pair.messages_high, 1.0),
                    (pair.messages_low, 0.0),
                ]:
                    inputs = self._tokenize_messages(version, pair.tools)
                    with cache.active():
                        self.model(**inputs)

                    last_idx = int(inputs["attention_mask"].sum(dim=1).item()) - 1

                    for sae_id, sae in self.sae_dict.items():
                        layer = self.layer_map[sae_id]
                        act = cache.get(layer)[0, last_idx, :]  # (hidden_dim,)
                        sae_dtype = self._sae_dtypes[sae_id]
                        feat = sae.encode(
                            act.unsqueeze(0).to(dtype=sae_dtype)
                        ).squeeze(0).float().cpu()  # (dict_size,)
                        accum[sae_id].append(feat)

                    labels.append(label)
                    cache.clear()

        labels_t = torch.tensor(labels)
        results = {}
        for sae_id in self.sae_dict:
            features = torch.stack(accum[sae_id])  # (2*n_pairs, dict_size)
            results[sae_id] = (features, labels_t)
            logger.info(
                "Probe features for %s: shape=%s, mean_sparsity=%.1f%%",
                sae_id, features.shape,
                100.0 * (features == 0).float().mean().item(),
            )

        return results


def train_probe_for_trait(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[LinearProbe, float, float]:
    """Train and evaluate a linear probe.

    Args:
        features: (n_samples, dict_size).
        labels: (n_samples,) — 0.0 or 1.0.
        alpha: Ridge regression regularization.
        train_frac: Fraction of data for training.
        seed: Random seed for train/test split.

    Returns:
        Tuple of (probe, train_r2, test_r2).
    """
    n = features.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    probe = LinearProbe(alpha=alpha)
    probe.fit(X_train, y_train)

    # Test R²
    y_pred_test = probe.predict(X_test)
    y_test_centered = y_test - y_test.mean()
    ss_res = ((y_test.float() - y_pred_test) ** 2).sum()
    ss_tot = (y_test_centered.float() ** 2).sum()
    test_r2 = float(1 - ss_res / (ss_tot + 1e-8))

    logger.info("Probe: train R²=%.4f, test R²=%.4f", probe.r_squared, test_r2)
    return probe, probe.r_squared, test_r2
