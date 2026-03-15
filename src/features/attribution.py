"""Logit attribution for SAE features.

Instead of ranking features by activation difference (TAS), this module
ranks them by their causal contribution to the model's output logit
difference between HIGH and LOW behavioral contexts.

The key insight: TAS measures correlation (features that activate
differently), but logit attribution measures causation (features that
change what the model says). A feature might activate differently in
HIGH vs LOW contexts but project onto a direction in residual stream
space that doesn't affect behaviorally-relevant output tokens.

Attribution score for feature f at layer L:
    attr_f = (a_HIGH_f - a_LOW_f) * ||W_unembed @ d_f||

Where:
    a_HIGH_f, a_LOW_f = feature activation in HIGH/LOW contexts
    d_f = decoder column for feature f (unit-normalized)
    W_unembed = model's unembedding matrix (lm_head)

This combines activation difference (like TAS) with a directional signal:
does this feature's decoder direction point toward tokens that matter?
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.data.contrastive import BehavioralTrait, ContrastivePair
from src.features.extraction import (
    FeatureExtractionResults,
    FeatureExtractor,
    FeaturePairResult,
)
from src.model.hooks import ActivationCache
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


def compute_decoder_logit_norms(
    sae: TopKSAE,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ||W_unembed @ d_f|| for each SAE feature.

    This measures how much each feature's decoder direction projects onto
    the output logit space. Features with high logit norm have more
    influence over the model's token predictions.

    Args:
        sae: The sparse autoencoder.
        lm_head_weight: The unembedding matrix, shape (vocab_size, hidden_dim).

    Returns:
        Tensor of shape (dict_size,) — L2 norm of each feature's logit vector.
    """
    # decoder.weight shape: (hidden_dim, dict_size)
    # lm_head_weight shape: (vocab_size, hidden_dim)
    # Result: (vocab_size, dict_size)
    with torch.no_grad():
        decoder_w = sae.decoder.weight.float()
        lm_head_w = lm_head_weight.float()
        feature_logit_vecs = lm_head_w @ decoder_w  # (vocab_size, dict_size)
        norms = feature_logit_vecs.norm(dim=0)  # (dict_size,)
    return norms


def compute_directed_logit_attribution(
    sae: TopKSAE,
    lm_head_weight: torch.Tensor,
    acts_high: torch.Tensor,
    acts_low: torch.Tensor,
    logits_high: torch.Tensor,
    logits_low: torch.Tensor,
    top_logit_k: int = 50,
) -> torch.Tensor:
    """Compute directed logit attribution for each SAE feature.

    For each contrastive pair, identifies the tokens with the largest
    logit difference (the "behavioral tokens"), then measures how much
    each SAE feature's decoder direction contributes to those specific
    logit differences.

    This is more targeted than the norm-based approach: it doesn't just
    ask "does this feature affect logits?" but "does this feature affect
    the specific logits that differ between HIGH and LOW behavior?"

    Args:
        sae: The sparse autoencoder.
        lm_head_weight: Unembedding matrix, shape (vocab_size, hidden_dim).
        acts_high: Residual stream activations for HIGH, shape (hidden_dim,).
        acts_low: Residual stream activations for LOW, shape (hidden_dim,).
        logits_high: Model logits for HIGH, shape (vocab_size,).
        logits_low: Model logits for LOW, shape (vocab_size,).
        top_logit_k: Number of top differing tokens to consider.

    Returns:
        Tensor of shape (dict_size,) — directed attribution per feature.
    """
    with torch.no_grad():
        # 1. Identify behavioral tokens: tokens with largest logit difference
        logit_diff = logits_high.float() - logits_low.float()  # (vocab_size,)
        _, top_token_idx = torch.topk(logit_diff.abs(), top_logit_k)

        # 2. Build behavioral direction: normalized logit diff restricted to top tokens
        behavioral_dir = torch.zeros_like(logit_diff)
        behavioral_dir[top_token_idx] = logit_diff[top_token_idx]
        behavioral_dir = behavioral_dir / (behavioral_dir.norm() + 1e-8)  # (vocab_size,)

        # 3. Compute feature activations
        sae_dtype = next(sae.parameters()).dtype
        features_high = sae.encode(acts_high.unsqueeze(0).to(dtype=sae_dtype)).squeeze(0)  # (dict_size,)
        features_low = sae.encode(acts_low.unsqueeze(0).to(dtype=sae_dtype)).squeeze(0)

        act_diff = (features_high - features_low).float()  # (dict_size,)

        # 4. Compute each feature's projection onto behavioral direction
        # feature_logit_proj[f] = behavioral_dir @ (lm_head @ decoder[:, f])
        decoder_w = sae.decoder.weight.float()  # (hidden_dim, dict_size)
        lm_head_w = lm_head_weight.float()  # (vocab_size, hidden_dim)
        feature_logit_vecs = lm_head_w @ decoder_w  # (vocab_size, dict_size)
        feature_behavioral_proj = behavioral_dir @ feature_logit_vecs  # (dict_size,)

        # 5. Attribution = activation_difference × behavioral_projection
        attribution = act_diff * feature_behavioral_proj  # (dict_size,)

    return attribution


class LogitAttributionExtractor:
    """Extracts logit attribution scores for SAE features across contrastive pairs.

    Extends FeatureExtractor's pattern but also computes attribution through
    the unembedding matrix. For each contrastive pair, produces both:
    - Standard feature activation differences (like TAS)
    - Logit-attributed feature importance scores
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

        # Get lm_head weight
        if hasattr(model, "lm_head"):
            self.lm_head_weight = model.lm_head.weight.detach()  # (vocab_size, hidden_dim)
        else:
            raise RuntimeError("Cannot find lm_head on model")

        # Pre-compute decoder logit norms for norm-based attribution
        self.decoder_logit_norms: dict[str, torch.Tensor] = {}
        for sae_id, sae in sae_dict.items():
            self.decoder_logit_norms[sae_id] = compute_decoder_logit_norms(
                sae, self.lm_head_weight
            )
            logger.info(
                "Decoder logit norms for %s: mean=%.3f, max=%.3f",
                sae_id,
                self.decoder_logit_norms[sae_id].mean().item(),
                self.decoder_logit_norms[sae_id].max().item(),
            )

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

    def extract_pair_attribution(
        self,
        pair: ContrastivePair,
        top_logit_k: int = 50,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Extract both norm-based and directed logit attribution for one pair.

        Args:
            pair: Contrastive pair.
            top_logit_k: Number of top differing logit tokens for directed attribution.

        Returns:
            Dict: sae_id → {
                "norm_attr": (dict_size,) — |act_diff| × logit_norm
                "directed_attr": (dict_size,) — act_diff × behavioral_projection
                "act_diff": (dict_size,) — raw activation difference (for TAS comparison)
            }
        """
        layers = list(set(self.layer_map.values()))
        cache = ActivationCache(self.model, layers=layers)
        results: dict[str, dict[str, torch.Tensor]] = {}

        with torch.no_grad():
            # Forward pass HIGH
            inputs_high = self._tokenize_messages(pair.messages_high, pair.tools)
            with cache.active():
                output_high = self.model(**inputs_high)
            logits_high = output_high.logits[0, -1, :]  # last token logits (vocab_size,)
            high_acts = {layer: cache.get(layer).clone() for layer in layers}
            cache.clear()

            # Forward pass LOW
            inputs_low = self._tokenize_messages(pair.messages_low, pair.tools)
            with cache.active():
                output_low = self.model(**inputs_low)
            logits_low = output_low.logits[0, -1, :]
            low_acts = {layer: cache.get(layer).clone() for layer in layers}
            cache.clear()

            # Compute attribution for each SAE
            for sae_id, sae in self.sae_dict.items():
                layer = self.layer_map[sae_id]
                sae_dtype = self._sae_dtypes[sae_id]

                # Get last-token activations
                last_idx_high = inputs_high["attention_mask"].sum(dim=1).item() - 1
                last_idx_low = inputs_low["attention_mask"].sum(dim=1).item() - 1
                act_high = high_acts[layer][0, int(last_idx_high), :]  # (hidden_dim,)
                act_low = low_acts[layer][0, int(last_idx_low), :]

                # Encode through SAE
                feat_high = sae.encode(act_high.unsqueeze(0).to(dtype=sae_dtype)).squeeze(0)
                feat_low = sae.encode(act_low.unsqueeze(0).to(dtype=sae_dtype)).squeeze(0)
                act_diff = (feat_high - feat_low).float()  # (dict_size,)

                # Norm-based attribution: |activation_diff| × decoder_logit_norm
                norm_attr = act_diff.abs() * self.decoder_logit_norms[sae_id].to(act_diff.device)

                # Directed attribution
                directed_attr = compute_directed_logit_attribution(
                    sae, self.lm_head_weight,
                    act_high, act_low,
                    logits_high, logits_low,
                    top_logit_k=top_logit_k,
                )

                results[sae_id] = {
                    "norm_attr": norm_attr.cpu(),
                    "directed_attr": directed_attr.cpu(),
                    "act_diff": act_diff.cpu(),
                }

        return results

    def extract_all_attribution(
        self,
        pairs: list[ContrastivePair],
        trait: BehavioralTrait,
        top_logit_k: int = 50,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Compute mean attribution scores across all contrastive pairs.

        Args:
            pairs: List of contrastive pairs.
            trait: Behavioral trait.
            top_logit_k: Number of top logit tokens for directed attribution.

        Returns:
            Dict: sae_id → {
                "mean_norm_attr": (dict_size,) — mean norm-based attribution
                "mean_directed_attr": (dict_size,) — mean directed attribution
                "mean_act_diff": (dict_size,) — mean activation diff (=TAS numerator)
                "std_directed_attr": (dict_size,) — std for significance testing
            }
        """
        accumulators: dict[str, dict[str, list[torch.Tensor]]] = {
            sae_id: {"norm_attr": [], "directed_attr": [], "act_diff": []}
            for sae_id in self.sae_dict
        }

        for i, pair in enumerate(pairs):
            if (i + 1) % 10 == 0:
                logger.info(
                    "Attribution extraction: %d / %d pairs", i + 1, len(pairs)
                )

            pair_results = self.extract_pair_attribution(pair, top_logit_k)
            for sae_id, attrs in pair_results.items():
                for key in ("norm_attr", "directed_attr", "act_diff"):
                    accumulators[sae_id][key].append(attrs[key])

        # Aggregate
        results: dict[str, dict[str, torch.Tensor]] = {}
        for sae_id in self.sae_dict:
            acc = accumulators[sae_id]
            norm_stack = torch.stack(acc["norm_attr"])  # (n_pairs, dict_size)
            dir_stack = torch.stack(acc["directed_attr"])
            diff_stack = torch.stack(acc["act_diff"])

            results[sae_id] = {
                "mean_norm_attr": norm_stack.mean(dim=0),
                "mean_directed_attr": dir_stack.mean(dim=0),
                "mean_act_diff": diff_stack.mean(dim=0),
                "std_directed_attr": dir_stack.std(dim=0, correction=1),
            }

            logger.info(
                "Attribution for %s / %s: norm_attr max=%.3f, "
                "directed_attr max=%.3f, act_diff max=%.3f",
                trait.value, sae_id,
                results[sae_id]["mean_norm_attr"].max().item(),
                results[sae_id]["mean_directed_attr"].abs().max().item(),
                results[sae_id]["mean_act_diff"].abs().max().item(),
            )

        return results


def compute_attribution_tas(
    attribution_results: dict[str, dict[str, torch.Tensor]],
    sae_id: str,
    method: str = "directed",
) -> torch.Tensor:
    """Compute attribution-based TAS (analogous to activation TAS).

    For directed attribution: TAS = mean / std (like Cohen's d).
    For norm attribution: just use mean (always positive).

    Args:
        attribution_results: Output of extract_all_attribution.
        sae_id: Which SAE.
        method: "directed" or "norm".

    Returns:
        Tensor of shape (dict_size,) — attribution-TAS per feature.
    """
    result = attribution_results[sae_id]

    if method == "directed":
        mean = result["mean_directed_attr"]
        std = result["std_directed_attr"]
        tas = torch.where(
            std > 1e-8,
            mean / std,
            torch.zeros_like(mean),
        )
        return tas
    elif method == "norm":
        return result["mean_norm_attr"]
    else:
        raise ValueError(f"Unknown method: {method}")


def rank_by_attribution(
    attribution_results: dict[str, dict[str, torch.Tensor]],
    sae_id: str,
    top_k: int = 20,
    method: str = "directed",
) -> list[tuple[int, float]]:
    """Rank features by attribution score.

    Args:
        attribution_results: Output of extract_all_attribution.
        sae_id: Which SAE.
        top_k: Number of top features.
        method: "directed" or "norm".

    Returns:
        List of (feature_index, attribution_score) sorted by |score| descending.
    """
    tas = compute_attribution_tas(attribution_results, sae_id, method)
    abs_tas = tas.abs()
    k = min(top_k, abs_tas.shape[0])
    topk_values, topk_indices = torch.topk(abs_tas, k)
    return [
        (int(idx.item()), float(tas[idx].item()))
        for idx, val in zip(topk_indices, topk_values)
    ]
