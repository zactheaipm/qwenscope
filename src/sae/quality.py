"""Reconstruction quality metrics for trained SAEs."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.hooks import ActivationCache
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


@contextmanager
def _intervention_hook(
    model: Any,
    layer: int,
    modify_fn: Any,
):
    """Context manager that registers a hook to modify activations at a layer.

    The hook replaces the layer's output with the result of ``modify_fn``.

    Args:
        model: The HuggingFace model.
        layer: Layer index to intervene on.
        modify_fn: Callable (hidden_states: Tensor) -> Tensor that transforms
            the residual stream.
    """
    def hook_fn(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified = modify_fn(hidden_states)
            return (modified,) + output[1:]
        return modify_fn(output)

    from src.model.loader import get_layers_module
    layers_module = get_layers_module(model)
    handle = layers_module[layer].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def _next_token_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute mean next-token cross-entropy loss over non-padding positions.

    Args:
        logits: Model output logits of shape (B, S, vocab_size).
        input_ids: Input token IDs of shape (B, S).
        attention_mask: Attention mask of shape (B, S).

    Returns:
        Scalar mean CE loss over valid (non-padding) next-token positions.
    """
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]   # (B, S-1, V)
    shift_labels = input_ids[:, 1:]    # (B, S-1)
    shift_mask = attention_mask[:, 1:] # (B, S-1)

    # Per-token CE loss
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    )  # (B*(S-1),)
    ce = ce.view(shift_labels.shape)   # (B, S-1)

    # Average over non-padding tokens only
    valid = shift_mask.bool()
    if not valid.any():
        return 0.0
    return float(ce[valid].mean().item())


def compute_reconstruction_metrics(
    model: Any,
    sae: TopKSAE,
    layer: int,
    eval_data: DataLoader,
    n_batches: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute MSE, explained variance, L0 sparsity, and loss recovered.

    Loss recovered (Anthropic, 2024) measures how much of the model's
    cross-entropy loss is preserved when SAE reconstructions replace the
    original activations:

        loss_recovered = 1 - (CE_sae - CE_orig) / (CE_zero - CE_orig)

    A value of 1.0 means the SAE reconstruction is functionally lossless.
    A value of 0.0 means it is as bad as zero-ablating the layer.

    Args:
        model: The base language model.
        sae: Trained SAE to evaluate.
        layer: Layer index to extract activations from.
        eval_data: DataLoader yielding tokenized batches.
        n_batches: Number of batches to evaluate.
        device: Device for computation.

    Returns:
        Dict containing:
            mse: Mean squared error between original and reconstruction.
            explained_variance: Fraction of variance explained by reconstruction.
            l0_sparsity: Average number of active features per token.
            dead_features: Number of features that never activate.
            dead_feature_pct: Percentage of dead features.
            ce_loss_original: Cross-entropy loss with no intervention.
            ce_loss_with_sae: Cross-entropy loss with SAE reconstruction.
            ce_loss_zero_ablation: Cross-entropy loss with zero ablation.
            loss_recovered: Fraction of loss recovered (1.0 = perfect).
    """
    sae.eval()
    cache = ActivationCache(model, layers=[layer])

    total_mse = 0.0
    # Per-dimension EV accumulators using sufficient statistics (Chan's formula).
    # Accumulating sum and sum-of-squares avoids the biased weighted-average-of-
    # per-batch-variances approach; global variance is recovered as E[X^2] - E[X]^2.
    total_residual_sum = torch.zeros(sae.hidden_dim)
    total_residual_sq_sum = torch.zeros(sae.hidden_dim)
    total_acts_sum = torch.zeros(sae.hidden_dim)
    total_acts_sq_sum = torch.zeros(sae.hidden_dim)
    total_l0 = 0.0
    total_tokens = 0
    feature_ever_active = torch.zeros(sae.dict_size, dtype=torch.bool)
    # Activation frequency: fraction of tokens each feature is active on.
    # Running sum of per-feature activation counts across all eval tokens.
    feature_activation_counts = torch.zeros(sae.dict_size, dtype=torch.long)

    # Loss recovered accumulators
    total_ce_original = 0.0
    total_ce_sae = 0.0
    total_ce_zero = 0.0
    total_ce_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_data):
            if batch_idx >= n_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # --- Pass 1: normal forward (captures activations + CE_original) ---
            with cache.active():
                output_original = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            ce_orig = _next_token_ce_loss(
                output_original.logits, input_ids, attention_mask
            )

            acts = cache.get(layer)  # (B, S, D)

            # Mask padding for activation-level metrics
            mask = attention_mask.unsqueeze(-1).bool()  # (B, S, 1)
            acts_flat = acts[mask.expand_as(acts)].view(-1, acts.shape[-1])  # (N, D)

            if acts_flat.shape[0] == 0:
                cache.clear()
                continue

            # Run through SAE (offline, for MSE/EV/L0)
            # Cast to SAE dtype (float32) for accurate reconstruction metrics
            sae_dtype = next(sae.parameters()).dtype
            acts_flat_sae = acts_flat.to(sae_dtype)
            reconstruction, features, _ = sae(acts_flat_sae)

            # MSE (computed in SAE dtype for numerical consistency)
            mse = (acts_flat_sae - reconstruction).pow(2).mean().item()
            total_mse += mse * acts_flat_sae.shape[0]

            # Accumulate sufficient statistics for global variance (Chan's formula).
            # Using sum and sum-of-squares avoids the bias of weighted-average-of-
            # per-batch-variances. Global variance = E[X^2] - E[X]^2.
            residual = acts_flat_sae - reconstruction
            total_residual_sum += residual.sum(dim=0).cpu().float()
            total_residual_sq_sum += residual.pow(2).sum(dim=0).cpu().float()
            total_acts_sum += acts_flat_sae.sum(dim=0).cpu().float()
            total_acts_sq_sum += acts_flat_sae.pow(2).sum(dim=0).cpu().float()

            # L0 sparsity
            active_features = (features.abs() > 0)                  # (N, dict_size)
            l0 = active_features.float().sum(dim=-1).mean().item()
            total_l0 += l0 * acts_flat.shape[0]

            # Dead feature tracking and activation frequency counts
            active_mask = active_features.any(dim=0).cpu()          # (dict_size,)
            feature_ever_active |= active_mask
            feature_activation_counts += active_features.sum(dim=0).cpu().long()

            total_tokens += acts_flat.shape[0]

            # --- Pass 2: SAE reconstruction replaces activations ---
            _attn_mask_pass2 = attention_mask  # captured for closure

            def sae_replace(hidden_states: torch.Tensor) -> torch.Tensor:
                orig_shape = hidden_states.shape  # (B, S, D)
                orig_dtype = hidden_states.dtype
                flat = hidden_states.reshape(-1, orig_shape[-1])
                recon, _, _ = sae(flat)
                recon = recon.reshape(orig_shape).to(orig_dtype)
                # Only replace at non-padding positions; keep original at padding.
                pad_mask = _attn_mask_pass2.unsqueeze(-1).bool()  # (B, S, 1)
                return torch.where(pad_mask, recon, hidden_states)

            with _intervention_hook(model, layer, sae_replace):
                output_sae = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            ce_sae = _next_token_ce_loss(
                output_sae.logits, input_ids, attention_mask
            )

            # --- Pass 3: zero ablation ---
            def zero_ablate(hidden_states: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(hidden_states)

            with _intervention_hook(model, layer, zero_ablate):
                output_zero = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            ce_zero = _next_token_ce_loss(
                output_zero.logits, input_ids, attention_mask
            )

            n_valid = int(attention_mask[:, 1:].sum().item())
            total_ce_original += ce_orig * n_valid
            total_ce_sae += ce_sae * n_valid
            total_ce_zero += ce_zero * n_valid
            total_ce_tokens += n_valid

            cache.clear()

    if total_tokens == 0:
        logger.warning("No tokens processed for quality evaluation")
        return {
            "mse": float("inf"),
            "explained_variance": 0.0,
            "l0_sparsity": 0.0,
            "dead_features": sae.dict_size,
            "dead_feature_pct": 100.0,
            "ce_loss_original": float("inf"),
            "ce_loss_with_sae": float("inf"),
            "ce_loss_zero_ablation": float("inf"),
            "loss_recovered": 0.0,
        }

    dead_features = int((~feature_ever_active).sum().item())

    avg_ce_orig = total_ce_original / max(total_ce_tokens, 1)
    avg_ce_sae = total_ce_sae / max(total_ce_tokens, 1)
    avg_ce_zero = total_ce_zero / max(total_ce_tokens, 1)

    ce_degradation = avg_ce_zero - avg_ce_orig
    if ce_degradation > 1e-8:
        raw_loss_recovered = 1.0 - (avg_ce_sae - avg_ce_orig) / ce_degradation
        if raw_loss_recovered < 0.0:
            logger.error(
                "loss_recovered is negative (%.4f): SAE reconstruction raises CE loss "
                "MORE than zeroing the layer entirely. This indicates a catastrophically "
                "broken SAE. CE_orig=%.4f, CE_sae=%.4f, CE_zero=%.4f",
                raw_loss_recovered,
                avg_ce_orig,
                avg_ce_sae,
                avg_ce_zero,
            )
        loss_recovered = max(0.0, min(1.0, raw_loss_recovered))
    else:
        # Layer contributes negligibly to loss; SAE is trivially lossless
        loss_recovered = 1.0

    # Compute global per-dimension variance from sufficient statistics (Chan's formula).
    # global_var = E[X^2] - E[X]^2 = (sq_sum / n) - (sum / n)^2
    n = total_tokens
    if n > 1:
        global_residual_var = (total_residual_sq_sum / n) - (total_residual_sum / n).pow(2)
        global_acts_var = (total_acts_sq_sum / n) - (total_acts_sum / n).pow(2)
        valid_dims = global_acts_var > 1e-8
        if valid_dims.any():
            per_dim_ev = 1.0 - global_residual_var[valid_dims] / global_acts_var[valid_dims]
            explained_variance = float(per_dim_ev.mean().item())
        else:
            explained_variance = 1.0
    else:
        explained_variance = 1.0

    # Activation frequency: fraction of eval tokens each feature fired on.
    # Used to detect power-law collapse (a handful of features dominating).
    activation_freq = feature_activation_counts.float() / max(total_tokens, 1)  # (dict_size,)

    metrics = {
        "mse": total_mse / total_tokens,
        "explained_variance": explained_variance,
        "l0_sparsity": total_l0 / total_tokens,
        "dead_features": dead_features,
        "dead_feature_pct": dead_features / sae.dict_size * 100,
        "ce_loss_original": avg_ce_orig,
        "ce_loss_with_sae": avg_ce_sae,
        "ce_loss_zero_ablation": avg_ce_zero,
        "loss_recovered": loss_recovered,
        # Frequency distribution summary statistics
        "freq_median": float(activation_freq[activation_freq > 0].median().item())
        if (activation_freq > 0).any()
        else 0.0,
        "freq_top1_pct": float(activation_freq.max().item()) * 100,
        "freq_gini": _gini(activation_freq),
    }

    logger.info(
        "Quality metrics: MSE=%.4f, EV=%.4f, L0=%.1f, Dead=%d (%.1f%%), "
        "CE_orig=%.4f, CE_sae=%.4f, CE_zero=%.4f, LossRecovered=%.4f, "
        "FreqGini=%.3f",
        metrics["mse"],
        metrics["explained_variance"],
        metrics["l0_sparsity"],
        metrics["dead_features"],
        metrics["dead_feature_pct"],
        metrics["ce_loss_original"],
        metrics["ce_loss_with_sae"],
        metrics["ce_loss_zero_ablation"],
        metrics["loss_recovered"],
        metrics["freq_gini"],
    )

    # Log activation frequency histogram to WandB if available
    try:
        import wandb
        if wandb.run is not None:
            # Log full distribution and a histogram of log10(freq) for live features
            live_freqs = activation_freq[activation_freq > 0].numpy()
            if len(live_freqs) > 0:
                import numpy as np
                wandb.log({
                    "eval/activation_freq_histogram": wandb.Histogram(
                        np.log10(live_freqs + 1e-10)
                    ),
                    "eval/freq_gini": metrics["freq_gini"],
                    "eval/freq_top1_pct": metrics["freq_top1_pct"],
                })
    except (ImportError, Exception):
        pass

    return metrics


def compute_per_trait_ev(
    model: Any,
    sae: TopKSAE,
    layer: int,
    tokenizer: Any,
    device: str = "cuda",
    max_seq_length: int = 2048,
) -> dict[str, dict[str, float]]:
    """Compute reconstruction EV and MSE broken down by behavioral trait.

    Uses contrastive pair conversations (both HIGH and LOW polarities) as
    trait-specific eval data. Each trait gets ~300+ conversations covering
    4 domains. This reveals whether the SAE reconstructs trait-relevant
    activations as well as general text.

    Only computes activation-level metrics (1 forward pass per batch).
    Does NOT compute loss_recovered (which needs 3 passes) — use
    ``compute_reconstruction_metrics`` for that.

    Args:
        model: The base language model.
        sae: Trained SAE to evaluate.
        layer: Layer index to extract activations from.
        tokenizer: Model tokenizer (for chat template).
        device: Device for computation.
        max_seq_length: Max sequence length for tokenization.

    Returns:
        Dict mapping trait name to {mse, explained_variance, n_tokens}.
    """
    from src.data.contrastive import BehavioralTrait, ContrastivePairGenerator

    sae.eval()
    cache = ActivationCache(model, layers=[layer])
    generator = ContrastivePairGenerator()
    all_pairs = generator.generate_all()

    results: dict[str, dict[str, float]] = {}

    for trait in BehavioralTrait:
        pairs = all_pairs[trait]

        # Tokenize all conversations (both high and low) for this trait
        tokenized_batches: list[dict[str, torch.Tensor]] = []
        for pair in pairs:
            for messages in (pair.messages_high, pair.messages_low):
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tools=pair.tools if pair.tools else None,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    text = tokenizer.apply_chat_template(
                        [{"role": m["role"], "content": m.get("content", "")} for m in messages],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                encoded = tokenizer(
                    text,
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokenized_batches.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                })

        if not tokenized_batches:
            results[trait.value] = {"mse": float("inf"), "explained_variance": 0.0, "n_tokens": 0}
            continue

        # Accumulate sufficient statistics across all conversations
        residual_sum = torch.zeros(sae.hidden_dim)
        residual_sq_sum = torch.zeros(sae.hidden_dim)
        acts_sum = torch.zeros(sae.hidden_dim)
        acts_sq_sum = torch.zeros(sae.hidden_dim)
        total_mse = 0.0
        total_tokens = 0

        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(tokenized_batches), batch_size):
                chunk = tokenized_batches[i : i + batch_size]
                input_ids = torch.stack([c["input_ids"] for c in chunk]).to(device)
                attention_mask = torch.stack([c["attention_mask"] for c in chunk]).to(device)

                with cache.active():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                acts = cache.get(layer)
                mask = attention_mask.unsqueeze(-1).bool()
                acts_flat = acts[mask.expand_as(acts)].view(-1, acts.shape[-1])

                if acts_flat.shape[0] == 0:
                    cache.clear()
                    continue

                reconstruction, _, _ = sae(acts_flat)

                residual = acts_flat - reconstruction
                total_mse += residual.pow(2).mean().item() * acts_flat.shape[0]
                residual_sum += residual.sum(dim=0).cpu()
                residual_sq_sum += residual.pow(2).sum(dim=0).cpu()
                acts_sum += acts_flat.sum(dim=0).cpu()
                acts_sq_sum += acts_flat.pow(2).sum(dim=0).cpu()
                total_tokens += acts_flat.shape[0]

                cache.clear()

        if total_tokens > 1:
            global_residual_var = (residual_sq_sum / total_tokens) - (residual_sum / total_tokens).pow(2)
            global_acts_var = (acts_sq_sum / total_tokens) - (acts_sum / total_tokens).pow(2)
            valid_dims = global_acts_var > 1e-8
            if valid_dims.any():
                per_dim_ev = 1.0 - global_residual_var[valid_dims] / global_acts_var[valid_dims]
                ev = float(per_dim_ev.mean().item())
            else:
                ev = 1.0
        else:
            ev = 1.0

        results[trait.value] = {
            "mse": total_mse / max(total_tokens, 1),
            "explained_variance": ev,
            "n_tokens": total_tokens,
        }
        logger.info(
            "Per-trait EV [%s]: EV=%.4f, MSE=%.4f, tokens=%d",
            trait.value, ev, results[trait.value]["mse"], total_tokens,
        )

    return results


def _gini(freqs: torch.Tensor) -> float:
    """Compute Gini coefficient of a feature frequency distribution.

    A Gini of 0 means all features fire equally often (uniform).
    A Gini near 1 means one feature dominates (power-law collapse).

    Args:
        freqs: Activation frequency tensor of shape (dict_size,).

    Returns:
        Gini coefficient in [0, 1].
    """
    f = freqs.float()
    n = f.shape[0]
    if n == 0 or f.sum() < 1e-12:
        return 0.0
    f_sorted = f.sort().values
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float((2 * (idx * f_sorted).sum() / (n * f_sorted.sum()) - (n + 1) / n).item())
