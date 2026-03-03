"""Run feature identification pipeline: TAS scoring across all traits/SAEs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify behavioral features")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from src.model.config import HOOK_POINTS, HOOK_POINTS_BY_ID
    from src.model.loader import load_model
    from src.sae.model import TopKSAE
    from src.data.contrastive import BehavioralTrait, load_contrastive_pairs
    from src.features.extraction import FeatureExtractor
    from src.features.scoring import compute_all_tas, compute_all_sub_behavior_tas, rank_features
    from safetensors.torch import save_file

    model, tokenizer = load_model(dtype="bfloat16", device=args.device)

    # Load all SAEs
    sae_dict = {}
    layer_map = {}
    for hp in HOOK_POINTS:
        sae_path = Path(f"data/saes/{hp.sae_id}")
        if (sae_path / "weights.safetensors").exists():
            sae_dict[hp.sae_id] = TopKSAE.load(sae_path, device=args.device)
            layer_map[hp.sae_id] = hp.layer

    if not sae_dict:
        logger.error("No trained SAEs found. Run 03_train_saes.py first.")
        return

    extractor = FeatureExtractor(model, tokenizer, sae_dict, layer_map, device=args.device)

    # Extract features for all traits
    all_extraction_results = {}
    for trait in BehavioralTrait:
        pairs = load_contrastive_pairs(trait)
        logger.info("Extracting features for %s (%d pairs)", trait.value, len(pairs))
        results = extractor.extract_all(pairs, trait)
        all_extraction_results[trait] = results

    # ---- Compute mean raw activations for mean-diff baseline ----
    # These are saved as safetensors for use by script 07's mean-diff baseline.
    # Keys are "{trait}_{sae_id}" -> (hidden_dim,) tensor.
    logger.info("Computing mean raw activations for mean-diff baseline...")
    mean_high_tensors: dict[str, torch.Tensor] = {}
    mean_low_tensors: dict[str, torch.Tensor] = {}

    # Build inverse layer_map: layer -> list of sae_ids at that layer
    layer_to_sae_ids: dict[int, list[str]] = {}
    for sae_id, layer in layer_map.items():
        layer_to_sae_ids.setdefault(layer, []).append(sae_id)

    for trait in BehavioralTrait:
        pairs = load_contrastive_pairs(trait)
        mean_high, mean_low = extractor.compute_mean_activations(pairs)
        for layer, tensor in mean_high.items():
            # Map layer back to sae_id(s) so script 07 can index by sae_id
            for sid in layer_to_sae_ids.get(layer, []):
                mean_high_tensors[f"{trait.value}_{sid}"] = tensor.cpu().contiguous()
        for layer, tensor in mean_low.items():
            for sid in layer_to_sae_ids.get(layer, []):
                mean_low_tensors[f"{trait.value}_{sid}"] = tensor.cpu().contiguous()

    save_file(mean_high_tensors, str(results_dir / "mean_activations_high.safetensors"))
    save_file(mean_low_tensors, str(results_dir / "mean_activations_low.safetensors"))
    logger.info(
        "Saved mean activations: %d trait×layer entries",
        len(mean_high_tensors),
    )

    # Compute TAS
    all_tas = compute_all_tas(all_extraction_results)

    # Save TAS scores
    tas_dir = results_dir / "tas_scores"
    tas_dir.mkdir(parents=True, exist_ok=True)

    for trait, sae_tas in all_tas.items():
        for sae_id, tas in sae_tas.items():
            save_file(
                {"tas": tas},
                str(tas_dir / f"{trait.value}_{sae_id}.safetensors"),
            )

    # Log top features per trait
    for trait in BehavioralTrait:
        logger.info("=== %s ===", trait.value)
        for sae_id in sae_dict:
            if sae_id in all_tas[trait]:
                top = rank_features(all_tas[trait][sae_id], top_k=5)
                logger.info("  %s: top features = %s", sae_id, top)

    # Compute sub-behavior TAS
    logger.info("Computing sub-behavior TAS scores...")
    all_sub_tas = compute_all_sub_behavior_tas(all_extraction_results)

    for sub_behavior, sae_tas in all_sub_tas.items():
        for sae_id, (tas, n_pairs, used_fallback) in sae_tas.items():
            if tas is not None and tas.numel() > 0:
                save_file(
                    {"tas": tas},
                    str(tas_dir / f"sub_{sub_behavior}_{sae_id}.safetensors"),
                )
        logger.info(
            "  %s: computed across %d SAEs",
            sub_behavior, len(sae_tas),
        )

    # ---- Position distribution analysis ----
    # For each trait, find the best SAE (highest mean |TAS| among top-20
    # features) and run position distribution analysis on those features.
    logger.info("Running position distribution analysis...")

    from src.features.interpretability import AutoInterp, PositionDistribution
    from src.features.scoring import rank_features as _rank_features

    position_dir = results_dir / "position_analysis"
    position_dir.mkdir(parents=True, exist_ok=True)

    position_summary: dict[str, dict[str, object]] = {}

    for trait in BehavioralTrait:
        if trait not in all_tas:
            continue

        # Select the best SAE for this trait by mean |TAS| of top-20 features
        best_sae_id: str | None = None
        best_mean_tas: float = -1.0
        for sae_id, tas in all_tas[trait].items():
            if tas.numel() == 0:
                continue
            top_features = _rank_features(tas, top_k=20)
            mean_abs = sum(abs(score) for _, score in top_features) / max(len(top_features), 1)
            if mean_abs > best_mean_tas:
                best_mean_tas = mean_abs
                best_sae_id = sae_id

        if best_sae_id is None:
            logger.warning("No valid SAE found for trait %s, skipping position analysis", trait.value)
            continue

        sae = sae_dict[best_sae_id]
        tas = all_tas[trait][best_sae_id]
        top_features = _rank_features(tas, top_k=20)
        feature_indices = [idx for idx, _ in top_features]
        layer = layer_map[best_sae_id]

        # Build chat_texts from contrastive pairs (both high and low messages)
        pairs = load_contrastive_pairs(trait)
        chat_texts: list[list[dict[str, object]]] = []
        for pair in pairs:
            chat_texts.append(pair.messages_high)
            chat_texts.append(pair.messages_low)

        interp = AutoInterp(model, tokenizer, layer=layer)
        distributions: list[PositionDistribution] = interp.analyze_position_distribution(
            sae=sae,
            feature_indices=feature_indices,
            chat_texts=chat_texts,
        )

        # Count system-dominated features
        n_system_dominated = sum(1 for d in distributions if d.is_system_dominated)

        logger.info(
            "Trait %s (best SAE=%s): %d / %d top features are system-dominated",
            trait.value,
            best_sae_id,
            n_system_dominated,
            len(distributions),
        )

        # Save per-trait position distribution results
        trait_results = {
            "trait": trait.value,
            "best_sae_id": best_sae_id,
            "n_features_analyzed": len(distributions),
            "n_system_dominated": n_system_dominated,
            "distributions": [
                {
                    "feature_idx": d.feature_idx,
                    "total_tokens": d.total_tokens,
                    "system_frac": d.system_frac,
                    "user_frac": d.user_frac,
                    "assistant_frac": d.assistant_frac,
                    "other_frac": d.other_frac,
                    "is_system_dominated": d.is_system_dominated,
                }
                for d in distributions
            ],
        }
        with open(position_dir / f"{trait.value}.json", "w") as f:
            json.dump(trait_results, f, indent=2)

        position_summary[trait.value] = {
            "best_sae_id": best_sae_id,
            "n_system_dominated": n_system_dominated,
            "n_features_analyzed": len(distributions),
        }

    logger.info("Position distribution analysis complete. Results at %s", position_dir)

    manifest = {
        "script": "06_identify_features",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "traits": [t.value for t in BehavioralTrait],
        "saes": list(sae_dict.keys()),
        "sub_behaviors": list(all_sub_tas.keys()),
        "position_analysis": position_summary,
    }
    with open(results_dir / "06_features.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Feature identification complete!")


if __name__ == "__main__":
    main()
