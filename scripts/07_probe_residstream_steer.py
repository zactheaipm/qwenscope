"""Probe-to-residual-stream steering.

Option A (single-layer): Maps probe weight vector through best SAE decoder
to get a dense residual stream steering vector. Uses MeanDiffSteeringEngine.

Option B (multi-layer): Trains probes at ALL 9 SAE layers, maps each through
its decoder, steers at all layers simultaneously.

Tests 3 position conditions × multiple multipliers per trait:
1. Decode-only (current approach)
2. All-positions (prefill + decode)
3. Prefill-only (steer during prefill, skip decode)

Requires GPU (model forward passes + generation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@contextmanager
def multi_engine_active(engines):
    """Context manager that activates multiple MeanDiffSteeringEngines."""
    with ExitStack() as stack:
        for engine in engines:
            stack.enter_context(engine.active())
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe-to-residual-stream steering")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument(
        "--multipliers", type=float, nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Multipliers to test (in activation-norm units)",
    )
    parser.add_argument(
        "--traits", type=str, nargs="*", default=None,
        help="Traits to test (default: all with probes)",
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["all_positions", "prefill_only"],
        help="Position conditions to test (default: all_positions prefill_only)",
    )
    parser.add_argument(
        "--skip-option-b", action="store_true",
        help="Skip multi-layer steering (Option B)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint, skipping completed conditions",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    phase2_dir = results_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.model.config import HOOK_POINTS
    from src.model.loader import load_model
    from src.sae.model import TopKSAE
    from src.data.contrastive import BehavioralTrait, load_contrastive_pairs
    from src.data.scenarios import build_extended_scenarios
    from src.evaluation.agent_harness import AgentHarness
    from src.steering.engine import MeanDiffSteeringEngine
    from src.features.probe import ProbeGuidedExtractor, train_probe_for_trait
    from safetensors.torch import load_file, save_file

    model, tokenizer = load_model(dtype="bfloat16", device=args.device)

    # Load SAEs
    sae_dict = {}
    layer_map = {}
    for hp in HOOK_POINTS:
        sae_path = Path(f"data/saes/{hp.sae_id}")
        if (sae_path / "weights.safetensors").exists():
            sae_dict[hp.sae_id] = TopKSAE.load(sae_path, device=args.device)
            layer_map[hp.sae_id] = hp.layer

    best_sae_per_trait = config.get("steering", {}).get("best_sae_per_trait", {})

    # Build scenarios
    scenarios = build_extended_scenarios()[:args.n_scenarios]
    harness = AgentHarness(model, tokenizer, temperature=0.6, seed=42)

    # Determine which traits to run
    trait_filter = set(args.traits) if args.traits else None

    all_results: dict[str, dict] = {}

    # ================================================================
    # Step 1: Train probes at ALL layers, compute steering vectors
    # ================================================================
    logger.info("=" * 60)
    logger.info("TRAINING PROBES AT ALL SAE LAYERS")
    logger.info("=" * 60)

    # Structure: trait -> sae_id -> {vector, layer, probe_r2, probe_weights}
    all_steering_vectors: dict[str, dict[str, dict]] = {}
    # Also track best single-layer per trait
    best_single_layer: dict[str, dict] = {}

    # We need contrastive pairs and probe features per trait
    # Extract features once, then train probes per SAE
    for trait in BehavioralTrait:
        if trait_filter and trait.value not in trait_filter:
            continue

        try:
            pairs = load_contrastive_pairs(trait)
        except Exception as e:
            logger.warning("No contrastive pairs for %s: %s", trait.value, e)
            continue

        logger.info("--- Probes for %s (%d pairs) ---", trait.value, len(pairs))
        all_steering_vectors[trait.value] = {}

        # Check for cached probes first
        cached_count = 0
        for sae_id, sae in sae_dict.items():
            probe_path = phase2_dir / f"probe_{trait.value}_{sae_id}.safetensors"
            if probe_path.exists():
                cached_count += 1

        # If we have cached probes for best SAE, try to load all cached ones
        # Otherwise, extract features and train fresh
        if cached_count > 0:
            logger.info("  Found %d cached probes, loading those first", cached_count)

        # Extract features for all SAEs (needed for any non-cached probes)
        need_extraction = any(
            not (phase2_dir / f"probe_{trait.value}_{sae_id}.safetensors").exists()
            for sae_id in sae_dict
        )

        probe_features_by_sae = {}
        if need_extraction:
            logger.info("  Extracting probe features for %d SAEs...", len(sae_dict))
            extractor = ProbeGuidedExtractor(
                model, tokenizer, sae_dict, layer_map, device=args.device,
            )
            probe_features_by_sae = extractor.extract_features_from_pairs(pairs)

        best_r2_for_trait = -float("inf")

        for sae_id, sae in sae_dict.items():
            layer = layer_map[sae_id]

            # Try cached probe
            probe_path = phase2_dir / f"probe_{trait.value}_{sae_id}.safetensors"
            if probe_path.exists():
                probe_data = load_file(str(probe_path))
                probe_weights = probe_data["weights"]
                # Load R² if saved (new format), otherwise None (legacy cache)
                if "test_r2" in probe_data:
                    test_r2 = float(probe_data["test_r2"].item())
                else:
                    test_r2 = None
                logger.info("  Loaded cached probe for %s / %s (R²=%s)", trait.value, sae_id,
                            f"{test_r2:.4f}" if test_r2 is not None else "unknown")
            elif sae_id in probe_features_by_sae:
                features, labels = probe_features_by_sae[sae_id]

                # Train with alpha search
                best_probe = None
                best_test_r2 = -float("inf")
                for alpha in [0.01, 0.1, 1.0, 10.0]:
                    probe, train_r2, test_r2_candidate = train_probe_for_trait(
                        features, labels, alpha=alpha,
                    )
                    if test_r2_candidate > best_test_r2:
                        best_test_r2 = test_r2_candidate
                        best_probe = probe

                if best_probe is None or best_probe.weights is None:
                    logger.warning("  Probe training failed for %s / %s", trait.value, sae_id)
                    continue

                probe_weights = best_probe.weights
                test_r2 = best_test_r2

                # Cache the probe (include R² as scalar tensor for later retrieval)
                save_file(
                    {
                        "weights": probe_weights.contiguous(),
                        "test_r2": torch.tensor([test_r2]),
                    },
                    str(probe_path),
                )

                logger.info(
                    "  Probe for %s / %s: test R²=%.4f",
                    trait.value, sae_id, test_r2,
                )
            else:
                continue

            # Map through decoder to residual stream
            with torch.no_grad():
                steering_vec = (
                    sae.decoder.weight.float()
                    @ probe_weights.float().to(sae.decoder.weight.device)
                )

            vec_norm = steering_vec.norm().item()

            all_steering_vectors[trait.value][sae_id] = {
                "vector": steering_vec,
                "layer": layer,
                "sae_id": sae_id,
                "raw_norm": vec_norm,
                "probe_r2": test_r2,
            }

            logger.info(
                "  Steering vector %s / %s: norm=%.4f, layer=%d",
                trait.value, sae_id, vec_norm, layer,
            )

            # Track best single layer by probe R² only.
            # Probes without known R² (legacy cache) are excluded from
            # best-layer selection to avoid comparing R² against norm.
            if test_r2 is not None and test_r2 > best_r2_for_trait:
                best_r2_for_trait = test_r2
                best_single_layer[trait.value] = all_steering_vectors[trait.value][sae_id]

    if not all_steering_vectors:
        logger.error("No steering vectors computed. Exiting.")
        return

    # Fallback: if best_single_layer is missing a trait (all probes were
    # legacy-cached without R²), use best_sae_per_trait from config.
    for trait_name, sae_vecs in all_steering_vectors.items():
        if trait_name not in best_single_layer and sae_vecs:
            fallback_sae = best_sae_per_trait.get(trait_name)
            if fallback_sae and fallback_sae in sae_vecs:
                best_single_layer[trait_name] = sae_vecs[fallback_sae]
                logger.info("  %s: using config fallback SAE %s (no R² available)",
                            trait_name, fallback_sae)
            else:
                # Last resort: pick first available
                first_sae = next(iter(sae_vecs))
                best_single_layer[trait_name] = sae_vecs[first_sae]
                logger.info("  %s: using first available SAE %s (no R² available)",
                            trait_name, first_sae)

    # Log summary
    logger.info("=" * 60)
    logger.info("STEERING VECTOR SUMMARY")
    logger.info("=" * 60)
    for trait_name, sae_vecs in all_steering_vectors.items():
        logger.info("  %s: %d layers", trait_name, len(sae_vecs))
        for sae_id, sv in sae_vecs.items():
            r2_str = f"R²={sv['probe_r2']:.4f}" if sv["probe_r2"] is not None else "R²=cached"
            logger.info("    %s (layer %d): norm=%.4f, %s", sae_id, sv["layer"], sv["raw_norm"], r2_str)
        best = best_single_layer.get(trait_name, {})
        logger.info("    BEST: %s", best.get("sae_id", "none"))

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = results_dir / "07_probe_resid_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)
            prev_results = checkpoint_data.get("results", {})
            for k, v in prev_results.items():
                if k not in all_results:
                    all_results[k] = v
            logger.info("Resumed from checkpoint: %d existing keys", len(prev_results))

    # ================================================================
    # Step 2: Unsteered baseline
    # ================================================================
    if "baseline" in all_results:
        logger.info("UNSTEERED BASELINE: SKIPPED (already in checkpoint)")
    else:
        logger.info("=" * 60)
        logger.info("UNSTEERED BASELINE (%d scenarios)", len(scenarios))
        logger.info("=" * 60)

        baseline_results = []
        for i, scenario in enumerate(scenarios):
            if (i + 1) % 10 == 0:
                logger.info("  Baseline: %d / %d", i + 1, len(scenarios))
            result = harness.run_scenario(scenario)
            baseline_results.append(result.model_dump())

        all_results["baseline"] = {
            "method": "unsteered",
            "n_scenarios": len(scenarios),
            "results": baseline_results,
        }
        _save_checkpoint(results_dir, all_results, args)

    # ================================================================
    # Step 3: OPTION A — Single-layer probe steering (best SAE per trait)
    # ================================================================
    ALL_CONDITIONS = {
        "decode_only": {"steer_all_positions": False, "prefill_only": False},
        "all_positions": {"steer_all_positions": True, "prefill_only": False},
        "prefill_only": {"steer_all_positions": False, "prefill_only": True},
    }
    CONDITIONS = [
        (name, ALL_CONDITIONS[name])
        for name in args.conditions
        if name in ALL_CONDITIONS
    ]

    logger.info("=" * 60)
    logger.info("OPTION A: SINGLE-LAYER PROBE STEERING")
    logger.info("=" * 60)

    for trait_name, sv_data in best_single_layer.items():
        steering_vec = sv_data["vector"]
        layer = sv_data["layer"]

        for multiplier in args.multipliers:
            for condition_name, condition_flags in CONDITIONS:
                key = f"probe_resid_{trait_name}_{condition_name}_mult{multiplier}"

                # Skip already-completed conditions
                if key in all_results:
                    logger.info("--- %s --- SKIPPED (already in checkpoint)", key)
                    continue

                logger.info("--- %s ---", key)

                engine = MeanDiffSteeringEngine(model, layer, steering_vec)
                engine.set_multiplier(multiplier)
                engine.steer_all_positions = condition_flags["steer_all_positions"]
                engine.prefill_only = condition_flags["prefill_only"]

                steered_results = []
                with engine.active():
                    for i, scenario in enumerate(scenarios):
                        if (i + 1) % 10 == 0:
                            logger.info("  %s: %d / %d", key, i + 1, len(scenarios))
                        result = harness.run_scenario(scenario)
                        steered_results.append(result.model_dump())

                all_results[key] = {
                    "method": "probe_residstream_single",
                    "condition": condition_name,
                    "trait": trait_name,
                    "sae_id": sv_data["sae_id"],
                    "layer": layer,
                    "multiplier": multiplier,
                    "raw_vector_norm": sv_data["raw_norm"],
                    "probe_r2": sv_data.get("probe_r2"),
                    "n_scenarios": len(scenarios),
                    "steered_results": steered_results,
                }
                logger.info("  Complete: %d scenarios", len(scenarios))
                _save_checkpoint(results_dir, all_results, args)

    # ================================================================
    # Step 4: OPTION B — Multi-layer probe steering (all SAEs per trait)
    # ================================================================
    if args.skip_option_b:
        logger.info("OPTION B: SKIPPED (--skip-option-b)")
    else:
        logger.info("=" * 60)
        logger.info("OPTION B: MULTI-LAYER PROBE STEERING")
        logger.info("=" * 60)

    for trait_name, sae_vecs in all_steering_vectors.items():
        if args.skip_option_b:
            break
        if len(sae_vecs) < 2:
            logger.info("  %s: only %d layers, skipping multi-layer", trait_name, len(sae_vecs))
            continue

        for multiplier in args.multipliers:
            for condition_name, condition_flags in CONDITIONS:
                key = f"probe_multi_{trait_name}_{condition_name}_mult{multiplier}"
                logger.info("--- %s (%d layers) ---", key, len(sae_vecs))

                # Build one engine per layer
                engines = []
                layer_info = []
                for sae_id, sv_data in sae_vecs.items():
                    engine = MeanDiffSteeringEngine(
                        model, sv_data["layer"], sv_data["vector"],
                    )
                    engine.set_multiplier(multiplier)
                    engine.steer_all_positions = condition_flags["steer_all_positions"]
                    engine.prefill_only = condition_flags["prefill_only"]
                    engines.append(engine)
                    layer_info.append({
                        "sae_id": sae_id,
                        "layer": sv_data["layer"],
                        "raw_norm": sv_data["raw_norm"],
                        "probe_r2": sv_data.get("probe_r2"),
                    })

                steered_results = []
                with multi_engine_active(engines):
                    for i, scenario in enumerate(scenarios):
                        if (i + 1) % 10 == 0:
                            logger.info("  %s: %d / %d", key, i + 1, len(scenarios))
                        result = harness.run_scenario(scenario)
                        steered_results.append(result.model_dump())

                all_results[key] = {
                    "method": "probe_residstream_multi",
                    "condition": condition_name,
                    "trait": trait_name,
                    "multiplier": multiplier,
                    "n_layers": len(engines),
                    "layers": layer_info,
                    "n_scenarios": len(scenarios),
                    "steered_results": steered_results,
                }
                logger.info("  Complete: %d scenarios across %d layers", len(scenarios), len(engines))
                _save_checkpoint(results_dir, all_results, args)

    # ================================================================
    # Save final results
    # ================================================================
    _save_results(results_dir, all_results, args)
    logger.info("Probe-to-residual-stream steering complete!")


def _save_checkpoint(results_dir: Path, all_results: dict, args) -> None:
    """Save intermediate checkpoint."""
    manifest = _build_manifest(args)
    manifest["checkpoint"] = True
    path = results_dir / "07_probe_resid_checkpoint.json"
    with open(path, "w") as f:
        json.dump({"manifest": manifest, "results": all_results}, f, indent=2, default=str)


def _save_results(results_dir: Path, all_results: dict, args) -> None:
    """Save final results."""
    manifest = _build_manifest(args)
    path = results_dir / "07_probe_resid_results.json"
    with open(path, "w") as f:
        json.dump({"manifest": manifest, "results": all_results}, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


def _build_manifest(args) -> dict:
    return {
        "script": "07_probe_residstream_steer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "multipliers": args.multipliers,
        "n_scenarios": args.n_scenarios,
    }


if __name__ == "__main__":
    main()
