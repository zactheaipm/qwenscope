"""Quick end-to-end pilot: 1 trait, 1 SAE, minimal scale.

Validates the full pipeline on:
- 1 SAE (sae_attn_mid, layer 35)
- 1 trait (autonomy)
- 5 contrastive pairs (not 80)
- 3 scenarios (not 20)
- 1 steering multiplier (5×)

If this works, the full experiment is just scaling up.
"""

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
    parser = argparse.ArgumentParser(description="Run end-to-end pilot")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from src.model.config import HOOK_POINTS_BY_ID
    from src.model.loader import load_model
    from src.sae.model import TopKSAE
    from src.sae.config import SAETrainingConfig
    from src.sae.trainer import SAETrainer
    from src.sae.activations import ActivationStream
    from src.data.contrastive import BehavioralTrait, ContrastivePairGenerator
    from src.data.scenarios import build_default_scenarios
    from src.features.extraction import FeatureExtractor
    from src.features.scoring import compute_tas, rank_features
    from src.steering.engine import SteeringEngine
    from src.evaluation.agent_harness import AgentHarness

    pilot_sae_id = "sae_attn_mid"
    pilot_trait = BehavioralTrait.AUTONOMY
    pilot_layer = HOOK_POINTS_BY_ID[pilot_sae_id].layer

    logger.info("=== PILOT: %s at layer %d for %s ===", pilot_sae_id, pilot_layer, pilot_trait.value)

    # Step 1: Load model
    logger.info("Step 1: Loading model...")
    model, tokenizer = load_model(dtype="bfloat16", device=args.device)

    # Step 2: Train a small SAE (reduced tokens)
    logger.info("Step 2: Training pilot SAE (reduced scale)...")
    sae_path = Path(f"data/saes/{pilot_sae_id}")
    if (sae_path / "weights.safetensors").exists():
        logger.info("  Loading existing SAE from %s", sae_path)
        sae = TopKSAE.load(sae_path, device=args.device)
    else:
        # Train with minimal tokens for pilot
        from src.data.training_data import SAETrainingDataBuilder

        config = SAETrainingConfig(
            layer=pilot_layer,
            sae_id=pilot_sae_id,
            training_tokens=1_000_000,  # 1M tokens for pilot (not 200M)
            checkpoint_every_tokens=500_000,
        )
        data_builder = SAETrainingDataBuilder(tokenizer, config)
        dataset = data_builder.build_dataset()

        def batch_iterator(data_iter, batch_size=8):
            batch = []
            for item in data_iter:
                batch.append(item)
                if len(batch) == batch_size:
                    yield {
                        "input_ids": torch.stack([b["input_ids"] for b in batch]),
                        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                    }
                    batch = []

        stream = ActivationStream(
            model=model, tokenizer=tokenizer, layer=pilot_layer,
            dataset_iter=batch_iterator(iter(dataset)), device=args.device,
        )

        sae = TopKSAE(hidden_dim=5120, dict_size=40960, k=64).to(args.device)
        trainer = SAETrainer(sae, config)
        sae = trainer.train(stream)
        sae.save(sae_path)
        logger.info("  Pilot SAE trained and saved")

    # Step 3: Generate contrastive pairs (5 only)
    logger.info("Step 3: Generating 5 contrastive pairs...")
    generator = ContrastivePairGenerator()
    all_pairs = generator.generate_all()
    pilot_pairs = all_pairs[pilot_trait][:5]
    logger.info("  Using %d pairs for %s", len(pilot_pairs), pilot_trait.value)

    # Step 4: Extract features and compute TAS
    logger.info("Step 4: Feature extraction and TAS computation...")
    sae_dict = {pilot_sae_id: sae}
    layer_map = {pilot_sae_id: pilot_layer}
    extractor = FeatureExtractor(model, tokenizer, sae_dict, layer_map, device=args.device)
    extraction_results = extractor.extract_all(pilot_pairs, pilot_trait)
    tas = compute_tas(extraction_results, pilot_trait, pilot_sae_id)
    top_features = rank_features(tas, top_k=20)
    logger.info("  Top 5 features by TAS: %s", top_features[:5])

    # Step 5: Steering
    logger.info("Step 5: Steering experiment...")
    engine = SteeringEngine(model, sae, pilot_layer)
    feature_indices = [idx for idx, _ in top_features]
    engine.set_steering(feature_indices, multiplier=5.0)

    scenarios = build_default_scenarios()[:3]
    harness = AgentHarness(model, tokenizer, temperature=0.0, seed=42)

    for scenario in scenarios:
        # Baseline
        baseline_traj = harness.run_scenario(scenario)
        logger.info("  Baseline %s: %d turns, terminated by %s",
                     scenario.id, baseline_traj.num_turns, baseline_traj.terminated_by)

        # Steered
        harness.steering_engine = engine
        steered_traj = harness.run_scenario(scenario)
        logger.info("  Steered %s: %d turns, terminated by %s",
                     scenario.id, steered_traj.num_turns, steered_traj.terminated_by)
        harness.steering_engine = None

    # Write manifest
    manifest = {
        "script": "run_pilot",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pilot_sae_id": pilot_sae_id,
        "pilot_layer": pilot_layer,
        "pilot_trait": pilot_trait.value,
        "num_pairs": len(pilot_pairs),
        "num_scenarios": len(scenarios),
        "top_features": top_features[:5],
        "status": "SUCCESS",
    }
    with open(results_dir / "pilot.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=== PILOT COMPLETE ===")
    logger.info("All pipeline components verified!")


if __name__ == "__main__":
    main()
