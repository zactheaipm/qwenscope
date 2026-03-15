"""Feature attribution analysis for steering vectors.

Decomposes each steering vector into its constituent SAE features
by examining the probe weights. Since:

    steering_vec = SAE.decoder.weight @ probe.weights

the probe.weights vector literally tells us which SAE features compose
each behavioral direction. Features with large positive weights push
the model toward HIGH trait values; large negative weights push LOW.

Also analyzes:
- Feature overlap between traits (shared vs unique features)
- Cosine similarity between probe vectors in SAE feature space
- Decoder column geometry (what each top feature "means" in residual stream)
- Feature concentration (how many features explain X% of the steering vector)

Does NOT require GPU — loads saved probe weights and SAE decoder only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRAITS = ["autonomy", "tool_use_eagerness", "persistence", "risk_calibration", "deference"]
TOP_K = 20  # Number of top features to report per trait


def load_probe_weights(phase2_dir: Path, trait: str, sae_id: str) -> torch.Tensor | None:
    """Load cached probe weights for a trait/SAE combination."""
    path = phase2_dir / f"probe_{trait}_{sae_id}.safetensors"
    if not path.exists():
        return None
    data = load_file(str(path))
    return data["weights"]


def load_sae_decoder(sae_dir: Path, sae_id: str) -> torch.Tensor | None:
    """Load SAE decoder weight matrix (hidden_dim × dict_size)."""
    path = sae_dir / sae_id / "weights.safetensors"
    if not path.exists():
        return None
    data = load_file(str(path))
    # decoder.weight has shape (hidden_dim, dict_size)
    return data["decoder.weight"]


def analyze_feature_concentration(weights: torch.Tensor) -> dict:
    """Analyze how concentrated the steering signal is across features.

    Returns the number of features needed to explain 50%, 80%, 90%, 95%
    of the total squared weight (i.e., variance explained).
    """
    sorted_sq, _ = torch.sort(weights.float() ** 2, descending=True)
    total = sorted_sq.sum().item()
    if total == 0:
        return {"n_50pct": 0, "n_80pct": 0, "n_90pct": 0, "n_95pct": 0, "gini": 0.0}

    cumsum = torch.cumsum(sorted_sq, dim=0) / total
    thresholds = {"n_50pct": 0.5, "n_80pct": 0.8, "n_90pct": 0.9, "n_95pct": 0.95}
    result = {}
    for name, thresh in thresholds.items():
        idx = (cumsum >= thresh).nonzero(as_tuple=True)[0]
        result[name] = int(idx[0].item()) + 1 if len(idx) > 0 else len(weights)

    # Gini coefficient of squared weights (measures inequality)
    n = len(sorted_sq)
    index = torch.arange(1, n + 1, dtype=torch.float32)
    gini = (2 * (index * sorted_sq).sum() / (n * sorted_sq.sum()) - (n + 1) / n).item()
    result["gini"] = gini

    return result


def main() -> None:
    phase2_dir = Path("data/results/phase2")
    sae_dir = Path("data/saes")
    meta_path = Path("data/results/steering_vectors_meta.json")

    if not meta_path.exists():
        logger.error("No steering vectors metadata at %s", meta_path)
        return

    with open(meta_path) as f:
        meta = json.load(f)

    # Load all probe weights and decoder matrices
    trait_data = {}
    for trait in TRAITS:
        if trait not in meta:
            logger.warning("No metadata for %s", trait)
            continue

        sae_id = meta[trait]["sae_id"]
        probe_w = load_probe_weights(phase2_dir, trait, sae_id)
        decoder_w = load_sae_decoder(sae_dir, sae_id)

        if probe_w is None:
            logger.warning("No probe weights for %s / %s", trait, sae_id)
            continue
        if decoder_w is None:
            logger.warning("No decoder weights for %s / %s", trait, sae_id)
            continue

        trait_data[trait] = {
            "sae_id": sae_id,
            "probe_weights": probe_w.float(),
            "decoder_weight": decoder_w.float(),
            "probe_r2": meta[trait]["probe_r2"],
            "layer": meta[trait]["layer"],
        }

    if not trait_data:
        logger.error("No trait data loaded")
        return

    # ================================================================
    # 1. Top features per trait
    # ================================================================
    print("\n" + "=" * 80)
    print("TOP FEATURES PER STEERING VECTOR")
    print("=" * 80)

    all_top_features = {}  # trait -> list of (feature_idx, weight)

    for trait, data in trait_data.items():
        probe_w = data["probe_weights"]
        decoder_w = data["decoder_weight"]  # (hidden_dim, dict_size)

        # Top features by absolute probe weight
        abs_weights = probe_w.abs()
        top_vals, top_idxs = torch.topk(abs_weights, TOP_K)

        print(f"\n--- {trait} (SAE: {data['sae_id']}, layer {data['layer']}, R²={data['probe_r2']:.3f}) ---")
        print(f"{'Rank':>4}  {'Feature':>8}  {'Weight':>10}  {'|w|':>8}  {'Dec Norm':>9}  {'Contrib':>9}")
        print("-" * 60)

        top_features = []
        for rank, (val, idx) in enumerate(zip(top_vals, top_idxs)):
            idx_int = idx.item()
            w = probe_w[idx_int].item()
            dec_col = decoder_w[:, idx_int]
            dec_norm = dec_col.norm().item()
            # Contribution = weight * decoder_norm (how much this feature moves residual stream)
            contrib = abs(w) * dec_norm

            print(f"{rank+1:>4}  {idx_int:>8}  {w:>+10.5f}  {abs(w):>8.5f}  {dec_norm:>9.4f}  {contrib:>9.5f}")
            top_features.append({"feature_idx": idx_int, "weight": w, "decoder_norm": dec_norm})

        all_top_features[trait] = top_features

    # ================================================================
    # 2. Feature concentration analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("FEATURE CONCENTRATION (how many features explain the direction?)")
    print("=" * 80)

    concentration_results = {}
    dict_sizes = {}
    for trait, data in trait_data.items():
        conc = analyze_feature_concentration(data["probe_weights"])
        concentration_results[trait] = conc
        dict_sizes[trait] = len(data["probe_weights"])
        print(f"\n{trait} ({len(data['probe_weights'])} total features):")
        print(f"  50% of variance: {conc['n_50pct']} features ({100*conc['n_50pct']/len(data['probe_weights']):.1f}%)")
        print(f"  80% of variance: {conc['n_80pct']} features ({100*conc['n_80pct']/len(data['probe_weights']):.1f}%)")
        print(f"  90% of variance: {conc['n_90pct']} features ({100*conc['n_90pct']/len(data['probe_weights']):.1f}%)")
        print(f"  95% of variance: {conc['n_95pct']} features ({100*conc['n_95pct']/len(data['probe_weights']):.1f}%)")
        print(f"  Gini coefficient: {conc['gini']:.4f} (1.0 = all weight on one feature)")

    # ================================================================
    # 3. Feature overlap between traits
    # ================================================================
    print("\n" + "=" * 80)
    print("FEATURE OVERLAP BETWEEN TRAITS")
    print("=" * 80)

    # Group traits by SAE (only traits using the same SAE can share features)
    sae_groups = {}
    for trait, data in trait_data.items():
        sae_id = data["sae_id"]
        if sae_id not in sae_groups:
            sae_groups[sae_id] = []
        sae_groups[sae_id].append(trait)

    print("\nTraits grouped by SAE:")
    for sae_id, group_traits in sae_groups.items():
        print(f"  {sae_id}: {', '.join(group_traits)}")

    # For traits sharing an SAE, compute overlap
    overlap_matrix = {}
    for sae_id, group_traits in sae_groups.items():
        if len(group_traits) < 2:
            continue

        print(f"\n--- Features shared within {sae_id} ---")

        for i, t1 in enumerate(group_traits):
            for t2 in group_traits[i+1:]:
                w1 = trait_data[t1]["probe_weights"]
                w2 = trait_data[t2]["probe_weights"]

                # Top features by abs weight
                _, top1 = torch.topk(w1.abs(), TOP_K)
                _, top2 = torch.topk(w2.abs(), TOP_K)

                set1 = set(top1.tolist())
                set2 = set(top2.tolist())
                shared = set1 & set2
                jaccard = len(shared) / len(set1 | set2) if set1 | set2 else 0

                print(f"\n  {t1} vs {t2}:")
                print(f"    Top-{TOP_K} overlap: {len(shared)} features ({jaccard:.1%} Jaccard)")
                if shared:
                    print(f"    Shared feature indices: {sorted(shared)}")
                    for fidx in sorted(shared):
                        w1_val = w1[fidx].item()
                        w2_val = w2[fidx].item()
                        same_sign = "SAME" if (w1_val > 0) == (w2_val > 0) else "OPPOSITE"
                        print(f"      Feature {fidx}: {t1}={w1_val:+.5f}, {t2}={w2_val:+.5f} ({same_sign} sign)")

                overlap_matrix[f"{t1}_vs_{t2}"] = {
                    "shared_top_k": len(shared),
                    "jaccard": jaccard,
                    "shared_features": sorted(shared),
                }

    # ================================================================
    # 4. Cosine similarity between probe vectors
    # ================================================================
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY BETWEEN PROBE VECTORS")
    print("(Only meaningful for traits using the same SAE)")
    print("=" * 80)

    cosine_matrix = {}
    for sae_id, group_traits in sae_groups.items():
        if len(group_traits) < 2:
            continue

        print(f"\n--- {sae_id} ---")
        for i, t1 in enumerate(group_traits):
            for t2 in group_traits[i+1:]:
                w1 = trait_data[t1]["probe_weights"].float()
                w2 = trait_data[t2]["probe_weights"].float()
                cos_sim = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
                print(f"  {t1} vs {t2}: cos_sim = {cos_sim:.4f}")
                cosine_matrix[f"{t1}_vs_{t2}"] = cos_sim

    # ================================================================
    # 5. Steering vector similarity in residual stream space
    # ================================================================
    print("\n" + "=" * 80)
    print("STEERING VECTOR SIMILARITY IN RESIDUAL STREAM SPACE")
    print("(Can compare across different SAEs since vectors are in same 2048-d space)")
    print("=" * 80)

    # Load steering vectors
    sv_path = Path("data/results/steering_vectors.safetensors")
    if sv_path.exists():
        sv_data = load_file(str(sv_path))
        sv_keys = [k for k in sv_data.keys() if not k.endswith("_probe_weights")]
        print(f"\nLoaded {len(sv_keys)} steering vectors: {sv_keys}")

        for i, k1 in enumerate(sv_keys):
            for k2 in sv_keys[i+1:]:
                v1 = sv_data[k1].float()
                v2 = sv_data[k2].float()
                cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                print(f"  {k1} vs {k2}: cos_sim = {cos_sim:+.4f}")
    else:
        logger.warning("No steering_vectors.safetensors found, computing from probes...")
        # Compute steering vectors from probe weights + decoder
        steering_vecs = {}
        for trait, data in trait_data.items():
            sv = data["decoder_weight"] @ data["probe_weights"]
            steering_vecs[trait] = sv

        traits_list = list(steering_vecs.keys())
        for i, t1 in enumerate(traits_list):
            for t2 in traits_list[i+1:]:
                v1 = steering_vecs[t1]
                v2 = steering_vecs[t2]
                cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                print(f"  {t1} vs {t2}: cos_sim = {cos_sim:+.4f}")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "description": "Feature attribution analysis for SAE-decoded probe steering vectors",
        "top_k": TOP_K,
        "top_features": {
            trait: features for trait, features in all_top_features.items()
        },
        "concentration": concentration_results,
        "feature_overlap": overlap_matrix,
        "cosine_similarity_probe_space": cosine_matrix,
    }

    output_path = Path("data/results/10_feature_attribution.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved feature attribution results to %s", output_path)


if __name__ == "__main__":
    main()
