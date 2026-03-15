"""Risk_calibration dissociation analysis.

The puzzle: risk_calibration and tool_use_eagerness use the SAME SAE
(sae_delta_mid_pos1, layer 21) with nearly identical probe R²
(0.795 vs 0.792), but tool_use steers successfully while risk_cal
only suppresses behavior.

Hypothesis: risk_cal's high R² is "borrowed" from its correlation
with tool_use in the training data. The risk_cal probe direction is
mostly parallel to the tool_use direction, with a weak orthogonal
residual that doesn't causally drive behavior.

This script:
1. Projects risk_cal probe vector onto tool_use probe vector
2. Decomposes into parallel and orthogonal components
3. Measures what fraction of risk_cal's variance is explained by tool_use
4. Analyzes the orthogonal component's properties
5. Compares decoder-mapped vectors in residual stream space

Does NOT require GPU.
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


def project(v: torch.Tensor, onto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project v onto the direction of `onto`. Returns (parallel, orthogonal)."""
    onto_unit = onto / onto.norm()
    parallel_magnitude = torch.dot(v, onto_unit)
    parallel = parallel_magnitude * onto_unit
    orthogonal = v - parallel
    return parallel, orthogonal


def main() -> None:
    phase2_dir = Path("data/results/phase2")
    sae_dir = Path("data/saes")
    meta_path = Path("data/results/steering_vectors_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    # Verify they share the same SAE
    rc_sae = meta["risk_calibration"]["sae_id"]
    tu_sae = meta["tool_use_eagerness"]["sae_id"]
    print(f"Risk calibration SAE: {rc_sae} (layer {meta['risk_calibration']['layer']})")
    print(f"Tool use eagerness SAE: {tu_sae} (layer {meta['tool_use_eagerness']['layer']})")

    if rc_sae != tu_sae:
        print("\nWARNING: Different SAEs — analysis still valid in residual stream space")

    # Load probe weights
    rc_probe = load_file(str(phase2_dir / f"probe_risk_calibration_{rc_sae}.safetensors"))
    tu_probe = load_file(str(phase2_dir / f"probe_tool_use_eagerness_{tu_sae}.safetensors"))

    rc_w = rc_probe["weights"].float()
    tu_w = tu_probe["weights"].float()

    rc_r2 = float(rc_probe["test_r2"].item()) if "test_r2" in rc_probe else meta["risk_calibration"]["probe_r2"]
    tu_r2 = float(tu_probe["test_r2"].item()) if "test_r2" in tu_probe else meta["tool_use_eagerness"]["probe_r2"]

    print(f"\nRisk calibration probe R² = {rc_r2:.4f}")
    print(f"Tool use eagerness probe R² = {tu_r2:.4f}")

    # ================================================================
    # 1. Probe vector geometry in SAE feature space
    # ================================================================
    print("\n" + "=" * 80)
    print("PROBE VECTOR GEOMETRY (SAE feature space, dim={})".format(len(rc_w)))
    print("=" * 80)

    cos_sim = torch.nn.functional.cosine_similarity(rc_w.unsqueeze(0), tu_w.unsqueeze(0)).item()
    print(f"\nCosine similarity (risk_cal, tool_use): {cos_sim:.4f}")
    print(f"  -> {abs(cos_sim):.1%} directional overlap")

    # Project risk_cal onto tool_use
    rc_parallel, rc_orthogonal = project(rc_w, tu_w)

    rc_total_var = (rc_w ** 2).sum().item()
    parallel_var = (rc_parallel ** 2).sum().item()
    ortho_var = (rc_orthogonal ** 2).sum().item()

    print(f"\nProjection of risk_cal onto tool_use direction:")
    print(f"  ||risk_cal||² = {rc_total_var:.4f}")
    print(f"  ||parallel||² = {parallel_var:.4f}  ({100*parallel_var/rc_total_var:.1f}% of total)")
    print(f"  ||orthogonal||² = {ortho_var:.4f}  ({100*ortho_var/rc_total_var:.1f}% of total)")
    print(f"  Parallel component magnitude: {rc_parallel.norm().item():.4f}")
    print(f"  Orthogonal component magnitude: {rc_orthogonal.norm().item():.4f}")

    # Also project tool_use onto risk_cal
    tu_parallel, tu_orthogonal = project(tu_w, rc_w)
    tu_total_var = (tu_w ** 2).sum().item()
    tu_parallel_var = (tu_parallel ** 2).sum().item()

    print(f"\nProjection of tool_use onto risk_cal direction:")
    print(f"  ||parallel||² / ||total||² = {100*tu_parallel_var/tu_total_var:.1f}%")
    print(f"  -> tool_use has {100*(1 - tu_parallel_var/tu_total_var):.1f}% unique variance")

    # ================================================================
    # 2. Decoder-mapped analysis (residual stream space)
    # ================================================================
    print("\n" + "=" * 80)
    print("STEERING VECTOR GEOMETRY (residual stream space, dim=2048)")
    print("=" * 80)

    # Load decoder
    if rc_sae == tu_sae:
        dec_data = load_file(str(sae_dir / rc_sae / "weights.safetensors"))
        decoder_w = dec_data["decoder.weight"].float()  # (2048, dict_size)

        rc_steer = decoder_w @ rc_w
        tu_steer = decoder_w @ tu_w

        # Also decode the parallel and orthogonal components
        rc_steer_parallel = decoder_w @ rc_parallel
        rc_steer_ortho = decoder_w @ rc_orthogonal
    else:
        # Different SAEs — load both decoders
        dec_rc = load_file(str(sae_dir / rc_sae / "weights.safetensors"))["decoder.weight"].float()
        dec_tu = load_file(str(sae_dir / tu_sae / "weights.safetensors"))["decoder.weight"].float()
        rc_steer = dec_rc @ rc_w
        tu_steer = dec_tu @ tu_w
        rc_steer_parallel = dec_rc @ rc_parallel
        rc_steer_ortho = dec_rc @ rc_orthogonal

    cos_sim_resid = torch.nn.functional.cosine_similarity(
        rc_steer.unsqueeze(0), tu_steer.unsqueeze(0)
    ).item()

    print(f"\nCosine similarity in residual stream: {cos_sim_resid:.4f}")
    print(f"  (vs {cos_sim:.4f} in SAE feature space)")

    rc_resid_norm = rc_steer.norm().item()
    tu_resid_norm = tu_steer.norm().item()
    rc_par_resid_norm = rc_steer_parallel.norm().item()
    rc_ort_resid_norm = rc_steer_ortho.norm().item()

    print(f"\nSteering vector norms:")
    print(f"  risk_cal: {rc_resid_norm:.4f}")
    print(f"  tool_use: {tu_resid_norm:.4f}")
    print(f"  risk_cal (parallel to tool_use): {rc_par_resid_norm:.4f}")
    print(f"  risk_cal (orthogonal to tool_use): {rc_ort_resid_norm:.4f}")

    # What fraction of risk_cal's steering POWER is parallel to tool_use?
    par_frac_resid = rc_par_resid_norm**2 / rc_resid_norm**2
    print(f"\nFraction of risk_cal's steering power parallel to tool_use:")
    print(f"  In SAE feature space: {parallel_var/rc_total_var:.1%}")
    print(f"  In residual stream:   {par_frac_resid:.1%}")

    # ================================================================
    # 3. Test: does the orthogonal component look "degenerate"?
    # ================================================================
    print("\n" + "=" * 80)
    print("ORTHOGONAL COMPONENT ANALYSIS")
    print("=" * 80)

    # How many features dominate the orthogonal component?
    ortho_abs = rc_orthogonal.abs()
    sorted_ortho, sorted_idx = torch.sort(ortho_abs, descending=True)

    cumvar_ortho = torch.cumsum(sorted_ortho ** 2, dim=0) / (rc_orthogonal ** 2).sum()
    n_50 = (cumvar_ortho >= 0.5).nonzero(as_tuple=True)[0][0].item() + 1
    n_80 = (cumvar_ortho >= 0.8).nonzero(as_tuple=True)[0][0].item() + 1
    n_90 = (cumvar_ortho >= 0.9).nonzero(as_tuple=True)[0][0].item() + 1

    print(f"\nOrthogonal component concentration:")
    print(f"  50% of variance in {n_50} features")
    print(f"  80% of variance in {n_80} features")
    print(f"  90% of variance in {n_90} features")

    # Compare with tool_use concentration
    tu_abs = tu_w.abs()
    sorted_tu, _ = torch.sort(tu_abs, descending=True)
    cumvar_tu = torch.cumsum(sorted_tu ** 2, dim=0) / (tu_w ** 2).sum()
    tu_n50 = (cumvar_tu >= 0.5).nonzero(as_tuple=True)[0][0].item() + 1
    tu_n80 = (cumvar_tu >= 0.8).nonzero(as_tuple=True)[0][0].item() + 1

    print(f"\nFor comparison, tool_use (which works) concentration:")
    print(f"  50% of variance in {tu_n50} features")
    print(f"  80% of variance in {tu_n80} features")

    # ================================================================
    # 4. Cross-check with other trait pairs
    # ================================================================
    print("\n" + "=" * 80)
    print("CROSS-CHECK: ALL TRAIT-PAIR PROJECTIONS")
    print("(Is risk_cal/tool_use unusually aligned compared to other pairs?)")
    print("=" * 80)

    # Load all probes for traits that share SAEs
    all_probes = {}
    for trait in ["autonomy", "tool_use_eagerness", "persistence", "risk_calibration", "deference"]:
        sae_id = meta[trait]["sae_id"]
        probe_path = phase2_dir / f"probe_{trait}_{sae_id}.safetensors"
        if probe_path.exists():
            all_probes[trait] = {
                "sae_id": sae_id,
                "weights": load_file(str(probe_path))["weights"].float(),
            }

    # For pairs sharing the same SAE, compute cosine similarity
    traits_list = list(all_probes.keys())
    print(f"\n{'Trait 1':<22} {'Trait 2':<22} {'Same SAE':>8} {'Cos Sim':>8}")
    print("-" * 65)
    for i, t1 in enumerate(traits_list):
        for t2 in traits_list[i+1:]:
            same_sae = all_probes[t1]["sae_id"] == all_probes[t2]["sae_id"]
            if same_sae:
                w1 = all_probes[t1]["weights"]
                w2 = all_probes[t2]["weights"]
                cs = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
                print(f"{t1:<22} {t2:<22} {'YES':>8} {cs:>+8.4f}")

    # Also compute residual stream cosine similarities (these work across SAEs)
    print(f"\n{'Trait 1':<22} {'Trait 2':<22} {'Cos Sim (resid)':>16}")
    print("-" * 65)

    sv_path = Path("data/results/steering_vectors.safetensors")
    if sv_path.exists():
        sv_data = load_file(str(sv_path))
        sv_keys = [k for k in sv_data.keys() if not k.endswith("_probe_weights")]
        for i, k1 in enumerate(sv_keys):
            for k2 in sv_keys[i+1:]:
                v1 = sv_data[k1].float()
                v2 = sv_data[k2].float()
                cs = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                print(f"{k1:<22} {k2:<22} {cs:>+16.4f}")

    # ================================================================
    # 5. Interpretation
    # ================================================================
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(f"""
The risk_calibration probe direction has {parallel_var/rc_total_var:.0%} of its variance
parallel to the tool_use direction in SAE feature space (cosine sim = {cos_sim:.3f}).

This means the probe found a direction that is largely a "shadow" of the
tool_use direction. The high R² ({rc_r2:.3f}) is likely because risk-taking
and tool-use eagerness are behaviorally correlated in the training data:
agents that take more risks also tend to use more tools.

When we map through the decoder to residual stream space, the parallel
fraction is {par_frac_resid:.0%} (cosine sim = {cos_sim_resid:.3f}).

The orthogonal component — the part unique to risk_calibration — has norm
{rc_ort_resid_norm:.2f} in residual stream (vs {tu_resid_norm:.2f} for full tool_use vector).
This is the component that SHOULD drive risk-specific behavior, but it's
{"too weak to overcome the noise floor" if rc_ort_resid_norm < tu_resid_norm * 0.5 else "present but may not be causally upstream of risk decisions"}.

Conclusion: R² measures correlation in the data distribution, not causal
influence on model computation. The risk_cal probe succeeded at PREDICTION
but the direction it found is mostly the tool_use direction in disguise.
""")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "description": "Risk calibration vs tool_use_eagerness dissociation analysis",
        "shared_sae": rc_sae == tu_sae,
        "sae_id": rc_sae,
        "probe_r2": {"risk_calibration": rc_r2, "tool_use_eagerness": tu_r2},
        "cosine_similarity": {
            "sae_feature_space": cos_sim,
            "residual_stream": cos_sim_resid,
        },
        "risk_cal_projection_onto_tool_use": {
            "parallel_variance_fraction_sae": parallel_var / rc_total_var,
            "parallel_variance_fraction_resid": par_frac_resid,
            "orthogonal_norm_resid": rc_ort_resid_norm,
            "tool_use_norm_resid": tu_resid_norm,
        },
        "orthogonal_concentration": {
            "n_features_50pct": n_50,
            "n_features_80pct": n_80,
            "n_features_90pct": n_90,
        },
    }

    output_path = Path("data/results/11_risk_cal_dissociation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved dissociation analysis to %s", output_path)


if __name__ == "__main__":
    main()
