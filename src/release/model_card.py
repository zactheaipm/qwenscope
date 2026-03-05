"""Generate HuggingFace model card for SAE release."""

from __future__ import annotations

import logging

from src.model.config import HOOK_POINTS

logger = logging.getLogger(__name__)


def _build_safety_section(redact_steering_data: bool) -> str:
    """Build the Safety and Ethical Considerations section for the model card.

    Args:
        redact_steering_data: Whether steering data has been redacted from
            this release.

    Returns:
        Markdown string for the safety section.
    """
    redaction_note = ""
    if redact_steering_data:
        redaction_note = """
### Redacted Data Notice

This release intentionally excludes pre-computed Trait Association Scores (TAS),
trait-associated feature lists, and steering multiplier recommendations.  These
artifacts could lower the barrier to misuse by enabling targeted behavioral
manipulation of language model agents without requiring the researcher to
understand or reproduce the underlying analysis.

Researchers who wish to replicate our steering results should run their own
contrastive analysis pipeline using the SAE weights provided here together with
the methodology described in our accompanying publication.  This ensures that
anyone performing behavioral steering has engaged with the full analysis process
and its limitations.
"""

    return f"""## Safety and Ethical Considerations

### Dual-Use Nature of Behavioral Steering

Sparse Autoencoders that decompose the residual stream of a language model into
interpretable features are a powerful tool for mechanistic interpretability.
When combined with contrastive analysis, they can identify features associated
with specific behavioral traits (e.g., autonomy, persistence, risk calibration)
and enable targeted behavioral steering through activation intervention.

This capability is inherently dual-use:

- **Beneficial uses** include safety auditing, alignment research, controllable
  agent deployment, red-teaming, and understanding how model behavior emerges
  from internal representations.
- **Potential misuse** includes covertly manipulating agent behavior to bypass
  safety guardrails, amplifying risk-taking or reducing deference in deployed
  systems, or weaponizing steering vectors to produce adversarial agent
  configurations.

### Why Trait-Associated Feature Lists Are Excluded

Publishing ready-to-use feature indices and steering multipliers would
significantly reduce the effort required to perform behavioral steering, making
it accessible without meaningful engagement with the underlying methodology or
its limitations.  By releasing only the trained SAE weights and reconstruction
quality metrics, we preserve the scientific value of this release — the
community can inspect, validate, and build on our SAE training — while
requiring that anyone who wishes to perform behavioral steering invest the
effort to run their own contrastive analysis and understand its assumptions.

This approach follows responsible disclosure practices for dual-use
mechanistic interpretability research.

### Recommendations for Responsible Use

1. **Do not deploy behavioral steering in production** without thorough safety
   evaluation and human oversight.
2. **Report any discovered misuse vectors** through responsible disclosure to
   the authors and the broader AI safety community.
3. **Validate steering effects empirically** — TAS scores identify correlations,
   not guaranteed causal levers.  Steering can have unpredictable side effects
   including cross-trait contamination.
4. **Consider downstream impacts** — behavioral traits interact with task
   context and domain.  Steering that appears beneficial in one domain may be
   harmful in another.
{redaction_note}"""


def generate_model_card(
    quality_metrics: dict[str, dict[str, float]],
    redact_steering_data: bool = True,
) -> str:
    """Generate the HuggingFace model card.

    Args:
        quality_metrics: sae_id → quality metric dict.
        redact_steering_data: When True (default), include a note explaining
            that behavioral steering data (TAS scores, trait-associated
            feature lists, steering multiplier recommendations) has been
            intentionally excluded for responsible disclosure.

    Returns:
        Model card as a markdown string.
    """
    # Build quality table
    quality_rows = []
    for sae_id, metrics in sorted(quality_metrics.items()):
        mse = metrics.get("mse", "N/A")
        ev = metrics.get("explained_variance", "N/A")
        l0 = metrics.get("l0_sparsity", "N/A")
        dead = metrics.get("dead_feature_pct", "N/A")

        mse_str = f"{mse:.4f}" if isinstance(mse, float) else str(mse)
        ev_str = f"{ev:.4f}" if isinstance(ev, float) else str(ev)
        l0_str = f"{l0:.1f}" if isinstance(l0, float) else str(l0)
        dead_str = f"{dead:.1f}%" if isinstance(dead, float) else str(dead)

        quality_rows.append(f"| {sae_id} | {mse_str} | {ev_str} | {l0_str} | {dead_str} |")

    quality_table = "\n".join(quality_rows)

    # Build the safety / ethical considerations section
    safety_section = _build_safety_section(redact_steering_data)

    # Generate hook points table dynamically from HOOK_POINTS
    n_saes = len(HOOK_POINTS)
    hook_rows = []
    for hp in HOOK_POINTS:
        hook_rows.append(
            f"| {hp.sae_id} | {hp.layer} | {hp.layer_type.value.title()} "
            f"| {hp.block} | {hp.description} |"
        )
    hook_table = "\n".join(hook_rows)

    return f"""---
license: mit
tags:
  - sparse-autoencoder
  - mechanistic-interpretability
  - qwen
  - behavioral-steering
language:
  - en
base_model: Qwen/Qwen3.5-27B
---

# Qwen 3.5 Scope — Sparse Autoencoders for Qwen 3.5-27B

The first set of Sparse Autoencoders (SAEs) trained on **Qwen 3.5-27B**, a hybrid
Gated DeltaNet + full attention architecture.

## Model Description

This release contains {n_saes} TopK SAEs trained at different positions in the Qwen 3.5-27B
residual stream, covering both DeltaNet (linear attention) and full attention layers
at early, early-mid, mid, and late depths. Includes a position-in-block control SAE for
isolating layer-type effects from positional confounds.

### Architecture

- **Base model:** Qwen 3.5-27B (64 layers, hybrid DeltaNet + Attention)
- **SAE type:** TopK (k varies by hook point: 64, 96, or 128)
- **Dictionary size:** 20,480 or 40,960 depending on hook point
- **Training methodology:** FAST (sequential instruction-following + tool-use data)
- **Training tokens:** 200M per SAE

### Hook Points

| SAE ID | Layer | Type | Block | Description |
|--------|-------|------|-------|-------------|
{hook_table}

### Reconstruction Quality

| SAE | MSE | Explained Variance | L0 Sparsity | Dead Features |
|-----|-----|-------------------|-------------|---------------|
{quality_table}

## Usage

```python
from safetensors.torch import load_file
import json, torch

# Load SAE
with open("sae_attn_mid/config.json") as f:
    config = json.load(f)
weights = load_file("sae_attn_mid/weights.safetensors")

# The SAE has encoder, decoder, and pre_bias
# Encode: sparse_features = topk(encoder(x - pre_bias))
# Decode: reconstruction = decoder(sparse_features) + pre_bias
```

See `demo.ipynb` for a complete usage example.

## Training Details

- **Training data:** FAST methodology — instruction-following (UltraChat 200k) +
  tool-use conversations, processed sequentially
- **Optimizer:** Adam, lr=5e-5 with warmup
- **Precision:** BF16

## Key Finding: DeltaNet vs Attention

This is the first SAE work on a hybrid architecture. Our analysis reveals that
behavioral traits are encoded differently in DeltaNet and attention layers.
See the accompanying blog post for full results.

{safety_section}

## Citation

```bibtex
@misc{{qwen35scope2026,
    title={{Qwen 3.5 Scope: Sparse Autoencoders for Hybrid DeltaNet-Attention Models}},
    year={{2026}},
    url={{https://huggingface.co/eigen-labs/qwen35-scope}}
}}
```
"""
