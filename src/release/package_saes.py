"""Package trained SAEs for HuggingFace release.

The public HuggingFace release
includes SAE weights and reconstruction quality metrics but does NOT include
pre-computed TAS scores, trait-associated feature lists, or steering multiplier
recommendations by default. This redaction can be overridden for internal use
by setting ``redact_steering_data=False``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.model.config import HOOK_POINTS
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


def package_for_huggingface(
    sae_dir: Path,
    output_dir: Path,
    quality_metrics: dict[str, dict[str, float]],
    feature_descriptions: dict[str, dict[int, str]] | None = None,
    redact_steering_data: bool = True,
    tas_scores: dict[str, dict[str, Any]] | None = None,
    steering_config: dict[str, Any] | None = None,
) -> None:
    """Package SAEs for HuggingFace upload.

    Creates:
    - README.md (model card)
    - config.json (architecture + training config per hook point)
    - sae_<type>_<depth>/ directories (safetensors + feature descriptions)
    - demo.ipynb (Colab notebook)

    When ``redact_steering_data`` is True (the default), the package
    intentionally excludes TAS scores, trait-associated feature lists, and
    steering multiplier recommendations per the responsible-disclosure policy.
    The model card will include a note explaining why this data is absent.

    When ``redact_steering_data`` is False, the provided ``tas_scores`` and
    ``steering_config`` will be written into the package.  This is intended
    only for internal/private releases.

    Args:
        sae_dir: Directory containing trained SAE subdirectories.
        output_dir: Output directory for HF upload.
        quality_metrics: sae_id → quality metric dict.
        feature_descriptions: Optional sae_id → {feature_idx: description}.
        redact_steering_data: When True (default), exclude TAS scores,
            trait-associated feature lists, and steering multiplier
            recommendations from the release package.
        tas_scores: Optional mapping of sae_id → trait → TAS data.  Only
            included in the package when ``redact_steering_data`` is False.
        steering_config: Optional dict containing steering multiplier
            recommendations and feature lists.  Only included in the package
            when ``redact_steering_data`` is False.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-hook-point config by reading each SAE's actual checkpoint config.
    # This avoids hardcoding dict_size/topk which differ across hook points
    # (e.g., early SAEs use dict_size=20480, topk=128; earlymid uses topk=96).
    hook_point_configs = {}
    for hp in HOOK_POINTS:
        hp_config: dict[str, Any] = {
            "layer": hp.layer,
            "layer_type": hp.layer_type.value,
            "block": hp.block,
            "description": hp.description,
        }
        # Read actual hyperparameters from SAE checkpoint config
        sae_config_path = sae_dir / hp.sae_id / "config.json"
        if sae_config_path.exists():
            with open(sae_config_path) as cf:
                sae_cfg = json.load(cf)
            hp_config["dict_size"] = sae_cfg.get("dict_size", sae_cfg.get("dictionary_size"))
            hp_config["topk"] = sae_cfg.get("k", sae_cfg.get("topk"))
            if hp_config["dict_size"] is None:
                logger.warning(
                    "SAE config for %s has no 'dict_size' or 'dictionary_size' key — "
                    "config.json will have null. Check %s",
                    hp.sae_id, sae_config_path,
                )
            if hp_config["topk"] is None:
                logger.warning(
                    "SAE config for %s has no 'k' or 'topk' key — "
                    "config.json will have null. Check %s",
                    hp.sae_id, sae_config_path,
                )
        hook_point_configs[hp.sae_id] = hp_config

    config = {
        "model_type": "topk_sae",
        "base_model": "Qwen/Qwen3.5-27B",
        "architecture": "Qwen 3.5-27B hybrid (DeltaNet + Attention)",
        "hidden_dim": 5120,
        "training_tokens": 200_000_000,
        "training_methodology": "FAST (sequential instruction-following + tool-use)",
        "hook_points": hook_point_configs,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy each SAE
    for hp in HOOK_POINTS:
        sae_path = sae_dir / hp.sae_id
        if not sae_path.exists():
            logger.warning("SAE not found: %s", sae_path)
            continue

        dest = output_dir / hp.sae_id
        dest.mkdir(parents=True, exist_ok=True)

        # Copy weights and config
        for filename in ["weights.safetensors", "config.json"]:
            src = sae_path / filename
            if src.exists():
                shutil.copy2(src, dest / filename)

        # Quality metrics
        if hp.sae_id in quality_metrics:
            with open(dest / "quality_metrics.json", "w") as f:
                json.dump(quality_metrics[hp.sae_id], f, indent=2)

        # Feature descriptions — only included when redact_steering_data is
        # False.  Descriptions may reference trait names (e.g., "autonomy-
        # related feature"), which would leak trait-associated feature lists
        # even when TAS scores are redacted.
        if not redact_steering_data and feature_descriptions and hp.sae_id in feature_descriptions:
            with open(dest / "feature_descriptions.json", "w") as f:
                json.dump(
                    {str(k): v for k, v in feature_descriptions[hp.sae_id].items()},
                    f,
                    indent=2,
                )
        elif redact_steering_data and feature_descriptions and hp.sae_id in feature_descriptions:
            logger.info(
                "Redacted feature descriptions for %s — descriptions may "
                "reference trait names, leaking trait-associated feature lists",
                hp.sae_id,
            )

        # TAS scores — only included when redact_steering_data is False
        if not redact_steering_data and tas_scores and hp.sae_id in tas_scores:
            with open(dest / "tas_scores.json", "w") as f:
                json.dump(tas_scores[hp.sae_id], f, indent=2)
            logger.info(
                "Included TAS scores for %s (redact_steering_data=False)",
                hp.sae_id,
            )
        elif redact_steering_data and tas_scores and hp.sae_id in tas_scores:
            logger.info(
                "Redacted TAS scores for %s per responsible-disclosure policy",
                hp.sae_id,
            )

        logger.info("Packaged SAE: %s", hp.sae_id)

    # Steering config — only included when redact_steering_data is False
    if not redact_steering_data and steering_config:
        with open(output_dir / "steering_config.json", "w") as f:
            json.dump(steering_config, f, indent=2)
        logger.info(
            "Included steering config (redact_steering_data=False)"
        )
    elif redact_steering_data and steering_config:
        logger.info(
            "Redacted steering config per responsible-disclosure policy — "
            "trait-associated feature lists and multiplier recommendations "
            "excluded from public release"
        )

    # Generate model card
    from src.release.model_card import generate_model_card

    model_card = generate_model_card(
        quality_metrics,
        redact_steering_data=redact_steering_data,
    )
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)

    # Generate demo notebook
    from src.release.demo_notebook import generate_demo_notebook

    generate_demo_notebook(output_dir / "demo.ipynb")

    logger.info("HuggingFace package ready at %s", output_dir)
