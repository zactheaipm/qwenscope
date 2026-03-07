"""Generate demo Colab notebook for the HuggingFace release."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NOTEBOOK_CELLS = [
    {
        "cell_type": "markdown",
        "source": [
            "# Qwen 3.5 Scope — SAE Demo\n",
            "\n",
            "This notebook demonstrates how to use the Sparse Autoencoders trained on Qwen 3.5-35B-A3B.\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eigen-labs/qwen35-scope/blob/main/demo.ipynb)\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Install dependencies\n",
            "!pip install -q torch transformers safetensors\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import json\n",
            "import torch\n",
            "from safetensors.torch import load_file\n",
            "from pathlib import Path\n",
            "\n",
            "# Load SAE configuration\n",
            "sae_id = 'sae_attn_mid'  # Change to explore other SAEs\n",
            "with open(f'{sae_id}/config.json') as f:\n",
            "    config = json.load(f)\n",
            "print(f'SAE config: {config}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Load SAE weights\n",
            "weights = load_file(f'{sae_id}/weights.safetensors')\n",
            "print('Loaded weights:', {k: v.shape for k, v in weights.items()})\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Simple TopK SAE implementation\n",
            "class TopKSAE:\n",
            "    def __init__(self, weights, k=None):\n",
            "        self.encoder_weight = weights['encoder.weight']\n",
            "        self.encoder_bias = weights['encoder.bias']\n",
            "        self.decoder_weight = weights['decoder.weight']\n",
            "        self.pre_bias = weights['pre_bias']\n",
            "        # Read k from config if not provided (SAEs use different k values)\n",
            "        self.k = k if k is not None else config.get('k', config.get('topk', 64))\n",
            "    \n",
            "    def encode(self, x):\n",
            "        x_centered = x - self.pre_bias\n",
            "        latents = x_centered @ self.encoder_weight.T + self.encoder_bias\n",
            "        latents = latents.clamp(min=0)  # ReLU before TopK\n",
            "        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)\n",
            "        sparse = torch.zeros_like(latents)\n",
            "        sparse.scatter_(-1, topk_idx, topk_vals)\n",
            "        return sparse\n",
            "    \n",
            "    def decode(self, features):\n",
            "        return features @ self.decoder_weight.T + self.pre_bias\n",
            "\n",
            "sae = TopKSAE(weights)\n",
            "print(f'SAE ready: {sae.encoder_weight.shape[0]} features, k={sae.k}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Test with random activations\n",
            "x = torch.randn(1, 10, 2048)  # (batch, seq_len, hidden_dim)\n",
            "features = sae.encode(x)\n",
            "reconstruction = sae.decode(features)\n",
            "\n",
            "print(f'Input shape: {x.shape}')\n",
            "print(f'Features shape: {features.shape}')\n",
            "print(f'Active features per token: {(features != 0).sum(dim=-1).float().mean():.0f}')\n",
            "print(f'Reconstruction MSE: {((x - reconstruction) ** 2).mean():.4f}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## Explore Feature Descriptions\n",
            "\n",
            "Feature descriptions may be included depending on the release configuration.\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# Load feature descriptions (if available)\n",
            "desc_path = Path(f'{sae_id}/feature_descriptions.json')\n",
            "if desc_path.exists():\n",
            "    with open(desc_path) as f:\n",
            "        descriptions = json.load(f)\n",
            "    print(f'Loaded {len(descriptions)} feature descriptions')\n",
            "    for idx, desc in list(descriptions.items())[:10]:\n",
            "        print(f'  Feature {idx}: {desc}')\n",
            "else:\n",
            "    print('No feature descriptions found for this SAE')\n",
        ],
    },
]


def generate_demo_notebook(output_path: Path) -> None:
    """Generate a demo Jupyter notebook.

    Args:
        output_path: Path to save the .ipynb file.
    """
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "cells": [],
    }

    for i, cell_def in enumerate(NOTEBOOK_CELLS):
        cell = {
            "cell_type": cell_def["cell_type"],
            "metadata": {},
            "source": cell_def["source"],
        }
        if cell_def["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["id"] = f"cell_{i}"
        notebook["cells"].append(cell)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    logger.info("Generated demo notebook: %s", output_path)
