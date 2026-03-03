# AgentGenome

**SAE-based behavioral decomposition and steering for Qwen 3.5-27B's hybrid GatedDeltaNet + Attention architecture.**

AgentGenome trains [Sparse Autoencoders](https://transformer-circuits.pub/2023/monosemantic-features) (SAEs) on the residual stream of [Qwen 3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) — a 64-layer hybrid model that interleaves GatedDeltaNet (linear attention) and standard attention layers — to identify, decompose, and steer five core behavioral traits of agentic language model behavior.

The central question: **How are behavioral traits encoded differently across DeltaNet vs. attention layers in a hybrid architecture?**

## Behavioral Traits

AgentGenome decomposes agentic behavior into 5 traits, each with 3 measurable sub-behaviors (15 total):

| Trait | Sub-behaviors | Description |
|-------|--------------|-------------|
| **Autonomy** | decision_independence, action_initiation, permission_avoidance | Tendency to act independently vs. seeking approval |
| **Tool Use Eagerness** | tool_reach, proactive_information_gathering, tool_diversity | Willingness to use and explore available tools |
| **Persistence** | retry_willingness, strategy_variation, escalation_reluctance | Tendency to retry and adapt vs. giving up |
| **Risk Calibration** | approach_novelty, scope_expansion, uncertainty_tolerance | Appetite for novel, expansive, or uncertain approaches |
| **Deference** | instruction_literalness, challenge_avoidance, suggestion_restraint | Tendency to follow instructions literally vs. pushing back |

## Architecture

Qwen 3.5-27B organizes its 64 layers into 16 blocks of 4 layers each. Within each block, the first 3 layers use GatedDeltaNet (a linear attention variant) and the 4th uses standard multi-head attention:

```
Block k: [DeltaNet₀, DeltaNet₁, DeltaNet₂, Attention₃] × 16 blocks = 64 layers
```

SAEs are trained at 7 hook points spanning early, mid, and late positions across both layer types:

| SAE ID | Layer | Type | Block | Purpose |
|--------|-------|------|-------|---------|
| `sae_delta_early` | 10 | DeltaNet | 2 | Early DeltaNet |
| `sae_attn_early` | 11 | Attention | 2 | Early attention |
| `sae_delta_mid_pos1` | 33 | DeltaNet | 8 | Mid DeltaNet (position 1 — within-block control) |
| `sae_delta_mid` | 34 | DeltaNet | 8 | Mid DeltaNet (position 2) |
| `sae_attn_mid` | 35 | Attention | 8 | Mid attention |
| `sae_delta_late` | 54 | DeltaNet | 13 | Late DeltaNet |
| `sae_attn_late` | 55 | Attention | 13 | Late attention |

The `sae_delta_mid_pos1` SAE exists as a control — comparing positions 1 and 2 within the same block isolates layer-type effects from positional confounds.

## Methodology

### SAE Architecture

Each SAE is a **TopK Sparse Autoencoder** (dictionary size 40,960 = 8× hidden dim, k=64):

- Encoder: 5120 → 40960 with learned pre-bias
- TopK sparsification with ReLU clamping
- Decoder: 40960 → 5120 (no bias, unit-normalized columns)
- Dead feature resampling every 5,000 steps via auxiliary-k loss

### Training Data Mix

SAEs are trained on 200M tokens per hook point with the following mix:

- **35% UltraChat 200k** — instruction-following conversations
- **35% WildChat-1M** — diverse real-world conversations (GDPR-filtered)
- **30% Synthetic tool-use** — multi-turn tool-calling conversations

Training uses FAST (Feature Alignment with Sequential Tokens) methodology: full conversations are processed sequentially with sequence packing for efficiency.

### Contrastive Feature Identification

Behavioral features are identified through contrastive activation analysis:

1. **1,520 contrastive prompt pairs** are generated (800 composite + 720 sub-behavior-specific + 60 null controls)
2. Each pair has HIGH and LOW variants — identical user messages but different system prompts that elicit different behavioral dispositions
3. Activations are extracted at the **last token** position (avoiding sequence-length confounds from different system prompt lengths)
4. **Trait Association Score (TAS)** = mean(high − low) / std(high − low) per feature, with cluster-robust standard errors, permutation-test p-values, and Benjamini-Hochberg FDR correction

### Steering

Behavioral steering modifies SAE features during autoregressive decoding:

```
steered = original + SAE.decode(modified_features) − SAE.decode(original_features)
```

Steering is applied **only during the decode phase** (sequence length = 1), never during prompt prefill, so the prompt representation stays uncorrupted. Non-target features are exactly preserved (the no-bias decoder ensures the delta exactly cancels non-target contributions).

### Evaluation

Steered model outputs are evaluated through:

- **Agent harness**: ReAct-style tool-calling loop using Qwen 3.5's native tool-call format with `enable_thinking=False` (suppresses `<think>` blocks so steering is causally meaningful)
- **LLM judge**: Scores trajectories on all 15 sub-behaviors (0.0–1.0 scale) using detailed rubrics
- **Safety evaluation**: Tests whether steering can override RLHF-trained safety behaviors on mild engineering scenarios (e.g., deploying without tests, skipping security review)

## Dataset

### Synthetic Tool-Use Conversations

The synthetic training data is not included in this repository due to size. You can regenerate it using the provided scripts (see [Generating Synthetic Data](#generating-synthetic-data) below).

The generated dataset consists of:

| Split | Examples | Description |
|-------|----------|-------------|
| `train_examples.jsonl` | ~10,000 | Training conversations |
| `eval_examples.jsonl` | ~1,000 | Held-out evaluation conversations |

Each example is a multi-turn conversation with system prompt, user message, assistant responses with tool calls, and tool responses. Generated with diversity from 20 scenario types × 15 domains × 4 tool-call counts = 1,200 unique prompt seeds.

### Contrastive Pairs

Generated at runtime by `05_build_contrastive_data.py` from templates in `src/data/contrastive.py`:

- **800 composite pairs**: 5 traits × 4 domains × 10 templates × 4 variations
- **720 sub-behavior pairs**: 15 sub-behaviors × 3 templates × 4 variations × 4 domains
- **60 null-trait control pairs**: For calibrating significance thresholds

The 4 task domains: Coding, Research, Communication, Data.

## Pipeline

The full pipeline is implemented as 10 numbered scripts, each writing a JSON manifest to `data/results/` for auditability:

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_setup_model.py` | Download Qwen 3.5-27B and verify activation hooks on all 64 layers |
| 02 | `02_extract_activations.py` | Extract small activation sample for spot-checks |
| 03 | `03_train_saes.py` | Train 7 TopK SAEs in parallel (200M tokens each, multi-GPU) |
| 04 | `04_evaluate_sae_quality.py` | Evaluate SAE quality (MSE, explained variance, L0, loss recovered) on both general chat and tool-use held-out data |
| 05 | `05_build_contrastive_data.py` | Generate 1,520 contrastive prompt pairs from templates |
| 06 | `06_identify_features.py` | Compute TAS scores, run permutation tests, FDR correction, cross-trait specificity checks |
| 07 | `07_run_steering.py` | Run all steering experiments (standard, layer-type comparison, cross-depth, baselines, safety) |
| 08 | `08_evaluate_behavior.py` | Score all trajectories with LLM judge |
| 09 | `09_analyze_results.py` | Generate analysis figures (contamination matrices, architecture heatmaps, effect sizes) |
| 10 | `10_package_release.py` | Package SAEs for HuggingFace release with responsible disclosure (redacts steering data by default) |

### Running the Pipeline

```bash
# Install dependencies (PyTorch 2.6+ required for fla compatibility)
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install causal-conv1d --no-build-isolation
pip install "git+https://github.com/fla-org/flash-linear-attention.git"
pip install accelerate
pip install -e ".[dev]"

# Set environment variables
export ANTHROPIC_API_KEY="your-key"    # For LLM judge and feature interpretation
export WANDB_API_KEY="your-key"        # For experiment tracking
export HF_TOKEN="your-token"           # For model download

# Run the full pipeline (requires H200 SXM or A100 80GB)
python scripts/01_setup_model.py
python scripts/02_extract_activations.py
python scripts/03_train_saes.py
python scripts/04_evaluate_sae_quality.py
python scripts/05_build_contrastive_data.py
python scripts/06_identify_features.py
python scripts/07_run_steering.py
python scripts/08_evaluate_behavior.py
python scripts/09_analyze_results.py
python scripts/10_package_release.py
```

### Generating Synthetic Data

To regenerate the synthetic tool-use training data:

```bash
# Using DeepSeek API
python scripts/generate_synthetic_data.py --split both --n-train 10000 --n-eval 1000

# Using a local vLLM server or any OpenAI-compatible endpoint
python scripts/generate_synthetic_data.py \
  --split both --n-train 10000 --n-eval 1000 \
  --provider openai \
  --api-base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model your-model-name

# Clean and deduplicate generated data
python scripts/clean_synthetic_data.py
```

### RunPod Setup

The recommended RunPod configuration:

| Setting | Value |
|---------|-------|
| GPU | NVIDIA H200 SXM (141 GB VRAM) |
| Volume | 200 GB network volume |
| Container Disk | 50 GB |
| vCPUs | ≥ 16 |
| RAM | ≥ 128 GB |
| Docker Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

After creating the pod:

```bash
# Sync code to pod
bash scripts/sync_to_pod.sh root@<POD_IP> <SSH_PORT>

# SSH in and run setup (upgrades PyTorch, installs fla + flash-attn + causal-conv1d)
ssh -p <SSH_PORT> root@<POD_IP>
cd /workspace/agentgenome
bash scripts/runpod_setup.sh
```

### Running a Pilot

To run a quick end-to-end validation (1 SAE, 1 trait, small sample):

```bash
python scripts/run_pilot.py
```

## Project Structure

```
agentgenome/
├── configs/
│   ├── experiment.yaml          # Traits, domains, steering experiments
│   ├── model.yaml               # Qwen 3.5-27B architecture parameters
│   ├── sae_training.yaml        # SAE hyperparameters and 7 hook points
│   └── eval.yaml                # Evaluation config (judge model, temperature)
├── src/
│   ├── model/                   # Model loading, architecture config, activation hooks
│   ├── sae/                     # TopK SAE model, trainer, activation buffer, quality metrics
│   ├── data/                    # Contrastive pairs, training data mix, scenarios, synthetic generator
│   ├── features/                # Feature extraction, TAS scoring, clustering, auto-interpretation
│   ├── steering/                # Residual steering engine, dose-response, multi-layer steering
│   ├── evaluation/              # Agent harness, LLM judge, behavioral metrics, safety evaluation
│   ├── analysis/                # Plots, effect sizes, architecture comparison, cost tracking
│   └── release/                 # HuggingFace packaging, model card generation, demo notebook
├── scripts/                     # Numbered pipeline scripts (01-10) + utilities
├── tests/                       # Unit and integration tests
├── data/
│   ├── synthetic/               # Generated synthetic tool-use conversations (not in repo, see below)
│   ├── activations/             # Extracted activations (generated, multi-GB)
│   ├── contrastive_pairs/       # Generated contrastive pairs
│   ├── scenarios/               # Evaluation scenarios
│   └── results/                 # Pipeline manifests and results
└── pyproject.toml
```

## Key Design Decisions

- **Last-token pooling** for feature extraction avoids sequence-length confounds from different system prompt lengths between HIGH/LOW variants
- **Decode-only steering** prevents corrupting the prompt representation during prefill
- **No-bias decoder** in the SAE ensures the steering delta exactly cancels for non-target features
- **Position-in-block control SAE** (`sae_delta_mid_pos1`) isolates layer-type effects from positional confounds within the 4-layer block
- **`enable_thinking=False`** suppresses Qwen 3.5's `<think>` blocks during evaluation, making steering causally meaningful
- **Hooks capture the residual stream after the full layer** (sublayer + FFN + skip connection) — findings are framed as "layers containing DeltaNet" rather than "DeltaNet itself"
- **Null controls** (60 pairs) calibrate significance thresholds for TAS scores
- **Responsible disclosure**: the default HuggingFace release redacts TAS scores, trait-associated feature lists, and steering multiplier recommendations

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.6 with CUDA 12.4
- [Flash Linear Attention (fla)](https://github.com/fla-org/flash-linear-attention) — provides fast CUDA kernels for Qwen 3.5's 48 GatedDeltaNet layers (without fla, these fall back to naive sequential recurrence)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) — fast causal 1D convolution kernel used by GatedDeltaNet layers
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) — for the 16 standard attention layers
- GPU: H200 SXM (141 GB VRAM) recommended. A100 80GB is the minimum for Qwen 3.5-27B in BFloat16 (~54 GB)
- Anthropic API key (for LLM judge evaluation and feature interpretation)
- ~200GB disk for model weights + activations

## Tests

```bash
pytest tests/
```

The test suite covers activation hooks, SAE training and roundtrip, steering correctness (including non-target feature preservation), tool-call parsing, TAS computation, and an end-to-end integration test.

## License

This project is released under the [MIT License](LICENSE).
