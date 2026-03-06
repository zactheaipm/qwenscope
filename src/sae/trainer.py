"""SAE training loop with FAST methodology."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.optim as optim

from src.sae.activations import ActivationStream
from src.sae.config import SAETrainingConfig
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


class CircularActivationBuffer:
    """Fixed-capacity circular buffer of activation vectors stored on CPU.

    Stores activations in a pre-allocated tensor to avoid repeated allocation.
    Supports random sampling without replacement for proper FAST-style mixing.

    Holding the buffer on CPU prevents it from competing with the base model and
    SAE for GPU memory; activations are moved to device only at training time.
    """

    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the buffer.

        Args:
            capacity: Maximum number of activation vectors to store.
            hidden_dim: Dimensionality of each activation vector.
            dtype: Storage dtype (bfloat16 keeps CPU footprint manageable).
        """
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self._buf = torch.zeros(capacity, hidden_dim, dtype=dtype, pin_memory=True)
        self._ptr = 0   # write pointer (next slot to fill)
        self._n = 0     # number of filled slots, capped at capacity

    def add(self, acts: torch.Tensor) -> None:
        """Add a batch of activations to the buffer.

        Writes wrap around when the buffer is full, overwriting the oldest data.

        Args:
            acts: Activations of shape (N, hidden_dim). Moved to CPU and cast to
                the buffer dtype before storing.
        """
        acts_cpu = acts.detach().cpu().to(self._buf.dtype)
        n = acts_cpu.shape[0]

        if n >= self.capacity:
            # Incoming batch is larger than entire buffer: keep the most recent
            self._buf[:] = acts_cpu[-self.capacity :]
            self._ptr = 0
            self._n = self.capacity
            return

        end = self._ptr + n
        if end <= self.capacity:
            self._buf[self._ptr : end] = acts_cpu
        else:
            first = self.capacity - self._ptr
            self._buf[self._ptr :] = acts_cpu[:first]
            self._buf[: n - first] = acts_cpu[first:]

        self._ptr = (self._ptr + n) % self.capacity
        self._n = min(self._n + n, self.capacity)

    def sample(self, n: int, device: str = "cuda") -> torch.Tensor:
        """Randomly sample n activation vectors from the buffer.

        Args:
            n: Number of vectors to sample. Clamped to min(n, self._n).
            device: Device to move samples to before returning.

        Returns:
            Tensor of shape (min(n, filled), hidden_dim) on `device`.
        """
        if self._n == 0:
            return torch.zeros(0, self.hidden_dim, dtype=self._buf.dtype, device=device)
        n = min(n, self._n)
        idx = torch.randperm(self._n)[:n]
        return self._buf[: self._n][idx].to(device, non_blocking=True)

    @property
    def is_ready(self) -> bool:
        """True once the buffer contains at least one element."""
        return self._n > 0

    def __len__(self) -> int:
        return self._n


class SAETrainer:
    """Trains a TopK SAE on streamed activations.

    Implements FAST methodology:
    - Sequential processing of instruction conversations
    - Circular activation buffer for proper random mixing
    - Dead feature tracking and resampling (Anthropic-style)
    - Aux-k loss for dead feature gradient signal (Gao et al. 2024)
    - Decoder column normalization after every gradient step
    - Gradient clipping for training stability
    - Warmup + constant + linear-decay LR schedule
    - Logs to WandB
    """

    def __init__(
        self,
        sae: TopKSAE,
        config: SAETrainingConfig,
        resample_dead_features: bool = True,
        resample_every_n_steps: int | None = None,
        aux_k_coeff: float = 1.0 / 32,
        buffer_capacity: int | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            sae: The SAE model to train.
            config: Training configuration.
            resample_dead_features: Whether to resample dead features periodically.
            resample_every_n_steps: How often to check and resample dead features.
                Defaults to config.resample_every_n_steps (5000). With 200M tokens
                and batch_size=4096 (~48K total steps), 5000-step intervals yield
                ~9 resampling events, vs only ~1 at the old 25K default.
            aux_k_coeff: Coefficient for the aux-k dead-feature reconstruction loss
                (Gao et al. 2024). Set to 0.0 to disable.
            buffer_capacity: Number of activation vectors in the circular buffer.
                Defaults to config.buffer_capacity (500K). At hidden_dim=2048 BF16
                this is ~2 GB CPU RAM and covers 0.25% of a 200M-token run.
        """
        self.sae = sae
        self.config = config
        self.resample_dead_features = resample_dead_features
        self.resample_every_n_steps = (
            resample_every_n_steps
            if resample_every_n_steps is not None
            else config.resample_every_n_steps
        )
        self.aux_k_coeff = aux_k_coeff

        self.optimizer = optim.Adam(sae.parameters(), lr=config.learning_rate)

        # LR schedule: linear warmup → constant → linear decay in final 20%.
        # Total steps is estimated from training_tokens / batch_size.
        total_steps = max(config.training_tokens // config.batch_size, 1)
        warmup_steps = config.lr_warmup_steps
        decay_start = int(0.8 * total_steps)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / max(total_steps - decay_start, 1)
            return max(1.0 - progress, 0.0)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        # Dead feature tracking
        # 10K steps × 4096 batch = ~40M tokens of silence before a feature is
        # considered dead. This is conservative enough to preserve rare-but-real
        # features (e.g., firing 1-in-10M tokens) while still allowing ~4
        # resampling events over a 200M-token run (resample_every=5K steps,
        # first eligible at step 10K). Previous threshold of 2K (~8M tokens)
        # was too aggressive and risked repeatedly killing interpretable features.
        self._dead_feature_threshold = 10_000
        self._feature_activity = torch.zeros(config.dictionary_size, dtype=torch.long)
        self._steps_since_last_active = torch.zeros(config.dictionary_size, dtype=torch.long)

        # Track high-loss examples for dead feature resampling
        self._high_loss_examples: torch.Tensor | None = None
        self._high_loss_threshold = 128

        # Circular activation buffer for FAST-style mixing.
        # Capacity determines shuffle window; caller can override via buffer_capacity arg
        # but the config value (default 500K) is the right production setting.
        _buffer_capacity = (
            buffer_capacity if buffer_capacity is not None else config.buffer_capacity
        )
        self._buffer = CircularActivationBuffer(
            capacity=_buffer_capacity,
            hidden_dim=sae.hidden_dim,
            dtype=torch.bfloat16,
        )
        logger.info(
            "Buffer capacity: %d vectors (%.1f GB CPU, shuffle window = %.2f%% of %d training tokens)",
            _buffer_capacity,
            _buffer_capacity * sae.hidden_dim * 2 / 1e9,  # BF16 = 2 bytes
            _buffer_capacity / max(config.training_tokens, 1) * 100,
            config.training_tokens,
        )

        # Training state
        self._step = 0
        self._tokens_seen = 0          # mini-batch tokens (batch_size × steps)
        self._stream_tokens_seen = 0   # actual activation-stream tokens received
        self._last_checkpoint_tokens = 0
        self._resampled_count = 0
        self._resume_skip_tokens = 0   # tokens to skip on resume (fast-forward)

        # Optional WandB
        self._wandb_run = None

    def init_wandb(self, project: str = "qwen35-scope", run_name: str | None = None) -> None:
        """Initialize Weights & Biases logging.

        Args:
            project: WandB project name.
            run_name: Optional run name. Auto-generated if None.
        """
        try:
            import wandb

            if run_name is None:
                run_name = f"sae_train_{self.config.sae_id}_{int(time.time())}"

            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=self.config.model_dump(),
                tags=["sae-training"],
            )
            logger.info("WandB initialized: %s", run_name)
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            logger.warning("WandB init failed: %s — training will continue without logging", e)

    def _save_training_state(self, ckpt_path: Path) -> None:
        """Save optimizer, scheduler, and training counters alongside SAE weights.

        Args:
            ckpt_path: Checkpoint directory (same as sae.save target).
        """
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self._step,
            "tokens_seen": self._tokens_seen,
            "stream_tokens_seen": self._stream_tokens_seen,
            "last_checkpoint_tokens": self._last_checkpoint_tokens,
            "resampled_count": self._resampled_count,
            "feature_activity": self._feature_activity,
            "steps_since_last_active": self._steps_since_last_active,
        }
        torch.save(state, ckpt_path / "training_state.pt")

    def resume_from_checkpoint(self, ckpt_path: Path) -> None:
        """Restore full training state from a checkpoint.

        Loads SAE weights, optimizer/scheduler state, dead feature tracking,
        and training counters. On the next call to ``train()``, the activation
        stream will fast-forward past already-processed tokens.

        Args:
            ckpt_path: Checkpoint directory containing weights.safetensors,
                config.json, and training_state.pt.
        """
        ckpt_path = Path(ckpt_path)

        # Load SAE weights — validates architecture match (hidden_dim, dict_size)
        sae_loaded = TopKSAE.load(ckpt_path, device="cpu")
        ckpt_dict_size = sae_loaded.dict_size
        if ckpt_dict_size != self.sae.dict_size:
            raise ValueError(
                f"Checkpoint dict_size ({ckpt_dict_size}) != current config "
                f"dict_size ({self.sae.dict_size}) for {ckpt_path}. "
                f"Cannot resume — delete old checkpoints and restart fresh."
            )
        self.sae.load_state_dict(sae_loaded.state_dict())
        logger.info("Loaded SAE weights from %s", ckpt_path)

        # Load training state
        state_file = ckpt_path / "training_state.pt"
        if not state_file.exists():
            logger.warning(
                "No training_state.pt in %s — SAE weights loaded but "
                "optimizer/scheduler state is reset. Training will restart "
                "from step 0 with fresh optimizer momentum.",
                ckpt_path,
            )
            return

        state = torch.load(state_file, map_location="cpu", weights_only=False)

        # Validate dead-feature tracking tensor shapes match current dict_size
        for key in ("feature_activity", "steps_since_last_active"):
            if state[key].shape[0] != self.config.dictionary_size:
                raise ValueError(
                    f"Checkpoint {key} has shape {state[key].shape} but current "
                    f"dictionary_size is {self.config.dictionary_size}. "
                    f"Cannot resume — delete old checkpoints and restart fresh."
                )

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self._step = state["step"]
        self._tokens_seen = state["tokens_seen"]
        self._stream_tokens_seen = state["stream_tokens_seen"]
        self._last_checkpoint_tokens = state["last_checkpoint_tokens"]
        self._resampled_count = state["resampled_count"]
        self._feature_activity = state["feature_activity"]
        self._steps_since_last_active = state["steps_since_last_active"]

        # On the next train() call, skip this many stream tokens.
        self._resume_skip_tokens = self._stream_tokens_seen

        logger.info(
            "Resumed from checkpoint: step=%d, stream_tokens=%d, "
            "mini_batch_tokens=%d, resampled=%d",
            self._step,
            self._stream_tokens_seen,
            self._tokens_seen,
            self._resampled_count,
        )

    def train(
        self,
        activation_stream: ActivationStream,
        checkpoint_dir: Path | None = None,
    ) -> TopKSAE:
        """Train the SAE on the activation stream.

        Logs to WandB: loss, MSE, explained variance, dead feature count.
        Checkpoints every N tokens.

        If ``resume_from_checkpoint()`` was called, the stream fast-forwards
        past already-processed tokens before training resumes.

        Args:
            activation_stream: Stream of activation batches.
            checkpoint_dir: Directory for saving checkpoints. Created if needed.

        Returns:
            The trained SAE.
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.sae.train()
        device = next(self.sae.parameters()).device
        sae_dtype = next(self.sae.parameters()).dtype

        _dtype_logged = False
        _skip_tokens_remaining = self._resume_skip_tokens
        self._resume_skip_tokens = 0  # consume so a second train() call doesn't re-skip

        for acts_batch in activation_stream.stream_tokens(self.config.training_tokens):
            # Fast-forward past already-processed tokens on resume.
            # Use >= 0 (not > 0) to skip the exact boundary batch too: if
            # _skip_tokens_remaining == acts_batch.shape[0], the entire batch
            # was already seen and should be discarded.
            if _skip_tokens_remaining > 0:
                _skip_tokens_remaining -= acts_batch.shape[0]
                if _skip_tokens_remaining >= 0:
                    continue
                # Boundary batch: discard the already-seen prefix, keep the rest.
                acts_batch = acts_batch[acts_batch.shape[0] + _skip_tokens_remaining :]
                _skip_tokens_remaining = 0
            if not _dtype_logged and acts_batch.dtype != sae_dtype:
                logger.info(
                    "Activation dtype %s differs from SAE dtype %s — "
                    "mini-batches will be cast to SAE dtype before each forward pass.",
                    acts_batch.dtype, sae_dtype,
                )
                _dtype_logged = True

            # Count actual stream tokens before mini-batching.
            self._stream_tokens_seen += acts_batch.shape[0]

            # Add new activations to the circular buffer (CPU).
            self._buffer.add(acts_batch)

            if not self._buffer.is_ready:
                continue

            # Train for as many mini-batches as there are new activations.
            # Each mini-batch is sampled fresh from the full buffer, ensuring
            # every gradient step sees a properly randomized mix.
            n_mini_batches = max(1, acts_batch.shape[0] // self.config.batch_size)

            for _ in range(n_mini_batches):
                mini_batch = self._buffer.sample(self.config.batch_size, device=device)
                mini_batch = mini_batch.to(dtype=sae_dtype)
                if mini_batch.shape[0] == 0:
                    continue

                self.optimizer.zero_grad()
                reconstruction, features, mse_loss = self.sae(mini_batch)

                # Aux-k loss: give dead features gradient signal so they can
                # recover (Gao et al. 2024 OpenAI TopK SAE paper).
                total_loss = mse_loss
                if self.aux_k_coeff > 0:
                    aux_loss = self._compute_aux_k_loss(mini_batch, reconstruction)
                    total_loss = mse_loss + self.aux_k_coeff * aux_loss

                total_loss.backward()

                # Gradient clipping: prevents exploding gradients at training
                # start and after dead-feature resampling events.
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                # Decoder column normalization: keeps decoder directions unit-norm
                # so encoder magnitudes are comparable across dictionary entries.
                self.sae.normalize_decoder()

                self._step += 1
                self._tokens_seen += mini_batch.shape[0]

                # Track dead features using topk_indices so zero-valued selected
                # features are counted as active (not missed by abs() > 0 check).
                with torch.no_grad():
                    _, topk_indices = self.sae._encode_with_indices(mini_batch)
                # topk_indices: (batch, k) — each value is a selected feature index
                active_mask = torch.zeros(
                    self.config.dictionary_size, dtype=torch.bool, device=topk_indices.device
                )
                active_mask.scatter_(0, topk_indices.reshape(-1), True)  # (dict_size,)
                self._feature_activity += active_mask.cpu().long()
                self._steps_since_last_active += 1
                self._steps_since_last_active[active_mask.cpu()] = 0

                # Track high-loss examples for resampling
                if self.resample_dead_features:
                    self._update_high_loss_examples(mini_batch, reconstruction)

                # Dead feature resampling
                if (
                    self.resample_dead_features
                    and self._step % self.resample_every_n_steps == 0
                    and self._step > 0
                ):
                    n_resampled = self._resample_dead_features()
                    self._resampled_count += n_resampled

                # Log metrics
                if self._step % 100 == 0:
                    dead_count = self._compute_dead_features()
                    explained_var = self._compute_explained_variance(
                        mini_batch, reconstruction
                    )

                    # Track activation RMS norm per layer type. DeltaNet and
                    # attention layers may have meaningfully different activation
                    # scales; large divergence from historical mean can signal
                    # that the learning rate or SAE scale is miscalibrated.
                    act_rms_norm = float(
                        mini_batch.norm(dim=-1).mean().item()
                    )

                    metrics = {
                        "loss/total": total_loss.item(),
                        "loss/mse": mse_loss.item(),
                        "explained_variance": explained_var,
                        "dead_features": dead_count,
                        "dead_feature_pct": dead_count / self.config.dictionary_size * 100,
                        "tokens_seen": self._tokens_seen,
                        "lr": self.scheduler.get_last_lr()[0],
                        "step": self._step,
                        "resampled_total": self._resampled_count,
                        "buffer_size": len(self._buffer),
                        "activations/rms_norm": act_rms_norm,
                    }

                    if self._wandb_run is not None:
                        import wandb

                        wandb.log(metrics, step=self._step)

                    if self._step % 1000 == 0:
                        logger.info(
                            "Step %d | Loss: %.4f | EV: %.4f | Dead: %d (%.1f%%) | "
                            "Resampled: %d | Stream tokens: %d | Mini-batch steps: %d | Buffer: %d | ActNorm: %.2f",
                            self._step,
                            metrics["loss/total"],
                            metrics["explained_variance"],
                            metrics["dead_features"],
                            metrics["dead_feature_pct"],
                            self._resampled_count,
                            self._stream_tokens_seen,
                            self._tokens_seen,
                            len(self._buffer),
                            metrics["activations/rms_norm"],
                        )

            # Checkpoint keyed by stream tokens (actual activations received from
            # the dataset), not mini-batch tokens. config.checkpoint_every_tokens
            # is expressed in stream-token units (e.g. 50M stream tokens), so
            # comparing against _stream_tokens_seen fires at the right frequency.
            if (
                checkpoint_dir is not None
                and self._stream_tokens_seen - self._last_checkpoint_tokens
                >= self.config.checkpoint_every_tokens
            ):
                ckpt_path = checkpoint_dir / f"checkpoint_{self._stream_tokens_seen}"
                self.sae.save(ckpt_path)
                self._save_training_state(ckpt_path)
                self._last_checkpoint_tokens = self._stream_tokens_seen
                logger.info(
                    "Checkpoint saved at %d stream tokens (%d mini-batch steps): %s",
                    self._stream_tokens_seen,
                    self._tokens_seen,
                    ckpt_path,
                )

        self.sae.eval()
        logger.info(
            "Training complete. Steps: %d, Stream tokens: %d, Mini-batch steps: %d, Resampled: %d",
            self._step, self._stream_tokens_seen, self._tokens_seen, self._resampled_count,
        )
        return self.sae

    def _compute_dead_features(self) -> int:
        """Count features that haven't activated in the last N steps.

        Returns:
            Number of dead features.
        """
        return int(
            (self._steps_since_last_active >= self._dead_feature_threshold).sum().item()
        )

    @staticmethod
    def _compute_explained_variance(
        original: torch.Tensor, reconstruction: torch.Tensor
    ) -> float:
        """Compute per-dimension explained variance ratio, then average.

        Computes variance per hidden dimension and averages. This is more
        informative than global variance (which inflates EV for high-variance
        dimensions) and is consistent with the eval metric in quality.py.

        Args:
            original: Original activations of shape (batch, hidden_dim).
            reconstruction: SAE reconstruction of shape (batch, hidden_dim).

        Returns:
            Explained variance ratio (1.0 = perfect reconstruction).
        """
        # Per-dimension variance: var across batch for each hidden dim
        residual_var = (original - reconstruction).var(dim=0)  # (hidden_dim,)
        total_var = original.var(dim=0)                         # (hidden_dim,)

        valid = total_var > 1e-8
        if not valid.any():
            return 1.0

        per_dim_ev = 1.0 - residual_var[valid] / total_var[valid]
        return float(per_dim_ev.mean().item())

    def _compute_aux_k_loss(
        self,
        x: torch.Tensor,           # (batch, hidden_dim) original activations
        reconstruction: torch.Tensor,  # (batch, hidden_dim) main SAE reconstruction
    ) -> torch.Tensor:
        """Aux-k loss from Gao et al. (OpenAI TopK SAE, 2024).

        Gives dead features gradient signal by asking them to reconstruct the
        residual that the main SAE failed to capture. Steps:
          1. Identify dead features (silent for >= _dead_feature_threshold steps).
          2. Among those features, select the top-k by encoder activation.
          3. Reconstruct from those dead features only.
          4. Loss = MSE(residual, dead_reconstruction).

        This forces dead features to orient toward directions the main SAE is
        missing, making them more likely to become useful on their next resampling.

        Args:
            x: Original activation vectors.
            reconstruction: Main SAE reconstruction (gradient-tracked).

        Returns:
            Scalar auxiliary loss. Returns 0 if no dead features exist.
        """
        dead_mask = self._steps_since_last_active >= self._dead_feature_threshold
        n_dead = int(dead_mask.sum().item())

        if n_dead == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Residual the main SAE failed to reconstruct
        residual = (x - reconstruction).detach()  # stop grad through main path

        # Encoder activations for all features
        x_centered = x - self.sae.pre_bias           # (batch, hidden_dim)
        latents = self.sae.encoder(x_centered)        # (batch, dict_size)

        # Among dead features only, take top-k (same k as main SAE)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0].to(x_centered.device)  # (n_dead,)
        dead_latents = latents[:, dead_indices]              # (batch, n_dead)

        k_aux = min(self.sae.k, n_dead)
        topk_vals, topk_cols = dead_latents.topk(k_aux, dim=-1)  # (batch, k_aux)
        topk_vals = topk_vals.clamp(min=0)

        # Scatter back into full dictionary space
        dead_features = torch.zeros_like(latents)            # (batch, dict_size)
        global_indices = dead_indices[topk_cols]             # (batch, k_aux)
        dead_features.scatter_(-1, global_indices, topk_vals)

        # Reconstruct from dead features
        dead_reconstruction = self.sae.decoder(dead_features)  # (batch, hidden_dim)

        return (residual - dead_reconstruction).pow(2).mean()

    def _update_high_loss_examples(
        self, original: torch.Tensor, reconstruction: torch.Tensor
    ) -> None:
        """Track examples with highest reconstruction loss for resampling.

        Args:
            original: Original activations of shape (batch, hidden_dim).
            reconstruction: SAE reconstruction of shape (batch, hidden_dim).
        """
        with torch.no_grad():
            per_example_loss = (original - reconstruction).pow(2).mean(dim=-1)  # (batch,)

            n_keep = min(self._high_loss_threshold, original.shape[0])
            top_indices = per_example_loss.topk(n_keep).indices
            new_high_loss = original[top_indices].detach()

            if self._high_loss_examples is None:
                self._high_loss_examples = new_high_loss
            else:
                combined = torch.cat([self._high_loss_examples, new_high_loss], dim=0)
                recon, _, _ = self.sae(combined)
                losses = (combined - recon).pow(2).mean(dim=-1)
                keep = min(self._high_loss_threshold, combined.shape[0])
                top_idx = losses.topk(keep).indices
                self._high_loss_examples = combined[top_idx].detach()

    def _resample_dead_features(self) -> int:
        """Resample dead features from high-loss examples.

        For each dead feature:
        1. Sample a high-loss example
        2. Set the encoder weight to point toward that example
        3. Set the decoder weight to the normalized direction
        4. Reset the bias terms
        5. Reset the optimizer state for those parameters only

        Returns:
            Number of features resampled.
        """
        dead_mask = self._steps_since_last_active >= self._dead_feature_threshold
        n_dead = int(dead_mask.sum().item())

        if n_dead == 0 or self._high_loss_examples is None:
            return 0

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        device = next(self.sae.parameters()).device

        with torch.no_grad():
            for i, feat_idx in enumerate(dead_indices):
                example_idx = i % self._high_loss_examples.shape[0]
                example = self._high_loss_examples[example_idx].to(device)

                centered = example - self.sae.pre_bias
                direction = centered / (centered.norm() + 1e-8)

                self.sae.encoder.weight.data[feat_idx] = direction * 0.1
                self.sae.encoder.bias.data[feat_idx] = 0.0
                self.sae.decoder.weight.data[:, feat_idx] = direction

                self._steps_since_last_active[feat_idx] = 0
                self._feature_activity[feat_idx] = 0

        # Reset optimizer state ONLY for resampled feature indices to avoid
        # catastrophic LR spikes on live features (see trainer history).
        named_params = dict(self.sae.named_parameters())
        for param_name in ["encoder.weight", "encoder.bias", "decoder.weight"]:
            param = named_params.get(param_name)
            if param is None:
                continue
            state = self.optimizer.state.get(param)
            if not state:
                continue
            for feat_idx in dead_indices:
                feat_idx_int = int(feat_idx.item()) if isinstance(feat_idx, torch.Tensor) else int(feat_idx)
                for key in ["exp_avg", "exp_avg_sq"]:
                    if key not in state:
                        continue
                    if param_name == "encoder.weight":
                        state[key][feat_idx_int].zero_()
                    elif param_name == "encoder.bias":
                        state[key][feat_idx_int] = 0.0
                    elif param_name == "decoder.weight":
                        state[key][:, feat_idx_int].zero_()

        logger.info("Resampled %d dead features at step %d", n_dead, self._step)
        return n_dead
