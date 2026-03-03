"""Parallel SAE training across multiple GPUs.

Loads the model ONCE and extracts activations at all 7 hook points per forward
pass, then distributes to parallel SAE training workers via multiprocessing queues.

Speedup sources:
  1. Single model load instead of 7 sequential loads (~5 min saved per load)
  2. One forward pass captures all 7 layers (vs. 7 separate full forward passes)
  3. SAE training runs concurrently across available GPUs

Architecture:
  Main process (producer):  Model on GPU 0 → forward pass → activations to queues
  Worker processes (consumers): Each trains one SAE from its queue on assigned GPU

Usage:
    python scripts/03_train_saes.py                       # All 7 SAEs, auto-detect GPUs
    python scripts/03_train_saes.py --n-gpus 4            # Limit to 4 GPUs
    python scripts/03_train_saes.py --sae-ids sae_attn_mid sae_delta_mid  # Subset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Timeout for queue operations (seconds). If the producer hasn't sent a batch
# in this long, the worker assumes something went wrong.
QUEUE_TIMEOUT_S = 600


class QueueActivationStream:
    """Adapts an mp.Queue to the ActivationStream interface expected by SAETrainer.

    SAETrainer.train() calls stream.stream_tokens(max_tokens) and iterates over
    the yielded (N, hidden_dim) tensors. This class wraps an mp.Queue so that
    a producer process can send activation batches and the trainer consumes them.
    """

    def __init__(self, queue: MPQueue, device: str) -> None:
        """Initialize the queue-backed stream.

        Args:
            queue: Multiprocessing queue carrying (N, hidden_dim) CPU tensors,
                terminated by a None sentinel.
            device: Target device for activations (e.g. "cuda:1").
        """
        self._queue = queue
        self._device = device
        self._tokens_processed = 0

    @property
    def tokens_processed(self) -> int:
        """Total activation vectors consumed so far."""
        return self._tokens_processed

    def stream_tokens(self, max_tokens: int) -> Iterator[torch.Tensor]:
        """Yield activation batches from the queue until token limit or sentinel.

        Args:
            max_tokens: Stop after consuming this many activation vectors.

        Yields:
            Activation tensors of shape (N, hidden_dim) on self._device.
        """
        while self._tokens_processed < max_tokens:
            try:
                item = self._queue.get(timeout=QUEUE_TIMEOUT_S)
            except Exception:
                logger.warning(
                    "Queue timeout after %ds — producer may have crashed.",
                    QUEUE_TIMEOUT_S,
                )
                break

            if item is None:  # Sentinel from producer
                logger.info(
                    "Received end-of-stream sentinel at %d tokens.",
                    self._tokens_processed,
                )
                break

            acts = item.to(self._device)  # (N, hidden_dim)
            self._tokens_processed += acts.shape[0]
            yield acts

    # Alias so it can also be used as a plain iterable
    def stream(self) -> Iterator[torch.Tensor]:
        """Yield all batches (no token limit)."""
        while True:
            try:
                item = self._queue.get(timeout=QUEUE_TIMEOUT_S)
            except Exception:
                break
            if item is None:
                break
            yield item.to(self._device)


def sae_worker(
    sae_id: str,
    layer: int,
    queue: MPQueue,
    device: str,
    config_path: str,
    output_dir: str,
    error_event: MPEvent,
) -> None:
    """Worker process: trains a single SAE from queued activations.

    Args:
        sae_id: SAE identifier (e.g. "sae_attn_mid").
        layer: Layer index this SAE is trained on.
        queue: Queue delivering (N, hidden_dim) activation batches.
        device: CUDA device for this worker (e.g. "cuda:1").
        config_path: Path to sae_training.yaml.
        output_dir: Base directory for saving trained SAEs.
        error_event: Shared event — set if any process encounters an error.
    """
    worker_logger = logging.getLogger(f"worker.{sae_id}")

    try:
        # Imports inside worker to avoid CUDA init before spawn
        from src.sae.config import SAETrainingConfig
        from src.sae.model import TopKSAE
        from src.sae.trainer import SAETrainer

        config = SAETrainingConfig.from_yaml(Path(config_path), sae_id)

        sae = TopKSAE(
            hidden_dim=config.hidden_dim,
            dict_size=config.dictionary_size,
            k=config.topk,
        ).to(device)

        trainer = SAETrainer(sae, config)
        trainer.init_wandb(run_name=f"sae_parallel_{sae_id}_{int(time.time())}")

        stream = QueueActivationStream(queue, device)
        checkpoint_dir = Path(output_dir) / sae_id / "checkpoints"
        trained_sae = trainer.train(stream, checkpoint_dir=checkpoint_dir)

        # Save final model
        sae_dir = Path(output_dir) / sae_id
        sae_dir.mkdir(parents=True, exist_ok=True)
        trained_sae.save(sae_dir)
        worker_logger.info("SAE %s trained and saved to %s", sae_id, sae_dir)

    except Exception:
        worker_logger.exception("Worker %s failed", sae_id)
        error_event.set()
        raise


def run_producer(
    model_device: str,
    queues: dict[int, MPQueue],
    layers: list[int],
    training_tokens: int,
    error_event: MPEvent,
) -> None:
    """Load model and produce activations for all target layers simultaneously.

    Registers hooks on all target layers, runs forward passes on training data,
    and dispatches masked+flattened activations to per-layer queues.

    Args:
        model_device: Device for the model (e.g. "cuda:0").
        queues: Map from layer index to its output queue.
        layers: Layer indices to capture.
        training_tokens: Total tokens to produce before stopping.
        error_event: Shared error event for early termination.
    """
    from src.data.training_data import SAETrainingDataBuilder
    from src.model.hooks import ActivationCache
    from src.model.loader import load_model
    from src.sae.config import SAETrainingConfig

    logger.info("Producer: loading model on %s...", model_device)
    model, tokenizer = load_model(dtype="bfloat16", device=model_device)
    logger.info("Producer: model loaded.")

    # Build training data — use first SAE's config for data parameters
    config = SAETrainingConfig.from_yaml(
        Path("configs/sae_training.yaml"),
        "sae_attn_mid",  # Any SAE ID works; data config is shared
    )
    data_builder = SAETrainingDataBuilder(tokenizer, config)
    dataset = data_builder.build_dataset()

    # Register hooks on ALL target layers at once
    cache = ActivationCache(model, layers=layers)

    tokens_processed = 0
    batch_size = 16
    batch: list[dict[str, torch.Tensor]] = []
    start_time = time.monotonic()

    for item in iter(dataset):
        if error_event.is_set():
            logger.warning("Producer: error event detected, stopping early.")
            break

        batch.append(item)
        if len(batch) < batch_size:
            continue

        input_ids = torch.stack([b["input_ids"] for b in batch]).to(model_device)
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(model_device)
        batch = []

        # Single forward pass captures all layers
        with torch.no_grad():
            with cache.active():
                model(input_ids=input_ids, attention_mask=attention_mask)

        # Mask padding and dispatch to per-layer queues
        mask = attention_mask.unsqueeze(-1).bool()  # (B, S, 1)
        for layer in layers:
            acts = cache.get(layer)  # (B, S, D)
            acts_masked = acts[mask.expand_as(acts)].view(-1, acts.shape[-1])  # (N, D)
            queues[layer].put(acts_masked.cpu())

        cache.clear()
        tokens_processed += int(attention_mask.sum().item())

        if tokens_processed % 1_000_000 < batch_size * 2048:
            elapsed = time.monotonic() - start_time
            rate = tokens_processed / elapsed if elapsed > 0 else 0
            logger.info(
                "Producer: %d / %d tokens (%.0f tok/s)",
                tokens_processed,
                training_tokens,
                rate,
            )

        if tokens_processed >= training_tokens:
            break

    # Send sentinels to all workers
    for layer in layers:
        queues[layer].put(None)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Producer finished: %d tokens in %.1fs (%.0f tok/s)",
        tokens_processed,
        elapsed,
        tokens_processed / elapsed if elapsed > 0 else 0,
    )


def assign_devices(
    sae_ids: list[str],
    n_gpus: int,
) -> tuple[str, dict[str, str]]:
    """Assign model and SAE workers to GPUs.

    Strategy:
      - 1 GPU:  Model + all SAEs on cuda:0 (still benefits from single model load
                 and single forward pass producing all layers).
      - 2+ GPUs: Model on cuda:0, SAEs round-robin across cuda:1..N-1.
                  GPU 0 is kept free for the model's forward passes.

    Args:
        sae_ids: List of SAE identifiers to train.
        n_gpus: Number of available GPUs.

    Returns:
        Tuple of (model_device, {sae_id: device}).
    """
    model_device = "cuda:0"

    if n_gpus == 1:
        sae_devices = {sid: "cuda:0" for sid in sae_ids}
    else:
        worker_gpus = [f"cuda:{i}" for i in range(1, n_gpus)]
        sae_devices = {
            sid: worker_gpus[i % len(worker_gpus)]
            for i, sid in enumerate(sae_ids)
        }

    return model_device, sae_devices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SAEs in parallel across GPUs",
    )
    parser.add_argument(
        "--sae-ids",
        nargs="+",
        help="Specific SAE IDs to train (default: all 7)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        help="Number of GPUs to use (default: auto-detect)",
    )
    parser.add_argument(
        "--config",
        default="configs/sae_training.yaml",
        help="Path to SAE training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="data/saes",
        help="Base directory for trained SAE outputs",
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", "data/results"),
        help="Directory for result manifests",
    )
    args = parser.parse_args()

    # CUDA multiprocessing requires spawn
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    n_gpus = args.n_gpus or torch.cuda.device_count()
    if n_gpus == 0:
        logger.error("No GPUs detected. This script requires at least 1 GPU.")
        return

    from src.model.config import HOOK_POINTS, HOOK_POINTS_BY_ID

    # Select which SAEs to train
    if args.sae_ids:
        hook_points = [HOOK_POINTS_BY_ID[sid] for sid in args.sae_ids]
    else:
        hook_points = list(HOOK_POINTS)

    # Skip already-trained SAEs
    to_train = []
    for hp in hook_points:
        sae_dir = Path(args.output_dir) / hp.sae_id
        if (sae_dir / "weights.safetensors").exists():
            logger.info("SAE %s already trained at %s — skipping.", hp.sae_id, sae_dir)
        else:
            to_train.append(hp)

    if not to_train:
        logger.info("All requested SAEs already trained!")
        return

    # Assign GPUs
    sae_ids = [hp.sae_id for hp in to_train]
    model_device, sae_devices = assign_devices(sae_ids, n_gpus)

    logger.info("=== Parallel SAE Training ===")
    logger.info("GPUs available: %d", n_gpus)
    logger.info("SAEs to train: %d", len(to_train))
    logger.info("Model device: %s", model_device)
    for sid, dev in sae_devices.items():
        logger.info("  %s → %s", sid, dev)

    # Create per-layer queues (maxsize bounds memory: ~8 batches × ~80MB = ~640MB per queue)
    queues: dict[int, MPQueue] = {}
    layer_to_sae: dict[int, str] = {}
    for hp in to_train:
        queues[hp.layer] = mp.Queue(maxsize=8)
        layer_to_sae[hp.layer] = hp.sae_id

    error_event = mp.Event()
    layers = [hp.layer for hp in to_train]

    # Start SAE worker processes
    workers: list[mp.Process] = []
    for hp in to_train:
        p = mp.Process(
            target=sae_worker,
            args=(
                hp.sae_id,
                hp.layer,
                queues[hp.layer],
                sae_devices[hp.sae_id],
                args.config,
                args.output_dir,
                error_event,
            ),
            name=f"sae-{hp.sae_id}",
        )
        p.start()
        workers.append(p)
        logger.info("Started worker %s (PID %d) on %s", hp.sae_id, p.pid, sae_devices[hp.sae_id])

    # Run producer in main process (blocks until all tokens produced)
    start = time.monotonic()
    from src.sae.config import SAETrainingConfig

    config = SAETrainingConfig.from_yaml(Path(args.config), to_train[0].sae_id)
    run_producer(
        model_device=model_device,
        queues=queues,
        layers=layers,
        training_tokens=config.training_tokens,
        error_event=error_event,
    )

    # Wait for all workers to finish training
    logger.info("Producer done. Waiting for SAE workers to finish training...")
    failed = []
    for p in workers:
        p.join()
        if p.exitcode != 0:
            logger.error("Worker %s exited with code %d", p.name, p.exitcode)
            failed.append(p.name)

    elapsed = time.monotonic() - start

    if failed:
        logger.error("FAILED workers: %s", ", ".join(failed))
    else:
        logger.info("All %d SAEs trained successfully in %.1fs", len(to_train), elapsed)

    # Write manifest
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "03_train_saes",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_gpus": n_gpus,
        "sae_ids": sae_ids,
        "device_assignment": sae_devices,
        "failed": failed,
    }
    with open(results_dir / "03_train_saes.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
