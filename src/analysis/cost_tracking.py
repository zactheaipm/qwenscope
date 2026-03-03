"""Computational cost tracking and estimation.

Provides:
- ``CostReport``: per-phase cost breakdown (GPU hours, API calls, wall clock).
- ``PipelineCostSummary``: aggregate cost across all phases.
- ``CostTracker``: runtime tracker with context manager, GPU memory sampling,
  API call counting, and save/load.
- ``estimate_pipeline_cost``: static cost estimator for planning (no GPU needed).
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class CostReport(BaseModel):
    """Cost breakdown for a single pipeline phase.

    Attributes:
        phase: Pipeline phase name (e.g., "sae_training", "evaluation").
        gpu_hours: Total GPU-hours consumed.
        api_calls: Number of LLM API calls (e.g., Claude judge invocations).
        api_cost_usd: Estimated API spend in USD.
        wall_clock_seconds: Elapsed wall-clock time in seconds.
        hardware: Hardware description (e.g., "NVIDIA A100 80GB").
        notes: Free-text notes (dataset size, batch size, etc.).
    """

    phase: str
    gpu_hours: float = 0.0
    api_calls: int = 0
    api_cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0
    hardware: str = ""
    notes: str = ""


class PipelineCostSummary(BaseModel):
    """Aggregate computational cost across all pipeline phases.

    Attributes:
        phases: Individual phase reports.
        total_gpu_hours: Sum of GPU-hours across phases.
        total_api_calls: Sum of API calls across phases.
        total_api_cost_usd: Sum of estimated API spend across phases.
        total_wall_clock_seconds: Sum of wall-clock seconds across phases.
        hardware: Hardware used (assumed uniform across phases).
        estimated: Whether these numbers are *estimated* (True) or *measured*.
        timestamp: ISO-8601 timestamp of when the summary was generated.
    """

    phases: list[CostReport] = Field(default_factory=list)
    total_gpu_hours: float = 0.0
    total_api_calls: int = 0
    total_api_cost_usd: float = 0.0
    total_wall_clock_seconds: float = 0.0
    hardware: str = ""
    estimated: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_reports(
        cls,
        reports: list[CostReport],
        *,
        estimated: bool = False,
    ) -> PipelineCostSummary:
        """Aggregate a list of ``CostReport`` objects into a summary.

        Args:
            reports: Individual phase cost reports.
            estimated: Mark the summary as estimated (planning mode).

        Returns:
            Aggregated pipeline cost summary.
        """
        hardware_set = {r.hardware for r in reports if r.hardware}
        hardware = ", ".join(sorted(hardware_set)) if hardware_set else ""

        return cls(
            phases=reports,
            total_gpu_hours=sum(r.gpu_hours for r in reports),
            total_api_calls=sum(r.api_calls for r in reports),
            total_api_cost_usd=sum(r.api_cost_usd for r in reports),
            total_wall_clock_seconds=sum(r.wall_clock_seconds for r in reports),
            hardware=hardware,
            estimated=estimated,
        )


# ---------------------------------------------------------------------------
# Runtime cost tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Records computational costs during pipeline execution.

    Usage::

        tracker = CostTracker(hardware="NVIDIA A100 80GB")

        with tracker.track("sae_training"):
            train_sae(...)

        tracker.record_api_call(cost_usd=0.003)

        tracker.save(Path("data/results/cost_report.json"))

    The ``track`` context manager records wall-clock time and optionally
    samples ``torch.cuda.memory_allocated`` on a background thread to
    provide a rough GPU utilisation signal (useful for detecting idle
    periods, NOT a substitute for ``nvidia-smi`` profiling).
    """

    def __init__(
        self,
        hardware: str = "",
        gpu_sample_interval_seconds: float = 5.0,
    ) -> None:
        """Initialise the tracker.

        Args:
            hardware: Human-readable hardware description
                (e.g., "NVIDIA A100 80GB").
            gpu_sample_interval_seconds: How often to sample GPU memory
                while a ``track`` context is active.  Set to 0 to disable.
        """
        if not hardware:
            hardware = _detect_hardware()

        self.hardware: str = hardware
        self.gpu_sample_interval: float = gpu_sample_interval_seconds
        self._reports: list[CostReport] = []

        # Mutable state for the *active* tracking context
        self._active_phase: str | None = None
        self._phase_start: float = 0.0
        self._phase_api_calls: int = 0
        self._phase_api_cost: float = 0.0

        # GPU memory sampling
        self._gpu_thread: threading.Thread | None = None
        self._gpu_stop_event: threading.Event = threading.Event()
        self._gpu_samples: list[float] = []

    # -- context manager -----------------------------------------------------

    @contextmanager
    def track(self, phase_name: str) -> Generator[None, None, None]:
        """Record wall-clock time and GPU memory for *phase_name*.

        Args:
            phase_name: Name of the pipeline phase being tracked.

        Yields:
            None — the caller runs its workload inside the ``with`` block.
        """
        if self._active_phase is not None:
            raise RuntimeError(
                f"Cannot start phase {phase_name!r}: "
                f"phase {self._active_phase!r} is already active. "
                "Nested tracking is not supported."
            )

        self._active_phase = phase_name
        self._phase_start = time.monotonic()
        self._phase_api_calls = 0
        self._phase_api_cost = 0.0
        self._gpu_samples = []

        # Start background GPU memory sampling (best-effort)
        self._start_gpu_sampling()

        logger.info("Cost tracking started for phase: %s", phase_name)

        try:
            yield
        finally:
            elapsed = time.monotonic() - self._phase_start
            self._stop_gpu_sampling()

            gpu_hours = _estimate_gpu_hours(
                elapsed_seconds=elapsed,
                memory_samples=self._gpu_samples,
            )

            report = CostReport(
                phase=phase_name,
                gpu_hours=gpu_hours,
                api_calls=self._phase_api_calls,
                api_cost_usd=self._phase_api_cost,
                wall_clock_seconds=elapsed,
                hardware=self.hardware,
            )
            self._reports.append(report)

            logger.info(
                "Phase %s complete — wall clock: %.1fs, GPU hours: %.4f, "
                "API calls: %d ($%.4f)",
                phase_name,
                elapsed,
                gpu_hours,
                self._phase_api_calls,
                self._phase_api_cost,
            )

            self._active_phase = None

    # -- API call counting ---------------------------------------------------

    def record_api_call(self, cost_usd: float = 0.0) -> None:
        """Increment the API call counter for the active phase.

        Can be called outside a ``track`` context; in that case the call is
        attached to a catch-all ``_untracked_api`` phase on save.

        Args:
            cost_usd: Estimated cost in USD for this single API call.
        """
        self._phase_api_calls += 1
        self._phase_api_cost += cost_usd
        logger.debug(
            "API call recorded (phase=%s, cost=$%.4f)",
            self._active_phase or "<untracked>",
            cost_usd,
        )

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write all recorded cost reports to a JSON file.

        If there are un-flushed API calls recorded outside a ``track``
        context they are attached as an ``_untracked_api`` phase.

        Args:
            path: Destination file path (created / overwritten).
        """
        self._flush_untracked_api_calls()
        summary = PipelineCostSummary.from_reports(
            self._reports, estimated=False
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Cost report saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> PipelineCostSummary:
        """Load a previously saved cost summary from JSON.

        Args:
            path: Path to the JSON file written by ``save``.

        Returns:
            Deserialized ``PipelineCostSummary``.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelineCostSummary.model_validate(data)

    # -- estimation ----------------------------------------------------------

    def estimate_full_pipeline_cost(self) -> PipelineCostSummary:
        """Estimate costs for the full AgentGenome pipeline.

        Uses project-specific constants derived from CLAUDE.md:

        * SAE training: 200M tokens/SAE, 7 SAEs.  Empirical throughput on a
          single A100-80GB is ~12k tokens/s (BF16, TopK SAE with 8x
          expansion).  This gives ~4.6 GPU-hours per SAE.
        * Feature identification: 400 contrastive pairs x 7 SAEs,
          inference-only.
        * Steering experiments: ~2750 trajectory generations (50 scenarios
          x 5 traits x 11 multiplier levels including baseline).
        * LLM judge evaluation: 2750 trajectories x 3 repeats.

        Returns:
            ``PipelineCostSummary`` with ``estimated=True``.
        """
        return estimate_pipeline_cost(
            n_saes=7,
            n_scenarios=50,
            n_traits=5,
            n_multipliers=4,
            judge_repeats=3,
            hardware=self.hardware,
        )

    # -- internal helpers ----------------------------------------------------

    def _flush_untracked_api_calls(self) -> None:
        """Attach any API calls made outside a ``track`` block."""
        if self._active_phase is None and self._phase_api_calls > 0:
            report = CostReport(
                phase="_untracked_api",
                api_calls=self._phase_api_calls,
                api_cost_usd=self._phase_api_cost,
                hardware=self.hardware,
            )
            self._reports.append(report)
            self._phase_api_calls = 0
            self._phase_api_cost = 0.0

    def _start_gpu_sampling(self) -> None:
        """Launch a daemon thread that periodically samples GPU memory."""
        if self.gpu_sample_interval <= 0:
            return

        if not _cuda_available():
            return

        self._gpu_stop_event.clear()

        def _sample_loop() -> None:
            import torch

            while not self._gpu_stop_event.is_set():
                try:
                    mem_bytes = torch.cuda.memory_allocated()
                    self._gpu_samples.append(float(mem_bytes))
                except Exception:
                    pass
                self._gpu_stop_event.wait(timeout=self.gpu_sample_interval)

        self._gpu_thread = threading.Thread(
            target=_sample_loop, daemon=True, name="cost-gpu-sampler"
        )
        self._gpu_thread.start()

    def _stop_gpu_sampling(self) -> None:
        """Signal the GPU sampling thread to stop and wait for it."""
        self._gpu_stop_event.set()
        if self._gpu_thread is not None and self._gpu_thread.is_alive():
            self._gpu_thread.join(timeout=2.0)
        self._gpu_thread = None


# ---------------------------------------------------------------------------
# Static pipeline cost estimator
# ---------------------------------------------------------------------------


# -- Cost constants ----------------------------------------------------------
# These constants come from empirical measurements and vendor pricing.

# SAE training throughput: tokens per second on a single A100-80GB running
# BF16 TopK SAE training with 8x expansion (dict_size=40960, hidden=5120).
_SAE_TOKENS_PER_SEC: float = 12_000.0

# Tokens per SAE (from configs/sae_training.yaml)
_TOKENS_PER_SAE: int = 200_000_000

# Average tokens per contrastive pair (high + low prompts, ~512 tokens each)
_TOKENS_PER_CONTRASTIVE_PAIR: int = 1024

# Average tokens per steering trajectory (scenario + up to 5 ReAct turns)
_TOKENS_PER_TRAJECTORY: int = 1024

# Inference throughput for Qwen 3.5-27B on A100-80GB (tokens/s, BF16)
_INFERENCE_TOKENS_PER_SEC: float = 50.0

# Anthropic Claude Sonnet API pricing (per call, ~1500 input + 200 output tokens)
# Input: $3/MTok, Output: $15/MTok  =>  ~$0.0075 per call
_JUDGE_COST_PER_CALL_USD: float = 0.0075

# Approximate A100-80GB on-demand cost (cloud, $/hr)
_GPU_COST_PER_HOUR_USD: float = 2.21


def estimate_pipeline_cost(
    n_saes: int = 7,
    n_scenarios: int = 50,
    n_traits: int = 5,
    n_multipliers: int = 4,
    judge_repeats: int = 3,
    hardware: str = "NVIDIA A100 80GB",
) -> PipelineCostSummary:
    """Estimate full pipeline cost without running anything.

    Useful for grant proposals, compute budgeting, and reviewer questions
    about computational requirements.

    The estimate covers four pipeline phases:

    1. **SAE training** — training ``n_saes`` TopK SAEs on 200M tokens each.
    2. **Feature identification** — running 400 contrastive pairs through
       the model + each SAE.
    3. **Steering experiments** — generating trajectories for all
       scenario x trait x multiplier combinations (+ baseline).
    4. **LLM judge evaluation** — scoring trajectories with Claude API.

    Args:
        n_saes: Number of SAEs to train.
        n_scenarios: Number of evaluation scenarios.
        n_traits: Number of behavioral traits.
        n_multipliers: Number of non-zero steering multipliers
            (e.g., 2x, 5x, 10x, 0x).  Baseline (1x) is always included.
        judge_repeats: Number of repeated judgings per trajectory.
        hardware: Hardware description for the report.

    Returns:
        ``PipelineCostSummary`` with ``estimated=True``.
    """
    reports: list[CostReport] = []

    # --- Phase 1: SAE training ---
    tokens_total = _TOKENS_PER_SAE * n_saes
    training_seconds = tokens_total / _SAE_TOKENS_PER_SEC
    training_gpu_hours = training_seconds / 3600.0

    reports.append(CostReport(
        phase="sae_training",
        gpu_hours=training_gpu_hours,
        wall_clock_seconds=training_seconds,
        hardware=hardware,
        notes=(
            f"{n_saes} SAEs x {_TOKENS_PER_SAE:,} tokens each = "
            f"{tokens_total:,} tokens total. "
            f"Throughput assumption: {_SAE_TOKENS_PER_SEC:,.0f} tok/s."
        ),
    ))

    # --- Phase 2: Feature identification ---
    n_contrastive_pairs = 400  # 5 traits x 4 domains x 20 pairs
    feat_tokens = n_contrastive_pairs * _TOKENS_PER_CONTRASTIVE_PAIR * n_saes
    feat_seconds = feat_tokens / _INFERENCE_TOKENS_PER_SEC
    feat_gpu_hours = feat_seconds / 3600.0

    reports.append(CostReport(
        phase="feature_identification",
        gpu_hours=feat_gpu_hours,
        wall_clock_seconds=feat_seconds,
        hardware=hardware,
        notes=(
            f"{n_contrastive_pairs} pairs x {n_saes} SAEs x "
            f"{_TOKENS_PER_CONTRASTIVE_PAIR} tok/pair = {feat_tokens:,} tokens."
        ),
    ))

    # --- Phase 3: Steering experiments ---
    # Total trajectory count: scenarios x traits x (multipliers + 1 baseline)
    n_trajectory_configs = n_scenarios * n_traits * (n_multipliers + 1)
    steer_tokens = n_trajectory_configs * _TOKENS_PER_TRAJECTORY
    steer_seconds = steer_tokens / _INFERENCE_TOKENS_PER_SEC
    steer_gpu_hours = steer_seconds / 3600.0

    reports.append(CostReport(
        phase="steering",
        gpu_hours=steer_gpu_hours,
        wall_clock_seconds=steer_seconds,
        hardware=hardware,
        notes=(
            f"{n_trajectory_configs} trajectories "
            f"({n_scenarios} scenarios x {n_traits} traits x "
            f"{n_multipliers + 1} multiplier levels) x "
            f"{_TOKENS_PER_TRAJECTORY} tok/traj."
        ),
    ))

    # --- Phase 4: LLM judge evaluation ---
    n_judge_calls = n_trajectory_configs * judge_repeats
    judge_cost = n_judge_calls * _JUDGE_COST_PER_CALL_USD
    # Judge API calls are I/O-bound; estimate ~2s per call with rate limiting
    judge_seconds = n_judge_calls * 2.0

    reports.append(CostReport(
        phase="evaluation",
        gpu_hours=0.0,
        api_calls=n_judge_calls,
        api_cost_usd=judge_cost,
        wall_clock_seconds=judge_seconds,
        hardware=hardware,
        notes=(
            f"{n_judge_calls} judge calls "
            f"({n_trajectory_configs} trajectories x {judge_repeats} repeats) "
            f"@ ${_JUDGE_COST_PER_CALL_USD:.4f}/call."
        ),
    ))

    summary = PipelineCostSummary.from_reports(reports, estimated=True)

    # Compute total estimated dollar cost including GPU time
    total_gpu_cost = summary.total_gpu_hours * _GPU_COST_PER_HOUR_USD
    total_cost = total_gpu_cost + summary.total_api_cost_usd

    logger.info(
        "Pipeline cost estimate: %.1f GPU-hours ($%.2f) + %d API calls ($%.2f) "
        "= $%.2f total",
        summary.total_gpu_hours,
        total_gpu_cost,
        summary.total_api_calls,
        summary.total_api_cost_usd,
        total_cost,
    )

    return summary


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    """Check whether CUDA is available without importing torch at module level."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_hardware() -> str:
    """Best-effort hardware detection string.

    Returns:
        Human-readable hardware description, or empty string if
        detection fails.
    """
    parts: list[str] = []

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_bytes = torch.cuda.get_device_properties(0).total_mem
            mem_gb = mem_bytes / (1024 ** 3)
            parts.append(f"{name} ({mem_gb:.0f}GB)")
    except Exception:
        pass

    if not parts:
        parts.append(platform.processor() or platform.machine())

    return ", ".join(parts)


def _estimate_gpu_hours(
    elapsed_seconds: float,
    memory_samples: list[float],
) -> float:
    """Convert wall-clock seconds to GPU-hours.

    If GPU memory samples are available and the *mean* allocated memory is
    below 100 MB, we assume the GPU was idle (e.g., CPU-only data loading)
    and return 0.  Otherwise we count the full elapsed time as GPU time.

    This is a coarse heuristic.  For precise GPU utilisation, use
    ``nvidia-smi`` or ``torch.cuda.Event`` based profiling.

    Args:
        elapsed_seconds: Wall-clock seconds.
        memory_samples: Periodic ``torch.cuda.memory_allocated()`` readings
            in bytes.

    Returns:
        Estimated GPU-hours (0 if GPU appears idle).
    """
    if not memory_samples:
        # No GPU sampling available — assume full GPU utilisation if CUDA
        # is present, 0 otherwise.
        if _cuda_available():
            return elapsed_seconds / 3600.0
        return 0.0

    mean_mem = sum(memory_samples) / len(memory_samples)
    idle_threshold_bytes = 100 * 1024 * 1024  # 100 MB

    if mean_mem < idle_threshold_bytes:
        logger.debug(
            "GPU appears idle (mean alloc %.1f MB < 100 MB threshold), "
            "recording 0 GPU-hours",
            mean_mem / (1024 ** 2),
        )
        return 0.0

    return elapsed_seconds / 3600.0
