"""Time-boxed training run for the Tier-1_000 model, with GPU/CPU/RAM resource monitoring —
the concrete "run training for N minutes and report resource usage" entrypoint.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
import psutil
import tensorflow as tf
from application.model_implementations.shared.base import setup_gpu
from application.model_implementations.tier1_000.datafeeder_input3_outcome1 import (
    build_dataset,
    make_tf_dataset,
    split_bundle,
)
from application.model_implementations.tier1_000.model import TIER1_000_CONFIG, build_tier1000_model
from GPUtil import getGPUs
from infrastructure.model_artifacts import ModelArtifactRepository
from tensorflow import keras as tf_keras


class TrainingPresenter(Protocol):
    """Calling points a Presentation-layer caller (one per design set, e.g.
    `presentation/ai_models/tier1_000_training.py`) can hook into `run_training()` — keeps this
    Application-layer module free of presentation concerns (print/plot/report formatting) itself, per
    the code-layers convention that presentation output belongs in the presentation layer, not
    application. `run_training()` accepts `presenter: TrainingPresenter | None`, defaulting to a no-op,
    so it still runs standalone (tests, ad hoc calls) without one."""

    def on_resume(self, step: int) -> None: ...

    def on_fresh_start(self) -> None: ...

    def on_checkpoint_saved(self, step: int, path: str) -> None: ...

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None: ...

    def on_run_complete(self, report: dict[str, object]) -> None: ...


class _NullPresenter:
    """Default no-op `TrainingPresenter` — lets `run_training()` skip a `presenter is not None` check at
    every calling point."""

    def on_resume(self, step: int) -> None: ...

    def on_fresh_start(self) -> None: ...

    def on_checkpoint_saved(self, step: int, path: str) -> None: ...

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None: ...

    def on_run_complete(self, report: dict[str, object]) -> None: ...


class TimeBudgetCallback(tf_keras.callbacks.Callback):  # type: ignore[misc]  # TensorFlow's Callback is untyped.
    """Stops training as soon as the wall-clock budget is spent — checked every batch (not just every
    epoch), since steps_per_epoch is small enough that several epoch boundaries occur within the
    budget, but the check must still be precise regardless of that choice."""

    def __init__(self, budget_seconds: float) -> None:
        super().__init__()
        self.budget_seconds = budget_seconds
        self.start_time: float = 0.0
        self.total_batches = 0

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        self.start_time = time.time()

    def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        self.total_batches += 1
        if time.time() - self.start_time >= self.budget_seconds:
            self.model.stop_training = True


class StepTimingCallback(tf_keras.callbacks.Callback):  # type: ignore[misc]  # TensorFlow's Callback is untyped.
    """Per-step wall-clock timing — the input-pipeline-stall signal `ResourceSampler`'s fixed 2s poll
    can't resolve, since that poll isn't synced to step boundaries (see
    docs/ML_Forecasting_System_Design/06-ML-Ops.md § saturation & error signals). Excludes the first
    `warmup_steps` (shuffle-buffer fill, cuDNN autotune/graph tracing — one-time costs, not stalls) from
    the coefficient-of-variation computation."""

    def __init__(self, warmup_steps: int = 5) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self._step_start = 0.0
        self._durations: list[float] = []

    def on_train_batch_begin(self, batch: int, logs: dict[str, float] | None = None) -> None:
        self._step_start = time.time()

    def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        self._durations.append(time.time() - self._step_start)

    def summary(self) -> dict[str, float]:
        steady_state = np.array(self._durations[self.warmup_steps :])
        if steady_state.size == 0:
            return {"step_time_mean_seconds": 0.0, "step_time_cv": 0.0}
        mean = float(steady_state.mean())
        return {"step_time_mean_seconds": mean, "step_time_cv": float(steady_state.std() / mean) if mean > 0 else 0.0}


class PeriodicCheckpointCallback(tf_keras.callbacks.Callback):  # type: ignore[misc]  # TensorFlow's Callback is untyped.
    """Saves model+optimizer state every `interval_seconds` of wall-clock training time — the resume
    mechanism this project's single-GPU, potentially hours-long runs need against an interrupted process
    (crash, manual stop, WSL2 restart) without losing already-trained progress. A final save also runs
    once training stops (`run_training()`'s `finally` block), so at most `interval_seconds` of progress
    since the last periodic save is ever at risk, not a whole run."""

    def __init__(
        self,
        repository: ModelArtifactRepository,
        checkpoint: tf.train.Checkpoint,
        step_counter: tf.Variable,
        interval_seconds: float,
        presenter: TrainingPresenter,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.checkpoint = checkpoint
        self.step_counter = step_counter
        self.interval_seconds = interval_seconds
        self.presenter = presenter
        self._last_save_time = 0.0

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        self._last_save_time = time.time()

    def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        self.step_counter.assign_add(1)
        now = time.time()
        if now - self._last_save_time >= self.interval_seconds:
            path = self.repository.save(self.checkpoint)
            self._last_save_time = now
            self.presenter.on_checkpoint_saved(int(self.step_counter.numpy()), path)


class PresenterEpochCallback(tf_keras.callbacks.Callback):  # type: ignore[misc]  # TensorFlow's Callback is untyped.
    """Forwards Keras' own epoch-end signal (already carrying both `loss` and `val_loss` in `logs`,
    since `validation_data` is passed to `model.fit()`) to the presentation calling point — the
    Application layer stays print-free, per `TrainingPresenter`'s docstring."""

    def __init__(self, presenter: TrainingPresenter) -> None:
        super().__init__()
        self.presenter = presenter

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        self.presenter.on_epoch_end(epoch, dict(logs or {}))


@dataclass
class ResourceSampler:
    """Background-thread poller for GPU/CPU/RAM during model.fit() — Keras callbacks only fire between
    steps, which wouldn't capture usage *during* a step's GPU-bound compute."""

    interval_seconds: float = 2.0
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _poll_loop(self) -> None:
        process = psutil.Process()
        psutil.cpu_percent(interval=None)  # first call always returns 0.0 — prime it, discard
        while not self._stop_event.is_set():
            try:
                gpu = getGPUs()[0]
                vm = psutil.virtual_memory()
                self.samples.append(
                    {
                        "time": time.time(),
                        "cpu_percent_system": psutil.cpu_percent(interval=None),
                        "process_rss_mb": process.memory_info().rss / 1024**2,
                        "ram_used_gb": vm.used / 1024**3,
                        "gpu_util_percent": gpu.load * 100,
                        "gpu_mem_used_mb": gpu.memoryUsed,
                        "gpu_mem_total_mb": gpu.memoryTotal,
                    }
                )
            except (IndexError, ValueError):
                # NVML queries occasionally fail transiently under WSL2's GPU passthrough
                # ("Failed to initialize NVML: N/A", observed in practice) — skip this sample
                # rather than let one bad poll kill the whole background thread.
                pass
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"n_samples": 0}
        cols = {key: np.array([s[key] for s in self.samples]) for key in self.samples[0] if key != "time"}
        summary: dict[str, float] = {"n_samples": len(self.samples)}
        for key, values in cols.items():
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_max"] = float(values.max())
        return summary


def run_training(
    symbol: str = "BTCUSDT",
    date_range_str: str = "22-02-10.00-00T24-10-31.23-59",
    budget_seconds: float = 600.0,
    steps_per_epoch: int = 50,
    validation_steps: int = 10,
    run_key: str = "tier1_000",
    checkpoint_interval_seconds: float = 600.0,
    reset_params: bool = False,
    presenter: TrainingPresenter | None = None,
) -> dict[str, object]:
    setup_gpu()
    presenter = presenter or _NullPresenter()

    print(f"[data] building dataset for {symbol} {date_range_str} ...")
    t0 = time.time()
    bundle = build_dataset(symbol, date_range_str)
    data_build_seconds = time.time() - t0
    train_bundle, val_bundle = split_bundle(bundle, val_fraction=0.1)
    print(
        f"[data] built in {data_build_seconds:.1f}s: {bundle.n_samples} samples "
        f"({train_bundle.n_samples} train / {val_bundle.n_samples} val), "
        f"anchors {bundle.anchor_index.min()} .. {bundle.anchor_index.max()}"
    )

    config = TIER1_000_CONFIG
    batch_size = cast(int, config["batch_size"])
    train_ds = make_tf_dataset(train_bundle, batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_bundle, batch_size, shuffle=False)

    model = build_tier1000_model(config)
    print(f"[model] compiled, batch_size={batch_size} (param count available only after the first call/step)")

    # tf.train.Checkpoint, not model.save()/load_model() — the subclassed layers here have no
    # get_config()/from_config(), so only the object-based checkpoint API round-trips reliably (see
    # infrastructure/model_artifacts's module docstring). Must restore before the first model.fit() step
    # so TF's deferred restore can match model/optimizer variables that don't exist yet.
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    checkpoint = tf.train.Checkpoint(step=global_step, model=model, optimizer=model.optimizer)
    repository = ModelArtifactRepository(run_key)
    resumed = not reset_params and repository.restore_latest(checkpoint)
    if resumed:
        presenter.on_resume(int(global_step.numpy()))
        print(f"[checkpoint] resumed from step {int(global_step.numpy())} ({repository.checkpoint_dir})")
    else:
        presenter.on_fresh_start()
        reason = "reset_params=True" if reset_params else "no checkpoint found"
        print(f"[checkpoint] starting from freshly-initialized parameters ({reason})")

    time_budget_cb = TimeBudgetCallback(budget_seconds)
    checkpoint_cb = PeriodicCheckpointCallback(
        repository, checkpoint, global_step, checkpoint_interval_seconds, presenter
    )
    step_timing_cb = StepTimingCallback()
    sampler = ResourceSampler(interval_seconds=2.0)

    sampler.start()
    train_start = time.time()
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1_000_000,  # unbounded — TimeBudgetCallback is the real stop condition
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[time_budget_cb, checkpoint_cb, step_timing_cb, PresenterEpochCallback(presenter)],
            verbose=2,
        )
    finally:
        train_elapsed = time.time() - train_start
        sampler.stop()
        # final save on the way out (including on an exception) — bounds the unsaved-progress window to
        # checkpoint_interval_seconds even if the run stops between periodic saves.
        final_checkpoint_path = repository.save(checkpoint)

    tf_gpu_mem = tf.config.experimental.get_memory_info("GPU:0")
    report: dict[str, object] = {
        "symbol": symbol,
        "date_range": date_range_str,
        "n_samples_total": bundle.n_samples,
        "n_samples_train": train_bundle.n_samples,
        "n_samples_val": val_bundle.n_samples,
        "model_params": model.count_params(),
        "batch_size": batch_size,
        "data_build_seconds": round(data_build_seconds, 1),
        "train_elapsed_seconds": round(train_elapsed, 1),
        "total_batches": time_budget_cb.total_batches,
        "samples_per_second": (
            round(time_budget_cb.total_batches * batch_size / train_elapsed, 2) if train_elapsed > 0 else 0.0
        ),
        "final_train_loss": float(history.history.get("loss", [float("nan")])[-1]),
        "final_val_loss": float(history.history.get("val_loss", [float("nan")])[-1]),
        "resumed_from_checkpoint": resumed,
        "global_step": int(global_step.numpy()),
        "final_checkpoint_path": final_checkpoint_path,
        "tf_gpu_mem_current_mb": round(tf_gpu_mem["current"] / 1024**2, 1),
        "tf_gpu_mem_peak_mb": round(tf_gpu_mem["peak"] / 1024**2, 1),
        **sampler.summary(),
        **step_timing_cb.summary(),
    }
    print("\n=== resource usage report ===")
    print(json.dumps(report, indent=2))
    presenter.on_run_complete(report)
    return report


if __name__ == "__main__":
    run_training()
