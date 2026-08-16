# mypy: ignore-errors
"""Time-boxed training run for the Tier-1_000 model, with GPU/CPU/RAM resource monitoring —
the concrete "run training for N minutes and report resource usage" entrypoint.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import tensorflow as tf
from ai_modelling.base import setup_gpu
from ai_modelling.tier1_000.datafeeder import build_dataset, make_tf_dataset, split_bundle
from ai_modelling.tier1_000.model import TIER1_000_CONFIG, build_tier1000_model
from GPUtil import getGPUs
from tensorflow import keras as tf_keras


class TimeBudgetCallback(tf_keras.callbacks.Callback):
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
) -> dict[str, object]:
    setup_gpu()

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
    batch_size = int(config["batch_size"])
    train_ds = make_tf_dataset(train_bundle, batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_bundle, batch_size, shuffle=False)

    model = build_tier1000_model(config)
    print(f"[model] compiled, batch_size={batch_size} (param count available only after the first call/step)")

    time_budget_cb = TimeBudgetCallback(budget_seconds)
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
            callbacks=[time_budget_cb],
            verbose=2,
        )
    finally:
        train_elapsed = time.time() - train_start
        sampler.stop()

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
        "tf_gpu_mem_current_mb": round(tf_gpu_mem["current"] / 1024**2, 1),
        "tf_gpu_mem_peak_mb": round(tf_gpu_mem["peak"] / 1024**2, 1),
        **sampler.summary(),
    }
    print("\n=== resource usage report ===")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_training()
