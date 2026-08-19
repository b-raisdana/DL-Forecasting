"""Architecture + hyperparameter search, replacing the old flat DEAP-GA optimizer.

Design follows docs/planing.md "optimization strategy":
- "which base architecture" and "which hyperparameters within it" are searched as ONE
  study (architecture is just a categorical param with conditional sub-params), so bad
  architectures get pruned early instead of each getting an exhaustive GA run.
- TPE (sample-efficient on a single GPU) + Hyperband pruning stand in for plain GA for
  the main search. GA (NSGA-II) is kept for a later multi-objective refinement stage,
  once trading KPIs are available, where a Pareto front actually matters.
- Per-trial cost is measured (profile_trial_cost), not assumed as a flat 0.5h/1h.
- Fitness is currently val_loss (a training-time proxy) -- see compute_fitness() for
  why, and what has to change once a backtest module exists.
"""

import gc
import sys
import time

import numpy as np
import optuna
from archive_not_used_trash.application.dataset_generation.feeders.ram_batch import build_ram_dataset
from archive_not_used_trash.application.model_implementations.cnn_lstm.cnn_lstm_model import CNNLSTMModel
from archive_not_used_trash.application.model_implementations.cnn_lstm_attention.cnn_lstm_attention_model import (
    build_cnn_lstm_attention_model,
)
from helper.logging.do_log import log_e, log_i
from tensorflow import config as tf_config
from tensorflow import keras as tf_keras

from application.model_implementations.shared.base import build_model, master_x_shape, pre_train_model, setup_gpu

BATCH_SIZE = 80
Y_LEN = 2
ARCHITECTURES = ("cnn_lstm_attention", "cnn_lstm")

DEFAULT_PARAMS = {
    "cnn_lstm_attention": dict(
        architecture="cnn_lstm_attention",
        cnn_filters=64,
        cnn_count=3,
        dropout_rate=0.3,
        lstm_units_1=256,
        lstm_units_2=128,
        kernel_step=2,
        num_heads=8,
        key_dim=16,
    ),
    "cnn_lstm": dict(
        architecture="cnn_lstm",
        cnn_filters=64,
        cnn_count=2,
        dropout_rate=0.3,
        lstm_units_1=128,
        lstm_units_2=64,
        cnn_kernel_growing_steps=2,
        dense_units=128,
    ),
}

_dataset = None


def get_dataset(batch_size: int = BATCH_SIZE):
    global _dataset
    if _dataset is None:
        _dataset = build_ram_dataset(batch_size=batch_size)
    return _dataset


# --------------------------------------------------------------------------- search space


def suggest_params(trial: optuna.Trial) -> dict:
    architecture = trial.suggest_categorical("architecture", ARCHITECTURES)
    params = {
        "architecture": architecture,
        "cnn_filters": trial.suggest_int("cnn_filters", 32, 128, step=16),
        "cnn_count": trial.suggest_int("cnn_count", 1, 4),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "lstm_units_1": trial.suggest_int("lstm_units_1", 16, 512, step=16),
        "lstm_units_2": trial.suggest_int("lstm_units_2", 16, 512, step=16),
    }
    if architecture == "cnn_lstm_attention":
        params["kernel_step"] = trial.suggest_int("kernel_step", 1, 4)
        params["num_heads"] = trial.suggest_int("num_heads", 2, 8)
        params["key_dim"] = trial.suggest_int("key_dim", 8, 32)
    else:
        params["cnn_kernel_growing_steps"] = trial.suggest_int("cnn_kernel_growing_steps", 1, 4)
        params["dense_units"] = trial.suggest_int("dense_units", 32, 256, step=32)
    return params


def build_model_from_params(params: dict, y_len: int, x_shape: dict, batch_size: int = BATCH_SIZE):
    if params["architecture"] == "cnn_lstm_attention":
        model = build_cnn_lstm_attention_model(
            y_len=y_len,
            input_shapes=x_shape,
            cnn_filters=params["cnn_filters"],
            lstm_units=[params["lstm_units_1"], params["lstm_units_2"]],
            cnn_count=params["cnn_count"],
            kernel_step=params["kernel_step"],
            dropout_rate=params["dropout_rate"],
            num_heads=params["num_heads"],
            key_dim=params["key_dim"],
        )
    elif params["architecture"] == "cnn_lstm":
        model = CNNLSTMModel(
            y_len=y_len,
            cnn_filters=params["cnn_filters"],
            lstm_units_list=[params["lstm_units_1"], params["lstm_units_2"]],
            dense_units=params["dense_units"],
            cnn_count=params["cnn_count"],
            cnn_kernel_growing_steps=params["cnn_kernel_growing_steps"],
            dropout_rate=params["dropout_rate"],
        )
    else:
        raise ValueError(f"Unknown architecture {params['architecture']!r}")
    build_model(batch_size, model, x_shape)
    return model


# --------------------------------------------------------------------------- profiling / budget (item 3 & 4)


def profile_trial_cost(params: dict, y_len: int, x_shape: dict, dataset, steps: int = 5) -> dict:
    """Measure real per-step wall-clock and peak GPU memory for one config, instead of
    assuming a flat per-trial time. Run once per architecture before sizing a study."""
    model = build_model_from_params(params, y_len, x_shape)
    model.fit(dataset.take(1), steps_per_epoch=1, epochs=1, verbose=0)  # warmup / XLA compile
    try:
        tf_config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass
    start = time.time()
    model.fit(dataset, steps_per_epoch=steps, epochs=1, verbose=0)
    elapsed = time.time() - start
    peak_mb = None
    try:
        peak_mb = tf_config.experimental.get_memory_info("GPU:0")["peak"] / (1024**2)
    except Exception:
        pass
    tf_keras.backend.clear_session()
    gc.collect()
    return {
        "architecture": params["architecture"],
        "seconds_per_step": elapsed / steps,
        "peak_gpu_mb": peak_mb,
    }


def estimate_total_budget(
    seconds_per_step: float, steps_per_epoch: int, epochs_per_trial: int, n_trials: int, label: str = ""
) -> float:
    seconds_per_trial = seconds_per_step * steps_per_epoch * epochs_per_trial
    total_seconds = seconds_per_trial * n_trials
    log_i(
        f"[budget:{label}] ~{seconds_per_trial / 60:.1f} min/trial (if it runs to the full "
        f"epoch budget uncapped by pruning) x {n_trials} trials <= {total_seconds / 3600:.2f} "
        f"GPU-hours ({total_seconds / 86400:.2f} days). Actual total will be lower: Hyperband "
        f"prunes most trials well before epochs_per_trial."
    )
    return total_seconds


def max_trials_for_budget(
    seconds_per_step: float, steps_per_epoch: int, epochs_per_trial: int, budget_hours: float
) -> int:
    seconds_per_trial = seconds_per_step * steps_per_epoch * epochs_per_trial
    return max(1, int(budget_hours * 3600 / seconds_per_trial))


# --------------------------------------------------------------------------- training / pruning / fitness


class OptunaPruningCallback(tf_keras.callbacks.Callback):
    """Reports val_loss to the trial each epoch and stops training early (TrialPruned)
    once Optuna's pruner judges the trial unpromising relative to others -- this is the
    early-stopping/pruning mechanism docs/planing.md asks for, and it also does double
    duty as the "cheap architecture screening" step: bad architectures get cut here too,
    since architecture is just another param in the same study."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        if np.isnan(value) or np.isinf(value):
            raise optuna.TrialPruned(f"{self.monitor} is {value} at epoch {epoch}")
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {epoch} with {self.monitor}={value:.5f}")


class TimeLimitCallback(tf_keras.callbacks.Callback):
    def __init__(self, time_limit_s: float):
        super().__init__()
        self.time_limit_s = time_limit_s
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if time.time() - self.start_time > self.time_limit_s:
            self.model.stop_training = True


def compute_fitness(history) -> float:
    """Training-time proxy fitness (val_loss). This is NOT the trading objective
    docs/planing.md wants for final selection (win rate / profit factor / Sortino /
    max drawdown from simulated trades) -- that needs a backtest/trade-simulator module
    that doesn't exist in this repo yet (see the open "evaluation & error metrics"
    section). Once it exists, swap this function's body for that KPI computation, or
    better, migrate the objective to run_kpi_refinement() below, which is already wired
    for it. Everything else (screening, pruning, budget calc) stays the same."""
    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        raise optuna.TrialPruned("no val_loss recorded")
    best = min(val_losses)
    if np.isnan(best):
        raise optuna.TrialPruned("nan val_loss")
    return best


def objective(
    trial: optuna.Trial,
    epochs: int,
    steps_per_epoch: int,
    validation_steps: int,
    time_limit_s: float,
    patience: int = 5,
) -> float:
    params = suggest_params(trial)
    log_i(f"[trial {trial.number}] {params}")
    model = build_model_from_params(params, Y_LEN, master_x_shape)
    dataset = get_dataset()
    model.fit(dataset.take(1), steps_per_epoch=1, epochs=1, verbose=0)  # warmup
    callbacks = [
        OptunaPruningCallback(trial),
        tf_keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        TimeLimitCallback(time_limit_s),
    ]
    try:
        history = model.fit(
            dataset,
            validation_data=dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=0,
        )
        return compute_fitness(history)
    finally:
        tf_keras.backend.clear_session()
        gc.collect()


# --------------------------------------------------------------------------- stage 1: TPE + Hyperband study


def run_study(
    n_trials: int = 40,
    timeout_hours: float | None = None,
    epochs: int = 30,
    steps_per_epoch: int = 50,
    validation_steps: int = 10,
    trial_time_limit_s: float = 3600,
    study_name: str = "cnn_lstm_search",
    storage: str | None = None,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs, reduction_factor=3)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            time_limit_s=trial_time_limit_s,
        ),
        n_trials=n_trials,
        timeout=int(timeout_hours * 3600) if timeout_hours else None,
        gc_after_trial=True,
    )
    log_i(f"[optuna] best trial #{study.best_trial.number}: value={study.best_value:.5f} params={study.best_params}")
    return study


# --------------------------------------------------------------------------- stage 2: NSGA-II KPI refinement


def run_kpi_refinement(
    kpi_fn,
    base_study: optuna.Study | None = None,
    top_k: int = 5,
    n_trials: int = 20,
    epochs: int = 30,
    steps_per_epoch: int = 50,
    validation_steps: int = 10,
    patience: int = 5,
    study_name: str = "cnn_lstm_kpi_refine",
    storage: str | None = None,
) -> optuna.Study:
    """Multi-objective (Sortino vs. max drawdown) refinement over the top candidates from
    run_study(), using NSGA-II -- this is where GA earns its keep, producing a Pareto
    front across competing trading KPIs instead of one scalar.

    kpi_fn(model, val_dataset) -> {"sortino": float, "max_drawdown": float} must be
    supplied by the caller: it requires simulating trades from the model's TP/SL
    predictions, which needs a backtest/trade-simulator module that isn't built yet
    (see docs/planing.md, "evaluation & error metrics"). Until that exists this function
    intentionally refuses to run rather than fabricate a KPI.
    """
    if kpi_fn is None:
        raise NotImplementedError(
            "run_kpi_refinement needs kpi_fn(model, val_dataset) -> {'sortino', 'max_drawdown'}, "
            "computed by simulating trades from the model's TP/SL predictions. That backtest "
            "module doesn't exist in this repo yet -- see the open 'evaluation & error metrics' "
            "section in docs/planing.md. Until it exists, use run_study()'s val_loss ranking."
        )

    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # maximize sortino, minimize max_drawdown
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    if base_study is not None:
        top_trials = sorted((t for t in base_study.trials if t.value is not None), key=lambda t: t.value)[:top_k]
        for t in top_trials:
            study.enqueue_trial(t.params)

    def _objective(trial: optuna.Trial):
        params = suggest_params(trial)
        model = build_model_from_params(params, Y_LEN, master_x_shape)
        dataset = get_dataset()
        model.fit(dataset.take(1), steps_per_epoch=1, epochs=1, verbose=0)
        model.fit(
            dataset,
            validation_data=dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[
                tf_keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
            ],
            verbose=0,
        )
        kpis = kpi_fn(model, dataset)
        tf_keras.backend.clear_session()
        gc.collect()
        return kpis["sortino"], kpis["max_drawdown"]

    study.optimize(_objective, n_trials=n_trials, gc_after_trial=True)
    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--trial-time-limit-min", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=None)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only profile the default config per architecture, print the "
        "projected budget, and exit without running the study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna_studies.db) for a "
        "resumable/parallel study instead of an in-memory one.",
    )
    args = parser.parse_args()

    setup_gpu()
    pre_train_model()

    for arch in ARCHITECTURES:
        try:
            cost = profile_trial_cost(DEFAULT_PARAMS[arch], Y_LEN, master_x_shape, get_dataset(), steps=5)
            log_i(f"[profile] {arch}: {cost['seconds_per_step']:.2f}s/step, peak GPU {cost['peak_gpu_mb']}MB")
            estimate_total_budget(
                cost["seconds_per_step"], args.steps_per_epoch, args.epochs, args.n_trials, label=arch
            )
        except Exception as e:
            log_e(f"[profile] {arch} failed: {e}")

    if args.profile_only:
        sys.exit(0)

    run_study(
        n_trials=args.n_trials,
        timeout_hours=args.timeout_hours,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        trial_time_limit_s=args.trial_time_limit_min * 60,
        storage=args.storage,
    )
