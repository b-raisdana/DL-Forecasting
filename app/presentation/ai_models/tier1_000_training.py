"""Presentation executor for the Tier-1_000 design set — the CLI entrypoint that drives
`application.ai_models.tier1_000.train.run_training()` and renders its calling points
(resume/fresh-start, periodic checkpoint saves, per-epoch loss, final report) to the console.

Each design set owns its own executor like this one, alongside its own
`application/ai_models/<design_set>/` module — e.g. a future Tier-1_001 gets
`presentation/ai_models/tier1_001_training.py`, not a shared abstraction guessed at ahead
of a second real user.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pytz
from application.model_implementations.tier1_000.train import TrainingPresenter, run_training
from config import app_config
from infrastructure.options_settings import get_oldest_available_timestamp


class ConsoleTrainingPresenter:
    """`TrainingPresenter` implementation for interactive/console use — the presentation-layer half of
    the split `train.py`'s `TrainingPresenter` protocol enforces (Application never prints presenter
    output itself)."""

    def on_resume(self, step: int) -> None:
        print(f"[presenter] resuming from a saved checkpoint at step {step}")

    def on_fresh_start(self) -> None:
        print("[presenter] no checkpoint resumed — training from freshly-initialized parameters")

    def on_checkpoint_saved(self, step: int, path: str) -> None:
        print(f"[presenter] checkpoint saved at step {step}: {path}")

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        print(f"[presenter] epoch {epoch}: loss={loss:.4f} val_loss={val_loss:.4f}")

    def on_run_complete(self, report: dict[str, object]) -> None:
        print(
            f"[presenter] run complete: {report['total_batches']} steps, "
            f"final_train_loss={report['final_train_loss']:.4f}, final_val_loss={report['final_val_loss']:.4f}, "
            f"global_step={report['global_step']}, checkpoint={report['final_checkpoint_path']}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tier-1_000 design set, with checkpoint/resume.")
    parser.add_argument(
        "--symbol",
        default=None,
        help=(
            "symbol to train on; if omitted, sequentially trains on every symbol in "
            "app_config.TRAIN_SYMBOLS (app_config.SYMBOLS minus the reserved validation symbol)"
        ),
    )
    parser.add_argument(
        "--date-range", default=None, help="date range for training (format: YY-MM-DD.HH-MMTYY-MM-DD.HH-MM)"
    )
    parser.add_argument("--budget-minutes", type=float, default=10.0, help="wall-clock training budget")
    parser.add_argument("--checkpoint-interval-minutes", type=float, default=10.0)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--run-key", default="tier1_000", help="checkpoint namespace under <data>/model_artifacts/")
    parser.add_argument(
        "--reset-params",
        action="store_true",
        help="ignore any saved checkpoint and start from freshly-initialized parameters",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    presenter: TrainingPresenter = ConsoleTrainingPresenter()
    symbols = [args.symbol] if args.symbol else app_config.TRAIN_SYMBOLS
    for symbol in symbols:
        run_key = args.run_key if args.symbol else f"{args.run_key}_{symbol}"

        # Determine date_range: use provided arg or fetch from broker/cache
        date_range_str = args.date_range
        if date_range_str is None:
            print(f"[presenter] fetching oldest available timestamp for {symbol}...")
            oldest_timestamp = get_oldest_available_timestamp(app_config.under_process_exchange.lower(), symbol)
            start_str = oldest_timestamp.strftime("%y-%m-%d.%H-%M")
            end = datetime.now(pytz.UTC)
            end_str = end.strftime("%y-%m-%d.%H-%M")
            date_range_str = f"{start_str}T{end_str}"
            print(f"[presenter] using date range from oldest available data: {date_range_str}")

        print(f"[presenter] === training {symbol} (run_key={run_key}) ===")
        run_training(
            symbol=symbol,
            date_range_str=date_range_str,
            budget_seconds=args.budget_minutes * 60.0,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps,
            run_key=run_key,
            checkpoint_interval_seconds=args.checkpoint_interval_minutes * 60.0,
            reset_params=args.reset_params,
            presenter=presenter,
        )
