"""Unit/integration tests for the checkpoint-resume and step-timing additions in
application/ai_models/tier1_000/train.py — not the full training loop (see test_model.py
for that layer), just the pieces that are cheaply testable in isolation: StepTimingCallback's CV math,
and ModelArtifactRepository's save/restore round-trip.
"""

import pytest
import tensorflow as tf
from application.model_implementations.tier1_000.train import StepTimingCallback
from config import app_config
from infrastructure.model_artifacts import ModelArtifactRepository


class TestStepTimingCallback:
    pytestmark = pytest.mark.unit

    def test_summary_computes_coefficient_of_variation_over_steady_state_steps(self) -> None:
        cb = StepTimingCallback(warmup_steps=2)
        cb._durations = [5.0, 5.0, 1.0, 1.0, 1.0, 1.0]  # first 2 are warmup, excluded

        summary = cb.summary()

        assert summary["step_time_mean_seconds"] == pytest.approx(1.0)
        assert summary["step_time_cv"] == pytest.approx(0.0)

    def test_summary_reports_nonzero_cv_for_uneven_step_durations(self) -> None:
        cb = StepTimingCallback(warmup_steps=0)
        cb._durations = [1.0, 1.0, 1.0, 5.0]  # one stalled step among steady ones

        summary = cb.summary()

        assert summary["step_time_cv"] > 0.5

    def test_summary_handles_no_steady_state_steps_without_dividing_by_zero(self) -> None:
        cb = StepTimingCallback(warmup_steps=5)
        cb._durations = [1.0, 1.0]  # fewer steps than the warmup window

        summary = cb.summary()

        assert summary == {"step_time_mean_seconds": 0.0, "step_time_cv": 0.0}


class TestModelArtifactRepository:
    pytestmark = pytest.mark.integration

    def test_restore_latest_returns_false_when_no_checkpoint_exists(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(app_config, "path_of_data", str(tmp_path))
        repository = ModelArtifactRepository("no_such_run")
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))

        assert repository.restore_latest(checkpoint) is False

    def test_save_then_restore_round_trips_variable_state(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(app_config, "path_of_data", str(tmp_path))
        saved_step = tf.Variable(0, dtype=tf.int64)
        saved_step.assign(42)
        ModelArtifactRepository("round_trip_run").save(tf.train.Checkpoint(step=saved_step))

        restored_step = tf.Variable(0, dtype=tf.int64)
        found = ModelArtifactRepository("round_trip_run").restore_latest(tf.train.Checkpoint(step=restored_step))

        assert found is True
        assert int(restored_step.numpy()) == 42

    def test_different_run_keys_do_not_collide(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(app_config, "path_of_data", str(tmp_path))
        ModelArtifactRepository("run_a").save(tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64)))

        restored_step = tf.Variable(0, dtype=tf.int64)
        found = ModelArtifactRepository("run_b").restore_latest(tf.train.Checkpoint(step=restored_step))

        assert found is False
