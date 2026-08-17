"""ModelArtifactRepository (.keras save/load) — closes docs/todos/03-infrastructure.md item 12.

Wraps `tf.train.Checkpoint`/`CheckpointManager` (TF's own object-based checkpoint idiom, per the
lib-first skill — not a hand-rolled save format) rather than `model.save()`: the model-implementation
classes here (see `application/model_implementations/tier1_000/model.py`) are subclassed `tf.keras.Model`s
with no `get_config`/`from_config` overrides, so a full `.keras`/SavedModel round-trip isn't reliable for
them, while a `tf.train.Checkpoint` only needs the variables to exist, and supports the deferred-restore
case (weights/optimizer slots not yet built because the caller restores before the first `model.fit()`
step) transparently.
"""

from __future__ import annotations

import os

import tensorflow as tf
from config import app_config


class ModelArtifactRepository:
    """One instance per `run_key` (e.g. a design-set name) — owns that run's checkpoint directory under
    `<data>/model_artifacts/<run_key>/`, so different design sets' saved parameters never collide."""

    def __init__(self, run_key: str, max_to_keep: int = 3) -> None:
        self.checkpoint_dir = os.path.join(app_config.path_of_data, "model_artifacts", run_key)
        self._max_to_keep = max_to_keep
        self._manager: tf.train.CheckpointManager | None = None

    def _manager_for(self, checkpoint: tf.train.Checkpoint) -> tf.train.CheckpointManager:
        if self._manager is None:
            self._manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_dir, max_to_keep=self._max_to_keep)
        return self._manager

    def restore_latest(self, checkpoint: tf.train.Checkpoint) -> bool:
        """Restores `checkpoint`'s tracked objects (model/optimizer/step) from the latest checkpoint in
        this run's directory, if any. Returns whether one was found and restored — callers fall back to
        the caller's already-freshly-initialized state when this is False, which is also the correct
        behavior for "no previous saved parameters found," with no separate flag needed for that case.
        Must be called before the tracked model/optimizer are first built (before the first `model.fit()`
        step) so TF's deferred restore can match variables created later.
        """
        manager = self._manager_for(checkpoint)
        latest = manager.latest_checkpoint
        if latest is None:
            return False
        checkpoint.restore(latest).expect_partial()
        return True

    def save(self, checkpoint: tf.train.Checkpoint) -> str:
        manager = self._manager_for(checkpoint)
        return manager.save()
