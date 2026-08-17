"""TensorFlow checkpoint persistence for model-training runs."""

from __future__ import annotations

import os

import tensorflow as tf
from config import app_config


class ModelArtifactRepository:
    """Own the checkpoint directory for one training run."""

    def __init__(self, run_key: str, max_to_keep: int = 3) -> None:
        self.checkpoint_dir = os.path.join(app_config.path_of_data, "model_artifacts", run_key)
        self._max_to_keep = max_to_keep
        self._manager: tf.train.CheckpointManager | None = None

    def _manager_for(self, checkpoint: tf.train.Checkpoint) -> tf.train.CheckpointManager:
        if self._manager is None:
            self._manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_dir, max_to_keep=self._max_to_keep)
        return self._manager

    def restore_latest(self, checkpoint: tf.train.Checkpoint) -> bool:
        """Restore the newest checkpoint, returning whether one existed."""
        latest = self._manager_for(checkpoint).latest_checkpoint
        if latest is None:
            return False
        checkpoint.restore(latest).expect_partial()
        return True

    def save(self, checkpoint: tf.train.Checkpoint) -> str:
        """Save the checkpoint and return TensorFlow's generated path."""
        return self._manager_for(checkpoint).save()
