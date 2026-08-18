"""Unit tests for Tier1000Model (app/ai_modelling/tier1_000/model.py) — architecture wiring and output
shapes, not training quality. Uses a deliberately small config (not TIER1_000_CONFIG's real "1-base"
sizes) so the test runs fast on CPU; the real sizes are exercised by the actual training run, not here.

Builds Tier1000Model directly rather than through build_tier1000_model() to avoid that factory's
side effect of setting Keras' global mixed-precision policy, which would leak into other tests running
in the same pytest process.
"""

import numpy as np
import pytest
import tensorflow as tf
from application.model_implementations.tier1_000.model import (
    AUX_FEATURE_DIM,
    BRANCH_TIMEFRAMES,
    BRANCH_WINDOW_LENGTHS,
    CANDLE_FEATURE_DIM,
    Tier1000Model,
    gaussian_nll_loss,
)

pytestmark = pytest.mark.unit

_SMALL_CONFIG: dict[str, object] = {
    "ModernTCN_kernel_size": 3,
    "ModernTCN_depth": 2,
    "ModernTCN_channels": 8,
    "LSTM_layers": 1,
    "LSTM_hidden_units": 8,
    "Perceiver_latent_tokens": 4,
    "Perceiver_latent_dim": 8,
    "Perceiver_cross_attention_layers": 1,
    "Perceiver_heads": 2,
    "GQA_layers": 1,
    "GQA_heads": 2,
    "GQA_kv_heads": 1,
    "MLP_depth": 1,
    "MLP_width": 8,
    "dropout": 0.0,
    "pooling_method": "last_token",
}
_BATCH = 3


@pytest.fixture
def sample_inputs() -> dict[str, np.ndarray]:
    inputs = {
        tf_name: np.random.randn(_BATCH, BRANCH_WINDOW_LENGTHS[tf_name], CANDLE_FEATURE_DIM).astype(np.float32)
        for tf_name in BRANCH_TIMEFRAMES
    }
    inputs["auxiliary_features"] = np.random.randn(_BATCH, AUX_FEATURE_DIM).astype(np.float32)
    return inputs


def test_forward_pass_produces_expected_output_shapes(sample_inputs: dict[str, np.ndarray]) -> None:
    model = Tier1000Model(_SMALL_CONFIG)

    out = model(sample_inputs, training=False)

    assert out["action"].shape == (_BATCH, 3)
    assert out["mfe_params"].shape == (_BATCH, 2)
    assert out["rer_params"].shape == (_BATCH, 2)


def test_action_head_is_a_valid_probability_distribution(sample_inputs: dict[str, np.ndarray]) -> None:
    model = Tier1000Model(_SMALL_CONFIG)

    action = model(sample_inputs, training=False)["action"].numpy()

    assert (action >= 0).all()
    np.testing.assert_allclose(action.sum(axis=1), 1.0, rtol=1e-5)


def test_mfe_and_rer_heads_respect_their_activations(sample_inputs: dict[str, np.ndarray]) -> None:
    """mfe: softplus mean/std (>=0). rer: sigmoid mean in [0,1], softplus std (>=0)."""
    model = Tier1000Model(_SMALL_CONFIG)

    out = model(sample_inputs, training=False)
    mfe_params = out["mfe_params"].numpy()
    rer_params = out["rer_params"].numpy()

    assert (mfe_params >= 0).all()
    assert (rer_params[:, 1] >= 0).all()  # std
    assert ((rer_params[:, 0] >= 0) & (rer_params[:, 0] <= 1)).all()  # mean


def test_train_on_batch_reduces_loss_after_several_steps(sample_inputs: dict[str, np.ndarray]) -> None:
    model = Tier1000Model(_SMALL_CONFIG)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss={
            "action": tf.keras.losses.CategoricalCrossentropy(),
            "mfe_params": gaussian_nll_loss,
            "rer_params": gaussian_nll_loss,
        },
    )
    targets = {
        "action": tf.one_hot([0, 1, 2], 3),
        "mfe_params": np.array([[1.0], [2.0], [1.5]], dtype=np.float32),
        "rer_params": np.array([[0.1], [0.05], [0.15]], dtype=np.float32),
    }

    first_loss = model.train_on_batch(sample_inputs, targets)[0]
    for _ in range(10):
        last_loss = model.train_on_batch(sample_inputs, targets)[0]

    assert last_loss < first_loss


def test_gaussian_nll_loss_is_lower_for_a_closer_mean() -> None:
    y_true = np.array([1.0, 1.0], dtype=np.float32)
    close_pred = np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32)
    far_pred = np.array([[5.0, 0.5], [5.0, 0.5]], dtype=np.float32)

    close_loss = gaussian_nll_loss(y_true, close_pred).numpy()
    far_loss = gaussian_nll_loss(y_true, far_pred).numpy()

    assert close_loss < far_loss
