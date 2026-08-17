"""Unit tests for add_mfe_mae_om_labels() (mfe_mae_om_labels.py) against
docs/ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md § TP / MAE / OM labels.

Scenarios are hand-constructed so the expected action is unambiguous by inspection (monotonic
rise/fall/flat), not captured from running the function — new spec-conformance code, not legacy
behavior to pin.
"""

import numpy as np
import pandas as pd
import pytest
from application.dataset_generation.mfe_mae_om_labels import HORIZON_BARS, RER_BOUND, add_mfe_mae_om_labels

pytestmark = pytest.mark.unit

_ANCHOR_COUNT = 20
_N = HORIZON_BARS + _ANCHOR_COUNT
_WARMUP_15MIN_CANDLES = 2000  # far more than ATR_FLOOR_LENGTH=255, so the ATR floor is never NaN


def _five_min_ohlc(close: np.ndarray, half_range: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp("2024-01-11", tz="UTC"), periods=len(close), freq="5min")
    df = pd.DataFrame(
        {
            "open": close - half_range * 0.5,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def warm_fifteen_min_ohlc() -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), periods=_WARMUP_15MIN_CANDLES, freq="15min")
    close = np.linspace(90, 200, _WARMUP_15MIN_CANDLES)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=idx)
    df.index.name = "date"
    return df


def test_monotonic_rise_is_all_long_actionable(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    close = np.linspace(100, 100 + _N, _N)
    labels = add_mfe_mae_om_labels(_five_min_ohlc(close), warm_fifteen_min_ohlc)

    assert len(labels) == _ANCHOR_COUNT
    assert labels["action_long"].eq(1.0).all()
    assert labels["action_short"].eq(0.0).all()
    assert labels["action_none"].eq(0.0).all()


def test_monotonic_fall_is_all_short_actionable(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    close = np.linspace(200, 200 - _N, _N)
    labels = add_mfe_mae_om_labels(_five_min_ohlc(close), warm_fifteen_min_ohlc)

    assert labels["action_short"].eq(1.0).all()
    assert labels["action_long"].eq(0.0).all()
    assert labels["action_none"].eq(0.0).all()


def test_flat_oscillating_price_is_all_none(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    close = 100 + np.sin(np.linspace(0, 6, _N)) * 0.5
    labels = add_mfe_mae_om_labels(_five_min_ohlc(close, half_range=0.2), warm_fifteen_min_ohlc)

    assert labels["action_none"].eq(1.0).all()


def test_mfe_is_never_negative(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    close = 100 + np.sin(np.linspace(0, 6, _N)) * 0.5
    labels = add_mfe_mae_om_labels(_five_min_ohlc(close, half_range=0.2), warm_fifteen_min_ohlc)

    assert (labels["mfe"] >= 0).all()


def test_rer_is_bounded_per_spec(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    """rer's (0, 1/4) bound only holds under the OM>5 gate per § model output targets — clipped to
    that bound for all rows (including non-actionable ones) so it stays a well-scaled regression target."""
    close = np.linspace(100, 100 + _N, _N)
    labels = add_mfe_mae_om_labels(_five_min_ohlc(close), warm_fifteen_min_ohlc)

    assert (labels["rer"] >= 0).all()
    assert (labels["rer"] <= RER_BOUND + 1e-9).all()


def test_action_one_hot_sums_to_one(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    close = 100 + np.sin(np.linspace(0, 6, _N)) * 0.5
    labels = add_mfe_mae_om_labels(_five_min_ohlc(close, half_range=0.2), warm_fifteen_min_ohlc)

    action_sum = labels[["action_long", "action_short", "action_none"]].sum(axis=1)
    np.testing.assert_allclose(action_sum.to_numpy(), 1.0)


def test_dropping_rows_beyond_an_anchors_horizon_does_not_change_its_label(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    """No-lookahead regression check: anchor i's label depends only on rows [i+1, i+1+HORIZON_BARS).
    Compute labels on a longer series, then again on a truncated one (last 10 rows dropped) — every
    anchor still valid in both must get byte-identical labels, since the dropped rows are strictly
    beyond that anchor's own horizon window (dropping is the simplest possible "perturb the future"
    edit — no shared value with the original data to accidentally match by coincidence)."""
    close = np.linspace(100, 100 + _N + 10, _N + 10)
    full = _five_min_ohlc(close)
    truncated = full.iloc[:_N]

    labels_full = add_mfe_mae_om_labels(full, warm_fifteen_min_ohlc)
    labels_truncated = add_mfe_mae_om_labels(truncated, warm_fifteen_min_ohlc)

    common_index = labels_truncated.index
    pd.testing.assert_frame_equal(labels_full.loc[common_index], labels_truncated)


def test_raises_when_too_few_rows_for_one_horizon(warm_fifteen_min_ohlc: pd.DataFrame) -> None:
    close = np.linspace(100, 110, HORIZON_BARS)  # exactly HORIZON_BARS rows -> zero valid anchors
    with pytest.raises(ValueError, match="needs >"):
        add_mfe_mae_om_labels(_five_min_ohlc(close), warm_fifteen_min_ohlc)
