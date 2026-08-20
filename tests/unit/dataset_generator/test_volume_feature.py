"""Unit tests for volume_feature.py against docs/input-features.md § candle feature schema.

Values are hand-derived from the spec formula (volume / RMA(volume, atr_timeperiod)) using pandas'
own alpha=1/length EWM definition, not captured from running the function — this is new
spec-conformance code, not legacy behavior to pin.
"""

import numpy as np
import pandas as pd
import pytest
from application.dataset_generation.volume_feature import (
    add_log_sma_volume_feature_column,
    add_volume_feature_columns,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def short_atr_period(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real app_config.atr_timeperiod (14) would need a much longer fixture just to clear
    # RMA's min_periods warm-up; shrink it here so a 4-row fixture is enough to hand-derive.
    from config import app_config

    monkeypatch.setattr(app_config, "atr_timeperiod", 2)


def test_volume_atr_is_volume_over_rma_volume() -> None:
    # RMA(length=2) == ewm(alpha=0.5, min_periods=2).mean(): a hand-verified recursive average,
    # e.g. row 2: (30 + 20*0.5 + 10*0.25) / (1 + 0.5 + 0.25) = 24.285714...
    ohlcv = pd.DataFrame({"volume": [10.0, 20.0, 30.0, 40.0]})
    result = add_volume_feature_columns(ohlcv)
    expected = [np.nan, 1.2, 1.2352941176470589, 1.2244897959183674]
    np.testing.assert_allclose(result["volume_atr"].to_numpy(), expected, rtol=1e-9, equal_nan=True)


def test_volume_atr_is_nan_before_atr_timeperiod_warms_up() -> None:
    ohlcv = pd.DataFrame({"volume": [10.0, 20.0, 30.0, 40.0]})
    result = add_volume_feature_columns(ohlcv)
    assert result["volume_atr"].iloc[:1].isna().all()
    assert result["volume_atr"].iloc[1:].notna().all()


def test_volume_atr_zero_volume_run_is_nan_not_inf() -> None:
    ohlcv = pd.DataFrame({"volume": [0.0, 0.0, 0.0, 0.0]})
    result = add_volume_feature_columns(ohlcv)
    assert result["volume_atr"].isna().all()
    assert not np.isinf(result["volume_atr"].fillna(0)).any()


# --- add_log_sma_volume_feature_column (additive sibling, gap 2) ---------------------------------


def test_log_volume_sma_ratio_matches_hand_derived_formula() -> None:
    ohlcv = pd.DataFrame({"volume": [10.0, 20.0, 30.0, 40.0]})
    result = add_log_sma_volume_feature_column(ohlcv, length=2)
    eps = np.finfo(np.float64).eps
    sma = pd.Series([10.0, 20.0, 30.0, 40.0]).rolling(2).mean()
    expected = np.log((ohlcv["volume"] + eps) / (sma + eps))
    np.testing.assert_allclose(
        result["log_volume_sma_ratio"].to_numpy(), expected.to_numpy(), rtol=1e-9, equal_nan=True
    )


def test_log_volume_sma_ratio_is_zero_when_volume_equals_its_own_sma() -> None:
    ohlcv = pd.DataFrame({"volume": [5.0, 5.0, 5.0, 5.0]})
    result = add_log_sma_volume_feature_column(ohlcv, length=2)
    np.testing.assert_allclose(result["log_volume_sma_ratio"].iloc[1:].to_numpy(), 0.0, atol=1e-12)


def test_length_exceeding_available_history_would_otherwise_crash_not_nan() -> None:
    """pandas_ta.sma returns None outright (not a NaN Series) when length > len(series) -- a real
    concern for the 1W branch, which has fewer than 256 cached candles (see
    datafeeder_input3_outcome1.py's _ATR_LENGTH_OVERRIDE, reused for this same reason). Documenting
    the failure mode here rather than papering over it inside this function, since the actual fix
    (a shorter override length) is the caller's responsibility, matching the existing ATR precedent."""
    ohlcv = pd.DataFrame({"volume": [10.0, 20.0, 30.0]})
    with pytest.raises(TypeError):
        add_log_sma_volume_feature_column(ohlcv, length=256)
