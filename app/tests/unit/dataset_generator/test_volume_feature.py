"""Unit tests for volume_feature.py against docs/input-features.md § candle feature schema.

Values are hand-derived from the spec formula (volume / RMA(volume, atr_timeperiod)) using pandas'
own alpha=1/length EWM definition, not captured from running the function — this is new
spec-conformance code, not legacy behavior to pin.
"""
import numpy as np
import pandas as pd
import pytest

from ai_modelling.dataset_generator.volume_feature import (
    add_volume_feature_columns,
    volume_feature_columns,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def short_atr_period(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real app_config.atr_timeperiod (14) would need a much longer fixture just to clear
    # RMA's min_periods warm-up; shrink it here so a 4-row fixture is enough to hand-derive.
    from Config import app_config
    monkeypatch.setattr(app_config, 'atr_timeperiod', 2)


def test_volume_feature_columns_lists_the_one_derived_field():
    assert volume_feature_columns() == ['volume_atr']


def test_volume_atr_is_volume_over_rma_volume():
    # RMA(length=2) == ewm(alpha=0.5, min_periods=2).mean(): a hand-verified recursive average,
    # e.g. row 2: (30 + 20*0.5 + 10*0.25) / (1 + 0.5 + 0.25) = 24.285714...
    ohlcv = pd.DataFrame({'volume': [10.0, 20.0, 30.0, 40.0]})
    result = add_volume_feature_columns(ohlcv)
    expected = [np.nan, 1.2, 1.2352941176470589, 1.2244897959183674]
    np.testing.assert_allclose(result['volume_atr'].to_numpy(), expected, rtol=1e-9, equal_nan=True)


def test_volume_atr_is_nan_before_atr_timeperiod_warms_up():
    ohlcv = pd.DataFrame({'volume': [10.0, 20.0, 30.0, 40.0]})
    result = add_volume_feature_columns(ohlcv)
    assert result['volume_atr'].iloc[:1].isna().all()
    assert result['volume_atr'].iloc[1:].notna().all()


def test_volume_atr_zero_volume_run_is_nan_not_inf():
    ohlcv = pd.DataFrame({'volume': [0.0, 0.0, 0.0, 0.0]})
    result = add_volume_feature_columns(ohlcv)
    assert result['volume_atr'].isna().all()
    assert not np.isinf(result['volume_atr'].fillna(0)).any()
