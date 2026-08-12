"""Unit tests for relative_candle.py against docs/input-features.md § candle feature schema.

Values are hand-derived from the spec formulas (C/ATR, (H-C)/ATR, (C-L)/ATR, gap, height/ATR), not
captured from running the function — this is new spec-conformance code, not legacy behavior to pin.
"""

import pandas as pd
import pytest

from ai_modelling.dataset_generator.relative_candle import (
    add_relative_candle_columns,
    relative_candle_columns,
)

pytestmark = pytest.mark.unit


def _assert_col(df, col, expected):
    pd.testing.assert_series_equal(
        df[col].reset_index(drop=True),
        pd.Series(expected, name=col),
        check_names=False,
        rtol=1e-4,
    )


@pytest.fixture
def base_ohlc(zigzag_ohlc):
    return zigzag_ohlc(n=5, atr=1.0)


@pytest.fixture
def after_relative_candle(base_ohlc):
    return add_relative_candle_columns(base_ohlc)


def test_relative_candle_columns_lists_the_five_derived_fields():
    assert relative_candle_columns() == [
        'rel_close', 'rel_high_close', 'rel_close_low', 'gap', 'rel_candle_height',
    ]


def test_rel_close_is_close_over_atr(after_relative_candle):
    _assert_col(after_relative_candle, 'rel_close', [102.0, 106.2, 108.0, 112.2, 114.0])


def test_rel_high_close_is_high_minus_close_over_atr(after_relative_candle):
    _assert_col(after_relative_candle, 'rel_high_close', [3.0, 4.8, 3.0, 4.8, 3.0])


def test_rel_close_low_is_close_minus_low_over_atr(after_relative_candle):
    _assert_col(after_relative_candle, 'rel_close_low', [7.0, 11.2, 7.0, 11.2, 7.0])


def test_gap_is_open_minus_prev_close_over_atr_and_nan_on_first_row(after_relative_candle):
    _assert_col(after_relative_candle, 'gap', [float('nan'), -2.2, -2.2, -2.2, -2.2])


def test_rel_candle_height_is_high_minus_low_over_atr(after_relative_candle):
    _assert_col(after_relative_candle, 'rel_candle_height', [10.0, 16.0, 10.0, 16.0, 10.0])


def test_scales_inversely_with_atr(zigzag_ohlc):
    unit_atr = add_relative_candle_columns(zigzag_ohlc(n=5, atr=1.0))
    double_atr = add_relative_candle_columns(zigzag_ohlc(n=5, atr=2.0))
    for col in relative_candle_columns():
        pd.testing.assert_series_equal(
            double_atr[col].reset_index(drop=True),
            (unit_atr[col] / 2).reset_index(drop=True),
            check_names=False,
            rtol=1e-4,
        )


def test_computes_atr_when_missing(zigzag_ohlc):
    ohlc = zigzag_ohlc(n=300).drop(columns=['atr'])
    result = add_relative_candle_columns(ohlc)
    assert 'atr' in result.columns
    pd.testing.assert_series_equal(
        result['rel_close'], result['close'] / result['atr'], check_names=False, rtol=1e-4,
    )
