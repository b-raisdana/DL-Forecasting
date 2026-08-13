"""Characterization tests for profit_loss_adder.py.

Pin today's *actual* output on a small deterministic fixture, ahead of the spec-alignment work in
docs/todos/02-training-data-labels.md. These are not spec-conformance checks — expected values were
captured by running the real functions, not hand-derived — so a genuine behavior change should make
exactly the affected test(s) fail; that failure is the point (confirm it's intentional, then re-capture).
"""

import numpy as np
import pandas as pd
import pytest
from ai_modelling.dataset_generator.profit_loss.profit_loss_adder import (
    long_n_short_drawdown,
    max_profit_n_loss,
    profit_n_loss,
    quantile_maxes,
    stop_loss,
)
from tests.conftest import ZigzagOhlcFactory

pytestmark = pytest.mark.characterization

POSITION_MAX_BARS = 5
ACTION_DELAY = 1
ROLLING_WINDOW = POSITION_MAX_BARS - ACTION_DELAY  # 4 — mirrors add_long_n_short_profit's own convention
QUANTILES = 2
ORDER_FEE = 0.005
MAX_RISK = 2.0  # high enough that the long side isn't force-capped to a "loser", unlike the short side
BAR_WIDTH_RISK_FREE_RATE = 0.0001

# profit_loss_adder.py's functions predate this repo's typed/strict-mypy convention (see the todo's
# spec-alignment rewrite) — the ignores below are calls into that still-untyped legacy module.


def _assert_col(df: pd.DataFrame, col: str, expected: list[float]) -> None:
    pd.testing.assert_series_equal(
        df[col].reset_index(drop=True),
        pd.Series(expected, name=col),
        check_names=False,
        rtol=1e-4,
    )


@pytest.fixture
def base_ohlc(zigzag_ohlc: ZigzagOhlcFactory) -> pd.DataFrame:
    return zigzag_ohlc(n=12)


@pytest.fixture
def after_max_profit_n_loss(base_ohlc: pd.DataFrame) -> pd.DataFrame:
    return max_profit_n_loss(  # type: ignore[no-untyped-call, no-any-return]
        base_ohlc,
        POSITION_MAX_BARS,
        ACTION_DELAY,
        ROLLING_WINDOW,
    )


@pytest.fixture
def after_quantile_maxes(after_max_profit_n_loss: pd.DataFrame) -> pd.DataFrame:
    return quantile_maxes(  # type: ignore[no-untyped-call, no-any-return]
        after_max_profit_n_loss,
        ROLLING_WINDOW,
        QUANTILES,
    )


@pytest.fixture
def after_long_n_short_drawdown(after_quantile_maxes: pd.DataFrame) -> pd.DataFrame:
    return long_n_short_drawdown(  # type: ignore[no-untyped-call, no-any-return]
        after_quantile_maxes,
        POSITION_MAX_BARS,
        QUANTILES,
        trigger_tf="15min",
    )


@pytest.fixture
def after_stop_loss(after_long_n_short_drawdown: pd.DataFrame) -> pd.DataFrame:
    return stop_loss(after_long_n_short_drawdown)  # type: ignore[no-untyped-call, no-any-return]


def test_max_profit_n_loss(after_max_profit_n_loss: pd.DataFrame) -> None:
    df = after_max_profit_n_loss
    _assert_col(
        df, "worst_long_open", [105.0, 111.0, 111.0, 117.0, 117.0, 123.0, 123.0, 129.0, 129.0, 135.0, 135.0, 141.0]
    )
    _assert_col(
        df, "worst_short_open", [95.0, 95.0, 101.0, 101.0, 107.0, 107.0, 113.0, 113.0, 119.0, 119.0, 125.0, 125.0]
    )
    _assert_col(df, "max_high", [123.0, 123.0, 129.0, 129.0, 135.0, 135.0, 141.0] + [np.nan] * 5)
    _assert_col(df, "min_low", [101.0, 101.0, 107.0, 107.0, 113.0, 113.0, 119.0] + [np.nan] * 5)
    _assert_col(df, "max_high_distance", [4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0] + [np.nan] * 5)
    _assert_col(df, "min_low_distance", [1.0] * 7 + [np.nan] * 5)


def test_quantile_maxes(after_quantile_maxes: pd.DataFrame) -> None:
    df = after_quantile_maxes
    _assert_col(
        df, "q1_max_high", [111.0, 117.0, 117.0, 123.0, 123.0, 129.0, 129.0, 135.0, 135.0, 141.0] + [np.nan] * 2
    )
    _assert_col(df, "q1_min_low", [95.0, 101.0, 101.0, 107.0, 107.0, 113.0, 113.0, 119.0, 119.0, 125.0] + [np.nan] * 2)
    _assert_col(df, "q1_max_high_distance", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] + [np.nan] * 2)
    _assert_col(df, "q1_min_low_distance", [0.0] * 10 + [np.nan] * 2)
    _assert_col(df, "q2_max_high", [117.0, 123.0, 123.0, 129.0, 129.0, 135.0, 135.0, 141.0] + [np.nan] * 4)
    _assert_col(df, "q2_min_low", [95.0, 101.0, 101.0, 107.0, 107.0, 113.0, 113.0, 119.0] + [np.nan] * 4)
    _assert_col(df, "q2_max_high_distance", [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0] + [np.nan] * 4)
    _assert_col(df, "q2_min_low_distance", [0.0] * 8 + [np.nan] * 4)


def test_long_n_short_drawdown(after_long_n_short_drawdown: pd.DataFrame) -> None:
    df = after_long_n_short_drawdown
    _assert_col(df, "max_high_quantile", [1.6, 1.2] * 3 + [1.6] + [np.nan] * 5)
    _assert_col(df, "min_low_quantile", [0.4] * 7 + [np.nan] * 5)
    _assert_col(df, "quantile_long_min_low", [95.0, 101.0, 101.0, 107.0, 107.0, 113.0, 113.0] + [np.nan] * 5)
    _assert_col(df, "quantile_short_max_high", [111.0, 117.0, 117.0, 123.0, 123.0, 129.0, 129.0] + [np.nan] * 5)
    _assert_col(df, "long_drawdown", [10.0] * 7 + [np.nan] * 5)
    _assert_col(df, "absolute_long_drawdown", [10.0] * 7 + [np.nan] * 5)
    _assert_col(df, "short_drawdown", [16.0, 22.0, 16.0, 22.0, 16.0, 22.0, 16.0] + [np.nan] * 5)
    _assert_col(df, "absolute_short_drawdown", [16.0, 22.0, 16.0, 22.0, 16.0, 22.0, 16.0] + [np.nan] * 5)


def test_stop_loss(after_stop_loss: pd.DataFrame) -> None:
    df = after_stop_loss
    # today's floor is max(1, drawdown) — never actually binds here since drawdown (10/16/22) > 1
    _assert_col(df, "long_sl_distance", [10.0] * 7 + [np.nan] * 5)
    _assert_col(df, "short_sl_distance", [16.0, 22.0, 16.0, 22.0, 16.0, 22.0, 16.0] + [np.nan] * 5)


def test_profit_n_loss(after_stop_loss: pd.DataFrame) -> None:
    df = profit_n_loss(  # type: ignore[no-untyped-call]
        after_stop_loss,
        bar_width_risk_free_rate=BAR_WIDTH_RISK_FREE_RATE,
        order_fee=ORDER_FEE,
        max_risk=MAX_RISK,
    )
    _assert_col(df, "long_profit", [18.0, 12.0, 18.0, 12.0, 18.0, 12.0, 18.0] + [np.nan] * 5)
    _assert_col(df, "short_profit", [-6.0] * 7 + [np.nan] * 5)
    _assert_col(df, "weighted_long_profit", [17.9946, 11.9947] * 3 + [17.9946] + [np.nan] * 5)
    _assert_col(df, "weighted_short_profit", [-6.0051] * 7 + [np.nan] * 5)
    # short is a structural loser here (weighted_short_profit <= 0, zigzag only trends up) regardless of max_risk
    _assert_col(df, "short_risk", [1.0] * 7 + [np.nan] * 5)
    _assert_col(df, "short_signal", [0.0] * 7 + [np.nan] * 5)
    # long clears max_risk=2.0, so risk/signal reflect the real formula instead of being force-capped
    _assert_col(df, "long_risk", [0.5557222722372267, 0.8337015515185874] * 3 + [0.5557222722372267] + [np.nan] * 5)
    _assert_col(df, "long_signal", [0.7994600000000002, 0.19947] * 3 + [0.7994600000000002] + [np.nan] * 5)
