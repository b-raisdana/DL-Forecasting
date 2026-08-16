"""Unit tests for datafeeder.py's cross-timeframe causal alignment (_last_closed_position) — the
piece responsible for never letting a still-forming higher-timeframe candle's eventual final OHLC
leak into an anchor's features. See that function's docstring for the shifted-merge_asof derivation.
"""

import numpy as np
import pandas as pd
import pytest
from ai_modelling.tier1_000.datafeeder import _last_closed_position

pytestmark = pytest.mark.unit


def test_base_timeframe_matches_the_anchor_candle_itself() -> None:
    """shift=0 for the base (5min) branch — the anchor's own "LAST" candle is itself, already closed."""
    anchors = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10"], tz="UTC")
    branch_index = anchors  # same series

    positions = _last_closed_position(anchors, branch_index, branch_tf_minutes=5.0, base_tf_minutes=5.0)

    np.testing.assert_array_equal(positions, [0, 1, 2])


def test_still_forming_higher_tf_candle_is_excluded() -> None:
    """A 15min candle starting at 00:00 is still forming for anchors at 00:00 and 00:05 (it only
    closes at 00:15) — those anchors must resolve to the PREVIOUS 15min candle (at 23:45 the day
    before), not the one currently in progress. The 00:10 anchor is a different, boundary case
    (its own close lands exactly on the 15min candle's close) — covered separately below, not here."""
    anchors = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:05"], tz="UTC")
    branch_index = pd.DatetimeIndex(["2023-12-31 23:45", "2024-01-01 00:00"], tz="UTC")  # 15min candles

    positions = _last_closed_position(anchors, branch_index, branch_tf_minutes=15.0, base_tf_minutes=5.0)

    np.testing.assert_array_equal(positions, [0, 0])


def test_simultaneous_close_is_allowed() -> None:
    """An anchor at 00:10 (5min) closes at 00:15 — the exact instant a 15min candle starting at 00:00
    also closes. Simultaneous availability is safe to use (not lookahead), so this anchor SHOULD
    resolve to that just-closed 15min candle, unlike the strictly-earlier anchors in the test above."""
    anchors = pd.DatetimeIndex(["2024-01-01 00:10"], tz="UTC")
    branch_index = pd.DatetimeIndex(["2023-12-31 23:45", "2024-01-01 00:00"], tz="UTC")

    positions = _last_closed_position(anchors, branch_index, branch_tf_minutes=15.0, base_tf_minutes=5.0)

    np.testing.assert_array_equal(positions, [1])


def test_no_valid_candle_yet_returns_negative_one() -> None:
    anchors = pd.DatetimeIndex(["2024-01-01 00:00"], tz="UTC")
    branch_index = pd.DatetimeIndex(["2024-06-01 00:00"], tz="UTC")  # branch history starts AFTER anchor

    positions = _last_closed_position(anchors, branch_index, branch_tf_minutes=15.0, base_tf_minutes=5.0)

    assert positions[0] == -1
