"""Unit tests for application/dataset_generation/extremum_features.py — Step C (extremum_weight) and
Step D (higher_extremum_distance's nearest_/last_ plus2TF/plus3TF pool logic) built on top of
domain/price_action/CausalExtremum.py's Step A/B primitives.

Fixtures build BranchExtremum directly (bypassing compute_true_extremum) so each test controls
exactly which events are "eligible" and when, isolating Step C/D's own logic from Step A's reach
computation (already covered by test_causal_extremum.py).
"""

import numpy as np
import pandas as pd
import pytest
from application.dataset_generation.extremum_features import (
    BranchExtremum,
    align_source_atr,
    compute_extremum_weight,
    compute_higher_extremum_distance,
)

pytestmark = pytest.mark.unit

_BASE = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def _ns_at(minutes: float) -> np.int64:
    return np.int64((_BASE + pd.Timedelta(minutes=minutes)).value)


def _time_index(minutes: list[float]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_BASE + pd.Timedelta(minutes=m) for m in minutes])


def _empty_source() -> BranchExtremum:
    return BranchExtremum(
        time_ns=np.array([], dtype=np.int64),
        time_index=pd.DatetimeIndex([]),
        high=np.array([], dtype=np.float64),
        low=np.array([], dtype=np.float64),
        atr=np.array([], dtype=np.float64),
        sign=np.array([], dtype=np.int64),
        true_extremum_tf_minutes=np.array([], dtype=np.float64),
    )


# --- compute_higher_extremum_distance (Step D) ----------------------------------------------------


def test_pool_eligibility_nearest_and_last_selection() -> None:
    """Three peak events (A@t=0 price=100, B@t=30min price=105, C@t=200min price=98), a 60-minute
    causal-confirmation window, and two anchors:

    - anchor0 (t=250min): cutoff = 250-60 = 190min. A and B are causally eligible (<=190); C is NOT
      (200 > 190) -- (a) an event isn't eligible until its own confirmation cutoff has passed.
      - position0 (t_i=50min, close=101): both A and B are also position-eligible (event_time<50).
        nearest_ by |close-price| picks A (|101-100|=1 < |101-105|=4), NOT the time-nearest B --
        (b) nearest_ picks smallest price distance, not smallest time distance.
        last_ (most recent event_time before t_i) picks B (t=30 > t=0) -- (c) last_ picks the most
        recent event in time.
      - position1 (t_i=10min, close=99): only A is position-eligible (B's t=30 is not < 10).
        nearest_ = last_ = A here (only one candidate).
    - anchor1 (t=50min): cutoff = 50-60 = -10min -- nothing is eligible at all (not even A@t=0).
      Both positions must fall back to the documented "no signal yet" 0.0 convention.
    """
    source = BranchExtremum(
        time_ns=np.array([_ns_at(0), _ns_at(30), _ns_at(200)]),
        time_index=_time_index([0, 30, 200]),
        high=np.array([100.0, 105.0, 98.0]),
        low=np.array([100.0, 105.0, 98.0]),
        atr=np.array([2.0, 2.0, 2.0]),
        sign=np.array([1, 1, 1], dtype=np.int64),
        true_extremum_tf_minutes=np.array([100.0, 100.0, 100.0]),
    )
    empty = _empty_source()

    encoding_close = np.array(
        [
            [101.0, 99.0],  # anchor0's two window positions
            [99.0, 99.0],  # anchor1's two window positions
        ]
    )
    encoding_time_ns = np.array(
        [
            [_ns_at(50), _ns_at(10)],  # anchor0
            [_ns_at(10), _ns_at(5)],  # anchor1
        ]
    )
    anchor_time_ns = np.array([_ns_at(250), _ns_at(50)])
    atr_aligned = np.full((2, 2), 2.0)

    result = compute_higher_extremum_distance(
        encoding_close,
        encoding_time_ns,
        anchor_time_ns,
        plus2_source=source,
        plus2_tf_minutes=60.0,
        plus2_atr_aligned=atr_aligned,
        plus3_source=empty,
        plus3_tf_minutes=60.0,
        plus3_atr_aligned=atr_aligned,
    )

    # anchor0, position0: nearest_ = A (price 100), last_ = B (t=30)
    np.testing.assert_allclose(result["price_normal_distance_plus2tf_peak"][0, 0], (101.0 - 100.0) / 2.0)
    np.testing.assert_allclose(result["time_distance_plus2tf_peak"][0, 0], np.log1p((50.0 - 30.0) / 60.0))

    # anchor0, position1: only A eligible -> nearest_ = last_ = A
    np.testing.assert_allclose(result["price_normal_distance_plus2tf_peak"][0, 1], (99.0 - 100.0) / 2.0)
    np.testing.assert_allclose(result["time_distance_plus2tf_peak"][0, 1], np.log1p((10.0 - 0.0) / 60.0))

    # anchor1: cutoff is negative -- nothing eligible at all -> "no signal yet" 0.0
    np.testing.assert_allclose(result["price_normal_distance_plus2tf_peak"][1], [0.0, 0.0])
    np.testing.assert_allclose(result["time_distance_plus2tf_peak"][1], [0.0, 0.0])

    # valley side never had any sign=-1 events -> always 0.0
    np.testing.assert_allclose(result["price_normal_distance_plus2tf_valley"], 0.0)
    np.testing.assert_allclose(result["time_distance_plus2tf_valley"], 0.0)

    # plus3 side used an entirely empty source -> always 0.0
    for key in (
        "price_normal_distance_plus3tf_peak",
        "price_normal_distance_plus3tf_valley",
        "time_distance_plus3tf_peak",
        "time_distance_plus3tf_valley",
    ):
        np.testing.assert_allclose(result[key], 0.0)


# --- compute_extremum_weight (Step C) ---------------------------------------------------------------


def test_extremum_weight_formula_and_zero_when_not_yet_confirmed() -> None:
    branch = BranchExtremum(
        time_ns=np.array([_ns_at(0), _ns_at(100)]),
        time_index=_time_index([0, 100]),
        high=np.array([100.0, 100.0]),
        low=np.array([100.0, 100.0]),
        atr=np.array([1.0, 1.0]),
        sign=np.array([1, -1], dtype=np.int64),
        true_extremum_tf_minutes=np.array([240.0, 1440.0]),  # "4h" and "1D" ladder rungs
    )
    gather_idx = np.array([[0, 1]])  # one anchor, two window positions (candles 0 and 1)
    anchor_time_ns = np.array([_ns_at(500)])  # well past both candles' own reach windows
    tf_minutes_native = 5.0

    weight = compute_extremum_weight(branch, gather_idx, anchor_time_ns, tf_minutes_native)

    age0 = 500.0 - 0.0
    observed0 = min(240.0, 240.0)  # floor_to_tf_ladder(500) = 240 ("4h" rung, since 240<=500<1440)
    expected0 = 1 * np.log1p(observed0 / tf_minutes_native) * min(1.0, age0 / observed0)
    age1 = 500.0 - 100.0
    # floor_to_tf_ladder(400) = 240 ("4h" rung, since 240<=400<1440) -> observed1 = min(1440, 240) = 240
    observed1 = min(1440.0, 240.0)
    expected1 = -1 * np.log1p(observed1 / tf_minutes_native) * min(1.0, age1 / observed1)

    np.testing.assert_allclose(weight[0, 0], expected0, rtol=1e-5)
    np.testing.assert_allclose(weight[0, 1], expected1, rtol=1e-5)


def test_extremum_weight_is_zero_when_reach_is_below_the_finest_ladder_rung() -> None:
    branch = BranchExtremum(
        time_ns=np.array([_ns_at(0)]),
        time_index=_time_index([0]),
        high=np.array([100.0]),
        low=np.array([100.0]),
        atr=np.array([1.0]),
        sign=np.array([1], dtype=np.int64),
        true_extremum_tf_minutes=np.array([1440.0]),
    )
    gather_idx = np.array([[0]])
    anchor_time_ns = np.array([_ns_at(1.0)])  # only 1 minute has elapsed -- below the 5min rung

    weight = compute_extremum_weight(branch, gather_idx, anchor_time_ns, tf_minutes_native=5.0)

    assert weight[0, 0] == 0.0


# --- align_source_atr -----------------------------------------------------------------------------


def test_align_source_atr_is_causal_backward_looking() -> None:
    """A source candle starting at t=0 with a 60-minute native width only 'closes' (and its ATR
    becomes usable) at t=60 -- query times before that must NOT see it."""
    source = BranchExtremum(
        time_ns=np.array([_ns_at(0)]),
        time_index=_time_index([0]),
        high=np.array([100.0]),
        low=np.array([100.0]),
        atr=np.array([7.0]),
        sign=np.array([0], dtype=np.int64),
        true_extremum_tf_minutes=np.array([0.0]),
    )
    query = _time_index([59, 60, 120])

    aligned = align_source_atr(query, source, target_tf_minutes=60.0)

    assert np.isnan(aligned[0])  # not yet closed
    assert aligned[1] == 7.0  # closes exactly at t=60
    assert aligned[2] == 7.0  # still the latest closed candle
