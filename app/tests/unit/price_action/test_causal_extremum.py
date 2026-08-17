"""Unit tests for domain/price_action/CausalExtremum.py — Step A (compute_true_extremum, the
full-hindsight monotonic-stack reach) and Step B (observed_extremum_tf_minutes, the causal cap).

Values are hand-derived from small, fully-controlled synthetic OHLC fixtures, not captured from
running the function — new spec-conformance code, not legacy behavior to pin. See
docs/todos/01-input-data-channels.md for why this causal-cap logic exists (the part of the schema
flagged as "most likely to silently leak future information if implemented naively").
"""

import numpy as np
import pandas as pd
import pytest
from domain.price_action.CausalExtremum import (
    TF_MINUTES,
    compute_true_extremum,
    floor_to_tf_ladder,
    observed_extremum_tf_minutes,
    plus2tf,
    plus3tf,
)

pytestmark = pytest.mark.unit


def _make_ohlc(high: list[float], low: list[float], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(high), freq=freq, tz="UTC")
    return pd.DataFrame({"high": high, "low": low}, index=idx)


# --- floor_to_tf_ladder / plus2tf / plus3tf -----------------------------------------------------


def test_floor_to_tf_ladder_snaps_down_to_the_largest_rung_not_exceeding_the_value() -> None:
    assert floor_to_tf_ladder(4.9) == 0.0
    assert floor_to_tf_ladder(5.0) == 5.0
    assert floor_to_tf_ladder(59.9) == 15.0
    assert floor_to_tf_ladder(60.0) == 60.0
    assert floor_to_tf_ladder(np.inf) == TF_MINUTES["1Y"]


def test_floor_to_tf_ladder_is_vectorized_over_an_array() -> None:
    result = floor_to_tf_ladder(np.array([0.0, 4.9, 5.0, 100_000.0]))
    np.testing.assert_array_equal(result, [0.0, 0.0, 5.0, TF_MINUTES["1M"]])


def test_plus2tf_plus3tf_match_the_spec_jsonc_examples() -> None:
    assert plus2tf("5min") == "1h"
    assert plus3tf("5min") == "4h"
    assert plus2tf("1h") == "1D"
    assert plus3tf("1h") == "1W"
    # extends past the 6 real cached branches, per model.py's documented judgment call
    assert plus2tf("1W") == "4M"
    assert plus3tf("1W") == "1Y"


# --- compute_true_extremum (Step A) -------------------------------------------------------------


def test_isolated_spike_and_dip_have_true_extremum_capped_at_top_ladder_rung() -> None:
    """A candle never beaten in either direction across the whole available series (peak at idx5,
    valley at idx2) reaches np.inf, which floors to the ladder's top rung (1Y)."""
    high = [5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5]
    low = [1, 1, -5, 1, 1, 1, 1, 1, 1, 1, 1]
    ohlc = _make_ohlc(high, low)

    result = compute_true_extremum(ohlc)

    assert result["extremum_sign"].iloc[5] == 1
    assert result["true_extremum_tf_minutes"].iloc[5] == TF_MINUTES["1Y"]
    assert result["extremum_sign"].iloc[2] == -1
    assert result["true_extremum_tf_minutes"].iloc[2] == TF_MINUTES["1Y"]


def test_tied_peak_and_valley_reach_breaks_to_peak() -> None:
    """Candle idx4 (same fixture as above): both its peak-reach and valley-reach are exactly one
    native step (5min, immediately matched by a flat neighbor) — a genuine tie. Peak wins ties, a
    documented judgment call (not specified either way in the input spec jsonc)."""
    high = [5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5]
    low = [1, 1, -5, 1, 1, 1, 1, 1, 1, 1, 1]
    ohlc = _make_ohlc(high, low)

    result = compute_true_extremum(ohlc)

    assert result["extremum_sign"].iloc[4] == 1
    assert result["true_extremum_tf_minutes"].iloc[4] == 5.0


def test_flat_run_of_equal_highs_yields_only_native_step_reach() -> None:
    """Mirrors PeakValley.py's own same-value-run convention (a run of identical highs shouldn't
    register as an isolated strong peak): every candle inside a flat run of equal highs is
    immediately "matched" by its neighbor (>=, not strict >), capping reach at exactly one native
    step, never something larger — no dedup preprocessing needed, the >=/<= comparison already
    handles it."""
    high = [5.0, 10.0, 10.0, 10.0, 5.0]
    low = [1.0, 1.0, 1.0, 1.0, 1.0]
    ohlc = _make_ohlc(high, low)

    result = compute_true_extremum(ohlc)

    for i in (1, 2, 3):
        assert result["true_peak_reach_minutes"].iloc[i] == 5.0


def test_reach_uses_real_elapsed_minutes_not_a_fixed_step_assumption() -> None:
    """A gap in the DatetimeIndex (e.g. missing candles) must reflect in the elapsed-minutes reach,
    not an assumed uniform step — compute_true_extremum reads real timestamp deltas."""
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 01:05"],
        tz="UTC",  # 60min gap before the 3rd candle
    )
    ohlc = pd.DataFrame({"high": [5.0, 20.0, 5.0], "low": [1.0, 1.0, 1.0]}, index=idx)

    result = compute_true_extremum(ohlc)

    # candle 0's nearest right-beating neighbor (high>=5) is candle 1 (20>=5), 5 minutes away
    assert result["true_peak_reach_minutes"].iloc[0] == 5.0


# --- observed_extremum_tf_minutes (Step B closed form) -------------------------------------------


def test_step_b_closed_form_matches_per_direction_causal_capping_reference() -> None:
    """The whole design leans on the closed form
    `observed = min(true_extremum_tf_minutes, floor_to_tf_ladder(age))` instead of re-running Step A
    per anchor. This is mathematically identical to capping EACH direction's own full-hindsight reach
    at `age` BEFORE combining via max — a standard min/max distributivity identity over a linear
    order: `max(min(P, a), min(V, a)) == min(max(P, V), a)` for any P, V, a — and that
    per-direction-capped computation is the actually-correct causal reference (a not-yet-beaten
    candle's confirmed-so-far reach is exactly `age`, a lower bound, not "infinite"/unconfirmed).

    Judgment call / caveat surfaced while writing this test: a NAIVE "slice the OHLC frame to rows
    <= anchor, then re-run Step A's stack scan directly on the slice" does NOT equal this closed form
    in general — slicing makes an as-yet-unbeaten candle's reach come back as literal infinity
    (nothing found within the truncated data), which overstates how little a causal observer actually
    knows once the OTHER sense (peak vs valley) has already been confirmed. Demonstrated concretely
    below: candle 5's peak is confirmed at 10min (well within the 50min anchor age); its valley's true
    eventual reach is 65min but isn't confirmed until 65min (beyond the 50min anchor). A naive
    truncate-and-rerun would report the valley side as literally unbeaten (inf) and combine via
    max(10, inf) = inf -> the top ladder rung (1Y) -- NOT the correct age-capped answer this test
    asserts (15, the "15min" rung). The per-direction-capped reference computed here is therefore the
    actual reference this test validates the closed form against, not a literal truncate-and-rerun.
    """
    n = 25
    high = np.full(n, 5.0)
    low = np.full(n, 1000.0)
    high[5] = 100.0
    high[7] = 150.0  # confirms peak reach = 10 minutes (2 candles * 5min)
    low[5] = 50.0
    low[18] = 40.0  # confirms valley reach = 65 minutes (13 candles * 5min) -- beyond the 50min anchor age
    ohlc = _make_ohlc(list(high), list(low))

    extremum = compute_true_extremum(ohlc)
    true_extremum = extremum["true_extremum_tf_minutes"].to_numpy()
    true_peak = extremum["true_peak_reach_minutes"].to_numpy()
    true_valley = extremum["true_valley_reach_minutes"].to_numpy()
    assert true_peak[5] == 10.0
    assert true_valley[5] == 65.0

    event_time_ns = ohlc.index.asi8
    anchor_time_ns = event_time_ns[15]  # 50 minutes after candle 5
    age_at_5 = (anchor_time_ns - event_time_ns[5]) / 60e9
    assert age_at_5 == 50.0

    observed = observed_extremum_tf_minutes(true_extremum, event_time_ns, anchor_time_ns)

    per_direction_capped_reference = floor_to_tf_ladder(max(min(true_peak[5], 50.0), min(true_valley[5], 50.0)))

    assert observed[5] == per_direction_capped_reference == 15.0


def test_no_lookahead_perturbing_future_never_changes_observed_reach_at_or_before_anchor() -> None:
    """The single most important property of this whole feature (see docs/todos/01-input-data-channels.md
    § step 1): perturbing FUTURE-slice data (rows strictly after some cutoff) must never change
    observed_extremum_tf_minutes for any candle at or before that cutoff, for an anchor at the cutoff.
    """
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    high = 100 + rng.normal(size=n).cumsum()
    low = high - rng.uniform(0.5, 2.0, size=n)
    ohlc_original = pd.DataFrame({"high": high, "low": low}, index=idx)

    cutoff_i = 15
    anchor_time_ns = idx.asi8[cutoff_i]

    extremum_before = compute_true_extremum(ohlc_original)
    observed_before = observed_extremum_tf_minutes(
        extremum_before["true_extremum_tf_minutes"].to_numpy(), ohlc_original.index.asi8, anchor_time_ns
    )

    ohlc_perturbed = ohlc_original.copy()
    future_mask = ohlc_perturbed.index > idx[cutoff_i]
    ohlc_perturbed.loc[future_mask, "high"] += 1000.0
    ohlc_perturbed.loc[future_mask, "low"] -= 1000.0

    extremum_after = compute_true_extremum(ohlc_perturbed)
    observed_after = observed_extremum_tf_minutes(
        extremum_after["true_extremum_tf_minutes"].to_numpy(), ohlc_perturbed.index.asi8, anchor_time_ns
    )

    np.testing.assert_array_equal(observed_before[: cutoff_i + 1], observed_after[: cutoff_i + 1])
