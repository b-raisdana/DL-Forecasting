"""Causal, hindsight-capped peak/valley "reach" detection — a fresh reimplementation for the
Tier-1_000 `higher_extremum_distance`/`extremum_weight` input fields (see
docs/ML_Forecasting_System_Design/designsets/input-data-feature/
Tier-1_000.atr_rel_ohlc_log_sma_v_extm_rel_6tf(handmade).input.jsonc), not a wrapper around
`PeakValley.py`. `PeakValley.py`'s own tf-confirmation (`calculate_strength`/`top_timeframe`) bakes in
unbounded *future* lookahead relative to any anchor candle that would consume it as a training feature
(see docs/ML_Forecasting_System_Design/todo/01-input-data-channels.md's peak/valley reuse decision) — exactly the leak this module
exists to avoid. Only the geometric idea of "local extremum via neighbor comparison" is shared; the
causal-capping machinery below is new.

Two-step design:

Step A (this module's `compute_true_extremum`) — full-hindsight, anchor-independent "reach": for each
candle, how long (in elapsed minutes, using the real DatetimeIndex, not an assumed fixed step) does its
high/low stay unbeaten looking both backward and forward across the *entire* available series. This is
the classic "nearest greater-or-equal element" problem, solved with an O(n) monotonic-stack single pass
(a plain Python loop over the stack — the documented `vectorized-pandas-numpy` skill carve-out for
"truly sequential/stateful logic"; the per-element body is O(1) amortized so this stays fast even for a
multi-year 5min series).

Step B (`observed_extremum_tf_minutes`) — the causal cap for a specific anchor time `A`: a candle's
full-hindsight reach `T` is only *knowable* by `A` once `A >= t_i + T` (T minutes have actually elapsed
since the candle, with nothing having beaten it yet in that elapsed window — which full-hindsight reach
already guarantees, since reach is defined as "elapsed time until first beaten"). So the causally-capped
reach is simply:

    observed_extremum_tf_minutes(i, A) = min(true_extremum_tf_minutes(i), floor_to_tf_ladder(A - t_i))

No need to re-run the expensive stack scan per anchor: Step A runs once per branch series, Step B is a
single cheap `min`/floor per (candle, anchor) pair. See
app/tests/unit/price_action/test_causal_extremum.py's closed-form regression test, which checks this
shortcut against a brute-force reference (re-running Step A restricted to data available as of a given
anchor) rather than trusting the derivation alone.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

# This codebase's lowercase-h timeframe convention (differs from the spec jsonc's "1H"/"4H", same
# meaning). 1M/4M/1Y have no native cached series anywhere in this codebase (app_config.timeframes
# stops at 1W) — they exist only as ladder rungs that a coarser branch's (1W's) own reach can validly
# extend into, per the input spec's plus2TF/plus3TF definitions.
TF_ORDER: list[str] = ["5min", "15min", "1h", "4h", "1D", "1W", "1M", "4M", "1Y"]

# time_assumptions per the input spec: 1M=30D, 1Y=365D.
TF_MINUTES: dict[str, float] = {
    "5min": 5.0,
    "15min": 15.0,
    "1h": 60.0,
    "4h": 240.0,
    "1D": 1440.0,
    "1W": 10080.0,
    "1M": 43200.0,
    "4M": 172800.0,
    "1Y": 525600.0,
}

_TF_RUNGS: npt.NDArray[np.float64] = np.array([TF_MINUTES[tf] for tf in TF_ORDER], dtype=np.float64)
_NS_PER_MINUTE = 60_000_000_000


def observed_extremum_tf_minutes(
    true_extremum_tf_minutes: npt.NDArray[np.float64],
    event_time_ns: npt.NDArray[np.int64],
    anchor_time_ns: int | np.int64 | npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Step B closed form (see module docstring for the derivation): the causally-capped reach as of
    `anchor_time_ns`, vectorized over events. `anchor_time_ns` may be a scalar (one anchor, many
    events) or an array broadcastable against `event_time_ns` (e.g. already-windowed per-anchor
    event times).
    """
    age_minutes = (np.asarray(anchor_time_ns) - np.asarray(event_time_ns)) / _NS_PER_MINUTE
    return np.minimum(true_extremum_tf_minutes, floor_to_tf_ladder(age_minutes))


# duplicated from app/domain/price_action/CausalExtremum.py (still live there; a dead function here depends on it)
def floor_to_tf_ladder(minutes: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """Snap a reach-in-minutes value down to the largest TF_MINUTES rung not exceeding it; 0 if below
    the smallest rung (5.0). Vectorized: accepts a scalar or an ndarray (incl. +/-inf and NaN — NaN
    propagates as 0 via the same "no rung reached" path since NaN comparisons are always False in
    np.searchsorted's underlying comparisons, matching "no signal yet" semantics used throughout this
    feature).

    Note: for any branch coarser than 5min, the smallest *possible* reach is already that branch's own
    native spacing (e.g. 60 minutes for a 1h series — you can't be "beaten" by a neighbor closer than
    one native step away), which already sits on a TF_MINUTES rung >= 5. So in practice this floor only
    ever produces exactly 0 for the 5min branch itself, or via Step B's anchor-age cap (an anchor whose
    age since a candle is itself < 5 minutes) — not from Step A's raw reach on a coarser branch. See
    this module's docstring and datafeeder_input3_outcome1.py / model.py's documented judgment calls.
    """
    arr = np.asarray(minutes, dtype=np.float64)
    idx = np.searchsorted(_TF_RUNGS, arr, side="right") - 1
    snapped = np.where(idx >= 0, _TF_RUNGS[np.clip(idx, 0, None)], 0.0)
    if arr.ndim == 0:
        return float(snapped)
    return snapped
