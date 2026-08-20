"""Causal, hindsight-capped peak/valley "reach" detection — a fresh reimplementation for the
Tier-1_000 `higher_extremum_distance`/`extremum_weight` input fields (see
docs/ML_Forecasting_System_Design/designsets/input-data-feature/
Tier-1_000.atr_rel_ohlc_log_sma_v_extm_rel_6tf(handmade).input.jsonc), not a wrapper around
`PeakValley.py`. `PeakValley.py`'s own tf-confirmation (`calculate_strength`/`top_timeframe`) bakes in
unbounded *future* lookahead relative to any anchor candle that would consume it as a training feature
(see docs/todos/01-input-data-channels.md's peak/valley reuse decision) — exactly the leak this module
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
from domain.schemas.price_action.CausalExtremum import CausalExtremumOHLC, CausalExtremumResult
from helper.importer import ptd

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


def plus2tf(tf: str) -> str:
    """2nd higher timeframe. eg. plus2tf('5min')='1h', plus2tf('1h')='1D'."""
    return TF_ORDER[TF_ORDER.index(tf) + 2]


def plus3tf(tf: str) -> str:
    """3rd higher timeframe. eg. plus3tf('5min')='4h', plus3tf('1h')='1W'."""
    return TF_ORDER[TF_ORDER.index(tf) + 3]


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


def _reach_minutes(
    values: npt.NDArray[np.float64], times_ns: npt.NDArray[np.int64], *, direction: str, sense: str
) -> npt.NDArray[np.float64]:
    """For each i, elapsed minutes to the nearest j (in `direction`) with values[j] on the extremum
    side of values[i] (`sense='peak'` -> values[j] >= values[i], `sense='valley'` -> values[j] <=
    values[i]). np.inf where no such j exists. O(n) amortized monotonic-stack single pass — the
    "nearest greater/smaller-or-equal element" problem; see module docstring for why a stack loop
    (not a vectorized numpy trick) is the right tool here.
    """
    n = len(values)
    reach = np.full(n, np.inf, dtype=np.float64)
    stack: list[int] = []
    index_order = range(n - 1, -1, -1) if direction == "right" else range(n)
    for i in index_order:
        v = values[i]
        if sense == "peak":
            while stack and values[stack[-1]] < v:
                stack.pop()
        else:
            while stack and values[stack[-1]] > v:
                stack.pop()
        if stack:
            j = stack[-1]
            reach[i] = abs(times_ns[j] - times_ns[i]) / _NS_PER_MINUTE
        stack.append(i)
    return reach


def compute_true_extremum(ohlc: ptd[CausalExtremumOHLC]) -> ptd[CausalExtremumResult]:
    """Step A: full-hindsight (anchor-independent) extremum reach for every candle in `ohlc` (a
    single-timeframe, single native-spacing series — one branch's own native series, not a
    multi-timeframe frame).

    Returns a DataFrame (same index as `ohlc`) with:
    - `true_peak_reach_minutes` / `true_valley_reach_minutes`: min(left, right) elapsed-minute reach
      in each sense (np.inf if unbeaten across the whole available series in both directions).
    - `extremum_sign`: +1 if this candle is "more of a peak" (true_peak_reach >= true_valley_reach —
      peak wins ties, a judgment call, see module/model.py docstrings), -1 if valley reach strictly
      won, 0 if floor_to_tf_ladder(the winning reach) == 0 (doesn't even beat its own series' immediate
      neighbor at the finest global rung — see floor_to_tf_ladder's docstring for why this is rare for
      non-5min branches).
    - `true_extremum_tf_minutes`: floor_to_tf_ladder(max(true_peak_reach, true_valley_reach)) — the
      unsigned reach magnitude (whichever of peak/valley reach is larger), ladder-snapped.
    """
    CausalExtremumOHLC.validate(ohlc, lazy=True)

    high = ohlc["high"].to_numpy(dtype=np.float64)
    low = ohlc["low"].to_numpy(dtype=np.float64)
    # Normalize the index to nanosecond UTC before reading integer timestamps. pandas 3.x stores
    # tz-aware datetimes at microsecond (or coarser) resolution, so `ohlc.index.asi8` can return
    # *microseconds*; dividing those by `_NS_PER_MINUTE` (nanoseconds/minute) yields reach values
    # ~1000x too small (e.g. 0.005 instead of 5.0) and makes `floor_to_tf_ladder(reach) == 0`, which
    # wrongly collapses `extremum_sign` to 0. Localize naive indexes, convert aware ones to UTC, then
    # force nanosecond precision so reach is always the real elapsed minutes between timestamps.
    index = ohlc.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    index = index.astype("datetime64[ns, UTC]")
    times_ns = index.asi8

    right_peak = _reach_minutes(high, times_ns, direction="right", sense="peak")
    left_peak = _reach_minutes(high, times_ns, direction="left", sense="peak")
    right_valley = _reach_minutes(low, times_ns, direction="right", sense="valley")
    left_valley = _reach_minutes(low, times_ns, direction="left", sense="valley")

    true_peak_reach = np.minimum(left_peak, right_peak)
    true_valley_reach = np.minimum(left_valley, right_valley)

    raw_magnitude = np.maximum(true_peak_reach, true_valley_reach)
    true_extremum_tf_minutes = floor_to_tf_ladder(raw_magnitude)

    peak_wins = true_peak_reach >= true_valley_reach  # peak wins ties (judgment call)
    extremum_sign = np.where(peak_wins, 1, -1)
    extremum_sign = np.where(true_extremum_tf_minutes == 0, 0, extremum_sign)

    result = ptd(
        {
            "true_peak_reach_minutes": true_peak_reach,
            "true_valley_reach_minutes": true_valley_reach,
            "extremum_sign": extremum_sign.astype(np.int64),
            "true_extremum_tf_minutes": true_extremum_tf_minutes,
        },
        index=index,
    )
    CausalExtremumResult.validate(result, lazy=True)
    return result


# def observed_extremum_tf_minutes(
# true_extremum_tf_minutes: npt.NDArray[np.float64],
# event_time_ns: npt.NDArray[np.int64],
# anchor_time_ns: int | np.int64 | npt.NDArray[np.int64],
# ) -> npt.NDArray[np.float64]:
# """Step B closed form (see module docstring for the derivation): the causally-capped reach as of
# `anchor_time_ns`, vectorized over events. `anchor_time_ns` may be a scalar (one anchor, many
# events) or an array broadcastable against `event_time_ns` (e.g. already-windowed per-anchor
# event times).
# """
# age_minutes = (np.asarray(anchor_time_ns) - np.asarray(event_time_ns)) / _NS_PER_MINUTE
# return np.minimum(true_extremum_tf_minutes, floor_to_tf_ladder(age_minutes))
