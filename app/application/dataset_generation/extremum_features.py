"""Steps C (`extremum_weight`) and D (`higher_extremum_distance`, 8 fields) on top of
domain/price_action/CausalExtremum.py's Step A/B primitives — feature engineering from an
already-computed domain concept, the same pattern relative_candle.py/volume_feature.py already
establish in this directory (both pure transforms living in `application/dataset_generation`, not
`domain/`). Purely additive: nothing here touches relative_candle.py's/volume_feature.py's own
columns, and this module has no other consumer today besides
model_implementations/tier1_000/datafeeder_input3_outcome1.py.

See CausalExtremum.py's module docstring for the causal-capping derivation this all builds on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from domain.price_action.CausalExtremum import compute_true_extremum, floor_to_tf_ladder

_NS_PER_MINUTE = 60_000_000_000


@dataclass(frozen=True)
class BranchExtremum:
    """Precomputed Step-A output for one branch's own native OHLC series, plus the raw fields Step
    C/D need (own ATR, high/low for event price, time index for causal alignment/eligibility)."""

    time_ns: npt.NDArray[np.int64]
    time_index: pd.DatetimeIndex
    high: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    atr: npt.NDArray[np.float64]
    sign: npt.NDArray[np.int64]
    true_extremum_tf_minutes: npt.NDArray[np.float64]


def build_branch_extremum(ohlc: pd.DataFrame) -> BranchExtremum:
    """`ohlc` must already carry an 'atr' column (relative_candle.py's add_relative_candle_columns
    side effect, already computed upstream in datafeeder_input3_outcome1.py's _branch_features)."""
    extremum = compute_true_extremum(ohlc)
    return BranchExtremum(
        time_ns=ohlc.index.asi8,
        time_index=ohlc.index,
        high=ohlc["high"].to_numpy(dtype=np.float64),
        low=ohlc["low"].to_numpy(dtype=np.float64),
        atr=ohlc["atr"].to_numpy(dtype=np.float64),
        sign=extremum["extremum_sign"].to_numpy(dtype=np.int64),
        true_extremum_tf_minutes=extremum["true_extremum_tf_minutes"].to_numpy(dtype=np.float64),
    )


def align_source_atr(
    query_time_index: pd.DatetimeIndex, source: BranchExtremum, target_tf_minutes: float
) -> npt.NDArray[np.float64]:
    """For every timestamp in `query_time_index`, the source branch's own ATR at the latest source
    candle whose OWN close (source_start + target_tf_minutes) is <= that timestamp. Same
    shifted-merge_asof causal idiom as datafeeder_input3_outcome1.py's `_last_closed_position` — here
    there's no separate base-tf subtraction because `query_time_index` entries are already absolute
    times (not anchor-relative), so the full shift is just the source's own timeframe width.
    Direction='backward', no lookahead. NaN where no source candle has closed yet.
    """
    shift = pd.Timedelta(minutes=target_tf_minutes)
    shifted = pd.DataFrame({"date": query_time_index - shift}).sort_values("date")
    source_positions = pd.DataFrame({"date": source.time_index, "position": np.arange(len(source.time_index))})
    merged = pd.merge_asof(shifted, source_positions, on="date", direction="backward")
    merged.index = shifted.index
    positions = merged["position"].reindex(range(len(query_time_index))).to_numpy()

    aligned_atr = np.full(len(query_time_index), np.nan, dtype=np.float64)
    valid = ~np.isnan(positions)
    aligned_atr[valid] = source.atr[positions[valid].astype(np.int64)]
    return aligned_atr


def compute_extremum_weight(
    branch: BranchExtremum,
    gather_idx: npt.NDArray[np.int64],
    anchor_time_ns: npt.NDArray[np.int64],
    tf_minutes_native: float,
) -> npt.NDArray[np.float32]:
    """Step C, fully vectorized across (n_anchors, window_len) at once (no anchor loop needed — unlike
    Step D's nearest_/last_ pool query, extremum_weight is a pure elementwise function of the
    causally-capped reach, so broadcasting the whole matrix in one shot is both correct and faster).

    `gather_idx`: (n_anchors, window_len) row positions into `branch`'s own native series (the same
    index matrix used to window every other per-branch feature). `anchor_time_ns`: (n_anchors,).
    """
    event_time_ns = branch.time_ns[gather_idx]  # (n_anchors, window_len)
    true_reach = branch.true_extremum_tf_minutes[gather_idx]
    sign = branch.sign[gather_idx]

    age_minutes = (anchor_time_ns[:, None] - event_time_ns) / _NS_PER_MINUTE
    observed = np.minimum(true_reach, floor_to_tf_ladder(age_minutes))

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_weight = sign * np.log1p(observed / tf_minutes_native) * np.minimum(1.0, age_minutes / observed)
    weight = np.where(observed == 0, 0.0, raw_weight)
    return weight.astype(np.float32)


def _threshold_filtered_pool(
    source: BranchExtremum, sign_value: int, threshold_minutes: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Fixed (anchor-independent) eligible-event pool for one (source branch, sign, target TF)
    combination: sign-matching events whose true (full-hindsight) reach already clears the target TF's
    threshold. `source.time_ns` is ascending, so boolean-masking preserves order — no explicit sort
    needed. Computed once per (target, sense) and reused across every anchor (only the anchor-specific
    causal-eligibility *cutoff*, applied via searchsorted in `_nearest_and_last`, varies per anchor)."""
    mask = (source.sign == sign_value) & (source.true_extremum_tf_minutes >= threshold_minutes)
    price = source.high if sign_value == 1 else source.low
    return source.time_ns[mask], price[mask]


def _nearest_and_last(
    close_i: npt.NDArray[np.float64],
    t_i: npt.NDArray[np.int64],
    pool_time: npt.NDArray[np.int64],
    pool_price: npt.NDArray[np.float64],
    cutoff_ns: np.int64,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One anchor's one (target, sense) query: for each window position i (time t_i, close close_i),
    find the price-nearest and time-last eligible pool event strictly before t_i, among pool events
    already causally eligible by `cutoff_ns` (anchor-fixed, computed by the caller).

    Returns (raw_signed_price_diff, raw_elapsed_minutes) — NOT yet normalized (caller divides by ATR /
    applies log1p across the whole array afterwards, vectorized). Both default to 0.0 where no eligible
    pool entry exists before t_i — the "no signal yet" convention (0.0 raw diff -> 0.0 once normalized;
    0.0 raw elapsed -> log1p(0/T) == 0.0 once normalized — the convention falls out of the math, no
    special-casing needed downstream).
    """
    window_len = len(t_i)
    end = int(np.searchsorted(pool_time, cutoff_ns, side="right"))
    if end == 0:
        zeros = np.zeros(window_len, dtype=np.float64)
        return zeros, zeros

    pt = pool_time[:end]
    pp = pool_price[:end]
    mask = pt[None, :] < t_i[:, None]  # (window_len, K): strictly-before-t_i, per position
    any_eligible = mask.any(axis=1)

    diff = close_i[:, None] - pp[None, :]
    abs_diff_masked = np.where(mask, np.abs(diff), np.inf)
    nearest_idx = np.argmin(abs_diff_masked, axis=1)
    raw_price_diff = np.where(any_eligible, diff[np.arange(window_len), nearest_idx], 0.0)

    time_masked = np.where(mask, pt[None, :], np.iinfo(np.int64).min)
    last_idx = np.argmax(time_masked, axis=1)
    last_event_time = pt[last_idx]
    raw_elapsed_minutes = np.where(any_eligible, (t_i - last_event_time) / _NS_PER_MINUTE, 0.0)

    return raw_price_diff, raw_elapsed_minutes


def _one_target(
    encoding_close: npt.NDArray[np.float64],
    encoding_time_ns: npt.NDArray[np.int64],
    anchor_time_ns: npt.NDArray[np.int64],
    source: BranchExtremum,
    target_tf_minutes: float,
    atr_aligned: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """One plusNTF target: loops over anchors (the granularity the design calls for — the eligible
    pool is fixed per anchor, not per window position, so each anchor does one vectorized
    (window_len x pool_size) op per sense, not a Python loop over window positions). Returns
    (price_peak, price_valley, time_peak, time_valley), each (n_anchors, window_len), fully
    normalized (ATR-divided / log1p'd).
    """
    n_anchors, window_len = encoding_close.shape
    price_peak = np.zeros((n_anchors, window_len), dtype=np.float64)
    price_valley = np.zeros((n_anchors, window_len), dtype=np.float64)
    time_peak = np.zeros((n_anchors, window_len), dtype=np.float64)
    time_valley = np.zeros((n_anchors, window_len), dtype=np.float64)

    peak_pool_time, peak_pool_price = _threshold_filtered_pool(source, 1, target_tf_minutes)
    valley_pool_time, valley_pool_price = _threshold_filtered_pool(source, -1, target_tf_minutes)
    cutoff_step_ns = np.int64(round(target_tf_minutes * _NS_PER_MINUTE))

    for a in range(n_anchors):
        cutoff_ns = anchor_time_ns[a] - cutoff_step_ns
        t_i = encoding_time_ns[a]
        close_i = encoding_close[a]

        price_peak[a], time_peak[a] = _nearest_and_last(close_i, t_i, peak_pool_time, peak_pool_price, cutoff_ns)
        price_valley[a], time_valley[a] = _nearest_and_last(
            close_i, t_i, valley_pool_time, valley_pool_price, cutoff_ns
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        price_peak = price_peak / atr_aligned
        price_valley = price_valley / atr_aligned
    time_peak = np.log1p(np.maximum(time_peak, 0.0) / target_tf_minutes)
    time_valley = np.log1p(np.maximum(time_valley, 0.0) / target_tf_minutes)

    return (
        price_peak.astype(np.float32),
        price_valley.astype(np.float32),
        time_peak.astype(np.float32),
        time_valley.astype(np.float32),
    )


def compute_higher_extremum_distance(
    encoding_close: npt.NDArray[np.float64],
    encoding_time_ns: npt.NDArray[np.int64],
    anchor_time_ns: npt.NDArray[np.int64],
    plus2_source: BranchExtremum,
    plus2_tf_minutes: float,
    plus2_atr_aligned: npt.NDArray[np.float64],
    plus3_source: BranchExtremum,
    plus3_tf_minutes: float,
    plus3_atr_aligned: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float32]]:
    """Step D, all 8 `higher_extremum_distance` fields for one encoding branch's window. `encoding_*`
    are the branch-being-encoded's own windowed close/time (n_anchors, window_len);
    `plus2_source`/`plus3_source` are the plus2TF(X)/plus3TF(X) *source* branches' BranchExtremum
    (the native branch if plus2TF/plus3TF(X) is one of the 6 real branches, else the 1W branch's own
    Step A/B output — see model.py's docstring for why 1W's own reach validly stands in for
    1M/4M/1Y-scale confirmation without a separate coarser series). `plus2_atr_aligned`/
    `plus3_atr_aligned`: (n_anchors, window_len), the source's own ATR causally aligned onto
    `encoding_time_ns` via `align_source_atr` (computed once per branch, not per anchor, then windowed
    with the same gather_idx as everything else — see datafeeder_input3_outcome1.py).

    Keys match CANDLE_FEATURE_COLUMNS' own naming exactly, so the caller can just look them up in
    order.
    """
    p2_peak, p2_valley, t2_peak, t2_valley = _one_target(
        encoding_close, encoding_time_ns, anchor_time_ns, plus2_source, plus2_tf_minutes, plus2_atr_aligned
    )
    p3_peak, p3_valley, t3_peak, t3_valley = _one_target(
        encoding_close, encoding_time_ns, anchor_time_ns, plus3_source, plus3_tf_minutes, plus3_atr_aligned
    )
    return {
        "price_normal_distance_plus2tf_peak": p2_peak,
        "price_normal_distance_plus2tf_valley": p2_valley,
        "price_normal_distance_plus3tf_peak": p3_peak,
        "price_normal_distance_plus3tf_valley": p3_valley,
        "time_distance_plus2tf_peak": t2_peak,
        "time_distance_plus2tf_valley": t2_valley,
        "time_distance_plus3tf_peak": t3_peak,
        "time_distance_plus3tf_valley": t3_valley,
    }
