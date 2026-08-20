"""Datafeeder for the Tier-1_000 architecture (see
docs/ML_Forecasting_System_Design/designsets/Tier-1_000.par_branch_mtcn_lstm_perc_gqa_mlp_lgbm(handmade).base.jsonc),
built specifically against
**input variation 3** of the input-data-feature category's handmade reference,
`Tier-1_000.atr_rel_ohlc_log_sma_v_extm_rel_6tf(handmade).input.jsonc` (256 candles on 5min/15min, 128 on
1h/4h, 64 on 1D/1W — see ./model.py's module docstring for why variation 3 and not the "1-base" row's
variation 1), and **outcome set 1** of the outcome-label-target-head category's handmade reference,
`Tier-1_000.action_mfe_rer(handmade).outcome.jsonc` (action_head + mean_std_pairs for
[mfe, rer], not the mean_std_skew_kurtosis_pairs of outcome set 2).

This is one input/outcome combination among several the Tier-1_000 architecture can be trained
against — a future `datafeeder_input<N>_outcome<M>.py` sibling module covers any other combination
tested later; this module's name pins down exactly which one `build_dataset()` here implements, so
callers never have to guess from a bare "datafeeder" which variation/outcome-set pairing they're
getting.

Builds the 6 fixed-timeframe branch windows + auxiliary_features + mfe/rer/action labels once per
training run, RAM-resident, then serves them through a tf.data pipeline — the "preload once,
in-memory feature cache, .prefetch(AUTOTUNE)" convention from 03-Model & Architecture Engineering.md
§ vram/ram budget split's pre-loading/prefetch pipeline. See the cache-or-generate skill for the
general pattern this follows.

Causality: a higher-timeframe candle is only usable for an anchor once it has actually closed by the
anchor's own close — not merely "started before" the anchor (a naive backward-nearest match on raw
timestamps would leak a still-forming higher-tf candle's eventual final OHLC into the anchor's
features). See `_last_closed_position` for the exact shifted-merge_asof this enforces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from application.dataset_generation.extremum_features import (
    BranchExtremum,
    align_source_atr,
    build_branch_extremum,
    compute_extremum_weight,
    compute_higher_extremum_distance,
)
from application.dataset_generation.mfe_mae_om_labels import add_mfe_mae_om_labels
from application.dataset_generation.relative_candle import add_relative_candle_columns
from application.dataset_generation.volume_feature import add_log_sma_volume_feature_column, add_volume_feature_columns
from application.model_implementations.tier1_000.model import (
    AUX_FEATURE_DIM,
    BRANCH_TIMEFRAMES,
    BRANCH_WINDOW_LENGTHS,
    CANDLE_FEATURE_COLUMNS,
)
from config import app_config
from domain.ohlcv.ohlcv import read_multi_timeframe_ohlcv
from domain.price_action.CausalExtremum import TF_MINUTES, plus2tf, plus3tf
from domain.schemas.common.OHLCV import OHLCV
from domain.schemas.price_action.extremum_features import BranchExtremumOHLC
from helper.data_preparation import single_timeframe
from helper.importer import ptd, ta

# The 5 static (non-anchor-dependent) relative-candle/volume columns gathered directly per branch —
# everything else in CANDLE_FEATURE_COLUMNS (relative_normal_close, the 8 higher_extremum_distance
# fields, extremum_weight) needs anchor-context (the training anchor's own base close/time) and is
# assembled separately in build_dataset() below.
_STATIC_CANDLE_COLUMNS: list[str] = [
    "rel_high_close",
    "rel_close_low",
    "gap",
    "rel_candle_height",
    "log_volume_sma_ratio",
]

# relative_candle.py's ATR_LENGTH=256 needs 256 *candles* of warmup — trivial on 5min/15min/1h/4h/1D
# given the cached history depth here, but 256 *weeks* (~4.9yr) exceeds the longest verified
# gap-free cached range (995 days/~142wk) for the 1W branch. Shorter, still-reasonable override for
# that one branch only — add_relative_candle_columns() skips its own ATR calc when 'atr' is already
# present, so precomputing it here is a clean, non-invasive hook (doesn't touch the shared module,
# which other consumers still get the real 256-period default from).
_ATR_LENGTH_OVERRIDE: dict[str, int] = {"1W": 20}


@dataclass
class DatasetBundle:
    """branch_windows/auxiliary_features shape per this module's input variation 3 (BRANCH_WINDOW_LENGTHS);
    mfe/rer/action shape per this module's outcome set 1 (a mean_std_pairs [mfe, rer] regression target
    per sample, not the mean_std_skew_kurtosis_pairs a future outcome-set-2 builder would carry)."""

    branch_windows: dict[str, npt.NDArray[np.float32]]  # tf -> (n_samples, window_len, feature_dim)
    auxiliary_features: npt.NDArray[np.float32]  # (n_samples, AUX_FEATURE_DIM)
    mfe: npt.NDArray[np.float32]  # (n_samples, 1)
    rer: npt.NDArray[np.float32]  # (n_samples, 1)
    action: npt.NDArray[np.float32]  # (n_samples, 3) one-hot [long, short, none]
    anchor_index: pd.DatetimeIndex

    @property
    def n_samples(self) -> int:
        return len(self.anchor_index)


def _branch_features(ohlc: ptd[OHLCV], tf_name: str) -> ptd[BranchExtremumOHLC]:
    """Returns the FULL processed frame (not sliced to CANDLE_FEATURE_COLUMNS) — build_dataset() needs
    raw 'close'/'atr'/'high'/'low' alongside the 5 static ratio columns, both to compute
    relative_normal_close's anchor-relative formula (gap 1) and to feed CausalExtremum's Step A/B
    (gap 3), neither of which are literal columns relative_candle.py itself produces."""
    ohlc = ohlc.copy()
    if tf_name in _ATR_LENGTH_OVERRIDE:
        ohlc["atr"] = ta.atr(
            high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], length=_ATR_LENGTH_OVERRIDE[tf_name]
        )
    ohlc = add_relative_candle_columns(ohlc)  # sets 'atr' (if not already) + the 5 relative-HLC ratios
    ohlc = add_volume_feature_columns(ohlc)  # unchanged legacy volume_atr column (unused downstream here)
    # Same 256-period-exceeds-available-history problem _ATR_LENGTH_OVERRIDE already works around for
    # the 1W branch's ATR (pandas_ta.sma, like pandas_ta.atr, returns None outright rather than a NaN
    # series when length > len(series) — reusing the same override avoids that crash).
    ohlc = add_log_sma_volume_feature_column(ohlc, length=_ATR_LENGTH_OVERRIDE.get(tf_name, 256))
    return ohlc


def _last_closed_position(
    anchor_index: pd.DatetimeIndex, branch_index: pd.DatetimeIndex, branch_tf_minutes: float, base_tf_minutes: float
) -> npt.NDArray[np.int64]:
    """Row position (into branch_index) of the latest branch-tf candle fully closed by each anchor's
    own close. shift = (branch_tf - base_tf) so a branch candle starting at `anchor - shift` closes at
    `anchor - shift + branch_tf == anchor + base_tf` — exactly the anchor's own close. shift=0 for the
    base (5min) branch itself, matching each anchor to itself (the "LAST" candle, already closed)."""
    shift = pd.Timedelta(minutes=branch_tf_minutes - base_tf_minutes)
    shifted_anchors = ptd({"date": anchor_index - shift}).sort_values("date")
    branch_positions = ptd({"date": branch_index, "position": np.arange(len(branch_index))})
    merged = pd.merge_asof(shifted_anchors, branch_positions, on="date", direction="backward")
    # restore original anchor order (sort_values above was needed for merge_asof's monotonic requirement)
    merged.index = shifted_anchors.index
    positions = merged["position"].reindex(range(len(anchor_index)))
    return positions.fillna(-1).to_numpy(dtype=np.int64)


def _window_gather_indices(positions: npt.NDArray[np.int64], window_len: int) -> npt.NDArray[np.int64]:
    """positions: (n_anchors,) row index of each window's LAST candle. Returns (n_anchors, window_len)
    row-position matrix into the branch's own native series — the same index matrix every per-branch
    feature (static ratios, raw close/atr, extremum weight/distance) is gathered through, so every
    channel stays aligned to the same (anchor, window-position) grid."""
    offsets = np.arange(-(window_len - 1), 1)  # window_len values, e.g. [-255,...,-1,0]
    return positions[:, None] + offsets[None, :]  # (n_anchors, window_len)


def _gather_windows(
    features: npt.NDArray[np.float32], positions: npt.NDArray[np.int64], window_len: int
) -> npt.NDArray[np.float32]:
    """features: (n_candles, feature_dim); positions: (n_anchors,) row index of each window's LAST
    candle. Returns (n_anchors, window_len, feature_dim); positions < window_len-1 (insufficient
    history) must already be filtered out by the caller."""
    gather_idx = _window_gather_indices(positions, window_len)
    return features[gather_idx]


def build_dataset(symbol: str, date_range_str: str) -> DatasetBundle:
    app_config.under_process_symbol = symbol
    mt_ohlcv = read_multi_timeframe_ohlcv(date_range_str)

    base_ohlc = single_timeframe(mt_ohlcv, "5min")
    fifteen_min_ohlc = single_timeframe(mt_ohlcv, "15min")
    labels = add_mfe_mae_om_labels(base_ohlc, fifteen_min_ohlc)  # drops the last HORIZON_BARS rows

    features_by_tf = {
        tf_name: _branch_features(single_timeframe(mt_ohlcv, tf_name), tf_name) for tf_name in BRANCH_TIMEFRAMES
    }
    base_tf_minutes = 5.0
    tf_minutes = {"5min": 5.0, "15min": 15.0, "1h": 60.0, "4h": 240.0, "1D": 1440.0, "1W": 10080.0}

    anchor_index = labels.index
    positions_by_tf: dict[str, npt.NDArray[np.int64]] = {}
    valid = np.ones(len(anchor_index), dtype=bool)
    for tf_name in BRANCH_TIMEFRAMES:
        positions = _last_closed_position(
            anchor_index, features_by_tf[tf_name].index, tf_minutes[tf_name], base_tf_minutes
        )
        positions_by_tf[tf_name] = positions
        valid &= positions >= (BRANCH_WINDOW_LENGTHS[tf_name] - 1)

    anchor_index = anchor_index[valid]
    labels = labels.loc[anchor_index]

    # gap 1's LAST anchor: base_ohlc.loc[anchor_index, 'close'] computed AFTER the valid-filter above,
    # so it's already aligned to the filtered anchor set — one fixed close value per training sample,
    # subtracted from every candle across every branch's window for that sample.
    anchor_base_close = base_ohlc.loc[anchor_index, "close"].to_numpy(dtype=np.float64)
    # .asi8 reflects the index's own storage unit, not always nanoseconds — pandas >=3 no longer
    # always upcasts to 'ns', so force it before any _NS_PER_MINUTE-based arithmetic downstream.
    anchor_time_ns = anchor_index.as_unit("ns").asi8

    # gap 3: Step A (full-hindsight causal-capped reach) runs once per branch's own native series —
    # reused both for that branch's own extremum_weight (Step C) and, when this branch is itself a
    # plus2TF/plus3TF target for some OTHER branch, as that other branch's Step D source pool.
    branch_extremum_by_tf: dict[str, BranchExtremum] = {
        tf_name: build_branch_extremum(features_by_tf[tf_name]) for tf_name in BRANCH_TIMEFRAMES
    }

    def _source_for(target_tf: str) -> BranchExtremum:
        # 1M/4M/1Y have no native cached series (app_config.timeframes stops at 1W) — the 1W branch's
        # own reach validly stands in, see model.py's module docstring.
        return branch_extremum_by_tf.get(target_tf, branch_extremum_by_tf["1W"])

    def _source_native_minutes(target_tf: str) -> float:
        # The ATR-alignment shift must be the *source branch's own* native spacing (e.g. 1W's 10080),
        # not the target TF's width (e.g. 1M's 43200) — those coincide for the 6 real branches (target
        # IS the source) but diverge for 1M/4M/1Y targets, whose source is 1W: a "closed 1W candle"
        # only needs 10080 minutes to elapse, not the full synthetic 1M/4M/1Y width. The eligibility
        # threshold/causal-cutoff/time-distance normalization inside compute_higher_extremum_distance
        # correctly stay target-relative (TF_MINUTES[target_tf]) — this is a separate, narrower fix
        # scoped only to how the source's own ATR gets causally aligned onto t_i.
        return tf_minutes.get(target_tf, tf_minutes["1W"])

    branch_windows: dict[str, npt.NDArray[np.float32]] = {}
    last_candle_by_tf: list[npt.NDArray[np.float32]] = []
    for tf_name in BRANCH_TIMEFRAMES:
        positions = positions_by_tf[tf_name][valid]
        window_len = BRANCH_WINDOW_LENGTHS[tf_name]
        gather_idx = _window_gather_indices(positions, window_len)
        feat_df = features_by_tf[tf_name]

        static_windows = _gather_windows(
            feat_df[_STATIC_CANDLE_COLUMNS].to_numpy(dtype=np.float32), positions, window_len
        )  # (n_anchors, window_len, 5): rel_high_close, rel_close_low, gap, rel_candle_height, log_volume_sma_ratio

        close_arr = feat_df["close"].to_numpy(dtype=np.float64)
        atr_arr = feat_df["atr"].to_numpy(dtype=np.float64)
        windowed_close = close_arr[gather_idx]  # (n_anchors, window_len)
        windowed_atr = atr_arr[gather_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_normal_close = (windowed_close - anchor_base_close[:, None]) / windowed_atr
        relative_normal_close = relative_normal_close.astype(np.float32)

        windowed_time_ns = feat_df.index.as_unit("ns").asi8[gather_idx]

        weight = compute_extremum_weight(
            branch_extremum_by_tf[tf_name], gather_idx, anchor_time_ns, tf_minutes[tf_name]
        )

        plus2_tf = plus2tf(tf_name)
        plus3_tf = plus3tf(tf_name)
        plus2_source = _source_for(plus2_tf)
        plus3_source = _source_for(plus3_tf)
        # Alignment is purely a function of "which branch-X candle" (its own timestamp), independent
        # of anchor — computed once over the branch's FULL native index, then windowed with the same
        # gather_idx as every other per-branch channel (not recomputed per anchor).
        plus2_atr_full = align_source_atr(feat_df.index, plus2_source, _source_native_minutes(plus2_tf))
        plus3_atr_full = align_source_atr(feat_df.index, plus3_source, _source_native_minutes(plus3_tf))
        plus2_atr_windowed = plus2_atr_full[gather_idx]
        plus3_atr_windowed = plus3_atr_full[gather_idx]

        extremum_distance = compute_higher_extremum_distance(
            windowed_close,
            windowed_time_ns,
            anchor_time_ns,
            plus2_source,
            TF_MINUTES[plus2_tf],
            plus2_atr_windowed,
            plus3_source,
            TF_MINUTES[plus3_tf],
            plus3_atr_windowed,
        )

        windows = np.concatenate(
            [
                relative_normal_close[:, :, None],
                static_windows,
                extremum_distance["price_normal_distance_plus2tf_peak"][:, :, None],
                extremum_distance["price_normal_distance_plus2tf_valley"][:, :, None],
                extremum_distance["price_normal_distance_plus3tf_peak"][:, :, None],
                extremum_distance["price_normal_distance_plus3tf_valley"][:, :, None],
                extremum_distance["time_distance_plus2tf_peak"][:, :, None],
                extremum_distance["time_distance_plus2tf_valley"][:, :, None],
                extremum_distance["time_distance_plus3tf_peak"][:, :, None],
                extremum_distance["time_distance_plus3tf_valley"][:, :, None],
                weight[:, :, None],
            ],
            axis=2,
        ).astype(np.float32)
        assert windows.shape[2] == len(CANDLE_FEATURE_COLUMNS)

        branch_windows[tf_name] = windows
        last_candle_by_tf.append(windows[:, -1, :])

    auxiliary_features = np.concatenate(last_candle_by_tf, axis=1).astype(np.float32)
    assert auxiliary_features.shape[1] == AUX_FEATURE_DIM

    # Indicator warmup (e.g. the 1W branch's own ATR override still needs ~20 prior weeks) can leave
    # NaN in a window's oldest candles even after the position-based validity check above, which only
    # accounts for window *length*, not each feature's own warmup. Final blanket scrub, matching this
    # codebase's existing dropna-to-first-fully-valid-row convention (e.g. training_datasets.py's
    # not_na_range) rather than hand-deriving each indicator's exact warmup length.
    clean = np.ones(len(anchor_index), dtype=bool)
    for windows in branch_windows.values():
        clean &= ~np.isnan(windows).any(axis=(1, 2))
    clean &= ~np.isnan(auxiliary_features).any(axis=1)

    return DatasetBundle(
        branch_windows={tf_name: w[clean] for tf_name, w in branch_windows.items()},
        auxiliary_features=auxiliary_features[clean],
        mfe=labels[["mfe"]].to_numpy(dtype=np.float32)[clean],
        rer=labels[["rer"]].to_numpy(dtype=np.float32)[clean],
        action=labels[["action_long", "action_short", "action_none"]].to_numpy(dtype=np.float32)[clean],
        anchor_index=anchor_index[clean],
    )


def split_bundle(bundle: DatasetBundle, val_fraction: float = 0.1) -> tuple[DatasetBundle, DatasetBundle]:
    """Time-ordered split (validation = the most recent slice) — never shuffle before splitting, or
    validation samples' overlapping label horizons would leak into training."""
    n_val = max(1, int(bundle.n_samples * val_fraction))
    split_at = bundle.n_samples - n_val

    def _slice(sl: slice) -> DatasetBundle:  # type: ignore[explicit-any]
        # shapes: branch_windows[tf]=(n_samples, window_len, feature_dim),
        # auxiliary_features=(n_samples, AUX_FEATURE_DIM), mfe=(n_samples, 1),
        # rer=(n_samples, 1), action=(n_samples, 3), anchor_index=(n_samples,)
        return DatasetBundle(
            branch_windows={tf_name: arr[sl] for tf_name, arr in bundle.branch_windows.items()},
            auxiliary_features=bundle.auxiliary_features[sl],
            mfe=bundle.mfe[sl],
            rer=bundle.rer[sl],
            action=bundle.action[sl],
            anchor_index=bundle.anchor_index[sl],
        )

    return _slice(slice(0, split_at)), _slice(slice(split_at, None))


def make_tf_dataset(bundle: DatasetBundle, batch_size: int, shuffle: bool, seed: int = 0) -> tf.data.Dataset:
    inputs = {tf_name: bundle.branch_windows[tf_name] for tf_name in BRANCH_TIMEFRAMES}
    inputs["auxiliary_features"] = bundle.auxiliary_features
    targets = {"action": bundle.action, "mfe_params": bundle.mfe, "rer_params": bundle.rer}
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=bundle.n_samples, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.AUTOTUNE)
