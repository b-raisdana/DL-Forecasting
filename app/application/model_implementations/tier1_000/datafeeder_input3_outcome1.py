"""Datafeeder for the Tier-1_000 architecture (see docs/ML_Forecasting_System_Design/designsets/
Tier-1_000.par_branch_mtcn_lstm_perc_gqa_mlp_lgbm (handmade).base.jsonc), built specifically against
**input variation 3** of the input-data-feature category's handmade reference,
`Tier-1_000.atr_rel_ohlc_log_sma_v_extm_rel_6tf (handmade).input.jsonc` (256 candles on 5min/15min, 128 on
1h/4h, 64 on 1D/1W — see model.py's module docstring for why variation 3 and not the "1-base" row's
variation 1), and **outcome set 1** of the outcome-label-target-head category's handmade reference,
`Tier-1_000.action_mfe_rer (handmade).outcome.jsonc` (action_head + mean_std_pairs for
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
from application.dataset_generation.mfe_mae_om_labels import add_mfe_mae_om_labels
from application.dataset_generation.relative_candle import add_relative_candle_columns
from application.dataset_generation.volume_feature import add_volume_feature_columns
from application.model_implementations.tier1_000.model import (
    AUX_FEATURE_DIM,
    BRANCH_TIMEFRAMES,
    BRANCH_WINDOW_LENGTHS,
    CANDLE_FEATURE_COLUMNS,
)
from config import app_config
from helper.data_preparation import single_timeframe
from helper.importer import ta
from infrastructure.ohlcv.ohlcv import read_multi_timeframe_ohlcv

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


def _branch_features(ohlc: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    ohlc = ohlc.copy()
    if tf_name in _ATR_LENGTH_OVERRIDE:
        ohlc["atr"] = ta.atr(
            high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], length=_ATR_LENGTH_OVERRIDE[tf_name]
        )
    ohlc = add_relative_candle_columns(ohlc)
    ohlc = add_volume_feature_columns(ohlc)
    return ohlc[CANDLE_FEATURE_COLUMNS].astype(np.float32)


def _last_closed_position(
    anchor_index: pd.DatetimeIndex, branch_index: pd.DatetimeIndex, branch_tf_minutes: float, base_tf_minutes: float
) -> npt.NDArray[np.int64]:
    """Row position (into branch_index) of the latest branch-tf candle fully closed by each anchor's
    own close. shift = (branch_tf - base_tf) so a branch candle starting at `anchor - shift` closes at
    `anchor - shift + branch_tf == anchor + base_tf` — exactly the anchor's own close. shift=0 for the
    base (5min) branch itself, matching each anchor to itself (the "LAST" candle, already closed)."""
    shift = pd.Timedelta(minutes=branch_tf_minutes - base_tf_minutes)
    shifted_anchors = pd.DataFrame({"date": anchor_index - shift}).sort_values("date")
    branch_positions = pd.DataFrame({"date": branch_index, "position": np.arange(len(branch_index))})
    merged = pd.merge_asof(shifted_anchors, branch_positions, on="date", direction="backward")
    # restore original anchor order (sort_values above was needed for merge_asof's monotonic requirement)
    merged.index = shifted_anchors.index
    positions = merged["position"].reindex(range(len(anchor_index)))
    return positions.fillna(-1).to_numpy(dtype=np.int64)


def _gather_windows(
    features: npt.NDArray[np.float32], positions: npt.NDArray[np.int64], window_len: int
) -> npt.NDArray[np.float32]:
    """features: (n_candles, feature_dim); positions: (n_anchors,) row index of each window's LAST
    candle. Returns (n_anchors, window_len, feature_dim); positions < window_len-1 (insufficient
    history) must already be filtered out by the caller."""
    offsets = np.arange(-(window_len - 1), 1)  # window_len values, e.g. [-255,...,-1,0]
    gather_idx = positions[:, None] + offsets[None, :]  # (n_anchors, window_len)
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

    branch_windows: dict[str, npt.NDArray[np.float32]] = {}
    last_candle_by_tf: list[npt.NDArray[np.float32]] = []
    for tf_name in BRANCH_TIMEFRAMES:
        positions = positions_by_tf[tf_name][valid]
        windows = _gather_windows(
            features_by_tf[tf_name].to_numpy(dtype=np.float32), positions, BRANCH_WINDOW_LENGTHS[tf_name]
        )
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

    def _slice(sl: slice) -> DatasetBundle:
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
