from __future__ import annotations

import numpy as np
from config import app_config
from domain.schemas.common.OHLCV import OHLCV
from helper.importer import ptd, ta

__volume_feature_columns = ["volume_atr"]


# def volume_feature_columns() -> list[str]:
# return __volume_feature_columns


def add_volume_feature_columns(ohlcv: ptd[OHLCV]) -> ptd[OHLCV]:
    """volume / ATR(volume) per docs/input-features.md § candle feature schema.

    Volume has no high/low/close to derive a true-range from, so "ATR(volume)" is Wilder's RMA
    (the same smoothing ATR itself uses) applied directly to volume. The ratio is already
    ~1-centered, unlike raw 'volume' which scale_slice rescales.
    """
    volume_rma = ta.rma(ohlcv["volume"], length=app_config.atr_timeperiod)
    ohlcv["volume_atr"] = ohlcv["volume"] / volume_rma.replace(0, np.nan)
    return ohlcv


def add_log_sma_volume_feature_column(ohlcv: ptd[OHLCV], length: int = 256) -> ptd[OHLCV]:
    """log((volume + eps) / (SMA_length(volume) + eps)) per .../atr_rel_ohlc_log_sma_v_extm_rel_6tf
    (handmade).input.jsonc's `V` definition. Additive sibling to add_volume_feature_columns (which
    stays unchanged for training_datasets.py's separate pipeline) — a different smoothing (plain SMA,
    not Wilder's RMA), a different default length (256, not app_config.atr_timeperiod's 14), and a
    log/eps transform add_volume_feature_columns doesn't apply.
    """
    eps = np.finfo(np.float64).eps
    volume_sma = ta.sma(ohlcv["volume"], length=length)
    ohlcv["log_volume_sma_ratio"] = np.log((ohlcv["volume"] + eps) / (volume_sma + eps))
    return ohlcv


# def log_sma_volume_feature_columns() -> list[str]:
# return ["log_volume_sma_ratio"]
