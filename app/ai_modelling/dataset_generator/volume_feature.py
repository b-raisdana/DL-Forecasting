import numpy as np
import pandas as pd

from Config import app_config
from helper.importer import ta

__volume_feature_columns = ['volume_atr']


def volume_feature_columns():
    return __volume_feature_columns


def add_volume_feature_columns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """volume / ATR(volume) per docs/input-features.md § candle feature schema.

    Volume has no high/low/close to derive a true-range from, so "ATR(volume)" is Wilder's RMA
    (the same smoothing ATR itself uses) applied directly to volume. The ratio is already
    ~1-centered, unlike raw 'volume' which scale_slice rescales.
    """
    volume_rma = ta.rma(ohlcv['volume'], length=app_config.atr_timeperiod)
    ohlcv['volume_atr'] = ohlcv['volume'] / volume_rma.replace(0, np.nan)
    return ohlcv
