import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa
from config import app_config
from domain.schemas.common.MultiTimeframe import MultiTimeframe
from domain.schemas.common.OHLCV import OHLCV
from helper.data_preparation import single_timeframe
from pandera import typing as pt


def RMA(values: pd.DataFrame, length):
    # alpha = 1 / length
    # rma = np.zeros_like(values)
    # rma[0] = values[0]
    # for i in range(1, len(values)):
    #     rma[i] = alpha * values[i] + np.nan_to_num((1 - alpha) * rma[i - 1])
    #     pass
    # return rma
    alpha = 1 / length
    rma = pd.DataFrame(values.index, np.nan)  # Initialize with NaN

    # Find the first non-NaN value in the series
    first_valid_index = values.first_valid_index()
    if first_valid_index is None:
        return rma  # Return as all NaN if no valid values

    rma[first_valid_index] = values[first_valid_index]  # Start with the first valid value

    for i in range(first_valid_index + 1, len(values)):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]

    return rma


def insert_mt_volume_rma(mt_v: pt.DataFrame[MultiTimeframe]):
    for timeframe in mt_v.index.get_level_values(level="timeframe").unique():
        timeframe_indexes = single_timeframe(mt_v, timeframe, index_only=True)
        timeframe_v = mt_v[timeframe_indexes]
        mt_v.loc[timeframe_indexes, "volume_rma"] = insert_volume_rma(timeframe_v)
    return mt_v


def insert_volume_rma(timeframe_v: pt.DataFrame[OHLCV]):
    """
    timeframe_v['volume_rma'] = timeframe_v['volume'] / ta.rma(timeframe_v['volume'])
    Args:
        timeframe_v:

    Returns:

    """
    if len(timeframe_v) <= app_config.atr_timeperiod:
        timeframe_v["volume_rma"] = pd.NA
        return timeframe_v
    timeframe_v["volume_rma"] = timeframe_v["volume"] / ta.rma(timeframe_v["volume"], length=app_config.atr_timeperiod)
    return timeframe_v["volume_rma"]


# @measure_time
def insert_atr(timeframe_ohlcv: pt.DataFrame[OHLCV], mode: str = "pandas_ta") -> pd.DataFrame:
    if len(timeframe_ohlcv) <= app_config.atr_timeperiod:
        timeframe_ohlcv["atr"] = pd.NA
    else:
        if mode == "pandas_ta":
            timeframe_ohlcv["atr"] = timeframe_ohlcv.ta.atr(
                timeperiod=app_config.atr_timeperiod,
                # high='high',
                # low='low',
                # close='close',
                # mamode='ema',
            )
        else:
            raise Exception(f"Unsupported mode:{mode}")
    insert_volume_rma(timeframe_ohlcv)
    return timeframe_ohlcv
