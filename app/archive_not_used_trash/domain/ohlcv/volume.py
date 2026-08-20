import pandas as pd
import pandas_ta as ta  # noqa
from config import app_config
from domain.schemas.common.MultiTimeframe import MultiTimeframe
from domain.schemas.common.OHLCV import OHLCV
from helper.data_preparation import single_timeframe
from pandera import typing as pt


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
