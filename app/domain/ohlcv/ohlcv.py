from typing import cast

import pandas as pd
from config import app_config
from domain.schemas.common.OHLCV import OHLCV
from helper.data_preparation import after_under_process_date, times_tester
from helper.pandera import pandera_validate
from helper.schema_casting import cast_and_validate
from infrastructure.datastore_engine.disk_cache_layout import CachableDataset
from pandera import typing as pt

OHLCV_DATASET = CachableDataset(dataset_folder_name="ohlcv")
MULTI_TIMEFRAME_OHLCV_DATASET = CachableDataset(dataset_folder_name="multi_timeframe_ohlcv")


@pandera_validate
def build_base_timeframe_ohlcv(
    raw_ohlcv: list[object], date_range_str: str, base_timeframe: str | None = None
) -> pt.DataFrame[OHLCV]:
    df = pd.DataFrame(raw_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("date")
    df = df.drop(columns=["timestamp"])
    cast_and_validate(df, OHLCV, zero_size_allowed=after_under_process_date(date_range_str))
    assert times_tester(df, date_range_str, timeframe=app_config.timeframes[0])
    return cast(pt.DataFrame[OHLCV], df)
