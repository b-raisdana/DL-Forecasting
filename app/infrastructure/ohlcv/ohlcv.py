import pandas as pd
from config import app_config
from domain.schemas.common.OHLCV import OHLCV, MultiTimeframeOHLCV
from helper.data_preparation import (
    after_under_process_date,
    concat,
    multi_timeframe_times_tester,
    single_timeframe,
    times_tester,
    trim_to_date_range,
)
from helper.logging import profile_it
from helper.schema_casting import cast_and_validate
from infrastructure.market_data_fetch.kucoin.fetch_ohlcv import fetch_ohlcv_by_range
from infrastructure.ohlcv.disk_cache import cache_on_disk
from pandera import typing as pt


def cache_times(result):
    for timeframe in app_config.timeframes:
        app_config.GLOBAL_CACHE[f"valid_times_{timeframe}"] = single_timeframe(result, timeframe).index


@profile_it
@cache_on_disk(file_name_prefix="ohlcv", windowed=True)
def get_base_timeframe_ohlcv(date_range_str: str = None, base_timeframe=None) -> pt.DataFrame[OHLCV]:
    """
    Generates exactly one cache window's worth of base-timeframe OHLCV (see
    cache_on_disk(windowed=True) / infrastructure/ohlcv/README.md § windowing);
    disk_cache.read_file_windowed() decomposes an arbitrary requested date_range_str into windows,
    fetches/generates each through this function, and stitches the result back together.
    """
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    raw_ohlcv = fetch_ohlcv_by_range(date_range_str)
    df = pd.DataFrame(raw_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("date")
    df = df.drop(columns=["timestamp"])
    cast_and_validate(df, OHLCV, zero_size_allowed=after_under_process_date(date_range_str))
    assert times_tester(df, date_range_str, timeframe=app_config.timeframes[0])
    return df


@profile_it
@cache_on_disk(file_name_prefix="multi_timeframe_ohlcv", after_read=cache_times, windowed=True)
def get_multi_timeframe_ohlcv(date_range_str: str = None) -> MultiTimeframeOHLCV:
    """One window's worth of multi-timeframe OHLCV, aggregated from get_base_timeframe_ohlcv() — see
    get_base_timeframe_ohlcv()'s docstring for how windowing decomposes an arbitrary date_range_str."""
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    ohlcv = get_base_timeframe_ohlcv(date_range_str)

    multi_timeframe_ohlcv = ohlcv.copy()
    multi_timeframe_ohlcv.insert(0, "timeframe", app_config.timeframes[0])
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.set_index("timeframe", append=True)
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.swaplevel()
    for _, timeframe in enumerate(app_config.timeframes[1:]):
        if timeframe == "1W":
            frequency = "W-MON"
        elif timeframe == "M":
            frequency = "MS"
        else:
            frequency = timeframe
        _timeframe_ohlcv = ohlcv.groupby(pd.Grouper(freq=frequency)).agg(
            {
                "open": "first",
                "close": "last",
                "low": "min",
                "high": "max",
                "volume": "sum",
            }
        )
        if len(_timeframe_ohlcv.index) > 0:
            _timeframe_ohlcv.insert(0, "timeframe", timeframe)
            _timeframe_ohlcv = _timeframe_ohlcv.set_index("timeframe", append=True)
            _timeframe_ohlcv = _timeframe_ohlcv.swaplevel()
            multi_timeframe_ohlcv = concat(multi_timeframe_ohlcv, _timeframe_ohlcv)
    multi_timeframe_ohlcv = trim_to_date_range(date_range_str, multi_timeframe_ohlcv)
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.sort_index(level="date")
    assert multi_timeframe_times_tester(multi_timeframe_ohlcv, date_range_str)
    return multi_timeframe_ohlcv


read_multi_timeframe_ohlcv = get_multi_timeframe_ohlcv
