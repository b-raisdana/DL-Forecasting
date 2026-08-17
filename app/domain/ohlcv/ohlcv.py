import pandas as pd
from config import app_config
from domain.ohlcv.multi_timeframe import aggregate_multi_timeframe_ohlcv
from domain.schemas.common.OHLCV import OHLCV, MultiTimeframeOHLCV
from helper.data_preparation import after_under_process_date, single_timeframe, times_tester
from helper.logging import profile_it
from helper.schema_casting import cast_and_validate
from infrastructure.disk_cache import cache_on_disk
from infrastructure.market_data_fetch.kucoin.fetch_ohlcv import fetch_ohlcv_by_range
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
    return build_base_timeframe_ohlcv(raw_ohlcv, date_range_str, base_timeframe)


@profile_it
@cache_on_disk(file_name_prefix="multi_timeframe_ohlcv", after_read=cache_times, windowed=True)
def get_multi_timeframe_ohlcv(date_range_str: str = None) -> MultiTimeframeOHLCV:
    """One window's worth of multi-timeframe OHLCV, aggregated from get_base_timeframe_ohlcv() — see
    get_base_timeframe_ohlcv()'s docstring for how windowing decomposes an arbitrary date_range_str."""
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    ohlcv = get_base_timeframe_ohlcv(date_range_str)
    return aggregate_multi_timeframe_ohlcv(ohlcv, date_range_str)


read_multi_timeframe_ohlcv = get_multi_timeframe_ohlcv


def build_base_timeframe_ohlcv(raw_ohlcv, date_range_str: str, base_timeframe=None) -> pt.DataFrame[OHLCV]:
    """
    Shapes a raw (timestamp, open, high, low, close, volume) row list — as returned by an exchange
    fetch — into a validated, date-indexed OHLCV DataFrame. Pure transform, no I/O: the caller is
    responsible for actually fetching `raw_ohlcv`.
    """
    df = pd.DataFrame(raw_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("date")
    df = df.drop(columns=["timestamp"])
    cast_and_validate(df, OHLCV, zero_size_allowed=after_under_process_date(date_range_str))
    assert times_tester(df, date_range_str, timeframe=app_config.timeframes[0])
    return df
