from datetime import datetime, timedelta

import pandas as pd
import pytz
from br_py.profiling import profile_it
from Config import app_config
from data_processing.disk_cache import cache_on_disk
from data_processing.fetch_ohlcv import fetch_ohlcv_by_range
from helper.data_preparation import (
    after_under_process_date,
    cast_and_validate,
    concat,
    empty_df,
    multi_timeframe_times_tester,
    single_timeframe,
    times_tester,
    trim_to_date_range,
)
from helper.functions import date_range
from pandera import typing as pt
from PanderaDFM.OHLCV import OHLCV, MultiTimeframeOHLCV


def cache_times(result):
    for timeframe in app_config.timeframes:
        app_config.GLOBAL_CACHE[f"valid_times_{timeframe}"] = single_timeframe(result, timeframe).index


@cache_on_disk(file_name_prefix="multi_timeframe_ohlcv", after_read=cache_times)
def core_get_multi_timeframe_ohlcv(date_range_str: str) -> MultiTimeframeOHLCV:
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


def read_daily_multi_timeframe_ohlcv(day: datetime) -> MultiTimeframeOHLCV:
    # Format the date_range_str for the given day
    start_str = day.strftime("%y-%m-%d.00-00")
    end_str = day.strftime("%y-%m-%d.23-59")
    day_date_range_str = f"{start_str}T{end_str}"

    if day.replace(hour=0, minute=0, second=0, microsecond=0) > datetime.now(tz=pytz.UTC):
        return empty_df(MultiTimeframeOHLCV)
    return core_get_multi_timeframe_ohlcv(day_date_range_str)


@profile_it
@cache_on_disk(file_name_prefix="multi_timeframe_ohlcv", after_read=cache_times)
def get_multi_timeframe_ohlcv(date_range_str: str = None) -> MultiTimeframeOHLCV:
    start, end = date_range(date_range_str)

    # Split the date range into individual days
    day_to_fetch = start
    daily_dataframes = []

    while day_to_fetch.date() <= end.date():
        # For each day, get the data and append to daily_dataframes list
        daily_dataframes.append(read_daily_multi_timeframe_ohlcv(day_to_fetch))
        day_to_fetch += timedelta(days=1)

    # Concatenate the daily data
    df = pd.concat(daily_dataframes)
    df = df.sort_index(level="date")
    df = trim_to_date_range(date_range_str, df)
    assert multi_timeframe_times_tester(df, date_range_str)
    return df


read_multi_timeframe_ohlcv = get_multi_timeframe_ohlcv


@cache_on_disk(file_name_prefix="ohlcv")
def core_get_ohlcv(date_range_str: str = None) -> pt.DataFrame[OHLCV]:
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    # if config.do_not_fetch_prices:
    #     raw_ohlcv = []
    # else:
    raw_ohlcv = fetch_ohlcv_by_range(date_range_str)
    df = pd.DataFrame(raw_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("date")
    df = df.drop(columns=["timestamp"])
    cast_and_validate(df, OHLCV, zero_size_allowed=after_under_process_date(date_range_str))
    assert times_tester(df, date_range_str, timeframe=app_config.timeframes[0])
    return df


def read_daily_ohlcv(day: datetime, base_timeframe=None) -> pt.DataFrame[OHLCV]:
    # Format the date_range_str for the given day
    start_str = day.strftime("%y-%m-%d.00-00")
    end_str = day.strftime("%y-%m-%d.23-59")
    day_date_range_str = f"{start_str}T{end_str}"

    return core_get_ohlcv(day_date_range_str)


@profile_it
@cache_on_disk(file_name_prefix="ohlcv")
def get_base_timeframe_ohlcv(date_range_str: str = None, base_timeframe=None) -> pt.DataFrame[OHLCV]:
    start, end = date_range(date_range_str)
    # Split the date range into individual days
    current_day = start
    daily_dataframes = []
    while current_day.date() <= end.date():
        # For each day, get the data and append to daily_dataframes list
        daily_dataframes.append(read_daily_ohlcv(current_day, base_timeframe))
        current_day += timedelta(days=1)
    # Concatenate the daily data
    df = pd.concat(daily_dataframes)
    df = df.sort_index(level="date")
    df = trim_to_date_range(date_range_str, df)
    assert times_tester(df, date_range_str, app_config.timeframes[0])
    return df
