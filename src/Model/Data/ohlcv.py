import os
from datetime import timedelta, datetime

import pandas as pd
import pytz
from pandera import typing as pt

from Config import config
# from MetaTrader import MT
from PanderaDFM.OHLCV import MultiTimeframeOHLCV, OHLCV
from Model.Data.fetch_ohlcv import fetch_ohlcv_by_range
from helper.data_preparation import read_file, single_timeframe, cast_and_validate, trim_to_date_range, to_timeframe, \
    after_under_process_date, multi_timeframe_times_tester, times_tester, empty_df, concat
from helper.helper import measure_time, date_range, date_range_to_string


def core_generate_multi_timeframe_ohlcv(date_range_str: str, file_path: str = None):
    if file_path is None:
        file_path = config.path_of_data
    biggest_timeframe = config.timeframes[-1]
    start, end = date_range(date_range_str)
    round_to_biggest_timeframe_end = to_timeframe(end, biggest_timeframe, ignore_cached_times=True, do_not_warn=True)
    if round_to_biggest_timeframe_end >= start:
        extended_end = (
                    to_timeframe(end, biggest_timeframe, ignore_cached_times=True, do_not_warn=True) + pd.to_timedelta(
                biggest_timeframe) -
                    pd.to_timedelta(config.timeframes[0]))
        extended_timerange_str = date_range_to_string(start=start, end=extended_end)
    else:
        extended_timerange_str = date_range_str

    ohlcv = read_base_timeframe_ohlcv(extended_timerange_str)

    # ohlcv['timeframe '] = config.timeframes[0]
    multi_timeframe_ohlcv = ohlcv.copy()
    multi_timeframe_ohlcv.insert(0, 'timeframe', config.timeframes[0])
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.set_index('timeframe', append=True)
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.swaplevel()
    for _, timeframe in enumerate(config.timeframes[1:]):
        if timeframe == '1W':
            frequency = 'W-MON'
        elif timeframe == 'M':
            frequency = 'MS'
        else:
            frequency = timeframe
        # if pd.to_timedelta(timeframe) >= timedelta(days=1):
        #     pass
        _timeframe_ohlcv = ohlcv.groupby(pd.Grouper(freq=frequency)) \
            .agg({'open': 'first',
                  'close': 'last',
                  'low': 'min',
                  'high': 'max',
                  'volume': 'sum', })
        if len(_timeframe_ohlcv.index) > 0:
            _timeframe_ohlcv.insert(0, 'timeframe', timeframe)
            _timeframe_ohlcv = _timeframe_ohlcv.set_index('timeframe', append=True)
            _timeframe_ohlcv = _timeframe_ohlcv.swaplevel()
            multi_timeframe_ohlcv = concat(multi_timeframe_ohlcv, _timeframe_ohlcv)
    multi_timeframe_ohlcv = trim_to_date_range(date_range_str, multi_timeframe_ohlcv)
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.sort_index(level='date')
    assert multi_timeframe_times_tester(multi_timeframe_ohlcv, date_range_str)
    # plot_multi_timeframe_ohlcv(multi_timeframe_ohlcv, date_range_str)
    multi_timeframe_ohlcv.to_csv(os.path.join(file_path, f'multi_timeframe_ohlcv.{date_range_str}.zip'),
                                 compression='zip')


# @measure_time
def core_read_multi_timeframe_ohlcv(date_range_str: str = None) -> MultiTimeframeOHLCV:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = read_file(date_range_str, 'multi_timeframe_ohlcv', core_generate_multi_timeframe_ohlcv,
                       MultiTimeframeOHLCV)
    cache_times(result)
    return result


def read_daily_multi_timeframe_ohlcv(day: datetime) -> MultiTimeframeOHLCV:
    # Format the date_range_str for the given day
    start_str = day.strftime('%y-%m-%d.00-00')
    end_str = day.strftime('%y-%m-%d.23-59')
    day_date_range_str = f'{start_str}T{end_str}'

    if day.replace(hour=0, minute=0, second=0, microsecond=0) > datetime.now(tz=pytz.UTC):
        return empty_df(MultiTimeframeOHLCV)
    # Fetch the data for the given day using the old function
    return core_read_multi_timeframe_ohlcv(day_date_range_str)


def read_multi_timeframe_ohlcv(date_range_str: str) -> pt.DataFrame[MultiTimeframeOHLCV]:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = read_file(date_range_str, 'multi_timeframe_ohlcv', generate_multi_timeframe_ohlcv,
                       MultiTimeframeOHLCV)
    cache_times(result)
    return result


def cache_times(result):
    for timeframe in config.timeframes:
        # if f'valid_times_{timeframe}' not in config.GLOBAL_CACHE.keys():
        config.GLOBAL_CACHE[f'valid_times_{timeframe}'] = \
            single_timeframe(result, timeframe).index.get_level_values('date').tolist()


@measure_time
def generate_multi_timeframe_ohlcv(date_range_str: str = None, file_path: str = None) -> None:
    if date_range_str == '23-09-13.00-00T23-12-27.23-59':
        pass
    if file_path is None:
        file_path = config.path_of_data
    start, end = date_range(date_range_str)

    # Split the date range into individual days
    current_day = start
    daily_dataframes = []

    while current_day.date() <= end.date():
        # For each day, get the data and append to daily_dataframes list
        daily_dataframes.append(read_daily_multi_timeframe_ohlcv(current_day))
        current_day += timedelta(days=1)

    # Concatenate the daily data
    df = pd.concat(daily_dataframes)
    df = df.sort_index(level='date')
    df = trim_to_date_range(date_range_str, df)
    assert multi_timeframe_times_tester(df, date_range_str)
    df.to_csv(os.path.join(file_path, f'multi_timeframe_ohlcv.{date_range_str}.zip'),
              compression='zip')


# @measure_time
def core_read_ohlcv(date_range_str: str = None, base_timeframe=None) -> pt.DataFrame[OHLCV]:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = read_file(date_range_str, f'ohlcv', core_generate_ohlcv, OHLCV,
                       generator_params={'base_timeframe': base_timeframe})
    return result


def read_daily_ohlcv(day: datetime, base_timeframe=None) -> pt.DataFrame[OHLCV]:
    # Format the date_range_str for the given day
    start_str = day.strftime('%y-%m-%d.00-00')
    end_str = day.strftime('%y-%m-%d.23-59')
    day_date_range_str = f'{start_str}T{end_str}'

    # Fetch the data for the given day using the old function
    return core_read_ohlcv(day_date_range_str, base_timeframe=base_timeframe)


def read_base_timeframe_ohlcv(date_range_str: str, base_timeframe=None) \
        -> pt.DataFrame[OHLCV]:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = read_file(date_range_str, f'ohlcv', generate_base_timeframe_ohlcv, OHLCV,
                       generator_params={'base_timeframe': base_timeframe})
    return result


@measure_time
def generate_base_timeframe_ohlcv(date_range_str: str = None, file_path: str = None, base_timeframe=None) -> None:
    if file_path is None:
        file_path = config.path_of_data
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
    df = df.sort_index(level='date')
    df = trim_to_date_range(date_range_str, df)
    assert times_tester(df, date_range_str, config.timeframes[0])
    df.to_csv(os.path.join(file_path, f'ohlcv.{date_range_str}.zip'), compression='zip')


def core_generate_ohlcv(date_range_str: str = None, file_path: str = None, base_timeframe=None):
    if date_range_str is None:
        date_range_str = config.processing_date_range
    if file_path is None:
        file_path = config.path_of_data
    raw_ohlcv = fetch_ohlcv_by_range(date_range_str, base_timeframe=base_timeframe)
    df = pd.DataFrame(raw_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('date')
    df = df.drop(columns=['timestamp'])
    cast_and_validate(df, OHLCV, zero_size_allowed=after_under_process_date(date_range_str))
    assert times_tester(df, date_range_str, timeframe=config.timeframes[0])
    df.to_csv(os.path.join(file_path, f'ohlcv.{date_range_str}.zip'), compression='zip')
    # if config.load_data_to_meta_trader:
    #     MT.extract_to_data_path(os.path.join(file_path, f'ohlcv.{date_range_str}.zip'))
    #     MT.load_rates()


# @measure_time
# def load_ohlcv_to_meta_trader(date_range_str: str = None, file_path: str = None):
#     if file_path is None:
#         file_path = config.path_of_data
#     if config.load_data_to_meta_trader:
#         MT.extract_to_data_path(os.path.join(file_path, f'ohlcv.{date_range_str}.zip'))
#         MT.load_rates()
