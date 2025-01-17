import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa
import pytz
from pandera import typing as pt

from app.Config import app_config
from app.PanderaDFM.MultiTimeframe import MultiTimeframe
from app.PanderaDFM.OHLCV import OHLCV
from app.PanderaDFM.OHLCVA import MultiTimeframeOHLCVA
from app.data_processing.fragmented_data import symbol_data_path
from app.helper.data_preparation import read_file, trim_to_date_range, single_timeframe, expand_date_range, \
    multi_timeframe_times_tester, empty_df, concat
from app.helper.helper import date_range
from app.data_processing.ohlcv import read_multi_timeframe_ohlcv, cache_times


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
def insert_mt_volume_rma(mt_v:pt.DataFrame[MultiTimeframe]):
    for timeframe in mt_v.index.get_level_values(level='timeframe').unique():
        timeframe_indexes = single_timeframe(mt_v, timeframe, index_only=True)
        timeframe_v = mt_v[timeframe_indexes]
        mt_v.loc[timeframe_indexes, 'volume_rma'] = insert_volume_rma(timeframe_v)
    return mt_v

def insert_volume_rma(timeframe_v: pt.DataFrame[OHLCV]):
    '''
    timeframe_v['volume_rma'] = timeframe_v['volume'] / ta.rma(timeframe_v['volume'])
    Args:
        timeframe_v:

    Returns:

    '''
    if len(timeframe_v) <= app_config.atr_timeperiod:
        timeframe_v['volume_rma'] = pd.NA
        return timeframe_v
    timeframe_v['volume_rma'] = timeframe_v['volume'] / ta.rma(timeframe_v['volume'], length=app_config.atr_timeperiod)
    return timeframe_v['volume_rma']

# @measure_time
def insert_atr(timeframe_ohlcv: pt.DataFrame[OHLCV], mode: str = 'pandas_ta') -> pd.DataFrame:
    if len(timeframe_ohlcv) <= app_config.atr_timeperiod:
        timeframe_ohlcv['atr'] = pd.NA
    else:
        if mode == 'pandas_ta':
            timeframe_ohlcv['atr'] = timeframe_ohlcv.ta.atr(timeperiod=app_config.atr_timeperiod,
                                                            # high='high',
                                                            # low='low',
                                                            # close='close',
                                                            # mamode='ema',
                                                            )
        else:
            raise Exception(f"Unsupported mode:{mode}")
    insert_volume_rma(timeframe_ohlcv)
    return timeframe_ohlcv


def generate_multi_timeframe_ohlcva(date_range_str: str = None, file_path: str = None) -> None:
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    if file_path is None:
        file_path = symbol_data_path()
    start, end = date_range(date_range_str)

    # Split the date range into individual days
    current_day = start
    daily_dataframes = []

    while current_day.date() <= end.date():
        # For each day, get the data and append to daily_dataframes list
        daily_dataframes.append(read_daily_multi_timeframe_ohlcva(current_day))
        current_day += timedelta(days=1)

    # Concatenate the daily data
    df = pd.concat(daily_dataframes)
    df = df.sort_index(level='date')
    df = trim_to_date_range(date_range_str, df)
    # assert not df.index.duplicated().any()
    multi_timeframe_times_tester(df, date_range_str)
    df.to_csv(os.path.join(file_path, f'multi_timeframe_ohlcva.{date_range_str}.zip'),
              compression='zip')


def read_multi_timeframe_ohlcva(date_range_str: str = None) -> pt.DataFrame[MultiTimeframeOHLCVA]:
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    result = read_file(date_range_str, 'multi_timeframe_ohlcva', generate_multi_timeframe_ohlcva,
                       MultiTimeframeOHLCVA)
    cache_times(result)
    return result


# @measure_time
def core_generate_multi_timeframe_ohlcva(date_range_str: str = None, file_path: str = None) -> None:
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    if file_path is None:
        file_path = symbol_data_path()
    multi_timeframe_ohlcva = empty_df(MultiTimeframeOHLCVA)
    for _, timeframe in enumerate(app_config.timeframes):
        if timeframe == '4h':
            pass
        expanded_date_range = \
            expand_date_range(date_range_str,
                              time_delta=((app_config.atr_timeperiod + 2) * pd.to_timedelta(timeframe) *
                                          app_config.atr_safe_start_expand_multipliers),
                              mode='start')
        expanded_date_multi_timeframe_ohlcv = read_multi_timeframe_ohlcv(expanded_date_range)
        timeframe_ohlcv = single_timeframe(expanded_date_multi_timeframe_ohlcv, timeframe)
        timeframe_ohlcva = insert_atr(timeframe_ohlcv)
        timeframe_ohlcva = timeframe_ohlcva.dropna(subset=['atr']).copy()
        timeframe_ohlcva['timeframe'] = timeframe
        timeframe_ohlcva = timeframe_ohlcva.set_index('timeframe', append=True)
        timeframe_ohlcva = timeframe_ohlcva.swaplevel()
        multi_timeframe_ohlcva = concat(multi_timeframe_ohlcva, timeframe_ohlcva)
    multi_timeframe_ohlcva = multi_timeframe_ohlcva.sort_index(level='date')
    multi_timeframe_ohlcva = trim_to_date_range(date_range_str, multi_timeframe_ohlcva)
    assert multi_timeframe_times_tester(multi_timeframe_ohlcva, date_range_str)
    # plot_multi_timeframe_ohlcva(multi_timeframe_ohlcva)
    multi_timeframe_ohlcva.to_csv(os.path.join(file_path, f'multi_timeframe_ohlcva.{date_range_str}.zip'),
                                  compression='zip')


def core_read_multi_timeframe_ohlcva(date_range_str: str = None) \
        -> pt.DataFrame[MultiTimeframeOHLCVA]:
    result = read_file(date_range_str, 'multi_timeframe_ohlcva', core_generate_multi_timeframe_ohlcva,
                       MultiTimeframeOHLCVA)
    cache_times(result)
    return result


def read_daily_multi_timeframe_ohlcva(day: datetime, timezone='GMT') -> pt.DataFrame[MultiTimeframeOHLCVA]:
    # Format the date_range_str for the given day
    start_str = day.strftime('%y-%m-%d.00-00')
    end_str = day.strftime('%y-%m-%d.23-59')
    day_date_range_str = f'{start_str}T{end_str}'

    if day.replace(hour=0, minute=0, second=0, microsecond=0) > datetime.now(tz=pytz.UTC):
        return empty_df(MultiTimeframeOHLCVA)
    # Fetch the data for the given day using the old function
    return core_read_multi_timeframe_ohlcva(day_date_range_str)
