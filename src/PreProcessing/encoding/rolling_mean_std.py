import os
from datetime import timedelta

import pandas as pd

from Config import config
from PanderaDFM.MtRollingMeanStdOHLCV import MtRollingMeanStdOHLCV
from data_processing.fragmented_data import data_path
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.data_preparation import expand_date_range, read_file, trim_to_date_range
from helper.helper import date_range_to_string
from helper.importer import pt, ta

columns_list = ['open', 'close', 'high', 'low', 'volume']
timeframe_normalization_length = {
    '1min': 256,
    '5min': 128,
    '15min': 128,
    '1h': 64,
    '4h': 32,
    '1D': 16,
    '1W': 4,
}

def reverse_rolling_mean_std(rolling_mean_std: pt.DataFrame[MtRollingMeanStdOHLCV]):
    # reconstructed_ohlcv = pd.DataFrame(index=rolling_mean_std.index)
    t_rolling_mean_std = rolling_mean_std.copy()
    for col in columns_list:
        t_rolling_mean_std.drop(col, axis='columns', inplace=True)
        t_rolling_mean_std.loc[:, col] = \
            t_rolling_mean_std[f'mean_{col}'] + t_rolling_mean_std[f'std_{col}'] * t_rolling_mean_std[f'n_{col}']
    return t_rolling_mean_std

def reverse_mt_rolling_mean_std(mt_rolling_mean_std):
    reconstructed_mt_ohlcv = pd.DataFrame(index=mt_rolling_mean_std.index)
    for timeframe in mt_rolling_mean_std.index.get_level_values(level='timeframe').unique():
        rolling_mean_std = mt_rolling_mean_std.loc[
            pd.IndexSlice[timeframe, :], :]
        for col in columns_list:
            reconstructed_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], col] = \
                rolling_mean_std[f'mean_{col}'] + rolling_mean_std[f'std_{col}'] * rolling_mean_std[f'n_{col}']
    return reconstructed_mt_ohlcv


def generate_multi_timeframe_rolling_mean_std_ohlcv(date_range_str: str, file_path: str = None) -> None:
    if file_path is None:
        file_path = data_path()
    expander_duration = sum(
        [pd.to_timedelta(tf) * (lenght + 1) for tf, lenght in timeframe_normalization_length.items()],
        timedelta())
    expanded_date_range = expand_date_range(date_range_str,
                                            time_delta=expander_duration,
                                            mode='start')
    mt_ohlcv = read_multi_timeframe_ohlcv(expanded_date_range)
    trans_mt_ohlcv = mt_ohlcv.copy()
    for timeframe in trans_mt_ohlcv.index.get_level_values(level='timeframe').unique():
        for col in columns_list:
            t = trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], [col]]
            t[f'pre_{col}'] = t[col].shift(1)
            t[f'mean_{col}'] = ta.ema(t[f'pre_{col}'], lenght=timeframe_normalization_length[timeframe])
            t[f'std_{col}'] = t[f'pre_{col}'].rolling(window=timeframe_normalization_length[timeframe]).std()
            t[f'n_{col}'] = (t[col] - t[f'mean_{col}']) / t[f'std_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'pre_{col}'] = t[f'pre_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'mean_{col}'] = t[f'mean_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'std_{col}'] = t[f'std_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'n_{col}'] = t[f'n_{col}']
    trans_mt_ohlcv = trim_to_date_range(date_range_str, trans_mt_ohlcv)
    if trans_mt_ohlcv.isna().any().any():
        raise AssertionError('trans_mt_ohlcv.isna().any().any()')
    trans_mt_ohlcv.to_csv(os.path.join(file_path, f'rolling_mean_std_multi_timeframe_ohlcv.{date_range_str}.zip'),
                          compression='zip')


def read_multi_timeframe_rolling_mean_std_ohlcv(date_range_str: str = None) -> MtRollingMeanStdOHLCV:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = read_file(date_range_str, 'rolling_mean_std_multi_timeframe_ohlcv',
                       generate_multi_timeframe_rolling_mean_std_ohlcv,
                       MtRollingMeanStdOHLCV)
    # result = MtRollingMeanStdOHLCV.validate(result)
    return result


# config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
#                                                     end=pd.to_datetime('09-01-24'))
# df = read_multi_timeframe_rolling_mean_std_ohlcv()
# print(df.describe())
