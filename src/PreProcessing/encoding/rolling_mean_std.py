import os
from datetime import timedelta

import pandas as pd

from Config import config
from PanderaDFM.OHLCV import MultiTimeframeOHLCV
from PanderaDFM.RollingMeanStdOHLCV import RollingMeanStdOHLCV
from data_processing.fragmented_data import data_path
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.data_preparation import expand_date_range, read_file, trim_to_date_range
from helper.helper import date_range
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


# def read_multi_timeframe_rolling_mean_std_ohlcv(mt_ohlcv: pt.DataFrame[MultiTimeframeOHLCV]):
def generate_multi_timeframe_rolling_mean_std_ohlcv(date_range_str: str, file_path: str = None) -> None:
    if file_path is None:
        file_path = data_path()
    expander_duration = sum([pd.to_timedelta(tf) * lenght for tf, lenght in timeframe_normalization_length.items()], timedelta())
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
            t[f't_{col}'] = (t[col] - t[f'mean_{col}'] )/ t[f'std_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'pre_{col}'] = t[f'pre_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'mean_{col}'] = t[f'mean_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f'std_{col}'] = t[f'std_{col}']
            trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], f't_{col}'] = t[f't_{col}']
    trans_mt_ohlcv = trim_to_date_range(date_range_str, trans_mt_ohlcv)
    if trans_mt_ohlcv.isna().any().any():
        raise AssertionError('trans_mt_ohlcv.isna().any().any()')
    trans_mt_ohlcv.to_csv(os.path.join(file_path, f'multi_timeframe_ohlcv.{date_range_str}.zip'),
              compression='zip')

def read_multi_timeframe_rolling_mean_std_ohlcv(date_range_str: str) -> pt.DataFrame[RollingMeanStdOHLCV]:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = read_file(date_range_str, 'rolling_mean_std_multi_timeframe_ohlcv',
                       generate_multi_timeframe_rolling_mean_std_ohlcv,
                       RollingMeanStdOHLCV)
    return result

def regenerate_original_mt_ohlcv(trans_mt_ohlcv: pd.DataFrame, mt_ohlcv: pd.DataFrame):
    original_mt_ohlcv = trans_mt_ohlcv.copy()
    for timeframe in trans_mt_ohlcv.index.get_level_values(level='timeframe').unique():
        for col in columns_list:
            t_col = trans_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], [col]]
            ema = ta.ema(mt_ohlcv.loc[pd.IndexSlice[timeframe, :], col],
                         length=timeframe_normalization_length[timeframe])
            rolling_std = mt_ohlcv.loc[pd.IndexSlice[timeframe, :], col].rolling(
                window=timeframe_normalization_length[timeframe]).std()
            t_col[col] = t_col[col] * rolling_std
            original_mt_ohlcv.loc[pd.IndexSlice[timeframe, :], col] = t_col[col] + ema
    return original_mt_ohlcv
