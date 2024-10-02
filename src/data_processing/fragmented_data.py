import os

import pandas as pd

from Config import config


def load_ohlcv_list():
    if 'ohlcv_list' not in globals() and os.path.exists(os.path.join(config.path_of_data, 'ohlcva_summary.zip')):
        ohlcv_list = pd.read_csv(os.path.join(config.path_of_data, 'ohlcva_summary.zip'), compression='zip')


def data_path(path_of_data:str = None, exchange:str = None,  market:str = None,  trading_pair:str = None, )->str:
    if path_of_data is None:
        path_of_data = config.path_of_data
    if exchange is None:
        path_of_data = config.exchange
    if market is None:
        path_of_data = config.market
    if trading_pair is None:
        path_of_data = config.trading_pair
    return os.path.join(path_of_data, exchange, market, trading_pair)
