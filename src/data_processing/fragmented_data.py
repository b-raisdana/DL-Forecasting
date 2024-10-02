import os

import pandas as pd

from Config import config


def load_ohlcv_list():
    if 'ohlcv_list' not in globals() and os.path.exists(os.path.join(data_path(), 'ohlcva_summary.zip')):
        ohlcv_list = pd.read_csv(os.path.join(data_path(), 'ohlcva_summary.zip'), compression='zip')


def data_path(file_path:str = None, exchange:str = None, market:str = None, trading_pair:str = None, )->str:
    if file_path is None:
        file_path = data_path()
    if exchange is None:
        exchange = config.exchange
    if market is None:
        market = config.market
    if trading_pair is None:
        trading_pair = config.trading_pair
    return os.path.join(file_path, exchange, market, trading_pair)
