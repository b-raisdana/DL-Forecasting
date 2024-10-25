import os

import pandas as pd

from Config import config, Config


def load_ohlcv_list():
    if 'ohlcv_list' not in globals() and os.path.exists(os.path.join(symbol_data_path(), 'ohlcva_summary.zip')):
        ohlcv_list = pd.read_csv(os.path.join(symbol_data_path(), 'ohlcva_summary.zip'), compression='zip')


def symbol_data_path(path_of_data:str = None, exchange:str = None, market:str = None, trading_pair:str = None, )->str:
    if path_of_data is None:
        path_of_data = Config.path_of_data
    if exchange is None:
        exchange = config.under_process_exchange
    if market is None:
        market = config.under_process_market
    if trading_pair is None:
        trading_pair = config.under_process_symbol
    return os.path.join(path_of_data, exchange, market, trading_pair)

