import os

import pandas as pd

from Config import config


def load_ohlcv_list():
    if 'ohlcv_list' not in globals() and os.path.exists(os.path.join(config.path_of_data, 'ohlcva_summary.zip')):
        ohlcv_list = pd.read_csv(os.path.join(config.path_of_data, 'ohlcva_summary.zip'), compression='zip')
