import os.path

import pandas as pd

from PanderaDFM.OHLCV import MultiTimeframeOHLCV
from helper.data_preparation import single_timeframe
from helper.helper import date_range_to_string
from Config import config
from tensor

config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
                                                    end=pd.to_datetime('09-01-24'))
# devided by rolling mean, std
n_mt_ohlcv = pd.read_csv(
    os.path.join(r"C:\Code\dl-forcasting\data\Kucoin\Spot\BTCUSDT",
                 f"n_mt_ohlcv.{config.processing_date_range}.csv.zip"), compression='zip')
n_mt_ohlcv

