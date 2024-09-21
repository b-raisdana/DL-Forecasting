import pandas as pd
from pandas import Timedelta

from Config import config
from Model.Data.atr import read_multi_timeframe_ohlcva
from helper.data_preparation import expand_date_range
from helper.helper import log_d, log_w
from Model.Data.ohlcv import read_multi_timeframe_ohlcv
from helper.helper import date_range_to_string

log_d('Start')
config.processing_date_range = date_range_to_string(start=pd.to_datetime('07-01-23'),
                                                     end=pd.to_datetime('09-01-24'))
ohlcva = read_multi_timeframe_ohlcva(config.processing_date_range)





nop = 1
