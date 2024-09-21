import pandas as pd

from Model.Data.atr import read_multi_timeframe_ohlcva
from helper.helper import log_d, log_w
from Model.Data.ohlcv import read_multi_timeframe_ohlcv
from helper.helper import date_range_to_string

log_d('Start')
ohlcva = read_multi_timeframe_ohlcva(date_range_to_string(start=pd.to_datetime('02-07-24'),
                                                     end=pd.to_datetime('08-07-24')))





nop = 1
