import pandas as pd
from helper.helper import log_d, log_w
from Model.Data.ohlcv import read_multi_timeframe_ohlcv
from helper.helper import date_range_to_string

log_d('Start')
ohlcv = read_multi_timeframe_ohlcv(date_range_to_string(start=pd.to_datetime('02-07-24'),
                                                     end=pd.to_datetime('08-07-24')))
for timeframe in ohlcv
shifted_ohlcv = df.shift(1)






nop = 1
