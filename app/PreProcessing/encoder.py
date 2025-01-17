import pandas as pd

from app.Config import app_config
from app.data_processing.atr import read_multi_timeframe_ohlcva
from app.helper.helper import log_d
from app.helper.helper import date_range_to_string

log_d('Start')
app_config.processing_date_range = date_range_to_string(start=pd.to_datetime('07-01-23'),
                                                        end=pd.to_datetime('09-01-24'))
ohlcva = read_multi_timeframe_ohlcva(app_config.processing_date_range)





nop = 1
