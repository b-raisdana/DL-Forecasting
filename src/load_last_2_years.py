from datetime import timedelta

import pandas as pd

from Config import config
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.helper import date_range_to_string

for end in pd.date_range(pd.to_datetime('08-01-22'), pd.to_datetime('09-01-24'), freq=timedelta(days=30)):
    config.processing_date_range = date_range_to_string(start=end - timedelta(days=60),
                                                        end=end)
    config.under_process_symbol = 'SOLUSDT'
    read_multi_timeframe_ohlcv(config.processing_date_range)