from datetime import datetime

import pandas as pd

from app.Config import config
from app.data_processing.atr import read_multi_timeframe_ohlcva
from app.helper.helper import date_range_to_string

if __name__ == "__main__":
    config.processing_date_range = date_range_to_string(start=pd.to_datetime('02-07-24'),
                                                        end=datetime.now())  # , end=datetime(year=2023, month=3, day=8))
    ohlcva = read_multi_timeframe_ohlcva(config.processing_date_range)

