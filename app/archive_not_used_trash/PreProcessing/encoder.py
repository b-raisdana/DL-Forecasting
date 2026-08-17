import pandas as pd
from config import app_config
from helper.functions import date_range_to_string, log_d
from infrastructure.ohlcv.atr import read_multi_timeframe_ohlcva

log_d("Start")
app_config.processing_date_range = date_range_to_string(
    start=pd.to_datetime("07-01-23"), end=pd.to_datetime("09-01-24")
)
ohlcva = read_multi_timeframe_ohlcva(app_config.processing_date_range)


nop = 1
