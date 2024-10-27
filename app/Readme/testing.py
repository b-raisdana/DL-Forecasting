import pandas as pd

from app.data_processing.atr import read_multi_timeframe_ohlcva, insert_mt_volume_rma
from app.helper.helper import date_range_to_string

mt_ohlcva = read_multi_timeframe_ohlcva(
    date_range_to_string(start=pd.to_datetime('02-07-24'), end=pd.to_datetime('08-07-24')))
insert_mt_volume_rma(mt_ohlcva)
