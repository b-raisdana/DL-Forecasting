import pandas as pd
from helper.functions import date_range_to_string
from infrastructure.ohlcv.atr import insert_mt_volume_rma, read_multi_timeframe_ohlcva

mt_ohlcva = read_multi_timeframe_ohlcva(
    date_range_to_string(start=pd.to_datetime("02-07-24"), end=pd.to_datetime("08-07-24"))
)
insert_mt_volume_rma(mt_ohlcva)
