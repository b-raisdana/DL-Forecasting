# from __future__ import annotations
from datetime import datetime
from typing import Annotated, Tuple

import backtrader as bt
import pandas as pd
import pandera
import pytz
from pandas import Timestamp
from pandera import typing as pt

from PanderaDFM.ExtendedDf import ExtendedDf, BaseDFM
from Model.Order import OrderSide
from helper.helper import log_w


class SignalDFM(BaseDFM):
    # from start of exact this candle the signal is valid
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]  # = pandera.Field(title='date')
    ref_date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    ref_timeframe: pt.Index[str]
    side: pt.Index[str]

    original_index: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    # from start of exact this candle the signal is in-valid
    end: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    base_asset_amount: pt.Series[float] = pandera.Field(nullable=True, default=None)
    # the worst acceptable price for order execution.
    limit_price: pt.Series[float] = pandera.Field(nullable=True, default=None)
    stop_loss: pt.Series[float] = pandera.Field(nullable=True, default=None)  # , ignore_na=False
    take_profit: pt.Series[float] = pandera.Field(nullable=True, default=None)
    # the condition direction is reverse of limit direction.
    trigger_price: pt.Series[float] = pandera.Field(nullable=True, default=None)
    # trigger_satisfied: pt.Series[bool] = pandera.Field(nullable=True, default=True)
    original_order_id: pt.Series[str] = pandera.Field(nullable=True, default=None)
    stop_loss_order_id: pt.Series[str] = pandera.Field(nullable=True, default=None)
    take_profit_order_id: pt.Series[str] = pandera.Field(nullable=True, default=None)
    order_is_active: pt.Series[bool] = pandera.Field(nullable=True, default=None)  # , ignore_na=False

    # todo: if the signal end changed we have to update signal orders
    # updated: pt.Series[bool] = pandera.Field(nullable=True, default=True)

    # class Config:
    #     # to resolve pandera.errors.SchemaError: column ['XXXX'] not in dataframe
    #     add_missing_columns = True
    #     # to resolve pandera.errors.SchemaError: expected series ['XXXX'/None] to have type datetime64[ns, UTC]
    #     # , got object
    #     coerce = True

    # @pandera.dataframe_check
    # def end_after_start_check(cls, df, *args, **kwargs):
    #     # Custom check to ensure 'end' is after 'date'
    #     return df['end'] > df['date']


class SignalDf(ExtendedDf):
    schema_data_frame_model = SignalDFM
    _sample_df = None
    _empty_obj = None

    @staticmethod
    def execution_type(signal):
        signal_dict = signal.to_dict()
        if pd.notna(signal_dict['trigger_price']) and pd.notna(signal_dict['limit_price']):
            execution_type = "StopLimit"  # bt.Order.StopLimit
        elif pd.notna(signal_dict['stop_loss']):
            execution_type = "Stop"
            log_w("Not tested")
        elif pd.notna(signal_dict['limit_price']):
            execution_type = "Limit"
            log_w("Not tested")
        elif (pd.isna(signal_dict['trigger_price']) and pd.isna(signal_dict['stop_loss'])
              and pd.isna(signal_dict['limit_price'])):
            execution_type = "Market"
            log_w("Not tested")
        else:
            execution_type = "Custom"
        return execution_type

    @classmethod
    def to_str(cls, signal_index: Tuple[Timestamp, Timestamp, str, str], signal: pt.Series[SignalDFM]):
        values = signal.to_dict()
        index_dict = dict(zip(SignalDFM.to_schema().index.names, list(signal_index)))
        side_indicator = f"{('Buy' if index_dict['side'] == OrderSide.Buy.value else 'Sell')}"
        result = (f"Signal{index_dict['date'].strftime('%m-%d.%H-%M')}{side_indicator}"
                  f"@{index_dict['ref_date'].strftime('%m-%d.%H-%M')}/{index_dict['ref_timeframe']}"
                  f"End{pd.to_datetime(values['end']).strftime('%m/%d.%H:%M')}"
                  f"{SignalDf.execution_type(signal)}"
                  f"{values['base_asset_amount']:.4f}@{values['limit_price']:.2f}SL{values['stop_loss']:.2f}"
                  f"TP{values['take_profit']:.2f}TR{values['trigger_price']:.2f}"
                  # f"Ref{ref_timeframe}/{ref_date}"
                  )
        return result


_sample_df = pd.DataFrame({
    'date': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'original_index': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'ref_date': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'ref_timeframe': ['1min'],

    'end': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=2).replace(tzinfo=pytz.UTC))],
    'side': ['buy'],

})
SignalDf._sample_df = _sample_df.set_index(['date', 'ref_date', 'ref_timeframe', 'side'])
