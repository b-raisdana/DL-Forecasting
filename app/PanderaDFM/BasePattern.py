from datetime import datetime
from typing import Annotated

import numpy as np
import pandas as pd
import pandera
from pandera import typing as pt

from app.PanderaDFM.MultiTimeframe import MultiTimeframe


class BasePattern(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(
        check_name=True)  # the last candle in the sequence
    # start: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(check_name=True) # the first candle in which the pattern is valid
    end: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    ttl: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    atr: pt.Series[float]
    zero_trigger_candle: pt.Series[bool]
    a_pattern_atr: pt.Series[float]
    a_trigger_atr: pt.Series[float]
    internal_high: pt.Series[float]
    internal_low: pt.Series[float]
    upper_band_activated: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    below_band_activated: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    size: pt.Series[float] = pandera.Field(nullable=True, default=np.NAN)
    base_timeframe_atr: pt.Series[float] = pandera.Field(nullable=True, default=np.NAN)
    ignore_backtesting: pt.Series[bool] = pandera.Field(nullable=True, default=np.NAN)


class MultiTimeframeBasePattern(BasePattern, MultiTimeframe):

    @classmethod
    def repr(cls, _start: datetime, timeframe: str, pattern):
        effective_end = '/'.join([
            pattern["end"].strftime("%H:%M") if pd.notna(pattern["end"]) else "",
            pattern["ttl"].strftime("%H:%M") if pd.notna(pattern["ttl"]) else ""
        ])
        text = f'BASE{timeframe}/' \
               f'{_start.strftime("%H:%M")}-{effective_end} ' \
               f'{pattern["internal_low"]}-{pattern["internal_high"]}' \
               f'={pattern["internal_high"] - pattern["internal_low"]:.1f}' \
               f' atr={pattern["atr"]:.1f}'
        if hasattr(pattern, 'upper_band_activated') and pd.notna(pattern['upper_band_activated']):
            text += f"U@{pattern['upper_band_activated'].strftime('%H:%M')}"
        if hasattr(pattern, 'below_band_activated') and pd.notna(pattern['below_band_activated']):
            text += f"B@{pattern['below_band_activated'].strftime('%H:%M')}"
        return text

    @classmethod
    def str(cls, _start: datetime, timeframe: str, pattern):
        # effective_end = ''.join([
        #     f'E{pattern["end"].strftime("%y%m/%d-%H:%M")}'  if pd.notna(pattern["end"]) else "",
        #     f'T{pattern["ttl"].strftime("%y%m/%d-%H:%M")}'  if pd.notna(pattern["ttl"]) else ""
        # ])
        text = f'BASE{timeframe}@' \
               f'{_start.strftime("%y/%m/%d-%H:%M")}'  # ' {effective_end} '
        return text
