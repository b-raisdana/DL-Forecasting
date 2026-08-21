from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd
import pandera
from pandera import typing as pt

from PanderaDFM.MultiTimeframe import MultiTimeframe


class BullBearSide(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]  # start
    bull_bear_side: pt.Series[str]
    end: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    internal_high: pt.Series[float]
    high_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    internal_low: pt.Series[float]
    low_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    movement: pt.Series[float]
    # strength: pt.Series[float]
    atr: pt.Series[float]
    # duration: pt.Series[timedelta]
    movement_start_value: pt.Series[float] = pandera.Field(nullable=True)
    movement_end_value: pt.Series[float] = pandera.Field(nullable=True)
    movement_start_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    movement_end_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)

    @classmethod
    def repr(cls, _start: datetime, _trend):
        return bull_bear_side_repr(_start, _trend)


def bull_bear_side_repr(_start: datetime, _trend):
    text = f'{_trend["bull_bear_side"].replace("_TREND", "")} ' \
           f'{_start.strftime("%H:%M")}-{_trend["end"].strftime("%H:%M")}:'
    if hasattr(_trend, "movement"):
        text += f'M:{_trend["movement"]:.2f}'
    if hasattr(_trend, "duration"):
        text += f'D:{_trend["duration"] / timedelta(hours=1):.2f}h'
    if hasattr(_trend, "strength"):
        text += f'S:{_trend["strength"]:.2f}'
    if hasattr(_trend, "atr"):
        text += f'atr:{_trend["atr"]:.2f}'
    return text


class MultiTimeframeBullBearSide(BullBearSide, MultiTimeframe):
    pass
