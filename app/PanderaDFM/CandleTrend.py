from typing import Annotated

import pandas as pd
import pandera
from pandera import typing as pt

from PanderaDFM.MultiTimeframe import MultiTimeframe


class CandleTrend(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    open: pt.Series[float]
    close: pt.Series[float]
    low: pt.Series[float]
    high: pt.Series[float]
    volume: pt.Series[float]
    previous_peak_index: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    previous_peak_value: pt.Series[float] = pandera.Field(nullable=True)
    next_peak_index: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    next_peak_value: pt.Series[float] = pandera.Field(nullable=True)
    previous_valley_index: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    previous_valley_value: pt.Series[float] = pandera.Field(nullable=True)
    next_valley_index: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    next_valley_value: pt.Series[float] = pandera.Field(nullable=True)
    bull_bear_side: pt.Series[str]
    is_final: pt.Series[bool]


class MultiTimeframeCandleTrend(CandleTrend, MultiTimeframe):
    pass
