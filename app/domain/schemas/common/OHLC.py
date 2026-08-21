from typing import Annotated  # python 3.9+

import pandas as pd
import pandera
from domain.schemas.common.MultiTimeframe import MultiTimeframe
from pandera import typing as pt


class OHLC(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    open: pt.Series[float]
    close: pt.Series[float]
    high: pt.Series[float]
    low: pt.Series[float]


class MultiTimeframeOHLC(OHLC, MultiTimeframe):
    pass
