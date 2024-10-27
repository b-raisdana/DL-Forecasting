from typing import Annotated  # python 3.9+

import pandas as pd
import pandera
from pandera import typing as pt

from app.PanderaDFM.MultiTimeframe import MultiTimeframe


class OHLCV(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    open: pt.Series[float]
    close: pt.Series[float]
    high: pt.Series[float]
    low: pt.Series[float]
    volume: pt.Series[float]


class MultiTimeframeOHLCV(OHLCV, MultiTimeframe):
    pass
