from typing import Annotated

import pandas as pd
from pandera import typing as pt

from PanderaDFM.MultiTimeframe import MultiTimeframe
from helper.importer import pa


class RollingMeanStdOHLCV(pa.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]

    t_open: pt.Series[float]
    t_close: pt.Series[float]
    t_high: pt.Series[float]
    t_low: pt.Series[float]
    t_volume: pt.Series[float]

    mean_open: pt.Series[float]
    mean_close: pt.Series[float]
    mean_high: pt.Series[float]
    mean_low: pt.Series[float]
    mean_volume: pt.Series[float]

    std_open: pt.Series[float]
    std_close: pt.Series[float]
    std_high: pt.Series[float]
    std_low: pt.Series[float]
    std_volume: pt.Series[float]

    pre_open: pt.Series[float]
    pre_close: pt.Series[float]
    pre_high: pt.Series[float]
    pre_low: pt.Series[float]
    pre_volume: pt.Series[float]

class MtRollingMeanStdOHLCV(RollingMeanStdOHLCV, MultiTimeframe):
    pass