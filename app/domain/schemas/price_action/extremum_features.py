import pandera
from pandera import typing as pt


class BranchExtremumOHLC(pandera.DataFrameModel):
    class Config:
        coerce = True

    open: pt.Series[float]
    high: pt.Series[float]
    low: pt.Series[float]
    close: pt.Series[float]
    volume: pt.Series[float]
    atr: pt.Series[float] = pandera.Field(nullable=True)
