import pandera
from pandera import typing as pt


class SummaryOHLCV(pandera.DataFrameModel):
    end: pt.Series[str]
    start: pt.Series[str]
    low: pt.Series[float]
    high: pt.Series[float]
    rows: pt.Series[int]
    hash: pt.Series[str]
    zip_hash: pt.Series[str]


class SummaryOHLCA(SummaryOHLCV):
    atr: pt.Series[float]
