import pandera
from domain.schemas.common.MultiTimeframe import MultiTimeframe
from domain.schemas.common.OHLCV import OHLCV
from pandera import typing as pt


class OHLCVA(OHLCV):
    atr: pt.Series[float] = pandera.Field(nullable=True)
    volume_rma: pt.Series[float] = pandera.Field(nullable=True)


class MultiTimeframeOHLCVA(OHLCVA, MultiTimeframe):
    pass
