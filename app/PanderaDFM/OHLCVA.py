import pandera
from pandera import typing as pt

from app.PanderaDFM.OHLCV import OHLCV
from app.PanderaDFM.MultiTimeframe import MultiTimeframe


class OHLCVA(OHLCV):
    atr: pt.Series[float] = pandera.Field(nullable=True)
    volume_rma: pt.Series[float] = pandera.Field(nullable=True)


class MultiTimeframeOHLCVA(OHLCVA, MultiTimeframe):
    pass
