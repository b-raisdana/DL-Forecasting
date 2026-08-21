from domain.schemas.common.MultiTimeframe import MultiTimeframe
from domain.schemas.common.OHLC import OHLC
from pandera import typing as pt


class OHLCV(OHLC):
    volume: pt.Series[float]


class MultiTimeframeOHLCV(OHLCV, MultiTimeframe):
    pass
