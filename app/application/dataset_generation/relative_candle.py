from __future__ import annotations

import pandas_ta as ta
from domain.schemas.common.OHLCV import OHLCV
from helper.pandera import pandera_transform
from pandera import typing as pt

ATR_LENGTH = 256

__relative_candle_columns = [
    "rel_close",
    "rel_high_close",
    "rel_close_low",
    "open_gap",
    "rel_candle_height",
]


# def relative_candle_columns() -> list[str]:
# return __relative_candle_columns


@pandera_transform
def add_relative_candle_columns(ohlc: pt.DataFrame[OHLCV]) -> pt.DataFrame[OHLCV]:
    """relative-HLC block per docs/input-features.md § candle feature schema.

    close/ATR, (high-close)/ATR, (close-low)/ATR, open_gap=(open-prev_close)/ATR, candle_height/ATR.
    Absolute close is already present as the raw 'close' column, so it isn't duplicated here.
    """
    if "atr" not in ohlc.columns:
        ohlc["atr"] = ta.atr(high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], length=ATR_LENGTH)
    ohlc["rel_close"] = ohlc["close"] / ohlc["atr"]
    ohlc["rel_high_close"] = (ohlc["high"] - ohlc["close"]) / ohlc["atr"]
    ohlc["rel_close_low"] = (ohlc["close"] - ohlc["low"]) / ohlc["atr"]
    ohlc["open_gap"] = (ohlc["open"] - ohlc["close"].shift(1)) / ohlc["atr"]
    ohlc["rel_candle_height"] = (ohlc["high"] - ohlc["low"]) / ohlc["atr"]
    return ohlc
