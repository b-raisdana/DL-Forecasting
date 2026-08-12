import pandas as pd
import pandas_ta as ta

ATR_LENGTH = 256

__relative_candle_columns = [
    'rel_close', 'rel_high_close', 'rel_close_low', 'gap', 'rel_candle_height',
]


def relative_candle_columns():
    return __relative_candle_columns


def add_relative_candle_columns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """relative-HLC block per docs/input-features.md § candle feature schema.

    close/ATR, (high-close)/ATR, (close-low)/ATR, gap=(open-prev_close)/ATR, candle_height/ATR.
    Absolute close is already present as the raw 'close' column, so it isn't duplicated here.
    """
    if 'atr' not in ohlc.columns:
        ohlc['atr'] = ta.atr(high=ohlc['high'], low=ohlc['low'], close=ohlc['close'], length=ATR_LENGTH)
    ohlc['rel_close'] = ohlc['close'] / ohlc['atr']
    ohlc['rel_high_close'] = (ohlc['high'] - ohlc['close']) / ohlc['atr']
    ohlc['rel_close_low'] = (ohlc['close'] - ohlc['low']) / ohlc['atr']
    ohlc['gap'] = (ohlc['open'] - ohlc['close'].shift(1)) / ohlc['atr']
    ohlc['rel_candle_height'] = (ohlc['high'] - ohlc['low']) / ohlc['atr']
    return ohlc
