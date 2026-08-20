import pandas as pd
import pandas_ta as ta

ATR_LENGTH = 256

__relative_candle_columns = [
    "rel_close",
    "rel_high_close",
    "rel_close_low",
    "open_gap",
    "rel_candle_height",
]


def relative_candle_columns() -> list[str]:
    return __relative_candle_columns


