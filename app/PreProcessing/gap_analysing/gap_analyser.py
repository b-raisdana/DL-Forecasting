from typing import Literal

import pandas as pd

from app.PanderaDFM.OHLCV import OHLCV
from app.helper.importer import pt


def single_timeframe_find_gap(ohlcv: pt.DataFrame[OHLCV], timeframe,
                              mode: Literal['count', 'list', 'boolean'] = 'boolean'):
    ohlcv_sorted = ohlcv.sort_index()

    # Generate a complete date range based on the min and max dates and the provided frequency
    full_date_range = pd.date_range(start=ohlcv_sorted.index.min(),
                                    end=ohlcv_sorted.index.max(),
                                    freq=timeframe,
                                    tz=ohlcv_sorted.index.tz)

    # Identify the missing dates (gaps)
    missing_dates = full_date_range.difference(ohlcv_sorted.index)
    if mode == 'count':
        return len(list(missing_dates))
    elif mode == 'list':
        return missing_dates
    elif mode == 'boolean':
        return len(list(missing_dates)) > 0
    else:
        raise ValueError(f"mode={mode} not supported!")
