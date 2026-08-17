import pandas as pd
from config import app_config
from domain.schemas.common.OHLCV import OHLCV, MultiTimeframeOHLCV
from helper.data_preparation import concat, multi_timeframe_times_tester, trim_to_date_range
from pandera import typing as pt


def aggregate_multi_timeframe_ohlcv(ohlcv: pt.DataFrame[OHLCV], date_range_str: str) -> MultiTimeframeOHLCV:
    """
    Resamples a base-timeframe OHLCV DataFrame into every configured higher timeframe
    (`app_config.timeframes[1:]`) via calendar-aligned `pd.Grouper` aggregation. Pure transform, no
    I/O: the caller is responsible for actually fetching/generating the base-timeframe `ohlcv`.
    """
    multi_timeframe_ohlcv = ohlcv.copy()
    multi_timeframe_ohlcv.insert(0, "timeframe", app_config.timeframes[0])
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.set_index("timeframe", append=True)
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.swaplevel()
    for _, timeframe in enumerate(app_config.timeframes[1:]):
        if timeframe == "1W":
            frequency = "W-MON"
        elif timeframe == "M":
            frequency = "MS"
        else:
            frequency = timeframe
        _timeframe_ohlcv = ohlcv.groupby(pd.Grouper(freq=frequency)).agg(
            {
                "open": "first",
                "close": "last",
                "low": "min",
                "high": "max",
                "volume": "sum",
            }
        )
        # pd.Grouper materializes a row for every calendar bin in span, including ones with zero
        # underlying base-timeframe rows (real low-liquidity gaps) — forward-fill those as flat
        # zero-volume candles instead of leaving non-nullable OHLC columns NaN.
        _timeframe_ohlcv["close"] = _timeframe_ohlcv["close"].ffill()
        for price_column in ("open", "high", "low"):
            _timeframe_ohlcv[price_column] = _timeframe_ohlcv[price_column].fillna(_timeframe_ohlcv["close"])
        if len(_timeframe_ohlcv.index) > 0:
            _timeframe_ohlcv.insert(0, "timeframe", timeframe)
            _timeframe_ohlcv = _timeframe_ohlcv.set_index("timeframe", append=True)
            _timeframe_ohlcv = _timeframe_ohlcv.swaplevel()
            multi_timeframe_ohlcv = concat(multi_timeframe_ohlcv, _timeframe_ohlcv)
    multi_timeframe_ohlcv = trim_to_date_range(date_range_str, multi_timeframe_ohlcv)
    multi_timeframe_ohlcv = multi_timeframe_ohlcv.sort_index(level="date")
    assert multi_timeframe_times_tester(multi_timeframe_ohlcv, date_range_str)
    return multi_timeframe_ohlcv
