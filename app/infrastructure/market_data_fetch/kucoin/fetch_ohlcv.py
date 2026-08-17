from datetime import datetime

from infrastructure.market_data_fetch import ccxt_client
from infrastructure.market_data_fetch.ccxt_client import ccxt_symbol_map, map_to_ccxt_symbol, pandas_to_ccxt_timeframes

BROKER = "kucoin"

__all__ = [
    "BROKER",
    "ccxt_symbol_map",
    "map_to_ccxt_symbol",
    "pandas_to_ccxt_timeframes",
    "fetch_ohlcv_by_range",
    "fetch_ohlcv",
]


def fetch_ohlcv_by_range(
    date_range_str: str = None, symbol: str = None, base_timeframe=None, limit_to_under_process_period: bool = None
) -> list[object]:
    return ccxt_client.fetch_ohlcv_by_range(
        BROKER, date_range_str, symbol, base_timeframe, limit_to_under_process_period
    )


def fetch_ohlcv(
    symbol, timeframe: str = None, start: datetime = None, number_of_ticks=None, params=None
) -> list[object]:
    return ccxt_client.fetch_ohlcv(
        BROKER, symbol, timeframe=timeframe, start=start, number_of_ticks=number_of_ticks, params=params
    )
