from datetime import datetime, timedelta

import ccxt
import pandas as pd
from config import app_config
from helper.data_preparation import map_symbol
from helper.functions import date_range
from helper.logging import profile_it
from helper.logging.do_log import log_i

"""
Broker-agnostic ccxt fetch engine — the pandas<->ccxt timeframe/symbol maps shared by every
per-broker package under market_data_fetch/ (kucoin/, binance/, ...). Kucoin and Binance both
speak ccxt's unified BASE/QUOTE symbol format, so only the exchange class differs per broker;
that's SUPPORTED_BROKERS' whole job. Pagination and retry are delegated to ccxt's own
`paginate`/`fetch_paginated_call_deterministic` machinery (params={"paginate": True, ...})
instead of a hand-rolled batching loop, so every broker gets its native per-endpoint page size,
retry count, and rate limiting for free. Each per-broker fetch_ohlcv.py is a thin wrapper binding
its own broker id here.
"""

SUPPORTED_BROKERS: dict[str, type] = {
    "kucoin": ccxt.kucoin,
    "binance": ccxt.binance,
}

ccxt_symbol_map = {
    "BTCUSDT": "BTC/USDT",
    "ETHUSDT": "ETH/USDT",
    "BNBUSDT": "BNB/USDT",
    "EOSUSDT": "EOS/USDT",
    "TRXUSDT": "TRX/USDT",
    "TONUSDT": "TON/USDT",
    "SOLUSDT": "SOL/USDT",
}

# Dictionary mapping pandas timeframes to CCXT abbreviations
pandas_to_ccxt_timeframes = {
    "1sec": "1s",
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1M",  # Note: This is the CCXT abbreviation for 1 month, but it's not precise for trading.
}


def map_to_ccxt_symbol(symbol: str) -> str:
    return map_symbol(symbol, ccxt_symbol_map)


def broker_exchange(broker: str) -> ccxt.Exchange:
    try:
        exchange_class = SUPPORTED_BROKERS[broker.lower()]
    except KeyError:
        raise ValueError(f"Unsupported broker {broker!r}; supported brokers: {sorted(SUPPORTED_BROKERS)}") from None
    return exchange_class()


def fetch_ohlcv_by_range(
    broker: str,
    date_range_str: str = None,
    symbol: str = None,
    base_timeframe=None,
    limit_to_under_process_period: bool = None,
) -> list[object]:
    if limit_to_under_process_period is None:
        limit_to_under_process_period = app_config.limit_to_under_process_period
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    if symbol is None:
        symbol = map_to_ccxt_symbol(app_config.under_process_symbol)
    if base_timeframe is None:
        base_timeframe = app_config.timeframes[0]
    start, end = date_range(date_range_str)

    if limit_to_under_process_period:
        _, under_process_period_end = date_range(app_config.processing_date_range)
        if start > under_process_period_end:
            return []
    duration = end - start + pd.to_timedelta(app_config.timeframes[0])
    limit = int(duration / pd.to_timedelta(base_timeframe))

    response = fetch_ohlcv(
        broker,
        symbol,
        timeframe=base_timeframe,
        start=start,
        number_of_ticks=limit,
        params={"timeframe": base_timeframe},
    )
    return response


@profile_it
def fetch_ohlcv(
    broker: str, symbol, timeframe: str = None, start: datetime = None, number_of_ticks=None, params=None
) -> list[object]:
    if params is None:
        params = {}
    if start is None or start.tzinfo is None or start.utcoffset() != timedelta(0):
        raise ValueError("start must be a timezone-aware UTC datetime")
    if not number_of_ticks:
        return []
    exchange = broker_exchange(broker)
    if timeframe is None:
        timeframe = app_config.timeframes[0]

    # Convert pandas timeframe to CCXT timeframe
    ccxt_timeframe = pandas_to_ccxt_timeframes[timeframe]
    since = int(start.timestamp() * 1000)
    until = since + number_of_ticks * int(pd.to_timedelta(timeframe).total_seconds() * 1000)
    # ceil(number_of_ticks / 1000) + 1: a safe upper bound on the calls ccxt's own pagination
    # needs, regardless of the broker's actual per-request page size (1000-1500 candles).
    pagination_calls = -(-number_of_ticks // 1000) + 1

    # Driving fetch_paginated_call_deterministic directly (rather than via the params={"paginate":
    # True} convenience flag on exchange.fetch_ohlcv) sidesteps a ccxt bug: kucoin's fetch_ohlcv
    # routes through internal helpers (fetch_spot_ohlcv/fetch_contract_ohlcv/fetch_utaohlcv) whose
    # method name isn't the literal "fetchOHLCV" string that fetch_paginated_call_deterministic's
    # result filter checks for, so the convenience flag silently returns an empty list on kucoin.
    # Calling it with the literal method name (aliased by ccxt to fetch_ohlcv on every broker)
    # keeps ccxt's own pagination, retry, and rate-limit handling while avoiding that bug.
    response = exchange.fetch_paginated_call_deterministic(
        "fetchOHLCV",
        symbol,
        since,
        number_of_ticks,
        ccxt_timeframe,
        {
            "paginationCalls": pagination_calls,
            "maxRetries": 20,
            **params,
            "until": until,
        },
        None,
    )
    log_i(f"fetch_ohlcv@{broker}@{start}#{number_of_ticks}>{len(response)}")
    return response
