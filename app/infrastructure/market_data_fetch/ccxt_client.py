from datetime import UTC, datetime, timedelta

import ccxt
import pandas as pd
from ccxt import NetworkError, RequestTimeout
from config import app_config
from helper.data_preparation import map_symbol
from helper.functions import date_range
from helper.logging import profile_it
from helper.logging.do_log import log_e, log_i

"""
Broker-agnostic ccxt fetch engine — pagination/retry and the pandas<->ccxt timeframe/symbol maps,
shared by every per-broker package under market_data_fetch/ (kucoin/, binance/, ...). Kucoin and
Binance both speak ccxt's unified BASE/QUOTE symbol format, so only the exchange class differs
per broker; that's SUPPORTED_BROKERS' whole job. Each per-broker fetch_ohlcv.py is a thin wrapper
binding its own broker id here rather than re-implementing fetch/retry logic.
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
    exchange = broker_exchange(broker)
    if timeframe is None:
        timeframe = app_config.timeframes[0]

    # Convert pandas timeframe to CCXT timeframe
    ccxt_timeframe = pandas_to_ccxt_timeframes[timeframe]
    output_list = []
    width_of_timeframe = pd.to_timedelta(timeframe).total_seconds()
    max_query_size = 1000
    for batch_start in range(0, number_of_ticks, max_query_size):
        if start < datetime.now(UTC):
            start_timestamp = int((start.timestamp() + batch_start * width_of_timeframe) * 1000)
            this_query_size = min(number_of_ticks - batch_start, max_query_size)
            last_error = None
            for _ in range(20):
                try:
                    response = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=ccxt_timeframe,
                        since=start_timestamp,
                        limit=min(number_of_ticks - batch_start, this_query_size),
                        params=params,
                    )
                    break
                except RequestTimeout as e:
                    log_e("ccxt.RequestTimeout:" + str(e))
                    last_error = e
                except NetworkError as e:
                    log_e("ccxt.NetworkError:" + str(e))
                    last_error = e
            else:
                raise last_error
            log_i(
                "fetch_ohlcv@"
                f"{broker}@{datetime.fromtimestamp(start_timestamp / 1000)}#{this_query_size}>{len(response)}",
            )
            output_list.extend(response)

    return output_list
