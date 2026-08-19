from typing import cast

from config import app_config
from domain.ohlcv.ohlcv import OHLCV_DATASET, get_base_timeframe_ohlcv
from domain.schemas.common.OHLCV import OHLCV
from helper.functions import date_range
from helper.logging.do_log import log_i
from helper.schema_casting import empty_df
from infrastructure.datastore_engine.disk_cache import DATASET_DB, cleanup_redundant_cache_files, write_data_file
from infrastructure.datastore_engine.disk_cache_gaps import find_cache_gaps
from infrastructure.market_data_fetch.ccxt_client import SUPPORTED_BROKERS, fetch_ohlcv_by_range
from pandera import typing as pt

"""
Orchestrates fetch-and-cache-to-disk runs for one (broker, trading_pair, date_range_str): binds the
CLI's explicit parameters onto app_config (the boundary get_base_timeframe_ohlcv()'s app_config-driven
fetch/cache chain expects — see project-decisions skill § code layers) and delegates to it. No new
cache path — get_base_timeframe_ohlcv() already is the OHLCV cache-or-generate entry point.
"""


def _bind_broker(broker: str, trading_pair: str, market: str) -> None:
    if broker.lower() not in SUPPORTED_BROKERS:
        raise ValueError(f"Unsupported broker {broker!r}; supported brokers: {sorted(SUPPORTED_BROKERS)}")
    app_config.under_process_exchange = broker.capitalize()
    app_config.under_process_symbol = trading_pair
    app_config.under_process_market = market


def fetch_and_cache_ohlcv(
    broker: str,
    trading_pair: str,
    date_range_str: str,
    market: str = "Spot",
    base_timeframe: str | None = None,
) -> pt.DataFrame[OHLCV]:
    _bind_broker(broker, trading_pair, market)
    return cast("pt.DataFrame[OHLCV]", get_base_timeframe_ohlcv(date_range_str, base_timeframe=base_timeframe))


def fill_ohlcv_gaps(
    broker: str,
    trading_pair: str,
    date_range_str: str,
    market: str = "Spot",
    base_timeframe: str | None = None,
) -> list[str]:
    """
    Migration/backfill entry point: finds every daily gap in date_range_str for base-timeframe OHLCV
    (find_cache_gaps()) and fills each from the broker, newest gap first. A gap that's still empty
    after two direct broker asks is treated as confirmed-unavailable (e.g. pre-listing history) —
    marked with an empty cache file so it isn't rediscovered every run, logged, not raised. A genuine
    ccxt/network error from either ask propagates uncaught (never swallowed as "no data").

    Returns the gap ranges actually fetched (excludes confirmed-unavailable ones).
    """
    _bind_broker(broker, trading_pair, market)
    # file_path deliberately omitted below — find_cache_gaps()/write_data_file()/
    # cleanup_redundant_cache_files() all default to DATASET_DB (dataset_db_root(), data_frame_type-first)
    # on their own, same as every other real disk-cache call site (see DatasetDbSentinel).
    gaps = find_cache_gaps(OHLCV_DATASET, date_range_str)
    if not gaps:
        log_i(f"fill_ohlcv_gaps: {trading_pair}@{broker} already up to date over {date_range_str}")
        return []

    filled: list[str] = []
    for gap in sorted(gaps, key=lambda gap_range: date_range(gap_range)[0], reverse=True):
        rows = fetch_ohlcv_by_range(broker.lower(), gap, base_timeframe=base_timeframe)
        if not rows:
            rows = fetch_ohlcv_by_range(broker.lower(), gap, base_timeframe=base_timeframe)
        if not rows:
            log_i(f"fill_ohlcv_gaps: confirmed no {trading_pair}@{broker} data for {gap} after 2 broker asks")
            write_data_file(empty_df(OHLCV), OHLCV_DATASET.dataset_folder_name, gap, DATASET_DB)
            continue
        get_base_timeframe_ohlcv(gap, base_timeframe=base_timeframe)
        filled.append(gap)

    cleanup_redundant_cache_files(OHLCV_DATASET)
    return filled
