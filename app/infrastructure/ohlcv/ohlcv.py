from typing import cast

from config import app_config
from domain.ohlcv.multi_timeframe import aggregate_multi_timeframe_ohlcv
from domain.ohlcv.ohlcv import MULTI_TIMEFRAME_OHLCV_DATASET, OHLCV_DATASET, build_base_timeframe_ohlcv
from domain.schemas.common.OHLCV import OHLCV, MultiTimeframeOHLCV
from helper.data_preparation import single_timeframe
from helper.logging import profile_it
from helper.pandera import pandera_validate
from infrastructure.datastore_engine.disk_cache_windowed import read_file_windowed
from infrastructure.market_data_fetch.ccxt_client import fetch_ohlcv_by_range
from pandera import typing as pt

__all__ = [
    "cache_times",
    "get_base_timeframe_ohlcv",
    "get_multi_timeframe_ohlcv",
    "read_multi_timeframe_ohlcv",
    "OHLCV_DATASET",
    "MULTI_TIMEFRAME_OHLCV_DATASET",
]


@pandera_validate
def cache_times(result: pt.DataFrame[MultiTimeframeOHLCV]) -> None:
    for timeframe in app_config.timeframes:
        app_config.GLOBAL_CACHE[f"valid_times_{timeframe}"] = single_timeframe(result, timeframe).index


@pandera_validate
def _generate_base_timeframe_ohlcv(date_range_str: str, base_timeframe: str | None = None) -> pt.DataFrame[OHLCV]:
    raw_ohlcv = fetch_ohlcv_by_range(
        app_config.under_process_exchange.lower(), date_range_str, base_timeframe=base_timeframe
    )
    return build_base_timeframe_ohlcv(raw_ohlcv, date_range_str, base_timeframe)  # type: ignore[no-any-return]


@profile_it
@pandera_validate
def get_base_timeframe_ohlcv(
    date_range_str: str | None = None, base_timeframe: str | None = None
) -> pt.DataFrame[OHLCV]:
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    return cast(
        pt.DataFrame[OHLCV],
        read_file_windowed(
            date_range_str,
            OHLCV_DATASET.dataset_folder_name,
            _generate_base_timeframe_ohlcv,
            OHLCV,
            generator_params={"base_timeframe": base_timeframe},
        ),
    )


@pandera_validate
def _generate_multi_timeframe_ohlcv(date_range_str: str) -> MultiTimeframeOHLCV:
    ohlcv = cast(pt.DataFrame[OHLCV], get_base_timeframe_ohlcv(date_range_str))
    return aggregate_multi_timeframe_ohlcv(ohlcv, date_range_str)


@profile_it
@pandera_validate
def get_multi_timeframe_ohlcv(date_range_str: str | None = None) -> MultiTimeframeOHLCV:
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    result = cast(
        MultiTimeframeOHLCV,
        read_file_windowed(
            date_range_str,
            MULTI_TIMEFRAME_OHLCV_DATASET.dataset_folder_name,
            _generate_multi_timeframe_ohlcv,
            MultiTimeframeOHLCV,
        ),
    )
    cache_times(result)
    return result


read_multi_timeframe_ohlcv = get_multi_timeframe_ohlcv
