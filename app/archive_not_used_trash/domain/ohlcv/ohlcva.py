import pandas as pd
from config import app_config
from domain.ohlcv.ohlcv import cache_times, get_multi_timeframe_ohlcv
from archive_not_used_trash.domain.ohlcv.volume import insert_volume_rma
from domain.schemas.common.OHLCV import OHLCV
from domain.schemas.common.OHLCVA import MultiTimeframeOHLCVA
from helper.data_preparation import concat, multi_timeframe_times_tester, single_timeframe, trim_to_date_range
from archive_not_used_trash.helper.data_preparation import expand_date_range
from helper.schema_casting import empty_df
from infrastructure.datastore_engine.disk_cache import CachableDataset, cache_on_disk
from pandera import typing as pt

# Reused elsewhere instead of repeating the "multi_timeframe_ohlcva" data_frame_type string.
MULTI_TIMEFRAME_OHLCVA_DATASET = CachableDataset(dataset_folder_name="multi_timeframe_ohlcva")


@cache_on_disk(MULTI_TIMEFRAME_OHLCVA_DATASET, after_read=cache_times, windowed=True)
def get_multi_timeframe_ohlcva(date_range_str: str = None) -> pt.DataFrame[MultiTimeframeOHLCVA]:
    """
    One cache window's worth of multi-timeframe OHLCVA (see cache_on_disk(windowed=True) /
    infrastructure/ohlcv/README.md § windowing); disk_cache.read_file_windowed() decomposes an
    arbitrary requested date_range_str into windows, fetches/generates each through this function
    (expanding its own start by atr_timeperiod-driven lookback via get_multi_timeframe_ohlcv(),
    itself windowed, so the lookback context comes from already-cached window tiles), and stitches
    the result back together.
    """
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    multi_timeframe_ohlcva = empty_df(MultiTimeframeOHLCVA)
    for _, timeframe in enumerate(app_config.timeframes):
        expanded_date_range = expand_date_range(
            date_range_str,
            time_delta=(
                (app_config.atr_timeperiod + 2)
                * pd.to_timedelta(timeframe)
                * app_config.atr_safe_start_expand_multipliers
            ),
            mode="start",
        )
        expanded_date_multi_timeframe_ohlcv = get_multi_timeframe_ohlcv(expanded_date_range)
        timeframe_ohlcv = single_timeframe(expanded_date_multi_timeframe_ohlcv, timeframe)
        timeframe_ohlcva = insert_atr(timeframe_ohlcv)
        timeframe_ohlcva = timeframe_ohlcva.dropna(subset=["atr"]).copy()
        timeframe_ohlcva["timeframe"] = timeframe
        timeframe_ohlcva = timeframe_ohlcva.set_index("timeframe", append=True)
        timeframe_ohlcva = timeframe_ohlcva.swaplevel()
        multi_timeframe_ohlcva = concat(multi_timeframe_ohlcva, timeframe_ohlcva)
    multi_timeframe_ohlcva = multi_timeframe_ohlcva.sort_index(level="date")
    multi_timeframe_ohlcva = trim_to_date_range(date_range_str, multi_timeframe_ohlcva)
    assert multi_timeframe_times_tester(multi_timeframe_ohlcva, date_range_str)
    return multi_timeframe_ohlcva


def insert_atr(timeframe_ohlcv: pt.DataFrame[OHLCV], mode: str = "pandas_ta") -> pd.DataFrame:
    if len(timeframe_ohlcv) <= app_config.atr_timeperiod:
        timeframe_ohlcv["atr"] = pd.NA
    else:
        if mode == "pandas_ta":
            timeframe_ohlcv["atr"] = timeframe_ohlcv.ta.atr(
                timeperiod=app_config.atr_timeperiod,
                # high='high',
                # low='low',
                # close='close',
                # mamode='ema',
            )
        else:
            raise Exception(f"Unsupported mode:{mode}")
    insert_volume_rma(timeframe_ohlcv)
    return timeframe_ohlcv
