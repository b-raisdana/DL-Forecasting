import pandas as pd
from archive_not_used_trash.domain.price_action.PivotsHelper import level_ttl, pivots_level_n_margins
from archive_not_used_trash.domain.schemas.market_structure.BullBearSidePivot import BullBearSidePivot
from archive_not_used_trash.domain.schemas.market_structure.Pivot import MultiTimeframePivotDFM
from config import app_config
from helper.data_preparation import (
    after_under_process_date,
    anti_pattern_timeframe,
    concat,
    single_timeframe,
    trigger_timeframe,
)
from helper.logging import profile_it
from helper.schema_casting import cast_and_validate, empty_df
from infrastructure.datastore_engine.disk_cache import CachableDataset, cache_on_disk
from pandera import typing as pt

from domain.ohlcv.ohlcva import get_multi_timeframe_ohlcva
from domain.price_action.BullBearSide import (
    get_multi_timeframe_bull_bear_side_trends,
    previous_trend,
)
from domain.price_action.PeakValley import get_multi_timeframe_peaks_n_valleys, major_timeframe
from domain.schemas.market_structure.BullBearSide import BullBearSide

MULTI_TIMEFRAME_BULL_BEAR_SIDE_PIVOTS_DATASET = CachableDataset(
    dataset_folder_name="multi_timeframe_bull_bear_side_pivots"
)


def remove_overlapping_trends(timeframe_trends: pt.DataFrame[BullBearSide]) -> pt.DataFrame[BullBearSide]:
    """
    Select the trend with maximum `movement` from each overlapping date group.

    Args:
        timeframe_trends: DataFrame containing trend data.

    Returns:
        pd.DataFrame[application.backtesting.BullBearSide.BullBearSide]: A DataFrame with overlapping trends removed.
    """
    # Group the DataFrame by 'date' and find the index of the row with the maximum 'movement' within each group
    # max_movement_indices = timeframe_trends.groupby('movement_start_time')['movement'].idxmax()
    max_movement_indices = timeframe_trends.groupby("movement_end_time")["movement"].idxmax()

    # Select the rows with the maximum 'movement' based on the indices
    deduplicated_trends = timeframe_trends.loc[max_movement_indices]

    return deduplicated_trends


def multi_timeframe_bull_bear_side_pivots(
    date_range_str: str = None, structure_timeframe_shortlist: list["str"] = None
) -> pt.DataFrame[MultiTimeframePivotDFM]:
    """
    highest high of every Bullish and lowest low of every Bearish trend. for Trends
            conditions:
                >= 3 atr
                >= 1 atr reverse movement after the most significant top before a trend with the same direction.
                if a trend with the same direction before < 1 atr return found, merge these together.
            index = time of pivot candle (highest high for Bullish and lowest low for Bearish) mapped to timeframe
            exact_time = exact time of pivot candle
            timeframe = time frame of pivot candle
            value = highest high for Bullish and lowest low for Bearish
            inner_margin = [Bullish: high - ]/[Bearish: low +]
                max(distance from nearest body of pivot and adjacent candles, 1 atr in pivot timeframe)
            outer_margin =
    warning: if highest high is not the last peak of Bullish and lowest low is not the last Valley raise a warning log:
                timeframe, trend start time (index), time of last top
                time and high of highest high in Bullish and time and low of lowest low in Bearish,
    :return:
    """
    if date_range_str is None:
        date_range_str = app_config.processing_date_range

    multi_timeframe_trends = get_multi_timeframe_bull_bear_side_trends(date_range_str)
    multi_timeframe_peaks_n_valleys = get_multi_timeframe_peaks_n_valleys(date_range_str)
    multi_timeframe_ohlcva = get_multi_timeframe_ohlcva(date_range_str)
    multi_timeframe_pivots = empty_df(MultiTimeframePivotDFM)
    if structure_timeframe_shortlist is None:
        structure_timeframe_shortlist = app_config.structure_timeframes[::-1]
    for timeframe in structure_timeframe_shortlist:
        timeframe_peaks_n_valleys = major_timeframe(multi_timeframe_peaks_n_valleys, timeframe)
        timeframe_ohlcva = single_timeframe(multi_timeframe_ohlcva, timeframe)
        trigger_timeframe_ohlcva = single_timeframe(multi_timeframe_ohlcva, trigger_timeframe(timeframe))
        timeframe_trends = single_timeframe(multi_timeframe_trends, timeframe)
        if len(timeframe_trends) > 0:
            expected_movement_size = fine_tune_expected_movement_size(timeframe_trends["atr"])
            timeframe_trends["previous_trend"], timeframe_trends["previous_trend_movement"] = previous_trend(
                timeframe_trends
            )
            pivot_trends = timeframe_trends[
                (timeframe_trends["movement"] > expected_movement_size)
                & (timeframe_trends["previous_trend_movement"] > expected_movement_size * 3)
            ]
            pivot_trends = remove_overlapping_trends(pivot_trends)
            if len(pivot_trends) > 0:
                """
                start of a trend considered as a Pivot if:
                    the movement of trend is > expected_movement_size
                    the movement of previous trend is > expected_movement_size * 3
                """
                timeframe_pivots = pd.DataFrame(
                    data={"date": pivot_trends["movement_start_time"], "ttl": None, "hit": 0}
                ).set_index("date")
                timeframe_pivots["original_start"] = timeframe_pivots.index
                timeframe_pivots["movement_start_time"] = timeframe_trends.loc[
                    pivot_trends["previous_trend"], "movement_start_time"
                ].to_list()
                timeframe_pivots["movement_start_value"] = timeframe_trends.loc[
                    pivot_trends["previous_trend"], "movement_start_value"
                ].to_list()
                timeframe_pivots["return_end_time"] = pivot_trends["movement_end_time"].to_list()
                timeframe_pivots["return_end_value"] = pivot_trends["movement_end_value"].to_list()

                # find the Peaks and Valleys align with the Pivot
                pivot_peaks_n_valleys = timeframe_peaks_n_valleys.loc[
                    timeframe_peaks_n_valleys.index.get_level_values("date").isin(timeframe_pivots.index)
                ]
                timeframe_pivots = pivots_level_n_margins(
                    timeframe_pivots=timeframe_pivots,
                    pivot_time_peaks_n_valleys=pivot_peaks_n_valleys,
                    timeframe=timeframe,
                    candle_body_source=timeframe_ohlcva,
                    internal_atr_source=timeframe_ohlcva,
                    breakout_atr_source=trigger_timeframe_ohlcva,
                )
                timeframe_pivots["ttl"] = timeframe_pivots.index + level_ttl(timeframe)
                timeframe_pivots["deactivated_at"] = None
                timeframe_pivots["archived_at"] = None
                timeframe_pivots["master_pivot_timeframe"] = None
                timeframe_pivots["master_pivot_date"] = None
                timeframe_pivots = cast_and_validate(timeframe_pivots, BullBearSidePivot)
                # timeframe_pivots['timeframe'] = timeframe
                timeframe_pivots["timeframe"] = anti_pattern_timeframe(timeframe)
                timeframe_pivots = timeframe_pivots.set_index("timeframe", append=True)
                timeframe_pivots = timeframe_pivots.swaplevel()
                multi_timeframe_pivots = concat(multi_timeframe_pivots, timeframe_pivots)
    multi_timeframe_pivots = cast_and_validate(
        multi_timeframe_pivots, MultiTimeframePivotDFM, zero_size_allowed=after_under_process_date(date_range_str)
    )
    return multi_timeframe_pivots


@profile_it
@cache_on_disk(MULTI_TIMEFRAME_BULL_BEAR_SIDE_PIVOTS_DATASET)
def get_multi_timeframe_bull_bear_side_pivots(
    date_range_str: str = None, timeframe_shortlist: list["str"] = None
) -> pt.DataFrame[MultiTimeframePivotDFM]:
    """
    highest high of every Bullish and lowest low of every Bearish trend. for Trends
            conditions:
                >= 3 atr
                >= 1 atr reverse movement after the most significant top before a trend with the same direction.
                if a trend with the same direction before < 1 atr return found, merge these together.
            index = time of pivot candle (highest high for Bullish and lowest low for Bearish) mapped to timeframe
            exact_time = exact time of pivot candle
            timeframe = time frame of pivot candle
            value = highest high for Bullish and lowest low for Bearish
            inner_margin = [Bullish: high - ]/[Bearish: low +]
                max(distance from nearest body of pivot and adjacent candles, 1 atr in pivot timeframe)
            outer_margin =
    warning: if highest high is not the last peak of Bullish and lowest low is not the last Valley raise a warning log:
                timeframe, trend start time (index), time of last top
                time and high of highest high in Bullish and time and low of lowest low in Bearish,
    :return:
    """
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    multi_timeframe_pivots = multi_timeframe_bull_bear_side_pivots(date_range_str, timeframe_shortlist)
    multi_timeframe_pivots = multi_timeframe_pivots.sort_index(level="date")

    """
        in all boundaries with movement >= 1 atr:
            if
                movement of previous boundary >= 3 atr
                or distance of most significant peak to reverse high/low of boundary >= 1 atr
                or in boundary reverse tops after most significant top find distance of >= 1 atr
            then:
                the boundary most significant top is a static level pivot.
                if:
                    the pivot is inside a previous active or inactive level of the same or higher timeframe:
                then:
                    do not addd as a new level and increase hit count of the previous level.
                else:
                    add a new level for the pivot
    """
    # raise Exception('Not implemented')
    return multi_timeframe_pivots


def fine_tune_expected_movement_size(_list: list):
    return _list  # * CandleSize.Standard.value.min
