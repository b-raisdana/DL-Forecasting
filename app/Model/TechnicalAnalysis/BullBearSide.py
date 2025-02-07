import os
from datetime import timedelta
from typing import Tuple, List, Optional, Literal

import pandas as pd
import pandera.typing as pt

from Config import TopTYPE, app_config, TREND
from Model.TechnicalAnalysis.PeakValley import valleys_only, peaks_only, major_timeframe, \
    read_multi_timeframe_peaks_n_valleys, insert_previous_n_next_top
from PanderaDFM.BullBearSide import MultiTimeframeBullBearSide, BullBearSide, bull_bear_side_repr
from PanderaDFM.CandleTrend import MultiTimeframeCandleTrend, CandleTrend
from PanderaDFM.OHLCV import OHLCV
from PanderaDFM.OHLCVA import OHLCVA
from PanderaDFM.PeakValley import PeakValley, MultiTimeframePeakValley

from data_processing.atr import read_multi_timeframe_ohlcva
from data_processing.fragmented_data import symbol_data_path
from helper.data_preparation import read_file, single_timeframe, to_timeframe, cast_and_validate, empty_df, concat, \
    date_range_of_data
from helper.functions import log, profile_it, LogSeverity
from data_processing.ohlcv import read_multi_timeframe_ohlcv


def insert_previous_n_next_tops(single_timeframe_peaks_n_valleys: pt.DataFrame[PeakValley], ohlcv: pt.DataFrame[OHLCV]) \
        -> pt.DataFrame[OHLCV]:
    ohlcv = insert_previous_n_next_top(TopTYPE.PEAK, single_timeframe_peaks_n_valleys, ohlcv)
    ohlcv = insert_previous_n_next_top(TopTYPE.VALLEY, single_timeframe_peaks_n_valleys, ohlcv)
    return ohlcv


@profile_it
def single_timeframe_candles_trend(ohlcv: pt.DataFrame[OHLCV], timeframe_peaks_n_valley: pt.DataFrame[PeakValley]) \
        -> pt.DataFrame[CandleTrend]:
    candle_trend = insert_previous_n_next_tops(timeframe_peaks_n_valley, ohlcv)
    candle_trend['bull_bear_side'] = pd.NA
    candle_trend['is_final'] = False
    if len(ohlcv) == 0:
        return candle_trend
    candles_with_known_trend = candle_trend.loc[
        candle_trend['next_peak_value'].notna() &
        candle_trend['previous_peak_value'].notna() &
        candle_trend['next_valley_value'].notna() &
        candle_trend['previous_valley_value'].notna()].index

    if len(candles_with_known_trend) == 0:
        log(f'Not found any candle with possibly known trend '
            f'in ({ohlcv.index[0]}:{ohlcv.index[-1]}#{len(ohlcv)}={ohlcv.head(5)})!',
            severity=LogSeverity.WARNING, stack_trace=False)
        candle_trend['bull_bear_side'] = TREND.SIDE.value
        if candle_trend['is_final'].isna().any():
            pass
        return candle_trend
    candle_trend.loc[
        candles_with_known_trend,
        'bull_bear_side'] = TREND.SIDE.value
    bullish_candles = candle_trend[
        ((candle_trend['next_peak_value'] > candle_trend['previous_peak_value'])
         & (candle_trend['next_valley_value'] > candle_trend['previous_valley_value']))
    ].index
    if len(bullish_candles) > 0:
        candle_trend.loc[
            bullish_candles,  # todo: the higher peak should be after higher valley
            'bull_bear_side'] = TREND.BULLISH.value
    bearish_candles = candle_trend[
        ((candle_trend['next_peak_value'] < candle_trend['previous_peak_value'])
         & (candle_trend['next_valley_value'] < candle_trend['previous_valley_value']))
    ].index
    if len(bearish_candles) > 0:
        candle_trend.loc[
            bearish_candles,  # todo: the lower valley should be after lower peak
            'bull_bear_side'] = TREND.BEARISH.value
    candle_trend.loc[candle_trend['bull_bear_side'].notna(), 'is_final'] = True
    candle_trend['bull_bear_side'] = candle_trend['bull_bear_side'].ffill()
    candle_trend['bull_bear_side'] = candle_trend['bull_bear_side'].bfill()
    if candle_trend['is_final'].isna().any():
        pass
    return candle_trend


@profile_it
def expand_trend_by_near_tops(timeframe_bull_or_bear: pt.DataFrame[BullBearSide],
                              timeframe_peaks: pt.DataFrame[PeakValley],
                              timeframe_valleys: pt.DataFrame[PeakValley], trend: TREND):
    assert 'movement_start_time' in timeframe_bull_or_bear.columns
    assert 'movement_end_time' in timeframe_bull_or_bear.columns
    assert 'movement_start_value' in timeframe_bull_or_bear.columns
    assert 'movement_end_value' in timeframe_bull_or_bear.columns

    if len(timeframe_bull_or_bear) == 0:
        return timeframe_bull_or_bear
    if trend == TREND.BULLISH:
        end_significant_column = 'high'
        end_tops = timeframe_peaks
        start_significant_column = 'low'
        start_tops = timeframe_valleys

        def more_significant_end(x: float, y: float):
            return x >= y

        def more_significant_start(x: float, y: float):
            return x <= y
    elif trend == TREND.BEARISH:
        end_significant_column = 'low'
        end_tops = timeframe_valleys
        start_significant_column = 'high'
        start_tops = timeframe_peaks

        def more_significant_end(x: float, y: float):
            return x <= y

        def more_significant_start(x: float, y: float):
            return x >= y
    else:
        raise Exception(f"Invalid trend['bull_bear_side']={trend}")
    shifted_next_tops = shifted_time_and_value(end_tops, 'next', end_significant_column, 'top')
    shifted_previous_tops = shifted_time_and_value(start_tops, 'previous', start_significant_column, 'top')
    previous_round_movement_end_time = timeframe_bull_or_bear['movement_end_time']
    previous_round_movement_start_time = timeframe_bull_or_bear['movement_start_time']
    possible_end_expandable_indexes = timeframe_bull_or_bear.index
    possible_start_expandable_indexes = timeframe_bull_or_bear.index
    number_of_changed_ends = None
    number_of_changed_starts = None

    while (
            None in [number_of_changed_ends, number_of_changed_starts]
            or number_of_changed_ends > 0
            or number_of_changed_starts > 0
    ):
        # track if this iteration changed anything.
        timeframe_bull_or_bear = timeframe_bull_or_bear.drop(
            columns=['next_top_value', 'next_top_time', 'previous_top_time', 'previous_top_value'], errors='ignore')
        assert timeframe_bull_or_bear['movement_end_time'].notna().all()
        timeframe_bull_or_bear = pd.merge_asof(timeframe_bull_or_bear.sort_values(by='movement_end_time'),
                                               shifted_next_tops, direction='forward',
                                               left_on='movement_end_time', right_index=True)
        end_expandable_indexes = timeframe_bull_or_bear.loc[
            timeframe_bull_or_bear.index.isin(possible_end_expandable_indexes) &
            timeframe_bull_or_bear[f'next_top_value'].notna() &
            more_significant_end(
                timeframe_bull_or_bear[f'next_top_value'],
                timeframe_bull_or_bear[f'internal_{end_significant_column}'])
            ].index
        if len(timeframe_bull_or_bear) == 11:
            pass
        possible_end_expandable_indexes = end_expandable_indexes
        assert timeframe_bull_or_bear.loc[end_expandable_indexes, 'next_top_time'].notna().all()
        timeframe_bull_or_bear.loc[end_expandable_indexes, 'movement_end_time'] = \
            timeframe_bull_or_bear.loc[end_expandable_indexes, 'next_top_time']
        timeframe_bull_or_bear.loc[end_expandable_indexes, 'movement_end_value'] = \
            timeframe_bull_or_bear.loc[end_expandable_indexes, 'next_top_value']
        if timeframe_bull_or_bear.loc[end_expandable_indexes, 'next_top_value'].isna().any():
            pass
        timeframe_bull_or_bear.loc[end_expandable_indexes, f'internal_{end_significant_column}'] = \
            timeframe_bull_or_bear.loc[end_expandable_indexes, 'next_top_value']

        timeframe_bull_or_bear = pd.merge_asof(timeframe_bull_or_bear.sort_values(by='movement_start_time'),
                                               shifted_previous_tops, direction='backward',
                                               left_on='movement_start_time', right_index=True)
        start_expandable_indexes = timeframe_bull_or_bear.loc[
            timeframe_bull_or_bear.index.isin(possible_start_expandable_indexes) &
            timeframe_bull_or_bear[f'previous_top_value'].notna() &
            more_significant_start(
                timeframe_bull_or_bear[f'previous_top_value'],
                timeframe_bull_or_bear[f'internal_{start_significant_column}'])
            ].index
        possible_start_expandable_indexes = start_expandable_indexes
        if len(timeframe_bull_or_bear) == 11:
            pass
        assert timeframe_bull_or_bear.loc[start_expandable_indexes, 'previous_top_time'].notna().all()
        timeframe_bull_or_bear.loc[start_expandable_indexes, 'movement_start_time'] = \
            timeframe_bull_or_bear.loc[start_expandable_indexes, 'previous_top_time']
        timeframe_bull_or_bear.loc[start_expandable_indexes, 'movement_start_value'] = \
            timeframe_bull_or_bear.loc[start_expandable_indexes, 'previous_top_value']
        if timeframe_bull_or_bear.loc[start_expandable_indexes, 'previous_top_value'].isna().any():
            pass
        timeframe_bull_or_bear.loc[start_expandable_indexes, f'internal_{start_significant_column}'] = \
            timeframe_bull_or_bear.loc[start_expandable_indexes, 'previous_top_value']
        # update while loop condition parameters
        changed_ends = timeframe_bull_or_bear['movement_end_time'].sort_index().compare(
            previous_round_movement_end_time.sort_index())
        number_of_changed_ends = len(changed_ends)
        changed_starts = timeframe_bull_or_bear['movement_start_time'].sort_index().compare(
            previous_round_movement_start_time.sort_index())
        number_of_changed_starts = len(changed_starts)
        log(f"Changed start={number_of_changed_starts} ends={number_of_changed_ends}"
            f'possibly movable starts={len(possible_start_expandable_indexes)} '
            f'ends= {len(possible_end_expandable_indexes)} '
            , severity=LogSeverity.DEBUG, stack_trace=False)
        previous_round_movement_end_time = timeframe_bull_or_bear['movement_end_time'].copy()
        previous_round_movement_start_time = timeframe_bull_or_bear['movement_start_time'].copy()

    return timeframe_bull_or_bear


def shifted_time_and_value(df: pd.DataFrame, direction: Literal['next', 'previous'], source_value_column: str,
                           shifted_columns_prefix: str) -> pd.DataFrame:
    if direction == 'next':
        shift_value = -1
    elif direction == 'previous':
        shift_value = 1
    else:
        raise Exception(f'Invalid direction: {direction} should be "next" or "previous"')
    shifted_top_indexes = df.index.shift(shift_value, freq=app_config.timeframes[0])
    shifted_tops = pd.DataFrame(index=shifted_top_indexes).sort_index()
    shifted_tops[f'{direction}_{shifted_columns_prefix}_time'] = df.index.tolist()
    shifted_tops[f'{direction}_{shifted_columns_prefix}_value'] = df[source_value_column].tolist()
    return shifted_tops


def add_trends_movement(bbs_trends: pt.DataFrame[BullBearSide],
                        timeframe_peaks_n_valleys: pt.DataFrame[PeakValley],
                        ) -> pt.DataFrame[BullBearSide]:
    side_trends = bbs_trends[bbs_trends['bull_bear_side'] == TREND.SIDE.value].copy()
    bullish_trends = bbs_trends[bbs_trends['bull_bear_side'] == TREND.BULLISH.value].copy()
    bearish_trends = bbs_trends[bbs_trends['bull_bear_side'] == TREND.BEARISH.value].copy()
    bullish_trends = add_trend_movement(bullish_trends, timeframe_peaks_n_valleys, TREND.BULLISH)
    bearish_trends = add_trend_movement(bearish_trends, timeframe_peaks_n_valleys, TREND.BEARISH)
    bbs_trends = pd.concat([bullish_trends, bearish_trends, side_trends]).sort_index(level='date')

    return bbs_trends


def add_trend_movement(bull_or_bear_trends: pt.DataFrame[BullBearSide],
                       timeframe_peaks_n_valleys: pt.DataFrame[PeakValley],
                       trend: TREND
                       ) -> pt.DataFrame[BullBearSide]:
    assert (
            'movement_start_time' not in bull_or_bear_trends.columns
            and 'movement_end_time' not in bull_or_bear_trends.columns
            and 'movement_start_value' not in bull_or_bear_trends.columns
            and 'movement_end_value' not in bull_or_bear_trends.columns)
    bull_or_bear_trends['movement_end_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    bull_or_bear_trends['movement_start_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    bull_or_bear_trends['movement_end_value'] = pd.Series(dtype=float)
    bull_or_bear_trends['movement_start_value'] = pd.Series(dtype=float)
    if len(bull_or_bear_trends) == 12:
        pass
    if len(bull_or_bear_trends) == 0:
        return bull_or_bear_trends
    if trend == TREND.BULLISH:
        end_significant_column = 'high'
        end_top_type = TopTYPE.PEAK.value
        end_tops = peaks_only(timeframe_peaks_n_valleys)
        start_significant_column = 'low'
        start_top_type = TopTYPE.VALLEY.value
        start_tops = valleys_only(timeframe_peaks_n_valleys)
    elif trend == TREND.BEARISH:
        end_significant_column = 'low'
        end_top_type = TopTYPE.VALLEY.value
        end_tops = valleys_only(timeframe_peaks_n_valleys)
        start_significant_column = 'high'
        start_top_type = TopTYPE.PEAK.value
        start_tops = peaks_only(timeframe_peaks_n_valleys)
    else:
        raise Exception(f"Invalid trend['bull_bear_side']={trend}")
    next_start_tops = shifted_time_and_value(start_tops, 'next', start_significant_column,
                                             f'{start_top_type}_after_start')
    previous_end_tops = shifted_time_and_value(end_tops, 'previous', end_significant_column,
                                               f'{end_top_type}_before_end')
    bull_or_bear_trends = pd.merge_asof(bull_or_bear_trends, next_start_tops, left_index=True, right_index=True,
                                        direction='forward')
    bull_or_bear_trends = pd.merge_asof(bull_or_bear_trends, previous_end_tops, left_on='end', right_index=True,
                                        direction='backward')

    # invalid trends are trends without peak and valleys
    with_next_top_trends = bull_or_bear_trends[
        bull_or_bear_trends[f'next_{start_top_type}_after_start_time'] <
        bull_or_bear_trends['end']
        ].index
    no_next_top_trends = bull_or_bear_trends.index.difference(with_next_top_trends)
    with_previous_top_trends = bull_or_bear_trends[
        bull_or_bear_trends[f'previous_{end_top_type}_before_end_time'] >
        bull_or_bear_trends.index
        ].index
    no_previous_top_trends = bull_or_bear_trends.index.difference(with_previous_top_trends)

    bull_or_bear_trends.loc[no_next_top_trends, 'movement_start_time'] = no_next_top_trends.tolist()
    bull_or_bear_trends.loc[no_next_top_trends, 'movement_start_value'] = bull_or_bear_trends.loc[
        no_next_top_trends, f'internal_{start_significant_column}']
    bull_or_bear_trends.loc[no_previous_top_trends, 'movement_end_time'] = bull_or_bear_trends.loc[
        no_previous_top_trends, 'end']
    bull_or_bear_trends.loc[no_previous_top_trends, 'movement_end_value'] = bull_or_bear_trends.loc[
        no_previous_top_trends, f'internal_{end_significant_column}']

    bull_or_bear_trends.loc[with_next_top_trends, 'movement_start_time'] = bull_or_bear_trends.loc[
        with_next_top_trends, f'next_{start_top_type}_after_start_time']
    bull_or_bear_trends.loc[with_next_top_trends, 'movement_start_value'] = bull_or_bear_trends.loc[
        with_next_top_trends, f'next_{start_top_type}_after_start_value']
    bull_or_bear_trends.loc[with_previous_top_trends, 'movement_end_time'] = bull_or_bear_trends.loc[
        with_previous_top_trends, f'previous_{end_top_type}_before_end_time']
    bull_or_bear_trends.loc[with_previous_top_trends, 'movement_end_value'] = bull_or_bear_trends.loc[
        with_previous_top_trends, f'previous_{end_top_type}_before_end_value']
    return bull_or_bear_trends


@profile_it
def expand_trends_by_near_tops(_timeframe_bull_bear_side_trends: pt.DataFrame[BullBearSide],
                               _timeframe_peaks_n_valleys: pt.DataFrame[PeakValley],
                               ) -> pd.DataFrame:
    """
    Expand boundaries towards the previous trend top.

    This function expands boundaries that are not of 'SIDE' trend towards the previous top of the same trend
    (either bullish or bearish) inside the boundary. If the trend is bullish, it finds the next peak (first peak
    after the boundary finishes) and the last peak inside the boundary. If the next peak's high value is greater
    than the last peak, the boundary is expanded to include the next peak. Similarly, for a bearish trend, it finds
    the next valley (first valley after the boundary finishes) and the last valley inside the boundary. If the
    next valley's low value is lower than the last valley, the boundary is expanded to include the next valley.

    Parameters:
        _timeframe_bull_bear_side_trends (pd.DataFrame): DataFrame containing boundaries for the specified timeframe.
        _timeframe_peaks_n_valleys (pd.DataFrame): DataFrame containing peaks and valleys for the same timeframe.

    Returns:
        pd.DataFrame: The input DataFrame with added columns 'movement_start_value' and 'movement_end_value'.

    Raises:
        Exception: If the DataFrame index of single_timeframe_boundaries is not unique.

    Example:
        # Expand boundaries towards the previous trend top for a specific timeframe
        updated_boundaries = add_toward_top_to_trend(single_timeframe_boundaries,
                                                     single_timeframe_peaks_n_valleys,
                                                     '15min')
        log_d(updated_boundaries)
    """
    if len(_timeframe_bull_bear_side_trends) == 46:
        pass
    if len(_timeframe_bull_bear_side_trends) > 0:
        if not _timeframe_bull_bear_side_trends.index.is_unique:
            raise Exception('We expect the single_timeframe_boundaries index be unique but isn\'t. '
                            'So we may have unexpected results after renaming DataFrame index to move boundary start.')
        timeframe_peaks = peaks_only(_timeframe_peaks_n_valleys).sort_index(level='date')
        timeframe_valleys = valleys_only(_timeframe_peaks_n_valleys).sort_index(level='date')

        bullish_indexes = _timeframe_bull_bear_side_trends[
            _timeframe_bull_bear_side_trends['bull_bear_side'] == TREND.BULLISH.value
            ].index
        bearish_indexes = _timeframe_bull_bear_side_trends[
            _timeframe_bull_bear_side_trends['bull_bear_side'] == TREND.BEARISH.value
            ].index
        if len(bullish_indexes) > 0:
            _timeframe_bull_bear_side_trends.loc[bullish_indexes] = \
                expand_trend_by_near_tops(_timeframe_bull_bear_side_trends.loc[bullish_indexes],
                                          timeframe_peaks, timeframe_valleys, trend=TREND.BULLISH)
        if len(bearish_indexes) > 0:
            _timeframe_bull_bear_side_trends.loc[bearish_indexes] = \
                expand_trend_by_near_tops(_timeframe_bull_bear_side_trends.loc[bearish_indexes],
                                          timeframe_peaks, timeframe_valleys, trend=TREND.BEARISH)
    assert not _timeframe_bull_bear_side_trends.isna().all(axis=1).any()
    return _timeframe_bull_bear_side_trends


@profile_it
def multi_timeframe_bull_bear_side_trends(multi_timeframe_candle_trend: pt.DataFrame[MultiTimeframeCandleTrend],
                                          multi_timeframe_peaks_n_valleys: pt.DataFrame[MultiTimeframePeakValley],
                                          multi_timeframe_ohlcva: pt.DataFrame[OHLCVA],
                                          timeframe_shortlist: List['str'] = None) \
        -> pt.DataFrame[MultiTimeframeBullBearSide]:
    trends = empty_df(MultiTimeframeBullBearSide)

    if timeframe_shortlist is None:
        timeframe_shortlist = app_config.timeframes
    for timeframe in timeframe_shortlist:
        timeframe_candle_trend = single_timeframe(multi_timeframe_candle_trend, timeframe)
        if len(timeframe_candle_trend) < 3:
            log(f'multi_timeframe_candle_trend has less than 3 rows ({len(timeframe_candle_trend)}) for '
                f'{timeframe}/{date_range_of_data(multi_timeframe_ohlcva)}',
                stack_trace=False, severity=LogSeverity.WARNING)
            continue
        timeframe_peaks_n_valleys = major_timeframe(multi_timeframe_peaks_n_valleys, timeframe) \
            .reset_index(level='timeframe')
        timeframe_trends = single_timeframe_bull_bear_side_trends(timeframe_candle_trend,
                                                                  timeframe_peaks_n_valleys,
                                                                  single_timeframe(multi_timeframe_ohlcva, timeframe)
                                                                  , timeframe)
        timeframe_trends['timeframe'] = timeframe
        timeframe_trends = timeframe_trends.set_index('timeframe', append=True)
        timeframe_trends = timeframe_trends.swaplevel().sort_index(level='date')
        trends = concat(trends, timeframe_trends)
    trends = cast_and_validate(trends, MultiTimeframeBullBearSide)
    return trends


def add_trend_extremum(_boundaries, single_timeframe_peak_n_valley: pt.DataFrame[PeakValley],
                       ohlcv: pt.DataFrame[OHLCV]):
    _boundaries['internal_high'] = pd.Series(dtype=float)
    _boundaries['high_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    _boundaries['internal_low'] = pd.Series(dtype=float)
    _boundaries['low_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    ohlcv['bbs_index'] = pd.Series(dtype='datetime64[ns, UTC]')
    if len(_boundaries) == 0:
        return _boundaries
    _boundaries = _boundaries.copy()

    ohlcv.loc[_boundaries.index, 'bbs_index'] = _boundaries.index.tolist()
    ohlcv['bbs_index'] = ohlcv['bbs_index'].ffill()
    ohlcv['bbs_index'] = ohlcv['bbs_index'].bfill()
    grouped_ohlcv = ohlcv.groupby(by='bbs_index').agg({'low': 'min', 'high': 'max', })
    _boundaries[['internal_low', 'internal_high']] = grouped_ohlcv[['low', 'high']]
    ohlcv = ohlcv.merge(_boundaries[['internal_high', 'internal_low']], how='left', left_index=True, right_index=True)
    ohlcv = ohlcv.rename(columns={'internal_high': 'bbs_high', 'internal_low': 'bbs_low'})
    ohlcv[['bbs_high', 'bbs_low']] = ohlcv[['bbs_high', 'bbs_low']].ffill()
    ohlcv[['bbs_high', 'bbs_low']] = ohlcv[['bbs_high', 'bbs_low']].bfill()
    ohlcv['is_bbs_high'] = ohlcv['high'] == ohlcv['bbs_high']
    ohlcv['is_bbs_low'] = ohlcv['low'] == ohlcv['bbs_low']
    ohlcv.loc[ohlcv['is_bbs_high'], 'high_time'] = ohlcv[ohlcv['is_bbs_high']].index
    ohlcv.loc[ohlcv['is_bbs_low'], 'low_time'] = ohlcv[ohlcv['is_bbs_low']].index
    grouped_ohlcv = ohlcv.groupby(by='bbs_index').agg({'high_time': 'first', 'low_time': 'first', })
    _boundaries[['low_time', 'high_time']] = grouped_ohlcv[['low_time', 'high_time']]
    return _boundaries


def most_two_significant_tops(start, end, single_timeframe_peaks_n_valleys, tops_type: TopTYPE) -> pd.DataFrame:
    log('test most_two_significant_valleys', severity=LogSeverity.ERROR)
    filtered_valleys = single_timeframe_peaks_n_valleys.loc[
        (single_timeframe_peaks_n_valleys.index >= start) &
        (single_timeframe_peaks_n_valleys.index <= end) &
        (single_timeframe_peaks_n_valleys['peak_or_valley'] == tops_type.value)
        ].sort_values(by='strength')
    return filtered_valleys.iloc[:2].sort_index(level='date')


def trends_atr(timeframe_boundaries: pt.DataFrame[BullBearSide], ohlcva: pt.DataFrame[OHLCVA]):
    ohlcva.loc[timeframe_boundaries.index, 'bbs_index'] = timeframe_boundaries.index.tolist()
    ohlcva['bbs_index'] = ohlcva['bbs_index'].ffill()
    ohlcva['bbs_index'] = ohlcva['bbs_index'].bfill()
    _boundaries_atrs = ohlcva.groupby(by='bbs_index').agg({'atr': 'mean'})['atr']
    assert _boundaries_atrs.notna().all()
    return _boundaries_atrs


def trend_rate(boundaries: pt.DataFrame[BullBearSide], timeframe: str) -> pt.DataFrame[BullBearSide]:
    return boundaries['movement'] / (boundaries['duration'] / pd.to_timedelta(timeframe))


def trend_duration(boundaries: pt.DataFrame[BullBearSide]) -> pt.Series[timedelta]:
    durations = pd.to_datetime(boundaries['end']) - pd.to_datetime(boundaries.index)
    invali_length_boundaries = boundaries[boundaries['end'] <= boundaries.index].index
    if len(invali_length_boundaries) > 0:
        message_body = "\n".join([BullBearSide.repr(start, boundary)
                                  for start, boundary in boundaries.loc[invali_length_boundaries].itterrows()])
        raise Exception(f'Invalid duration(s) in:{message_body}')
    return durations


def ignore_weak_trend(boundaries: pt.DataFrame[BullBearSide]) -> pt.DataFrame[BullBearSide]:
    """
    Remove weak trends from the DataFrame.

    Parameters:
        boundaries (pt.DataFrame[Strategy.BullBearSide.BullBearSide]): A DataFrame containing trend boundary data.

    Returns:
        pt.DataFrame[Strategy.BullBearSide.BullBearSide]: A DataFrame with weak trends removed.

    Example:
        # Assuming you have a DataFrame '_boundaries' with the required columns and data
        filtered_boundaries = ignore_weak_trend(_boundaries)
    """
    boundaries.loc[boundaries['strength'] < app_config.momentum_trand_strength_factor, 'bull_bear_side'] \
        = TREND.SIDE.value
    return boundaries


def trend_movement(boundaries: pt.DataFrame[BullBearSide]) -> pt.Series[float]:
    """
        Calculate the trend movement as the difference between the highest high and the lowest low.

        Parameters:
            boundaries (pd.DataFrame): A DataFrame containing trend boundary data.

        Returns:
            pd.Series: A Series containing the calculated trend movement values.

        Example:
            # Assuming you have a DataFrame '_boundaries' with the required columns
            result = trend_movement(_boundaries)
        """
    if len(boundaries) > 0:
        assert 'movement_start_value' in boundaries.columns
        assert ('movement_end_value' in boundaries.columns)
        assert ('internal_high' in boundaries.columns)
        assert ('internal_low' in boundaries.columns)
        t = boundaries[['internal_high', 'movement_start_value', 'movement_end_value']].max(axis='columns') \
            - boundaries[['internal_low', 'movement_start_value', 'movement_end_value']].min(axis='columns')
        return t
    else:
        return pd.Series(dtype=float)


def trend_strength(boundaries):
    return boundaries['rate'] / boundaries['atr']


def previous_trend(trends: pt.DataFrame[BullBearSide]) -> Tuple[List[Optional[int]], List[Optional[float]]]:
    """
        Find the previous trend and its movement for each row in a DataFrame of trends.

        Args:
            trends (pd.DataFrame): A DataFrame containing trend data with columns 'movement_start_time' and 'movement_end_time'.

        Returns:
            Tuple[List[Optional[int]], List[Optional[float]]]: A tuple containing two lists:
                - A list of previous trend indices (int or None).
                - A list of the corresponding previous trend movements (float or None).
        """
    previous_trends = []
    previous_trends_movement = []
    for _start, this_trend in trends.iterrows():
        if this_trend['movement_start_time'] is not None:
            possible_previous_trends = trends[trends['movement_end_time'] == this_trend['movement_start_time']]
            if len(possible_previous_trends) > 0:
                previous_trends.append(possible_previous_trends['movement'].idxmax())
                previous_trends_movement.append(possible_previous_trends['movement'].max())
                continue
            else:
                log(f'did not find any previous trend for trend stat@{bull_bear_side_repr(_start, this_trend)})',
                    stack_trace=False)
        else:
            raise Exception(f"movement_start_time is not valid:{this_trend['movement_start_time']}")
        previous_trends.append(None)
        previous_trends_movement.append(None)
    return previous_trends, previous_trends_movement


def single_timeframe_bull_bear_side_trends(single_timeframe_candle_trend: pd.DataFrame,
                                           timeframe_peaks_n_valleys, ohlcva: pt.DataFrame[OHLCVA],
                                           timeframe: str) -> pd.DataFrame:
    if ohlcva['atr'].first_valid_index() is None:
        # return pd.DataFrame()
        return empty_df(BullBearSide)
    single_timeframe_candle_trend = single_timeframe_candle_trend.loc[ohlcva['atr'].first_valid_index():]
    trends = detect_trends(single_timeframe_candle_trend, timeframe)
    trends = add_trend_extremum(trends, timeframe_peaks_n_valleys, ohlcva)
    if timeframe == '5min':
        pass
    trends = add_trends_movement(trends, timeframe_peaks_n_valleys)
    trends = expand_trends_by_near_tops(trends, timeframe_peaks_n_valleys)
    trends['movement'] = trend_movement(trends)
    trends['atr'] = trends_atr(trends, ohlcva=ohlcva)

    trends['duration'] = trend_duration(trends)
    trends['rate'] = trend_rate(trends, timeframe)
    trends['strength'] = trend_strength(trends)
    trends = cast_and_validate(trends, BullBearSide, zero_size_allowed=True, unique_index=True)
    return trends


def detect_trends(single_timeframe_candle_trend, timeframe: str) -> pt.DataFrame[BullBearSide]:
    single_timeframe_candle_trend = single_timeframe_candle_trend.copy()
    if single_timeframe_candle_trend['bull_bear_side'].isna().any():
        pass
    if len(single_timeframe_candle_trend) < 2:
        boundaries = single_timeframe_candle_trend
        boundaries['bull_bear_side'] = TREND.SIDE.value
        boundaries['end'] = pd.Series(dtype='datetime64[ns, UTC]')
        return boundaries[['bull_bear_side', 'end']]
    single_timeframe_candle_trend['previous_trend'] = single_timeframe_candle_trend['bull_bear_side'].shift(1)
    boundaries = single_timeframe_candle_trend[
        single_timeframe_candle_trend['previous_trend'] != single_timeframe_candle_trend['bull_bear_side']
        ].copy()
    boundaries['end'] = pd.Series(dtype='datetime64[ns, UTC]')
    time_of_last_candle = single_timeframe_candle_trend.index.get_level_values('date')[-1]
    if len(boundaries) > 1:
        boundaries.loc[:boundaries.index[-2], 'end'] = \
            pd.to_datetime(to_timeframe(boundaries.index.get_level_values('date')[1:], timeframe).tolist())
    boundaries.loc[boundaries.index[-1], 'end'] = to_timeframe(time_of_last_candle, timeframe)
    assert not boundaries['end'].isna().any()
    if boundaries.iloc[-1]['end'] == boundaries.index[-1]:
        raise Exception('Not tested!')
        boundaries = boundaries.drop(boundaries.index[-1])
    return boundaries[['bull_bear_side', 'end']]


def read_multi_timeframe_bull_bear_side_trends(date_range_str: str = None) -> pt.DataFrame[MultiTimeframeBullBearSide]:
    result = read_file(
        date_range_str,
        'multi_timeframe_bull_bear_side_trends',
        generate_multi_timeframe_bull_bear_side_trends,
        MultiTimeframeBullBearSide)
    return result


def read_multi_timeframe_candle_trend(date_range_str: str = None) -> pt.DataFrame[MultiTimeframeCandleTrend]:
    result = read_file(date_range_str, 'multi_timeframe_candle_trend', generate_multi_timeframe_candle_trend,
                       MultiTimeframeCandleTrend)
    return result


@profile_it
def generate_multi_timeframe_bull_bear_side_trends(date_range_str: str = None, file_path: str = None,
                                                   timeframe_shortlist: List['str'] = None):
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    if file_path is None:
        file_path = symbol_data_path()
    multi_timeframe_ohlcva = read_multi_timeframe_ohlcva(date_range_str)

    multi_timeframe_peaks_n_valleys = read_multi_timeframe_peaks_n_valleys(date_range_str)
    multi_timeframe_candle_trend = read_multi_timeframe_candle_trend(date_range_str)
    trends = multi_timeframe_bull_bear_side_trends(multi_timeframe_candle_trend,
                                                   multi_timeframe_peaks_n_valleys,
                                                   multi_timeframe_ohlcva,
                                                   timeframe_shortlist=timeframe_shortlist)
    # Save multi-timeframe trend boundaries to a.zip file
    trends = trends.sort_index(level='date')
    trends.to_csv(os.path.join(file_path, f'multi_timeframe_bull_bear_side_trends.{date_range_str}.zip'),
                  compression='zip')


@profile_it
def generate_multi_timeframe_candle_trend(date_range_str: str, timeframe_shortlist: List['str'] = None,
                                          file_path: str = None):
    if file_path is None:
        file_path = symbol_data_path()
    multi_timeframe_ohlcv = read_multi_timeframe_ohlcv(date_range_str)
    multi_timeframe_peaks_n_valleys = read_multi_timeframe_peaks_n_valleys(date_range_str).sort_index(level='date')
    multi_timeframe_candle_trend = empty_df(MultiTimeframeCandleTrend)
    if timeframe_shortlist is None:
        timeframe_shortlist = app_config.timeframes
    for timeframe in timeframe_shortlist:  # peaks_n_valleys.index.unique(level='timeframe'):
        t_candle_trend = \
            single_timeframe_candles_trend(single_timeframe(multi_timeframe_ohlcv, timeframe),
                                           major_timeframe(multi_timeframe_peaks_n_valleys, timeframe))
        if len(t_candle_trend) > 0:
            t_candle_trend['timeframe'] = timeframe
            t_candle_trend = t_candle_trend.set_index('timeframe', append=True)
            t_candle_trend = t_candle_trend.swaplevel()
            multi_timeframe_candle_trend = concat(multi_timeframe_candle_trend, t_candle_trend)
    multi_timeframe_candle_trend = multi_timeframe_candle_trend.sort_index(level='date')
    multi_timeframe_candle_trend.to_csv(os.path.join(file_path, f'multi_timeframe_candle_trend.{date_range_str}.zip'),
                                        compression='zip')
    return multi_timeframe_candle_trend
