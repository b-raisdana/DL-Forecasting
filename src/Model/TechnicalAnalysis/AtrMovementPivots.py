import os
from typing import List, Annotated

import pandas as pd
from pandera import typing as pt

from ClassicPivot import insert_pivot_info, update_pivot_deactivation, insert_pivot_type_n_level
from Config import config, TopTYPE
from PanderaDFM.AtrTopPivot import MultiTimeframeAtrMovementPivotDFM, MultiTimeframeAtrMovementPivotDf, \
    AtrMovementPivotDf
from PanderaDFM.OHLCV import OHLCV
from PanderaDFM.OHLCVA import OHLCVA
from PanderaDFM.PeakValley import MultiTimeframePeakValley, PeakValley
from PeakValley import read_multi_timeframe_peaks_n_valleys, peaks_only, valleys_only, insert_crossing2, \
    major_timeframe
from data_processing.atr import read_multi_timeframe_ohlcva
from Model.TechnicalAnalysis.ftc import insert_multi_timeframe_pivots_real_start
from data_processing.fragmented_data import symbol_data_path
from helper.data_preparation import to_timeframe, single_timeframe, pattern_timeframe, trigger_timeframe
from helper.helper import profile_it, date_range, date_range_to_string


@profile_it
def find_multi_timeframe_atr_movement_pivots(mt_ohlcva, base_timeframe_ohlcva, mt_tops, structure_timeframe_shortlist,
                                             same_time_multiple_timeframes):
    all_mt_tops = mt_tops.copy()
    pivots = MultiTimeframeAtrMovementPivotDf.new()
    for timeframe in structure_timeframe_shortlist[::-1]:
        if timeframe in mt_tops.index.get_level_values('timeframe'):
            timeframe_tops: pt.DataFrame[PeakValley] = major_timeframe(mt_tops, timeframe).copy()
            timeframe_tops.reset_index(level='timeframe', inplace=True)
            ohlcva: pt.DataFrame[OHLCVA] = single_timeframe(mt_ohlcva, timeframe)
            timeframe_tops = pd.merge_asof(timeframe_tops, ohlcva[['atr']], left_index=True, right_index=True,
                                           direction='backward')
            insert_pivot_movements(timeframe_tops, base_timeframe_ohlcva)

            movement_satisfied_tops = timeframe_tops[
                timeframe_tops['movement_start_time'].notna() & timeframe_tops['return_end_time'].notna()].copy()
            if len(movement_satisfied_tops) > 0:
                insert_more_significant_tops(movement_satisfied_tops, all_mt_tops)

                timeframe_pivots = movement_satisfied_tops[
                    (movement_satisfied_tops['left_more_significant_top'].isna() |
                     (movement_satisfied_tops['left_more_significant_top'] <= movement_satisfied_tops[
                         'movement_start_time']))
                    &
                    (movement_satisfied_tops['right_more_significant_top'].isna() |
                     (movement_satisfied_tops['right_more_significant_top'] >= movement_satisfied_tops[
                         'return_end_time']))
                    ].copy()

                timeframe_pivots['timeframe'] = timeframe
                timeframe_pivots.set_index('timeframe', inplace=True, append=True)
                timeframe_pivots = timeframe_pivots.swaplevel()
                pivots = MultiTimeframeAtrMovementPivotDf.concat(pivots, timeframe_pivots)
                if config.check_assertions and not ((len(mt_tops.drop(
                        index=mt_tops[mt_tops.index.get_level_values('date') \
                                .isin(timeframe_pivots.index.get_level_values('date'))].index)) + len(timeframe_pivots))
                                                    == len(mt_tops)):
                    AssertionError("not ((len(mt_tops.drop(...")
                if not same_time_multiple_timeframes:
                    mt_tops.drop(
                        index=mt_tops[mt_tops.index.get_level_values('date') \
                            .isin(timeframe_pivots.index.get_level_values('date'))].index, inplace=True)
    return pivots


def generate_multi_timeframe_atr_movement_pivots(date_range_str: str = None, file_path: str = None,
                                                 timeframe_shortlist: List['str'] = None):
    if file_path is None:
        file_path = symbol_data_path()
    if date_range_str is None:
        date_range_str = config.processing_date_range
    if timeframe_shortlist is None:
        timeframe_shortlist = config.structure_timeframes
    else:
        # if any([t in timeframe_shortlist for t in config.timeframes[-2:]]):
        if any([t not in config.structure_timeframes for t in timeframe_shortlist]):
            raise Exception(f"timeframes {timeframe_shortlist} should be in structure_timeframes:"
                            f"{config.structure_timeframes}!")
    t_atr_movement_pivots = atr_movement_pivots(date_range_str=date_range_str,
                                                structure_timeframe_shortlist=timeframe_shortlist)
    t_atr_movement_pivots.to_csv(os.path.join(file_path, f'multi_timeframe_atr_movement_pivots.{date_range_str}.zip'),
                                 compression='zip')


def read_multi_timeframe_atr_movement_pivots(date_range_str: str = None) \
        -> pt.DataFrame[MultiTimeframeAtrMovementPivotDFM]:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    result = MultiTimeframeAtrMovementPivotDf.read_file(date_range_str, 'multi_timeframe_atr_movement_pivots',
                                                        generate_multi_timeframe_atr_movement_pivots)
    return result


@profile_it
def atr_movement_pivots(date_range_str: str = None, structure_timeframe_shortlist: List['str'] = None,
                        same_time_multiple_timeframes: bool = True) -> pt.DataFrame[MultiTimeframeAtrMovementPivotDFM]:
    """
    for peaks:
        (
            (find first candle before far >= 3*ATR:
                if there is not a peak between these 2)
                and
            (find first candle after far >= 1*ATR:
                if there is not a peak between these 2)
        ):
            the peak is a pivot
    :param same_time_multiple_timeframes:
    :param date_range_str:
    :param structure_timeframe_shortlist:
    :return:
    """
    if structure_timeframe_shortlist is None:
        structure_timeframe_shortlist = config.structure_timeframes
    if date_range_str is None:
        date_range_str = config.processing_date_range
    mt_tops: pt.DataFrame[MultiTimeframePeakValley] = read_multi_timeframe_peaks_n_valleys(date_range_str)
    start, end = date_range(date_range_str)
    mt_tops_mapped_time = [to_timeframe(date, timeframe, ignore_cached_times=True, do_not_warn=True)
                           for (timeframe, date), top_data in mt_tops.iterrows()]
    expanded_atr_start = min(mt_tops_mapped_time)
    expanded_atr_date_range_str = date_range_to_string(start=min(start, expanded_atr_start), end=end)
    mt_ohlcva = read_multi_timeframe_ohlcva(expanded_atr_date_range_str)
    base_timeframe_ohlcva = single_timeframe(mt_ohlcva, config.timeframes[0])
    peaks_and_valleys = read_multi_timeframe_peaks_n_valleys(date_range_str)
    # filter to tops of structure timeframes
    mt_tops = mt_tops[mt_tops.index.get_level_values('timeframe').isin(structure_timeframe_shortlist)]
    pivots = find_multi_timeframe_atr_movement_pivots(mt_ohlcva, base_timeframe_ohlcva, mt_tops,
                                                      structure_timeframe_shortlist, same_time_multiple_timeframes)
    insert_major_timeframe(pivots, structure_timeframe_shortlist)
    insert_pivot_type_n_level(pivots)
    insert_multi_timeframe_pivots_real_start(pivots, peaks_and_valleys)
    final_pivots = MultiTimeframeAtrMovementPivotDf.new()
    for timeframe in structure_timeframe_shortlist[::-1]:
        if timeframe in pivots.index.get_level_values(level='timeframe').unique():
            ohlcva: pt.DataFrame[OHLCVA] = single_timeframe(mt_ohlcva, timeframe)
            pattern_ohlcva: pt.DataFrame[OHLCVA] = single_timeframe(mt_ohlcva, pattern_timeframe(timeframe))
            trigger_ohlcva: pt.DataFrame[OHLCVA] = single_timeframe(mt_ohlcva, trigger_timeframe(timeframe))
            timeframe_pivots = single_timeframe(pivots, timeframe)
            timeframe_pivots = insert_pivot_info(timeframe_pivots, ohlcva, timeframe, pattern_ohlcva, trigger_ohlcva)
            timeframe_pivots = update_pivot_deactivation(timeframe_pivots, timeframe, ohlcva)
            timeframe_pivots = AtrMovementPivotDf.cast_and_validate(timeframe_pivots)
            # pivots_to_change_timeframe = timeframe_pivots[timeframe_pivots['timeframe_invalid'].astype(bool)].copy() # todo: test
            # pivots_to_change_timeframe['timeframe'] = timeframe
            # pivots_to_change_timeframe = \
            #     pivots_to_change_timeframe.reset_index().set_index(['timeframe', 'date', 'original_start'])
            # pivots.lov[pivots_to_change_timeframe.index, 'timeframe'] = pattern_timeframe(timeframe)
            # timeframe_pivots = timeframe_pivots[~timeframe_pivots['timeframe_invalid'].astype(bool)]
            timeframe_pivots['timeframe'] = timeframe
            timeframe_pivots = timeframe_pivots.reset_index().set_index(['timeframe', 'date', 'original_start'])
            final_pivots = MultiTimeframeAtrMovementPivotDf.concat(final_pivots, timeframe_pivots)
    final_pivots = MultiTimeframeAtrMovementPivotDf.cast_and_validate(final_pivots)
    # pivots = insert_ftc(pivots, mt_ohlcva)
    return final_pivots


@profile_it
def insert_major_timeframe(pivots, structure_timeframe_shortlist):
    for timeframe in structure_timeframe_shortlist[::-1][:-1]:
        timeframe_pivots = single_timeframe(pivots, timeframe)
        same_time_pattern_timeframe_pivots = pivots[
            pivots.index.get_level_values(level='date').isin(timeframe_pivots.index.get_level_values(level='date'))
            & (pivots.index.get_level_values(level='timeframe') == pattern_timeframe(timeframe))
            ]
        if config.check_assertions and len(timeframe_pivots) != len(same_time_pattern_timeframe_pivots):
            raise AssertionError("Expected every pivot overlaps with a pivot in pattern time. "
                                 "len(timeframe_pivots) != len(same_time_pattern_timeframe_pivots)")
    pivots['timeframe_timedelta'] = pd.to_timedelta(pivots.index.get_level_values(level='timeframe'))
    unique_date_pivots = pivots.groupby(level='date')['timeframe_timedelta'].idxmax().tolist()
    pivots['major_timeframe'] = False
    pivots.loc[unique_date_pivots, 'major_timeframe'] = True
    if config.check_assertions and len(pivots[pivots['major_timeframe']]) != len(
            pivots.index.get_level_values(level='date').unique()):
        raise AssertionError(
            "For each date/time which is a pivot expected to have one and only one major-timeframe pivot."
            "len(pivots[pivots['major_timeframe'] == True]) != pivots.index.get_level_values(level='date').unique()")


@profile_it
def insert_pivot_movements(timeframe_tops: pt.DataFrame[PeakValley],
                           base_ohlcv: pt.DataFrame[OHLCV]) -> pt.DataFrame[PeakValley]:
    """

    :param timeframe_tops:
    :param base_ohlcv:
    :return:
    adds:
    "left_crossing_time" = time of first candle before with 3 * ATR adds
    "right_crossing_time" = time of first candle before with 1 * ATR -
    'right_crossing_value', 'left_crossing_value'
    """
    if 'right_crossing_time' not in timeframe_tops.columns:
        timeframe_tops['right_crossing_time'] = pd.Series(Annotated[pd.DatetimeTZDtype, "ns", "UTC"])
    if 'right_crossing_value' not in timeframe_tops.columns:
        timeframe_tops['right_crossing_value'] = pd.Series(float)
    if 'left_crossing_time' not in timeframe_tops.columns:
        timeframe_tops['left_crossing_time'] = pd.Series(Annotated[pd.DatetimeTZDtype, "ns", "UTC"])
    if 'left_crossing_value' not in timeframe_tops.columns:
        timeframe_tops['left_crossing_value'] = pd.Series(float)

    peaks = peaks_only(timeframe_tops).copy()
    peaks_indexes = peaks.index
    timeframe_tops.loc[peaks_indexes, ['right_crossing_time', 'left_crossing_time',
                                       'right_crossing_value', 'left_crossing_value']] = \
        insert_pivot_movement_times(peaks, base_ohlcv, top_type=TopTYPE.PEAK
                                    )[['right_crossing_time', 'left_crossing_time',
                                       'right_crossing_value', 'left_crossing_value']]

    valleys = valleys_only(timeframe_tops).copy()
    valleys_indexes = valleys.index
    timeframe_tops.loc[valleys_indexes, ['right_crossing_time', 'left_crossing_time',
                                         'right_crossing_value', 'left_crossing_value']] = \
        insert_pivot_movement_times(valleys, base_ohlcv, top_type=TopTYPE.VALLEY
                                    )[['right_crossing_time', 'left_crossing_time',
                                       'right_crossing_value', 'left_crossing_value']]
    timeframe_tops['return_end_time'] = timeframe_tops['right_crossing_time'].astype('datetime64[ns, UTC]')
    timeframe_tops['movement_start_time'] = timeframe_tops['left_crossing_time'].astype('datetime64[ns, UTC]')
    timeframe_tops['return_end_value'] = timeframe_tops['right_crossing_value'].astype(float)
    timeframe_tops['movement_start_value'] = timeframe_tops['left_crossing_value'].astype(float)
    timeframe_tops.drop(columns=['right_crossing_time', 'left_crossing_time',
                                 'right_crossing_value', 'left_crossing_value'], inplace=True)
    return timeframe_tops


@profile_it
def insert_pivot_movement_times(timeframe_peak_or_valleys: pt.DataFrame[PeakValley], base_ohlcv: pt.DataFrame[OHLCV],
                                top_type: TopTYPE) -> pt.DataFrame[MultiTimeframePeakValley]:
    """
    find the first candle before with 3 * ATR distance and first candle next with 1 * ATR distance.
    :param base_ohlcv:
    :param timeframe_peak_or_valleys:
    :param top_type:
    :return:
    - right_crossing_time and left_crossing_time: Time index where the crossing occurs in the specified direction.
    - right_crossing_value and left_crossing_value: Value of the OHLCV data at the crossing point in the specified direction.
    """

    if top_type == TopTYPE.PEAK:
        high_low = 'high'

        def aggregate(high, n_atr):
            return high - n_atr
    else:  # top_type == TopTYPE.VALLEY
        high_low = 'low'

        def aggregate(low, n_atr):
            return low + n_atr

    peak_or_valley_indexes = \
        timeframe_peak_or_valleys[timeframe_peak_or_valleys['peak_or_valley'] == top_type.value].index
    peak_or_valleys = timeframe_peak_or_valleys.loc[peak_or_valley_indexes]

    peak_or_valleys['previous_target'] = \
        aggregate(peak_or_valleys[high_low], 3 * timeframe_peak_or_valleys.loc[peak_or_valley_indexes, 'atr'])
    if top_type == TopTYPE.PEAK:
        timeframe_peak_or_valleys.loc[peak_or_valley_indexes, ['left_crossing_time', 'left_crossing_value']] = \
            insert_crossing2(base=peak_or_valleys, base_target_column='previous_target', direction='left',
                             target=base_ohlcv, target_compare_column='low',
                             more_significant=lambda target, base: target < base,
                             )[['left_crossing_time', 'left_crossing_value']]
    else:  # top_type == TopTYPE.VALLEY
        timeframe_peak_or_valleys.loc[peak_or_valley_indexes, ['left_crossing_time', 'left_crossing_value']] = \
            insert_crossing2(base=peak_or_valleys, base_target_column='previous_target', direction='left',
                             target=base_ohlcv, target_compare_column='high',
                             more_significant=lambda target, base: target > base,
                             )[['left_crossing_time', 'left_crossing_value']]
    peak_or_valleys['next_target'] = aggregate(peak_or_valleys[high_low], peak_or_valleys['atr'])
    if top_type == TopTYPE.PEAK:
        timeframe_peak_or_valleys.loc[peak_or_valley_indexes, ['right_crossing_time', 'right_crossing_value']] = \
            insert_crossing2(base=peak_or_valleys, base_target_column='next_target', direction='right',
                             target=base_ohlcv, target_compare_column='low',
                             more_significant=lambda target, base: target < base,
                             )[['right_crossing_time', 'right_crossing_value']]
    else:  # top_type == TopTYPE.VALLEY
        timeframe_peak_or_valleys.loc[peak_or_valley_indexes, ['right_crossing_time', 'right_crossing_value']] = \
            insert_crossing2(base=peak_or_valleys, base_target_column='next_target', direction='right',
                             target=base_ohlcv, target_compare_column='high',
                             more_significant=lambda target, base: target > base,
                             )[['right_crossing_time', 'right_crossing_value']]

    return timeframe_peak_or_valleys


@profile_it
def insert_more_significant_tops(timeframe_tops: pt.DataFrame[PeakValley],
                                 compared_tops: pt.DataFrame[MultiTimeframePeakValley]) \
        -> pt.DataFrame[MultiTimeframePeakValley]:
    timeframe_tops.drop(
        columns=['right_crossing_time', 'left_crossing_time', 'right_crossing_value', 'left_crossing_value'],
        inplace=True, errors='ignore')
    insert_more_significant_top(timeframe_tops, compared_tops, TopTYPE.PEAK)
    insert_more_significant_top(timeframe_tops, compared_tops, TopTYPE.VALLEY)
    timeframe_tops.rename(columns={
        'right_crossing_time': 'right_more_significant_top',
        'left_crossing_time': 'left_more_significant_top',
        'right_crossing_value': 'right_more_significant_top_value',
        'left_crossing_value': 'left_more_significant_top_value',
    }, inplace=True)
    return timeframe_tops


def insert_more_significant_top(base_tops: pt.DataFrame[MultiTimeframePeakValley],
                                target_tops: pt.DataFrame[MultiTimeframePeakValley], top_type: TopTYPE) \
        -> pt.DataFrame[MultiTimeframePeakValley]:
    if top_type == TopTYPE.PEAK:
        high_low = 'high'
    else:  # top_type == TopTYPE.VALLEY
        high_low = 'low'

    no_timeframe_tops = target_tops.reset_index(level='timeframe')
    for direction in ['left', 'right']:
        if f'{direction}_crossing_time' not in base_tops.columns:
            base_tops[f'{direction}_crossing_time'] = pd.Series(dtype='datetime64[ns, UTC]')
        if f'{direction}_crossing_value' not in base_tops.columns:
            base_tops[f'{direction}_crossing_value'] = pd.Series(dtype=float)

    peak_or_valley_indexes = base_tops[base_tops['peak_or_valley'] == top_type.value].index
    base_tops.loc[peak_or_valley_indexes, 'target'] = base_tops.loc[peak_or_valley_indexes, high_low]
    if top_type == TopTYPE.PEAK:
        base_tops.loc[peak_or_valley_indexes, ['left_crossing_time', 'left_crossing_value']] = \
            insert_crossing2(base=base_tops.loc[peak_or_valley_indexes], target=no_timeframe_tops, direction='left',
                             target_compare_column='high',
                             base_target_column='target', more_significant=lambda target, base: target > base
                             )[['left_crossing_time', 'left_crossing_value']]
        base_tops.loc[peak_or_valley_indexes, ['right_crossing_time', 'right_crossing_value']] = \
            insert_crossing2(base=base_tops.loc[peak_or_valley_indexes], target=no_timeframe_tops, direction='right',
                             target_compare_column='high',
                             base_target_column='target', more_significant=lambda target, base: target > base
                             )[['right_crossing_time', 'right_crossing_value']]
    else:
        base_tops.loc[peak_or_valley_indexes, ['left_crossing_time', 'left_crossing_value']] = \
            insert_crossing2(base=base_tops.loc[peak_or_valley_indexes], target=no_timeframe_tops, direction='left',
                             target_compare_column='low',
                             base_target_column='target', more_significant=lambda target, base: target < base
                             )[['left_crossing_time', 'left_crossing_value']]
        base_tops.loc[peak_or_valley_indexes, ['right_crossing_time', 'right_crossing_value']] = \
            insert_crossing2(base=base_tops.loc[peak_or_valley_indexes], target=no_timeframe_tops, direction='right',
                             target_compare_column='low',
                             base_target_column='target', more_significant=lambda target, base: target < base
                             )[['right_crossing_time', 'right_crossing_value']]
    return base_tops
