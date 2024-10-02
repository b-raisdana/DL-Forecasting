import os
from typing import Literal

import pandas as pd
import pandera.typing as pt

from Config import config, TopTYPE
from FigurePlotter.plotter import INFINITY_TIME_DELTA
# from MetaTrader import MT
from PanderaDFM.OHLCV import OHLCV
from PanderaDFM.PeakValley import PeakValley, MultiTimeframePeakValley
from helper.data_preparation import read_file, cast_and_validate, trim_to_date_range, \
    expand_date_range, after_under_process_date, empty_df, nearest_match, concat, index_names
from helper.helper import measure_time, date_range
from data_processing.ohlcv import read_base_timeframe_ohlcv


def calculate_strength(peaks_or_valleys: pt.DataFrame[PeakValley], top_type: TopTYPE,
                       ohlcv: pt.DataFrame[OHLCV]) -> pt.DataFrame[PeakValley]:
    if len(peaks_or_valleys) == 0:
        return peaks_or_valleys
    start = ohlcv.index[0]
    end = ohlcv.index[-1]
    peaks_or_valleys = peaks_or_valleys.copy()
    peaks_or_valleys['strength'] = INFINITY_TIME_DELTA
    peaks_or_valleys['start_distance'] = peaks_or_valleys.index - start
    peaks_or_valleys['end_distance'] = end - peaks_or_valleys.index
    peaks_or_valleys = insert_distance(peaks_or_valleys, ohlcv, top_type, direction='right')
    assert not peaks_or_valleys.index.duplicated(False).any()
    peaks_or_valleys = insert_distance(peaks_or_valleys, ohlcv, top_type, direction='left')
    assert not peaks_or_valleys.index.duplicated(False).any()
    peaks_or_valleys['strength'] = peaks_or_valleys[['right_distance', 'left_distance']].min(axis='columns')
    assert not peaks_or_valleys.index.duplicated(False).any()
    # peaks_or_valleys['strength'] = fix_same_value_peaks_or_valleys(peaks_or_valleys, top_type)
    tops_with_unknown_strength = peaks_or_valleys[peaks_or_valleys['strength'].isna()]
    assert len(tops_with_unknown_strength) <= 1
    tops_with_strength = peaks_or_valleys[peaks_or_valleys['strength'].notna()]
    assert not tops_with_strength.index.duplicated(False).any()

    peaks_or_valleys.loc[tops_with_strength.index, 'permanent_strength'] = (
            (tops_with_strength['strength'] < tops_with_strength['start_distance']) &
            (tops_with_strength['strength'] < tops_with_strength['end_distance']))
    peaks_or_valleys['strength'] = peaks_or_valleys[['strength', 'start_distance', 'end_distance']].min(axis='columns')
    return peaks_or_valleys


def insert_distance(base: pt.DataFrame[PeakValley], target: pt.DataFrame[OHLCV], top_type: TopTYPE,
                    direction: Literal['right', 'left']) -> pt.DataFrame[PeakValley]:
    """
    Calculates the distance of OHLCV data points from peaks or valleys in a specified direction.

    Parameters:
    - ohlcv (pt.DataFrame[OHLCV]): DataFrame containing OHLCV data.
    - peaks_or_valleys (pt.DataFrame[PeakValley]): DataFrame containing peak/valley information.
    - top_type (TopTYPE): Enum specifying whether peaks or valleys are considered.
    - direction (Literal['right', 'left']): Direction to calculate distance ('right' for right, 'left' for left).
    - compare_column (str, optional): Column to compare for peak/valley values. Defaults to None.

    Returns:
    - pt.DataFrame[PeakValley]: DataFrame with calculated distances for each peak or valley.

    Columns Added to Returned DataFrame:
    - right_distance or left_distance: Distance of each peak or valley in the specified direction.
    - right_top_time or left_top_time: Time index of the top in the specified direction.
    - right_top_value or left_top_value: Value of the top in the specified direction.
    - right_crossing or left_crossing: Boolean indicating whether OHLCV data is crossing the peak/valley.
    - right_crossing_time or left_crossing_time: Time index where the crossing occurs in the specified direction.
    - right_crossing_value or left_crossing_value: Value of the OHLCV data at the crossing point in the specified direction.
    """
    if top_type == TopTYPE.PEAK:  # todo: test
        base = insert_crossing2(base=base, target=target, direction=direction, base_target_column='high',
                                target_compare_column='high', more_significant=lambda target, base: target > base)
    else:
        base = insert_crossing2(base=base, target=target, direction=direction, base_target_column='low',
                                target_compare_column='low', more_significant=lambda target, base: target < base)
    valid_crossing_times = base[base[f'{direction}_crossing_time'].notna()].index
    base.loc[valid_crossing_times, direction + '_distance'] = \
        abs(pd.to_datetime(valid_crossing_times) - base.loc[valid_crossing_times, f'{direction}_crossing_time'])
    return base


def insert_crossing2(base: pd.DataFrame, target: pd.DataFrame,
                     direction: Literal['right', 'left'],
                     more_significant,
                     target_compare_column,
                     base_target_column: str,
                     # cross_direction: Literal['out', 'in'],
                     # base_type: TopTYPE,
                     ) -> pt.DataFrame[PeakValley]:
    """
    find the fist crossing candle in 'target' which crosses 'base['base_compare_column']'  toward the 'direction'.
    according to 'base_type'=high/low it will consider to find first crossing in/out candle.
    crossing in for Peaks = target < base
    crossing out for Peaks = target > base
    crossing in for Valleys = target > base
    crossing out for Valleys = target < base

    Parameters:
    - target (pd.DataFrame): A single-timeframe DataFrame containing the compared-to data.
    - base (pt.DataFrame[PeakValley]): DataFrame containing peak/valley information.
    - base_type (TopTYPE): Enum specifying whether peaks or valleys are considered.
    - direction (Literal['right', 'left']): Direction to calculate distance ('right' for right, 'left' for left).
    - compare_column (str, optional): Column to compare for peak/valley values. Defaults to None.

    Returns:
    - pt.DataFrame[PeakValley]: DataFrame with calculated distances for each peak or valley.

    Columns Added to Returned DataFrame:
    - right_base_time or left_base_time: Time index of the base in the specified direction.
    - right_base_target or left_base_target: Value of the base in the specified direction.
    - right_crossing or left_crossing: Boolean indicating whether OHLCV data is crossing the peak/valley.
    - right_crossing_time or left_crossing_time: Time index where the crossing occurs in the specified direction.
    - right_crossing_value or left_crossing_value: Value of the OHLCV data at the crossing point in the specified direction.
    """
    if hasattr(target.index, 'names') and 'timeframe' in target.index.names:
        raise ValueError("Expected single-timeframe target but 'timeframe' in target.index.names")
    if direction not in ['right', 'left']:
        raise ValueError(f"direction({direction}) not in ['right', 'left']")
    target = target.copy()
    base = base.copy()
    if f'{direction}_crossing_time' not in base.columns:
        base.loc[:, f'{direction}_crossing_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    if f'{direction}_crossing_value' not in base.columns:
        base.loc[:, f'{direction}_crossing_value'] = pd.Series(dtype=float)
    base.loc[:, 'iloc'] = range(len(base))
    remained_bases = base.copy()
    indexes_backup = index_names(base)
    base.reset_index(inplace=True)
    base.set_index('iloc', inplace=True)
    drop_base_without_crossing2(bases_to_compare=remained_bases, base_target_column=base_target_column, target=target,
                                more_significant=more_significant, direction=direction,
                                target_compare_column=target_compare_column)
    while len(remained_bases) > 0:
        no_repeat_base_indexes = ~remained_bases.index.get_level_values('date').duplicated(keep='first')
        bases_to_compare = remained_bases[no_repeat_base_indexes]
        remained_bases = remained_bases[~no_repeat_base_indexes]
        bases_to_compare['target_index'] = nearest_match(bases_to_compare.index.get_level_values('date'), target.index,
                                                         direction=direction, shift=0)
        if bases_to_compare['target_index'].isna().any():
            raise AssertionError("bases_to_compare['target_index'].isna().any()")  # todo: test
        bases_with_known_crossing_target = empty_df(PeakValley)
        number_of_crossed_bases = 9999999999999
        while number_of_crossed_bases > 0 and len(bases_to_compare) > 0:
            # iteration preparation
            crossed_bases, target = \
                find_crossing_single_iteration(bases_to_compare, base_target_column, target,
                                               target_compare_column, direction, more_significant,
                                               bases_shall_have_crossing=config.check_assertions)
            if config.check_assertions and len(crossed_bases) == 0 and number_of_crossed_bases == 9999999999999:
                raise AssertionError("Did not found any crossing in first iteration")
            number_of_crossed_bases = len(crossed_bases)
            if number_of_crossed_bases > 0:
                # add crossing information to base
                insert_crossing_info(bases_to_compare, crossed_bases, direction, target, target_compare_column)
                if len(bases_with_known_crossing_target) == 0:
                    bases_with_known_crossing_target = bases_to_compare.loc[crossed_bases.index]
                else:
                    bases_with_known_crossing_target = concat(
                        bases_with_known_crossing_target, bases_to_compare.loc[crossed_bases.index])
                bases_to_compare = bases_to_compare.drop(crossed_bases.index)
        if config.check_assertions and len(bases_to_compare) > 0:
            raise AssertionError("Expected to find crossing for all bases after drop_base_without_crossing2...")
        if len(bases_with_known_crossing_target) > 0:
            bases_with_known_crossing_target.set_index('iloc', inplace=True)
            base.loc[
                bases_with_known_crossing_target.index, [f'{direction}_crossing_time', f'{direction}_crossing_value']] = \
                bases_with_known_crossing_target[[f'{direction}_crossing_time', f'{direction}_crossing_value']]
    # if 'original_start' in base.columns:
    #     base.set_index(['date', 'original_start'], inplace=True)
    # else:
    #     base.set_index('date', inplace=True)
    base.set_index(indexes_backup, inplace=True)
    return base


def drop_base_without_crossing(bases_to_compare, target, base_target_column, base_type, cross_direction, direction):
    n = len(target)
    if ((base_type == TopTYPE.PEAK and cross_direction == 'out')
            or (base_type == TopTYPE.VALLEY and cross_direction == 'in')):
        # target_compare_column == 'high':
        if direction == 'left':
            target['max_value'] = target['high'].rolling(window=n, min_periods=0, ).max()
        else:  # direction == 'right':
            target['max_value'] = target['high'].iloc[::-1].rolling(window=n, min_periods=0).max()
        without_crossings = bases_to_compare[
            bases_to_compare[base_target_column] > target.loc[bases_to_compare['target_index'], 'max_value']]
    else:
        # target_compare_column == 'low':
        if direction == 'left':
            target['min_value'] = target['low'].rolling(window=n, min_periods=0).min()
        else:  # direction == 'right':
            target['min_value'] = target['low'].iloc[::-1].rolling(window=n, min_periods=0).min()
        without_crossings = bases_to_compare[
            bases_to_compare[base_target_column] < target.loc[bases_to_compare['target_index'], 'min_value']]
    bases_to_compare.drop(index=without_crossings.index, inplace=True)
    return without_crossings


def drop_base_without_crossing2(bases_to_compare, target, base_target_column, more_significant,
                                direction: Literal['right', 'left'], target_compare_column: Literal['low', 'high']):
    n = len(target)
    if 'target_index' not in bases_to_compare.columns:
        # bases_to_compare['target_index'] = nearest_match(bases_to_compare.index.get_level_values('date'), target.index,
        #                                                  direction='forward', shift=0)
        # if direction == 'left':
        #     pass
        # else:
        #     pass
        bases_to_compare['target_index'] = nearest_match(bases_to_compare.index.get_level_values('date'), target.index,
                                                         direction=direction, shift=0)
    if bases_to_compare['target_index'].isna().any():
        pass
    if more_significant(target=2, base=1):
        if direction == 'left':
            target['max_value'] = target[target_compare_column].rolling(window=n, min_periods=0, ).max()
            target['max_value'] = target['max_value'].shift(1)
        else:  # direction == 'right':
            target['max_value'] = target[target_compare_column].iloc[::-1].rolling(window=n, min_periods=0).max()
            target['max_value'] = target['max_value'].shift(-1)
        bases_to_compare.loc[bases_to_compare['target_index'].notna(), 'target_max_value'] = \
            target.loc[bases_to_compare['target_index'].dropna(), 'max_value'].tolist()
        without_crossings = bases_to_compare[
            bases_to_compare['target_index'].isna() |
            bases_to_compare[base_target_column].isna() |
            bases_to_compare['target_max_value'].isna() |
            (bases_to_compare[base_target_column] >= bases_to_compare['target_max_value'])]
    else:  # more_significant(target=1, base=2)
        if direction == 'left':
            target['min_value'] = target[target_compare_column].rolling(window=n, min_periods=0).min()
            target['min_value'] = target['min_value'].shift(1)
        else:  # direction == 'right':
            target['min_value'] = target[target_compare_column].iloc[::-1].rolling(window=n, min_periods=0).min()
            target['min_value'] = target['min_value'].shift(-1)
        try:
            len(bases_to_compare['target_index'])
            nop = target.loc[bases_to_compare['target_index']]
        except:
            nop = 1
            pass
        bases_to_compare['target_min_value'] = target.loc[bases_to_compare['target_index'], 'min_value'].tolist()
        without_crossings = bases_to_compare[
            bases_to_compare[base_target_column].isna() |
            bases_to_compare['target_min_value'].isna() |
            (bases_to_compare[base_target_column] <= bases_to_compare['target_min_value'])]
    bases_to_compare.drop(index=without_crossings.index, inplace=True)
    return without_crossings


# def insert_crossing_info(base, bases_to_compare, crossed_bases, direction, target):
def insert_crossing_info(base, crossed_bases, direction, target, target_compare_column):
    # base.loc[crossed_bases, f'{direction}_crossing_time'] = \
    #     target.loc[bases_to_compare.loc[crossed_bases, 'target_index'], f'{direction}_crossing_time'].to_list()
    base.loc[crossed_bases.index, f'{direction}_crossing_time'] = crossed_bases['target_date']
    # base.loc[crossed_bases, f'{direction}_crossing_value'] = \
    #     target.loc[bases_to_compare.loc[crossed_bases, 'target_index'], f'{direction}_crossing_value'].to_list()
    base.loc[crossed_bases.index, f'{direction}_crossing_value'] = \
        target.loc[crossed_bases['target_date'].tolist(), target_compare_column].to_list()
    # return base


# pd.set_option('future.no_silent_downcasting', True)


def find_crossing_single_iteration(bases_to_compare, base_compare_column, target, target_compare_column, direction,
                                   more_significant,
                                   return_both: bool = True, bases_shall_have_crossing: bool = False):
    """

    :param bases_shall_have_crossing:
    :param bases_to_compare:
    :param base_compare_column:
    :param target:
    :param target_compare_column:
    :param direction:
    :param more_significant: f(target, base)
    :param return_both:
    :return:
    """
    if direction == 'right':
        reverse = 'left'
    elif direction == 'left':
        reverse = 'right'
    else:
        raise ValueError('Invalid direction')
    target = target.drop(columns=['right_base_index', 'right_base_target', 'left_base_index', 'left_base_target',
                                  'right_crossing', 'left_crossing',
                                  ], errors='ignore')
    bases_to_compare = bases_to_compare.drop(columns=[direction + '_distance'], errors='ignore')

    base_dates = bases_to_compare.index.get_level_values(level='date')
    if config.check_assertions and not base_dates.is_unique:
        raise AssertionError("find_crossings only implemented for unique base_dates!")
    if hasattr(target.index, 'names') and 'timeframe' in target.index.names:
        raise ValueError('timeframe' in target.index.names)
    target_dates = target.index.get_level_values(level='date')
    # find the adjacent target of bases
    # if direction == 'right':
    #     adjacent_target_dates = \
    #         nearest_match(needles=base_dates, reference=target_dates, direction='backward')
    # else:  # direction == 'left'
    #     adjacent_target_dates = \
    #         nearest_match(needles=base_dates, reference=target_dates, direction='forward')
    adjacent_target_dates = \
        nearest_match(needles=base_dates, reference=target_dates, direction=direction)

    bases_to_compare['adjacent_target_date'] = adjacent_target_dates
    bases_with_adjacent_target = bases_to_compare[bases_to_compare['adjacent_target_date'].notna()]
    bases_with_unique_adjacent_target = \
        bases_with_adjacent_target[~(bases_with_adjacent_target['adjacent_target_date'].duplicated(keep='first'))]
    adjacent_target_dates = \
        bases_with_unique_adjacent_target['adjacent_target_date'].tolist()
    if len(adjacent_target_dates) != len(set(adjacent_target_dates)):
        raise ValueError("find_crossings only implemented for unique adjacent_target_index!")
    # mark the adjacent target with base information
    target.loc[adjacent_target_dates, f'{reverse}_base_index'] = bases_with_unique_adjacent_target.index
    target.loc[adjacent_target_dates, reverse + '_base_target'] = \
        bases_with_unique_adjacent_target[base_compare_column].tolist()
    # propagate
    if direction == 'right':
        target['left_base_index'] = target['left_base_index'].ffill()
        target['left_base_target'] = target['left_base_target'].ffill()
    else:  # direction == 'left'
        target['right_base_index'] = target['right_base_index'].bfill()
        target['right_base_target'] = target['right_base_target'].bfill()
    # if high/low of 'target' is higher/lower than 'base' high/low it is crossing
    target[f'{direction}_crossing'] = \
        more_significant(target[target_compare_column], target[reverse + '_base_target'])
    target: pd.DataFrame
    if direction == 'right':
        date_chooser = 'min'
    else:
        date_chooser = 'max'
    target['target_date'] = target.index
    crossed_bases = target[target[f'{direction}_crossing']] \
        .groupby(by=[f'{reverse}_base_index']).agg({'target_date': date_chooser})
    if config.check_assertions and bases_shall_have_crossing:
        if crossed_bases['target_date'].isna().any().any():
            raise AssertionError("crossed_bases['target_date'].isna().any().any()")
    if return_both:
        target.loc[crossed_bases['target_date'], 'crossed_base_index'] = crossed_bases.index
        return crossed_bases, target
    else:
        bases_to_compare.loc[crossed_bases.index, 'target_date'] = crossed_bases['target_date']
        return bases_to_compare['target_date']


# def direction_parameters(direction, top_type, cross_direction: Literal['out', 'in']):
#     """
#     if direction.lower() == 'right':
#         reverse = 'left'
#         compare_column, les_significant, more_significant = top_operators(top_type)
#     elif direction.lower() == 'left':
#         reverse = 'right'
#         compare_column, les_significant, more_significant = top_operators(top_type, equal_is_significant=True)
#
#     top_operators(top_type, equal_is_significant=True):
#             if equal_is_significant:
#         if top_type == TopTYPE.PEAK:
#             compare_column = 'high'
#
#             def more_significant(x, y):
#                 return x >= y
#
#             def les_significant(x, y):
#                 return x <= y
#         else:
#             compare_column = 'low'
#
#             def more_significant(x, y):
#                 return x <= y
#
#             def les_significant(x, y):
#                 return x >= y
#     else:
#         if top_type == TopTYPE.PEAK:
#             compare_column = 'high'
#
#             def more_significant(x, y):
#                 return x > y
#
#             def les_significant(x, y):
#                 return x < y
#         else:
#             compare_column = 'low'
#
#             def more_significant(x, y):
#                 return x < y
#
#             def les_significant(x, y):
#                 return x > y
#     :param cross_direction:
#     :param direction:
#     :param top_type:
#     :return:
#     """
#     direction = direction.lower()
#     if direction.lower() == 'right':
#         reverse = 'left'
#         compare_column, les_significant, more_significant = top_operators(top_type, cross_direction)
#     elif direction.lower() == 'left':
#         reverse = 'right'
#         compare_column, les_significant, more_significant = \
#             top_operators(top_type, cross_direction, equal_is_significant=True)
#
#     else:
#         raise Exception(f'Invalid direction: {direction} only right and left are supported.')
#     return compare_column, direction, les_significant, more_significant, reverse


# def top_operators(top_type, cross_direction: Literal['out', 'in'] = 'out', equal_is_significant: bool = False):
#     if equal_is_significant:
#         if top_type == TopTYPE.PEAK:
#             compare_column = 'high'
#
#             def gt(x, y):
#                 return x >= y
#
#             def lt(x, y):
#                 return x <= y
#         else:
#             compare_column = 'low'
#
#             def gt(x, y):
#                 return x <= y
#
#             def lt(x, y):
#                 return x >= y
#     else:
#         if top_type == TopTYPE.PEAK:
#             compare_column = 'high'
#
#             def gt(x, y):
#                 return x > y
#
#             def lt(x, y):
#                 return x < y
#         else:
#             compare_column = 'low'
#
#             def gt(x, y):
#                 return x < y
#
#             def lt(x, y):
#                 return x > y
#     if cross_direction == 'out':
#         more_significant = gt
#         les_significant = lt
#     elif cross_direction == 'in':
#         if top_type == TopTYPE.PEAK:
#             compare_column = 'low'
#         else:
#             compare_column = 'high'
#         more_significant = lt
#         les_significant = gt
#     else:
#         raise ValueError(f"invalid cross_direction:{cross_direction}")
#     return compare_column, les_significant, more_significant
#

def map_strength_to_frequency(peaks_valleys: pd.DataFrame) -> pt.DataFrame[PeakValley]:
    # peaks_valleys.insert(len(peaks_valleys.columns), 'timeframe', None)
    peaks_valleys['timeframe'] = None

    for i in range(len(config.timeframes)):
        for t_peak_valley_index in peaks_valleys[
            peaks_valleys['strength'] > pd.to_timedelta(config.timeframes[i])
        ].index.values:
            peaks_valleys.at[t_peak_valley_index, 'timeframe'] = config.timeframes[i]
    peaks_valleys = peaks_valleys[pd.notna(peaks_valleys['timeframe'])]
    return peaks_valleys


def peaks_only(peaks_n_valleys: pt.DataFrame[PeakValley]) -> pt.DataFrame[PeakValley]:
    """
        Filter peaks from the DataFrame containing peaks and valleys data.

        Parameters:
            peaks_n_valleys (pd.DataFrame): DataFrame containing peaks and valleys data.

        Returns:
            pd.DataFrame: DataFrame containing only peaks data.
    """
    return peaks_n_valleys[peaks_n_valleys['peak_or_valley'] == TopTYPE.PEAK.value]


def valleys_only(peaks_n_valleys: pd.DataFrame) -> pt.DataFrame[PeakValley]:
    """
        Filter valleys from the DataFrame containing peaks and valleys data.

        Parameters:
            peaks_n_valleys (pd.DataFrame): DataFrame containing peaks and valleys data.

        Returns:
            pd.DataFrame: DataFrame containing only valleys data.
    """
    return peaks_n_valleys[peaks_n_valleys['peak_or_valley'] == TopTYPE.VALLEY.value]


def merge_tops(peaks: pd.DataFrame, valleys: pd.DataFrame) -> pd.DataFrame:
    return concat(peaks, valleys).sort_index(level='date')


def find_peaks_n_valleys(base_ohlcv: pd.DataFrame,
                         sort_index: bool = True) -> pt.DataFrame[PeakValley]:  # , max_cycles=100):
    mask_of_sequence_of_same_value = (base_ohlcv['high'] == base_ohlcv['high'].shift(1))
    list_of_same_high_lows_sequence = base_ohlcv.loc[mask_of_sequence_of_same_value].index

    none_repeating_ohlcv = base_ohlcv.drop(list_of_same_high_lows_sequence)
    mask_of_peaks = (none_repeating_ohlcv['high'] > none_repeating_ohlcv['high'].shift(1)) & (
            none_repeating_ohlcv['high'] > none_repeating_ohlcv['high'].shift(-1))
    _peaks = none_repeating_ohlcv.loc[mask_of_peaks].copy()
    _peaks['peak_or_valley'] = TopTYPE.PEAK.value

    mask_of_sequence_of_same_value = (base_ohlcv['low'] == base_ohlcv['low'].shift(1))
    list_of_same_high_lows_sequence = base_ohlcv.loc[mask_of_sequence_of_same_value].index
    none_repeating_ohlcv = base_ohlcv.drop(list_of_same_high_lows_sequence)

    mask_of_valleys = (none_repeating_ohlcv['low'] < none_repeating_ohlcv['low'].shift(1)) & (
            none_repeating_ohlcv['low'] < none_repeating_ohlcv['low'].shift(-1))
    _valleys = none_repeating_ohlcv.loc[mask_of_valleys].copy()
    _valleys['peak_or_valley'] = TopTYPE.VALLEY.value
    _peaks_n_valleys: pt.DataFrame[PeakValley] = concat(_peaks, _valleys)
    _peaks_n_valleys = _peaks_n_valleys.loc[:, ['open', 'high', 'low', 'close', 'volume', 'peak_or_valley']]
    return _peaks_n_valleys.sort_index(level='date') if sort_index else _peaks_n_valleys


@measure_time
def major_timeframe(multi_timeframe_df: pd.DataFrame, timeframe: str) \
        -> pt.DataFrame[MultiTimeframePeakValley]:
    """
    Filter rows from multi_timeframe_peaks_n_valleys with a timeframe equal to or greater than the specified timeframe.

    Parameters:
        multi_timeframe_df (pd.DataFrame): DataFrame containing peaks and valleys data with 'timeframe' index.
        timeframe (str): The timeframe to filter rows.

    Returns:
        pd.DataFrame: DataFrame containing rows with timeframe equal to or greater than the specified timeframe.
    """
    result = higher_or_eq_timeframe(multi_timeframe_df, timeframe)
    return result


def higher_or_eq_timeframe(multi_timeframe_df: pd.DataFrame, timeframe: str):
    try:
        index = config.timeframes.index(timeframe)
    except ValueError:
        raise Exception(f'timeframe:{timeframe} should be in [{config.timeframes}]!')
    # result = peaks_n_valleys.loc[peaks_n_valleys.index.isin(config.timeframes[index:], level='timeframe')]
    result = multi_timeframe_df.loc[multi_timeframe_df.index.get_level_values('timeframe').isin(config.timeframes[index:])]
    return result


def top_timeframe(tops: pt.DataFrame[PeakValley]) -> pt.DataFrame[PeakValley]:
    """
    _peaks_n_valleys['timeframe'] = [strength_to_timeframe(row['strength']) for index, row in
                                     _peaks_n_valleys.iterrows()]
    :param tops:
    :return:
    """
    tops['timeframe'] = config.timeframes[0]
    for _, timeframe in enumerate(config.timeframes[1:]):
        eq_or_higher_timeframe = (tops['strength'] >= pd.to_timedelta(timeframe).total_seconds() * 2)
        tops.loc[eq_or_higher_timeframe, 'timeframe'] = timeframe
        if eq_or_higher_timeframe.any() == 0:
            break
    return tops


def multi_timeframe_peaks_n_valleys(expanded_date_range: str) -> pt.DataFrame[MultiTimeframePeakValley]:
    base_ohlcv = read_base_timeframe_ohlcv(expanded_date_range)

    _peaks_n_valleys = find_peaks_n_valleys(base_ohlcv, sort_index=False)

    _peaks_n_valleys = calculate_strength_of_peaks_n_valleys(base_ohlcv, _peaks_n_valleys)

    _peaks_n_valleys = top_timeframe(_peaks_n_valleys)

    _peaks_n_valleys = _peaks_n_valleys.set_index('timeframe', append=True, )
    _peaks_n_valleys = _peaks_n_valleys.swaplevel()
    _peaks_n_valleys = _peaks_n_valleys.sort_index(level='date')

    _peaks_n_valleys = (
        cast_and_validate(_peaks_n_valleys, MultiTimeframePeakValley,
                          zero_size_allowed=after_under_process_date(expanded_date_range)))
    return _peaks_n_valleys


# @measure_time
def generate_multi_timeframe_peaks_n_valleys(date_range_str, file_path: str = None):
    if file_path is None:
        file_path = config.path_of_data
    biggest_timeframe = config.timeframes[-1]
    expanded_date_range = expand_date_range(date_range_str,
                                            time_delta=4 * pd.to_timedelta(biggest_timeframe),
                                            mode='both')
    _peaks_n_valleys = multi_timeframe_peaks_n_valleys(expanded_date_range)
    start, end = date_range(date_range_str)
    _peaks_n_valleys = _peaks_n_valleys.loc[
        (start < _peaks_n_valleys.index.get_level_values(level='date')) &
        (_peaks_n_valleys.index.get_level_values(level='date') < end)].copy()
    # plot_multi_timeframe_peaks_n_valleys(_peaks_n_valleys, date_range_str)
    _peaks_n_valleys = _peaks_n_valleys.sort_index(level='date')
    _peaks_n_valleys = trim_to_date_range(date_range_str, _peaks_n_valleys, ignore_duplicate_index=True)
    _peaks_n_valleys = \
        _peaks_n_valleys[['open', 'high', 'low', 'close', 'volume', 'peak_or_valley', 'strength', 'permanent_strength']]
    _peaks_n_valleys.to_csv(os.path.join(file_path, f'multi_timeframe_peaks_n_valleys.{date_range_str}.zip'),
                            compression='zip')
    # MT.extract_to_data_path(os.path.join(file_path, f'multi_timeframe_peaks_n_valleys.{date_range_str}.zip'))


# @measure_time
def calculate_strength_of_peaks_n_valleys(time_ohlcv: pt.DataFrame[OHLCV],
                                          time_peaks_n_valleys: pt.DataFrame[PeakValley]) \
        -> pt.DataFrame[PeakValley]:
    peaks = calculate_strength(peaks_only(time_peaks_n_valleys), TopTYPE.PEAK, time_ohlcv)
    valleys = calculate_strength(valleys_only(time_peaks_n_valleys), TopTYPE.VALLEY, time_ohlcv)
    peaks_and_valleys = concat(peaks, valleys).sort_index(level='date')
    peaks_and_valleys['strength'] = peaks_and_valleys['strength'].dt.total_seconds()
    return peaks_and_valleys


def read_multi_timeframe_peaks_n_valleys(date_range_str: str = None) \
        -> pt.DataFrame[MultiTimeframePeakValley]:
    result = read_file(date_range_str, 'multi_timeframe_peaks_n_valleys',
                       generate_multi_timeframe_peaks_n_valleys,
                       MultiTimeframePeakValley)
    return result


def old_insert_previous_n_next_top(top_type: TopTYPE, peaks_n_valleys, ohlcv):
    if f'previous_top_index' not in ohlcv.columns:
        ohlcv[f'previous_top_index'] = None
    if f'previous_top_value' not in ohlcv.columns:
        ohlcv[f'previous_top_value'] = None
    if f'next_top_index' not in ohlcv.columns:
        ohlcv[f'next_top_index'] = None
    if f'next_top_value' not in ohlcv.columns:
        ohlcv[f'next_top_value'] = None
    tops = peaks_n_valleys[peaks_n_valleys['peak_or_valley'] == top_type.value]
    high_or_low = 'high' if top_type == TopTYPE.PEAK else 'low'
    for i in range(len(tops)):
        if i == len(tops) - 1:
            is_previous_for_indexes = ohlcv.loc[ohlcv.index > tops.index.get_level_values('date')[i]].index
        else:
            is_previous_for_indexes = ohlcv.loc[(ohlcv.index > tops.index.get_level_values('date')[i]) &
                                                (ohlcv.index <= tops.index.get_level_values('date')[i + 1])].index
        if i == 0:
            is_next_for_indexes = ohlcv.loc[ohlcv.index <= tops.index.get_level_values('date')[i]].index
        else:
            is_next_for_indexes = ohlcv.loc[(ohlcv.index > tops.index.get_level_values('date')[i - 1]) &
                                            (ohlcv.index <= tops.index.get_level_values('date')[i])].index
        ohlcv.loc[is_previous_for_indexes, f'previous_top_index'] = tops.index.get_level_values('date')[i]
        ohlcv.loc[is_previous_for_indexes, f'previous_top_value'] = tops.iloc[i][high_or_low]
        ohlcv.loc[is_next_for_indexes, f'next_top_index'] = tops.index.get_level_values('date')[i]
        ohlcv.loc[is_next_for_indexes, f'next_top_value'] = tops.iloc[i][high_or_low]
    return ohlcv


@measure_time
def insert_previous_n_next_top(top_type, peaks_n_valleys: pt.DataFrame[PeakValley], ohlcv: pt.DataFrame[OHLCV]) \
        -> pt.DataFrame[OHLCV]:
    # Define columns
    prev_index_col = f'previous_{top_type.value}_index'
    prev_value_col = f'previous_{top_type.value}_value'
    next_index_col = f'next_{top_type.value}_index'
    next_value_col = f'next_{top_type.value}_value'
    # Filter the relevant tops
    tops = peaks_n_valleys[peaks_n_valleys['peak_or_valley'] == top_type.value].copy()
    high_or_low = 'high' if top_type == TopTYPE.PEAK else 'low'
    # Using `shift()` to create the previous and next columns
    tops[prev_index_col] = tops.index.get_level_values('date').unique().to_series().shift(1).tolist()
    tops[prev_value_col] = tops[high_or_low].shift(-1)
    previous_tops = pd.DataFrame(index=tops.index.get_level_values('date').shift(1, config.timeframes[0])).sort_index()
    previous_tops[prev_index_col] = tops.index.get_level_values('date').unique().tolist()
    previous_tops[prev_value_col] = tops[high_or_low].tolist()
    # merge the previous and next values
    ohlcv = pd.merge_asof(ohlcv.sort_index(), previous_tops,
                          left_index=True, right_index=True, direction='backward', suffixes=('_x', ''))
    tops[next_index_col] = tops.index.get_level_values('date').unique().to_series().shift(-1).tolist()
    tops[next_value_col] = tops[high_or_low].shift(1)
    next_tops = pd.DataFrame(index=tops.index.get_level_values('date').shift(-1, config.timeframes[0])).sort_index()
    next_tops[next_index_col] = tops.index.get_level_values('date').unique().tolist()
    next_tops[next_value_col] = tops[high_or_low].tolist()
    # merge the previous and next values
    ohlcv = pd.merge_asof(ohlcv.sort_index(), next_tops,
                          left_index=True, right_index=True, direction='forward', suffixes=('_x', ''))
    # Cleaning any duplicate columns
    for col in ohlcv.columns:
        if col.endswith('_x') or col.endswith('_y'):
            raise Exception(f'Duplicate merge on same column:{col}')

    return ohlcv
