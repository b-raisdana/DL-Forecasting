from typing import List

import pandas as pd
from pandera import typing as pt

from Config import config, TopTYPE
from PanderaDFM.BasePattern import MultiTimeframeBasePattern
from PanderaDFM.BullBearSide import BullBearSide
from PanderaDFM.PeakValley import PeakValley, MultiTimeframePeakValley
from PanderaDFM.Pivot import PivotDFM
from PanderaDFM.Pivot2 import MultiTimeframePivot2DFM, Pivot2DFM
from Model.TechnicalAnalysis.PeakValley import insert_crossing2, peaks_only, valleys_only, major_timeframe
from helper.data_preparation import single_timeframe


def merge_bbs_overlap(time_frame_bbs: pt.DataFrame[BullBearSide]):
    """
    find time_frame_bbs which overlaps with any other time_frame_bbs by comparing time_frame_bbs['movement_start_time']
    and time_frame_bbs['movement_end_time']. the overlapping  time_frame_bbs s have movement_start_time of one before
    movement_end_time of another.
    find the nearest time_frame_bbs with movement_start_time after movement_start_time of the row using pd.merage_asof()
    and add the index as 'next_movement_start_time'.
    All time_frame_bbs with next_movement_start_time < time_frame_bbs['movement_end_time'] indicates an overlap.
    :param time_frame_bbs:
    :param remove:
    :return:
    """
    if len(time_frame_bbs) <= 1:
        return time_frame_bbs
    next_movements = time_frame_bbs.copy()[['movement_start_time', 'movement_end_time']] \
        .rename(columns={'movement_start_time': 'next_movement_start_time',
                         'movement_end_time': 'next_movement_end_time'})
    next_movements['shifted_movement_start_time'] = \
        next_movements['next_movement_start_time'] - pd.to_timedelta(config.timeframes[0])
    next_movements['next_movement_index'] = next_movements.index
    next_movements.sort_values(by='next_movement_start_time', inplace=True)
    time_frame_bbs['date_backup'] = time_frame_bbs.index
    # drop overlaps with the same start
    merged_to_find_same_start = \
        time_frame_bbs.merge(next_movements, left_on='movement_start_time', right_on='next_movement_start_time',
                             how='left')
    valid_same_start_overlaps = merged_to_find_same_start[
        merged_to_find_same_start['next_movement_index'].notna()
        & (merged_to_find_same_start['next_movement_index'] != merged_to_find_same_start.index)
        & (merged_to_find_same_start['next_movement_end_time'] < merged_to_find_same_start['movement_end_time'])
        ]
    if len(valid_same_start_overlaps['next_movement_index'].dropna().unique()) > 0:
        pass
    time_frame_bbs.drop(labels=valid_same_start_overlaps['next_movement_index'].dropna().unique(),
                        inplace=True)
    next_movements.drop(labels=valid_same_start_overlaps['next_movement_index'].dropna().unique(),
                        inplace=True)
    # find overlaps which does not have the same start
    merged_to_find_overlap = \
        pd.merge_asof(left=time_frame_bbs.sort_values(by='movement_start_time'), right=next_movements,
                      left_on='movement_start_time', right_on='shifted_movement_start_time', direction='forward')
    overlapped_movements = merged_to_find_overlap[
        merged_to_find_overlap['next_movement_start_time'].notna()
        & (merged_to_find_overlap['next_movement_index'] != time_frame_bbs.index)
        & (merged_to_find_overlap['next_movement_start_time'] < merged_to_find_overlap['movement_end_time'])
        ]
    # time_frame_bbs.loc[invalid_overlap_movements.index, ['next_movement_start_time', 'next_movement_index']] = pd.NA

    # drop overlap_movements which does not need merging. These overlaps are inside the original trend boundaries.
    if any(overlapped_movements['next_movement_index'].isna()):
        raise AssertionError("any(overlapped_movements['next_movement_index'].isna())")
    no_merge_overlap_movements = overlapped_movements[
        overlapped_movements['next_movement_end_time'] <= overlapped_movements['movement_end_time']
        ]
    if len(no_merge_overlap_movements) > 0:
        time_frame_bbs.drop(labels=no_merge_overlap_movements['next_movement_index'].dropna().unique(),
                            inplace=True)
    # expand the end of first overlapping movement, to the end of second one to cover both

    # to_merge_overlap_movements = time_frame_bbs[
    #     time_frame_bbs['next_movement_index'].notna()
    #     # & (time_frame_bbs['next_movement_end_time'] > time_frame_bbs['movement_end_time'])
    # ]
    to_merge_overlap_movements = \
        overlapped_movements.loc[overlapped_movements.index.difference(no_merge_overlap_movements.index)]
    if len(to_merge_overlap_movements) > 0:
        time_frame_bbs.loc[to_merge_overlap_movements['date_backup'], 'movement_end_time'] = \
            to_merge_overlap_movements['next_movement_end_time'].to_list()
        # drop trends which covered by the original after expansion.
        time_frame_bbs.drop(
            labels=to_merge_overlap_movements['next_movement_index'].dropna().unique(), inplace=True)
    return time_frame_bbs


# def zz_var codePasteTime = new Date();
# var runAt = new Date(codePasteTime.getFullYear(), codePasteTime.getMonth(), codePasteTime.getDate(), 8, 44, 45, 0); // 8:44:45, 8:45:0.300, 8:45:0.330
# // var runAt = new Date(new Date().getTime() + 1 * 1000); // 8:44:45, 8:45:0.300, 8:45:0.330
# var durationSeconds = 30; // 120
# var clickInterval = 350; // 310=3/36!=8%, 320=6/48=12%, 350=3/45=7%
#
# var maxTries = durationSeconds * 1000 / 100, retryCounter = 1, sellButton, mode, config = {attributes: true};
# var stockName = document.querySelector('.instrument-name').textContent.split(' ').filter(function (item) {
#     return !['', 'اقدام', 'نظارتي', 'مجاز', 'پايان', 'مشاوره'].includes(item.trim());
# })
# var workerId = String(codePasteTime.getSeconds()) + codePasteTime.getMilliseconds();
#
# function appendValue(name, value) {
#     var currentValue = localStorage.getItem(name);
#     if (currentValue === null) {
#         localStorage.setItem(name, value);
#     } else {
#         // Append the new value to the existing one
#         localStorage.setItem(name, currentValue + ',' + value);
#     }
# }
#
# function arrayFromValue(name) {
#     var raw = localStorage.getItem(name);
#     if (raw === null) {
#         return null;
#     }
#     return raw.split(',').filter(function (item) {
#         return item.length > 0;
#     });
# }
#
# function continueConditions() {
#     if (retryCounter++ > maxTries) {
#         console.log('maxTries reached 1!');
#         sellBtnObserver.disconnect()
#         return false;
#     }
#     if (new Date().getTime() > (runAt.getTime() + durationSeconds * 1000)) {
#         console.log('Timeout reached 1!');
#         sellBtnObserver.disconnect()
#         return false;
#     }
#     return true;
# }
#
# function isPrimaryWorker(thisAttempt) {
#     attemptWorkers = arrayFromValue('sk_attempt_' + thisAttempt)
#     if (attemptWorkers === null || attemptWorkers.length < 1)
#         throw new Error('Can not see myself in the list of attempt workers1');
#     if (attemptWorkers.length > 0) {
#         return (workerId === attemptWorkers.sort()[0]);
#     } else
#         throw new Error('Can not see myself in the list of attempt workers2');
# }
#
# function bindWorkerToAttempt(thisAttempt) {
#     appendValue('sk_attempt_' + thisAttempt, workerId + ',');
#     var attemptWorker = localStorage.getItem('sk_attempt_' + thisAttempt)
#     while (attemptWorker === null || (attemptWorker !== null && attemptWorker.indexOf(workerId) === -1)) {
#         console.log('appendValue failed!' + attemptWorker);
#         setTimeout(function () {
#             bindWorkerToAttempt(thisAttempt)
#         }, 5);
#     }
# }
#
# function fineTuneTryOnce(thisAttempt) {
#     if (orderRequirements()) {
#         if (isPrimaryWorker(thisAttempt)) {
#             lastClickTs = localStorage.getItem('sk_lastClick');
#             var fineTuneDelay = 0;
#             if (lastClickTs !== null) {
#                 var nextAllowedClick = Number(lastClickTs) + clickInterval;
#                 if (nextAllowedClick > new Date().getTime()) {
#                     fineTuneDelay = nextAllowedClick - new Date().getTime();
#                 }
#                 console.log('lastClickTs', lastClickTs, 'nextAllowedClick', nextAllowedClick, 'now', new Date().getTime(), 'fineTuneDelay', fineTuneDelay);
#             }
#             if (fineTuneDelay > 0) setTimeout(function () {
#                 tryOnce(thisAttempt)
#             }, fineTuneDelay);
#             else tryOnce(thisAttempt);
#         } else {
#             console.log('!isPrimaryWorker', 'now', new Date().getTime());
#             setTimeout(function () {
#                 sellButtonEnabled()
#             }, 5);
#         }
#     } else console.log('Sell requirements not satisfied!!!');
#     if (!continueConditions())
#         sellBtnObserver.disconnect();
# }
#
# // function sellButtonEnabled() {}
# function sellButtonEnabled() {
#     console.log('sellButtonEnabled:', dateToStrMilliseconds(new Date()))
#     var bookedAttemptTs = localStorage.getItem('sk_nextAttempt');
#     var delayBeforeTry = 20, thisAttempt;
#     if (bookedAttemptTs !== null && Number(bookedAttemptTs) > 0 && Number(bookedAttemptTs) > (new Date().getTime() - 10000)) {
#         thisAttempt = new Date(Number(bookedAttemptTs)).getTime()
#         console.log('bookedAttempt:', thisAttempt, 'now:', new Date().getTime());
#         if (thisAttempt > new Date())
#             delayBeforeTry = thisAttempt - new Date().getTime();
#     } else {
#         thisAttempt = runAt.getTime();
#     }
#     var nextAttempt = new Date(thisAttempt + clickInterval).getTime();
#     bindWorkerToAttempt(thisAttempt);
#     console.log('delayBeforeTry:', delayBeforeTry, 'now:', dateToStrMilliseconds(new Date()), '/', new Date().getTime(), 'bookedAttempt:', thisAttempt, 'nextAttempt:', nextAttempt);
#     localStorage.setItem('sk_nextAttempt', nextAttempt);
#     if (!continueConditions()) {
#         sellBtnObserver.disconnect()
#         return;
#     }
#     if (delayBeforeTry > 0)
#         setTimeout(function () {
#             fineTuneTryOnce(thisAttempt)
#         }, delayBeforeTry)
#     else
#         fineTuneTryOnce(thisAttempt)
# }
#
# var sellBtnObserver = new MutationObserver(function (mutations, observer) {
#     mutations.forEach(function (mutation) {
#         if (mutation.attributeName === 'disabled' && !sellButton.disabled) sellButtonEnabled();
#         else if (mutation.attributeName === 'class' && !document.querySelector('.sell-btn.disabled')) sellButtonEnabled();
#         else console.log('sellButtonDisabled:', dateToStrMilliseconds(new Date()))
#     });
# });
#
# function dateToStrMilliseconds(date) {
#     if (typeof date !== 'undefined')
#         return date.toString().replace(" GMT+0330 (Iran Standard Time)", "") + '.' + date.getMilliseconds() + 'ms'
#     else
#         return 'undefined'
# }
#
# function findSellButton() {
#     if (document.querySelector('rh-online-order#online-order div.form-container.online-order-container form') !== null) {	// OneLine mode
#         // if (mode !== 'OneLine') {
#         //     mode = 'OneLine'; // console.log('mode',mode);
#         sellButton = document.querySelector('.sell-btn');
#         // }
#     } else if (document.querySelector('rh-online-order#online-order div.ng-star-inserted div.basic-container') !== null) {	// Tabbed view
#         // if (mode !== 'Tabbed') {
#         //     mode = 'Tabbed'; // console.log('mode',mode);
#         sellButton = document.querySelector('button.sell');
#         // }
#     } else console.log('Unexpected mode');
# };findSellButton();
#
# function orderRequirements() {
#     if (document.querySelector('rh-online-order input#orderPrice').value === "") {
#         alert('Price is not valid!');
#         sellBtnObserver.disconnect();
#         return false;
#     }
#     if (document.querySelector('rh-online-order input[name="orderQuantity"]').value === '') {
#         alert('Quantity is not valid!');
#         sellBtnObserver.disconnect();
#         return false;
#     }
#     if (typeof sellButton === 'undefined' || sellButton === null) {
#         alert('Sell button is not enabled!');
#         sellBtnObserver.disconnect();
#         return false;
#     }
#     return true;
# }
#
# function tryOnce(thisAttempt) {
#     if (orderRequirements()) {
#         sellButton.click();
#         setTimeout(function (){
#             localStorage.removeItem('sk_attempt_' + thisAttempt);
#         },10000)
#         localStorage.setItem('sk_lastClick', new Date().getTime())
#         console.log(stockName + ' Sell@' + dateToStrMilliseconds(new Date()) + ' by ' + workerId + '!');
#     } else console.log('Sell requirements not satisfied!!!');
#     if (!continueConditions())
#         sellBtnObserver.disconnect();
# }
#
# function retryIt() {
#     sellBtnObserver.observe(sellButton, config);
#     sellButtonEnabled();
#     setTimeout(function () {
#         sellBtnObserver.disconnect();
#     }, durationSeconds * 1000);
# }
#
# if (orderRequirements()) {
#     var delayToStart = runAt - codePasteTime, timeIsOk = true;
#     if (delayToStart < 0) {
#         delayToStart += 86400000;
#         timeIsOk = confirm('Timer set for tomorrow in ' + delayToStart / 1000 / 60 / 60 + ' hours! are you sure?');
#     }
#     if (timeIsOk === true) {
#         setTimeout(function () {
#             console.log(stockName + " Running in " + dateToStrMilliseconds(new Date()));
#             retryIt();
#         }, delayToStart);
#         console.log(stockName + ' Scheduled for ' + dateToStrMilliseconds(new Date()) + ' + ' + delayToStart + 'ms@' + dateToStrMilliseconds(runAt));
#     } else console.log('Time not confirmed!!!');
# } else console.log('Sell requirements not satisfied!!!');(timeframe_pivots: pt.DataFrame[PivotDFM],
#                                 time_frame_bbs_trends: pt.DataFrame[BullBearSide]):
#     """
#     Find the time_frame_bbs_boundary which matches with each pivot:
#     Remove overlapped boundaries to prevent finding multiple bbs_trends for each pivot.
#     :param time_frame_bbs_trends:
#     :param bull_bear_trends:
#     :param timeframe_pivots:
#     :param pivots:
#     :param bbs_boundaries:
#     :return:
#     """
#     bull_bear_trends = (
#         time_frame_bbs_trends[time_frame_bbs_trends['bull_bear_side'].isin([TREND.BULLISH.value, TREND.BEARISH.value])]
#         .copy())
#     no_overlap_bull_bear_trends = merge_bbs_overlap(time_frame_bbs=bull_bear_trends)
#     no_overlap_bull_bear_trends = \
#         no_overlap_bull_bear_trends.rename(columns={
#             'movement_start_time': 'bbs_movement_start_time',
#             'movement_start_value': 'bbs_movement_start_value',
#         })[['bbs_movement_start_time', 'bbs_movement_start_value', ]]
#     timeframe_pivots.loc[:, ['bbs_movement_start_time', 'bbs_movement_start_value']] = pd.merge_asof(
#         left=timeframe_pivots, right=no_overlap_bull_bear_trends.dropna(), left_index=True,
#         right_on='bbs_movement_start_time',
#         direction='forward'
#     )[['bbs_movement_start_time', 'bbs_movement_start_value']]
#     timeframe_pivots['bbs_movement'] = \
#         timeframe_pivots['bbs_movement_start_value'] - timeframe_pivots['level']  # todo: test


def zz_ftc_of_range_by_time(pivots_with_ftc_range_start_time, multi_timeframe_base_patterns, pivots_timeframe):
    if len(pivots_with_ftc_range_start_time) > 0:
        pass
    pivot_lower_timeframes = config.timeframes[config.timeframes.index(pivots_timeframe):0:-1]
    pivots_with_ftc_range_start_time['ftc_list'] = pd.NA
    remained_pivots = pivots_with_ftc_range_start_time.copy()
    for timeframe in pivot_lower_timeframes:
        if len(remained_pivots) > 0:
            timeframe_base_patterns = single_timeframe(multi_timeframe_base_patterns, timeframe)
            for pivot_start, pivot in remained_pivots.iterrows():
                remained_pivots.loc[pivot_start, 'ftc_list'] = timeframe_base_patterns[
                    (timeframe_base_patterns.index < pivot_start)
                    & (timeframe_base_patterns.index > pivot['ftc_range_start_time'])
                    ]
                pivots_with_ftc_range_start_time.loc[pivot_start, 'ftc_list'] = \
                    remained_pivots.loc[pivot_start, 'ftc_list']
            remained_pivots = remained_pivots[remained_pivots['ftc_list'].isna()]
        else:
            break


def ftc_of_range_by_price(pivots: pt.DataFrame[Pivot2DFM],
                          mt_base_patterns: pt.DataFrame[MultiTimeframeBasePattern], pivots_timeframe):
    if len(pivots['real_movement'].dropna()) > 0:
        pass
    pivot_lower_timeframes = config.timeframes[config.timeframes.index(pivots_timeframe)::-1]
    # todo: test
    if 'real_movement' not in pivots.columns:
        raise ValueError("'real_movement' not in pivots.columns")
    pivots['ftc_range_start_value'] = \
        pivots['level'] - pivots['real_movement'] * config.ftc_price_range_percentage
    pivots['ftc_range_low'] = pivots[['ftc_range_start_value', 'level']].min(axis='columns', skipna=False)
    pivots['ftc_range_high'] = pivots[['ftc_range_start_value', 'level']].max(axis='columns', skipna=False)

    pivots['ftc_list'] = pd.NA
    remained_pivots = pivots.copy()
    for timeframe in pivot_lower_timeframes:
        if len(remained_pivots) > 0:
            timeframe_base_patterns = single_timeframe(mt_base_patterns, timeframe, keep_timeframe=True)
            """
            in following code I receive ValueError: Incompatible indexer with Series for last line. 
            """
            for pivot_start, pivot_info in remained_pivots.iterrows():
                matched_bases = timeframe_base_patterns[
                    (timeframe_base_patterns['internal_low'] < pivot_info['ftc_range_high'])
                    # & (timeframe_base_patterns['internal_high'] > pivot_info['ftc_range_low'])
                    & (timeframe_base_patterns['internal_low'] > pivot_info['ftc_range_low'])
                    & (timeframe_base_patterns.index.get_level_values('date') <= pivot_start)
                    & (timeframe_base_patterns['ttl'] >= pivot_start)
                    ].copy().reset_index()
                if len(matched_bases) > 0:
                    remained_pivots.at[pivot_start, 'ftc_list'] = matched_bases.to_dict(orient='records')
                    pivots.at[pivot_start, 'ftc_list'] = \
                        remained_pivots.loc[pivot_start, 'ftc_list']
            remained_pivots = remained_pivots[remained_pivots['ftc_list'].isna()]
        else:
            break


def insert_multi_timeframe_pivots_real_start(mt_pivot: pt.DataFrame[MultiTimeframePivot2DFM],
                                             mt_peaks_n_valleys: pt.DataFrame[MultiTimeframePeakValley], ):
    """
    todo: change to:
    real start is most significant candle between pivot movement start and first pivot of the same or higher timeframe
    before movement start.
    real end is most significant candle between pivot movement end and first pivot of the same or higher timeframe
    after movement end.
    :param mt_pivot:
    :param mt_peaks_n_valleys:
    :return:
    """
    mt_pivot['real_start_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    mt_pivot['real_start_value'] = pd.Series(dtype=float)
    mt_pivot['real_start_permanent'] = pd.Series(dtype=bool)
    for timeframe in mt_pivot.index.get_level_values('timeframe').unique():
        timeframe_pivots = single_timeframe(mt_pivot, timeframe).copy()
        time_frame_peaks_n_valleys = major_timeframe(mt_peaks_n_valleys, timeframe)
        insert_pivots_real_start(timeframe_pivots, time_frame_peaks_n_valleys)

        timeframe_pivots['timeframe'] = timeframe
        timeframe_pivots = timeframe_pivots.reset_index().set_index(['timeframe', 'date'])  # , 'original_start'])
        mt_pivot.loc[timeframe_pivots.index, ['real_start_time', 'real_start_value', 'real_start_permanent']] = \
            timeframe_pivots[['real_start_time', 'real_start_value', 'real_start_permanent']]
    mt_pivot['real_movement'] = mt_pivot['level'] - mt_pivot['real_start_value']
    if mt_pivot[['real_start_time', 'real_start_value', 'real_start_permanent']].isna().any().any():
        raise AssertionError(
            "mt_pivot[['real_start_time', 'real_start_value', 'real_start_permanent']].isna().any().any('")


def multi_timeframe_ftc(
        mt_pivot: pt.DataFrame[MultiTimeframePivot2DFM],
        # mt_bbs_trend: pt.DataFrame[MultiTimeframeBullBearSide],
        # mt_peaks_n_valleys: pt.DataFrame[MultiTimeframePeakValley],
        # mt_ohlcv: pt.DataFrame[MultiTimeframeOHLCV],
        multi_timeframe_base_patterns: pt.DataFrame[MultiTimeframeBasePattern],
        timeframe_shortlist: List[str] = None):
    if timeframe_shortlist is None:
        timeframe_shortlist = config.structure_timeframes[::-1]
    else:  # filter and order
        timeframe_shortlist = [timeframe for timeframe in config.structure_timeframes[::-1]
                               if timeframe in timeframe_shortlist]
    mt_pivot.loc[:, 'ftc_list'] = pd.NA
    timeframe_shortlist = [timeframe for timeframe in timeframe_shortlist
                           if timeframe in mt_pivot.index.get_level_values(level='timeframe')]
    for timeframe in timeframe_shortlist:
        timeframe_pivots = single_timeframe(mt_pivot, timeframe)
        # timeframe_bbs = single_timeframe(mt_bbs_trend, timeframe)
        # time_frame_peaks_n_valleys = single_timeframe(mt_peaks_n_valleys, timeframe)
        # ohlcv = single_timeframe(mt_ohlcv, timeframe)
        pivots_with_ftc = insert_ftc(
            timeframe_pivots=timeframe_pivots,
            # time_frame_bbs_boundaries=timeframe_bbs,
            # time_frame_peaks_n_valleys=time_frame_peaks_n_valleys,
            # ohlcv=ohlcv,
            multi_timeframe_base_patterns=multi_timeframe_base_patterns,
            timeframe=timeframe)
        pivots_with_ftc['timeframe'] = timeframe
        pivots_with_ftc = pivots_with_ftc.reset_index().set_index(['timeframe', 'date', 'original_start'])
        mt_pivot.loc[pivots_with_ftc.index, 'ftc_list'] = pivots_with_ftc['ftc_list']
    # pass


def insert_pivot_real_start(support_resistance: pt.DataFrame[PivotDFM], peaks_or_valleys: pt.DataFrame[PeakValley],
                            top_type: TopTYPE):
    """
    todo:
        real start is most significant candle between pivot movement start and first pivot of the same kind and of the
        same or higher timeframe
        before movement start.
        real end is most significant candle between pivot movement end and first pivot of the same kind and of the
        same or higher timeframe
        after movement end.
    real_start of support/resistance pivot is the first valley/peak which in compare with previous valley/peak is
    less_significant: (valley < previous / peak > previous)
    :param top_type:
    :param support_resistance:
    :param peaks_or_valleys:
    :return:
    """
    if top_type == TopTYPE.PEAK:  # Resistance:  # todo: test
        more_significant = lambda top, previous_top: top > previous_top
        high_low = 'high'
    else:  # top_type == TopTYPE.VALLEY: #Support
        more_significant = lambda top, previous_top: top < previous_top
        high_low = 'low'

    if 'real_start_time' not in support_resistance.columns:
        raise AssertionError("'real_start_time' not in support_resistance.columns")
    if 'real_start_value' not in support_resistance.columns:
        raise AssertionError("'real_start_value' not in support_resistance.columns")
    if 'real_start_permanent' not in support_resistance.columns:
        raise AssertionError("'real_start_permanent' not in support_resistance.columns")
    # support_resistance['real_start_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    # support_resistance['real_start_value'] = pd.Series(dtype=float)
    # support_resistance['real_start_permanent'] = pd.Series(dtype=bool)
    if support_resistance.index.names != ['date']:  # , 'original_start']:
        raise AssertionError("support_resistance.index.names != ['date']")
        # raise AssertionError("support_resistance.index.names != ['date', 'original_start']")
    for pivot_time, pivot in support_resistance.iterrows():
        passed_tops = peaks_or_valleys[peaks_or_valleys.index.get_level_values('date') <= pivot['movement_start_time']] \
            .sort_index(level='date', ascending=False)
        if len(passed_tops) > 0:
            passed_tops['previous_top_value'] = passed_tops[high_low].shift(-1)
            movement_start = passed_tops[more_significant(passed_tops[high_low], passed_tops['previous_top_value'])]
            if len(movement_start) > 0:
                support_resistance.loc[pivot_time, 'real_start_time'] = movement_start.index.get_level_values('date')[0]
                support_resistance.loc[pivot_time, 'real_start_value'] = movement_start.iloc[0][high_low]
                support_resistance.loc[pivot_time, 'real_start_permanent'] = True
            else:
                support_resistance.loc[pivot_time, 'real_start_time'] = passed_tops.index.get_level_values('date')[-1]
                support_resistance.loc[pivot_time, 'real_start_value'] = passed_tops.iloc[-1][high_low]
                support_resistance.loc[pivot_time, 'real_start_permanent'] = False
    return support_resistance


def insert_pivots_real_start(original_pivots: pt.DataFrame[PivotDFM], peaks_n_valleys: pt.DataFrame[PeakValley]):
    """
    todo: change to:
    real start is most significant candle between pivot movement start and first pivot of the same or higher timeframe
    before movement start.
    real end is most significant candle between pivot movement end and first pivot of the same or higher timeframe
    after movement end.
    """
    if 'real_start_time' not in original_pivots.columns:
        raise AssertionError("'real_start_time' not in original_pivots.columns")
    if 'real_start_value' not in original_pivots.columns:
        raise AssertionError("'real_start_value' not in original_pivots.columns")
    if 'real_start_permanent' not in original_pivots.columns:
        raise AssertionError("'real_start_permanent' not in original_pivots.columns")
    # original_pivots['real_start_time'] = pd.Series(dtype='datetime64[ns, UTC]')
    # original_pivots['real_start_value'] = pd.Series(dtype=float)
    # original_pivots['real_start_permanent'] = pd.Series(dtype=bool)
    resistance_pivots = original_pivots[original_pivots['is_resistance'].astype(bool)].copy()
    valleys = valleys_only(peaks_n_valleys)
    original_pivots.loc[resistance_pivots.index, ['real_start_time', 'real_start_value', 'real_start_permanent']] = \
        insert_pivot_real_start(resistance_pivots, valleys, top_type=TopTYPE.VALLEY)[
            ['real_start_time', 'real_start_value', 'real_start_permanent']]
    support_pivots = original_pivots[~original_pivots['is_resistance'].astype(bool)].copy()
    peaks = peaks_only(peaks_n_valleys)
    original_pivots.loc[support_pivots.index, ['real_start_time', 'real_start_value', 'real_start_permanent']] = \
        insert_pivot_real_start(support_pivots, peaks, top_type=TopTYPE.PEAK)[
            ['real_start_time', 'real_start_value', 'real_start_permanent']]
    na_real_starts = original_pivots['real_start_time'].isna()
    original_pivots.loc[na_real_starts, 'real_start_time'] = original_pivots.loc[na_real_starts, 'movement_start_time']
    original_pivots.loc[na_real_starts, 'real_start_value'] = original_pivots.loc[
        na_real_starts, 'movement_start_value']
    original_pivots.loc[na_real_starts, 'real_start_permanent'] = False
    if original_pivots[['real_start_time', 'real_start_value', 'real_start_permanent']].isna().any().any():
        raise AssertionError(
            "original_pivots[['real_start_time', 'real_start_value', 'real_start_permanent']].isna().any().any('")


def insert_ftc(timeframe_pivots: pt.DataFrame[Pivot2DFM],
               multi_timeframe_base_patterns: pt.DataFrame[MultiTimeframeBasePattern], timeframe):
    if 'real_start_time' not in timeframe_pivots.columns:
        raise ValueError("'real_start' not in original_pivots.columns")  # todo: test
    if 'real_start_value' not in timeframe_pivots.columns:
        raise ValueError("'real_start' not in original_pivots.columns")  # todo: test
    # log_w("after testing properly, extend the operation to support duplicated pivots.")
    original_pivots = timeframe_pivots[
        timeframe_pivots.index.get_level_values(level='date')
        == timeframe_pivots.index.get_level_values(level='original_start')] \
        .copy().reset_index().set_index('date')
    # if any(timeframe_original_pivots.index.get_level_values(level='date') != \
    #         timeframe_original_pivots.index.get_level_values(level='original_start')):
    #     raise AssertionError("timeframe_pivots.index.get_level_values(level='date')!= "
    #                          "timeframe_pivots.index.get_level_values(level='original_start')")
    ftc_of_range_by_price(original_pivots, multi_timeframe_base_patterns, timeframe)
    if 'ftc_range_start_value' not in original_pivots.columns:
        raise AssertionError("'ftc_range_start_value' not in pivots_with_bbs_movement.columns")
    if 'ftc_range_low' not in original_pivots.columns:
        raise AssertionError("'ftc_range_low' not in pivots_with_bbs_movement.columns")
    if 'ftc_range_high' not in original_pivots.columns:
        raise AssertionError("'ftc_range_high' not in pivots_with_bbs_movement.columns")
    if original_pivots[['ftc_range_start_value', 'ftc_range_low', 'ftc_range_high']].isna().any().any():
        raise AssertionError(
            "original_pivots[['ftc_range_start_value', 'ftc_range_low', 'ftc_range_high']].isna().any().any()")
    return original_pivots


def zz_ftc_range_start_time(pivots_with_bbs_movement, ohlcv):
    if len(pivots_with_bbs_movement) > 0:
        pass
    resistance_pivots = pivots_with_bbs_movement[pivots_with_bbs_movement['is_resistance']]
    pivots_with_bbs_movement.loc[resistance_pivots.index, 'ftc_range_start_time'] = (
        insert_crossing2(base=resistance_pivots, base_target_column='ftc_range_start_value',
                         target=ohlcv, target_compare_column='low', direction='left',
                         more_significant=lambda target, base: target < base,
                         ))['left_crossing_time']
    support_pivots = pivots_with_bbs_movement[~pivots_with_bbs_movement['is_resistance']]
    pivots_with_bbs_movement.loc[support_pivots.index, 'ftc_range_start_time'] = (
        insert_crossing2(base=support_pivots, base_target_column='ftc_range_start_value',
                         target=ohlcv, target_compare_column='high', direction='left',
                         more_significant=lambda target, base: target > base,
                         ))['left_crossing_time']
