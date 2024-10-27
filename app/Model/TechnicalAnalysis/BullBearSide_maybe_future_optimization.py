from datetime import datetime

import pandas as pd
from pandera import typing as pt

from BullBearSide import bull_bear_side_repr
from app.Config import TREND
from app.PanderaDFM.BullBearSide import BullBearSide
from app.helper.helper import log


def test_bull_bear_side_trends(trends: pt.DataFrame[BullBearSide]):
    for start, trend in trends.iterrows():
        end = trend['end']
        direction = trend['bull_bear_side']

        trend_overlaps = trends[
            (trends.index <= end) & (trends['end'] >= start)
            ]
        if len(trend_overlaps) > 0:
            # none of trends should overlap with another trend of same direction
            if direction in trend_overlaps['bull_bear_side']:
                _msg = f"{bull_bear_side_repr(start, trend)} overlaps with same direction trends:"
                _msg = str.join([bull_bear_side_repr(o_start, o_trend) for o_start, o_trend in
                                 trend_overlaps[trend_overlaps['bull_bear_side'] == direction].iterrows()])
                raise Exception(_msg)
            if TREND.SIDE.value in trend_overlaps['bull_bear_side']:
                _msg = f"{bull_bear_side_repr(start, trend)} overlaps with SIDE trends:"
                _msg = str.join([bull_bear_side_repr(o_start, o_trend) for o_start, o_trend in
                                 trend_overlaps[trend_overlaps['bull_bear_side'] == direction].iterrows()])
                raise Exception(_msg)
            # at this point we are sure the trend only overlaps with a reverse BULL/BEAR trend
            start_overlaps = trends[
                (trends.index <= end) & (trends.index >= start)
                ]
            end_overlaps = trends[
                (trends['end'] <= end) & (trends['end'] >= start)
                ]
            if len(start_overlaps) > 0:
                if len(start_overlaps) > 1:
                    raise Exception(f"{bull_bear_side_repr(start, trend)} have more than one start overlaps:"
                                    f"{[bull_bear_side_repr(o_start, o_trend) for o_start, o_trend in start_overlaps]}")
                if start != start_overlaps.iloc[0]['end']:
                    raise Exception(f"start of {bull_bear_side_repr(start, trend)} does not match with end of "
                                    f"{bull_bear_side_repr(start_overlaps.index[0], start_overlaps.iloc[0])}")
            if len(end_overlaps) > 0:
                if len(end_overlaps) > 1:
                    raise Exception(f"{bull_bear_side_repr(start, trend)} have more than one end overlaps:"
                                    f"{[bull_bear_side_repr(o_start, o_trend) for o_start, o_trend in end_overlaps]}")
                if trend['end'] != end_overlaps.index[0]:
                    raise Exception(f"end of {bull_bear_side_repr(start, trend)} does not match with start of "
                                    f"{bull_bear_side_repr(end_overlaps.index[0], end_overlaps.iloc[0])}")


def test_side_trend_not_overlapping_another_side(start: datetime, trend: pd.Series,
                                                 overlapping_trends: pt.DataFrame[BullBearSide],
                                                 raise_exception: bool = False):
    if TREND.SIDE.value in overlapping_trends['bull_bear_side'].unique():
        _message = f'{bull_bear_side_repr(start, trend)} overlaps with"'
        for o_start, o_trend in overlapping_trends[overlapping_trends['bull_bear_side'] == TREND.SIDE.value]:
            _message += bull_bear_side_repr(o_start, o_trend)
        if raise_exception:
            raise Exception(_message)
        else:
            log(_message)
            return False
    return True
