import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pytz
from config import app_config
from domain.schemas.common.MultiTimeframe import MultiTimeframe, MultiTimeframe_Type
from helper.functions import date_range, date_range_to_string
from helper.logging.do_log import log_d, log_w
from pandas import DatetimeIndex, Timedelta, Timestamp
from pandas._typing import Axes
from pandera import typing as pt


def date_range_of_data(data: pd.DataFrame) -> str:
    """
    Generate a formatted date range string based on the first and last timestamps in the DataFrame's index.

    This function calculates and returns a formatted string representing the date range of the provided DataFrame.
    The string format is 'yy-mm-dd.HH-MMTyy-mm-dd.HH-MM', where the first timestamp corresponds to the start of the
    date range and the last timestamp corresponds to the end of the date range.

    Parameters:
        data (pd.DataFrame): The DataFrame for which to generate the date range string.

    Returns:
        str: The formatted date range string.

    Example:
        # Assuming you have a DataFrame 'data' with an index containing timestamps
        date_range = range_of_data(data)
        log_d(date_range)  # Output: 'yy-mm-dd.HH-MMTyy-mm-dd.HH-MM'
    """
    return (
        f"{data.index.get_level_values('date').min().strftime('%y-%m-%d.%H-%M')}T"
        f"{data.index.get_level_values('date').max().strftime('%y-%m-%d.%H-%M')}"
    )


def df_timedelta_to_str(input_time: str | Timedelta, hours=True, ignore_zero: bool = True) -> str:
    """
    Convert a pandas timedelta string or a pandas Timedelta object into a human-readable string representation.

    This function takes a pandas timedelta string or a pandas Timedelta object and converts it into a string format
    of hours and minutes. If the input is a string, it is converted to a Timedelta object. The resulting string
    represents the number of hours and minutes in the input timedelta.

    Parameters:
        input_time (Union[str, Timedelta]): The input timedelta, which can be a pandas timedelta string or a
                                           pandas Timedelta object.
        hours (bool, optional): If True (default), includes hours in the output. If False, only includes minutes.
        ignore_zero (bool, optional): If True (default), removes zero values from the output.

    Returns:
        str: A string representation of the input timedelta in the format "hours:minutes".

    Raises:
        ValueError: If the input is not a pandas timedelta string or a pandas Timedelta object.

    Example:
        # Convert a timedelta string to a human-readable string
        time_str = "2 days 03:30:00"
        result = df_timedelta_to_str(time_str)  # Result: "51:30"

        # Convert a Timedelta object to a human-readable string
        import pandas as pd
        time_delta = pd.Timedelta(days=2, hours=3, minutes=30)
        result = df_timedelta_to_str(time_delta)  # Result: "51:30"
    """
    if isinstance(input_time, str):
        timedelta_obj = Timedelta(input_time)
    elif (
        isinstance(input_time, Timedelta)
        or isinstance(input_time, np.timedelta64)
        or isinstance(input_time, pt.Timedelta)
    ):
        timedelta_obj = input_time
    elif isinstance(input_time, float):
        timedelta_obj = timedelta(seconds=input_time)
    else:
        raise ValueError(
            "Input should be either a pandas timedelta string, float(seconds) or a pandas Timedelta object."
        )

    total_minutes = timedelta_obj.total_seconds() // 60
    _hours = 0
    if hours:
        _hours = int(total_minutes // 60)
    _minutes = int(total_minutes % 60)

    if ignore_zero:
        _tuple = (_hours, _minutes)
        _tuple = (v if v > 0 else "" for v in _tuple)
        _hours, _minutes = _tuple

    return f"{_hours}:{_minutes}"


def timedelta_to_str(
    time_delta: timedelta,
    hours: bool = True,
    minutes: bool = True,
    seconds: bool = False,
    milliseconds: bool = False,
    microseconds: bool = False,
    ignore_zero: bool = True,
) -> str:
    """
    Convert a pandas timedelta string or a pandas Timedelta object into a human-readable string representation.

    This function takes a pandas timedelta string, a pandas Timedelta object, or a datetime.timedelta object
    and converts it into a string format of hours, minutes, seconds, milliseconds, and/or microseconds.
    If the input is a string, it is converted to a Timedelta object. The resulting string represents the time
    components specified by the function parameters.

    Parameters:
        time_delta (timedelta): The input timedelta, which should be a timedelta.
        hours (bool, optional): If True (default), includes hours in the output. If False, excludes hours.
        minutes (bool, optional): If True (default), includes minutes in the output. If False, excludes minutes.
        seconds (bool, optional): If True, includes seconds in the output. Default is False.
        milliseconds (bool, optional): If True, includes milliseconds in the output. Default is False.
        microseconds (bool, optional): If True, includes microseconds in the output. Default is False.
        ignore_zero (bool, optional): If True (default), removes zero values from the output.

    Returns:
        str: A string representation of the input timedelta in the specified format "hours:minutes:seconds:milliseconds".

    Raises:
        ValueError: If the input is not a pandas timedelta string, a pandas Timedelta object, or a datetime.timedelta object.

    Example:
        # Convert a timedelta string to a human-readable string
        time_str = "2 days 03:30:45.123456"
        result = timedelta_to_str(time_str, hours=True, minutes=True, seconds=True, milliseconds=True, microseconds=True)
        # Result: "51:30:45:123456"

        # Convert a Timedelta object to a human-readable string
        import pandas as pd
        time_delta = pd.Timedelta(days=2, hours=3, minutes=30, seconds=45, milliseconds=123, microseconds=456)
        result = timedelta_to_str(time_delta, hours=True, minutes=True, seconds=True, milliseconds=True, microseconds=True)
        # Result: "51:30:45:123456"
    """
    _hours, _minutes, _seconds, _seconds_fraction = [""] * 4
    remained_seconds = time_delta.total_seconds()
    if hours:
        _hours = int(remained_seconds // 60 * 60)
        remained_seconds -= _hours * 60 * 60
    if minutes:
        _minutes = int(remained_seconds // 60)
        remained_seconds -= _minutes * 60
    if seconds:
        _seconds = int(remained_seconds // 1)
        remained_seconds -= _seconds
    if microseconds:
        _seconds_fraction = int(remained_seconds // 0.000001) * 0.000001
    elif milliseconds:
        _seconds_fraction = int(remained_seconds // 0.001) * 0.001
        # remained_seconds -= _milliseconds
    if ignore_zero:
        _tuple = (_hours, _minutes, _seconds, _seconds_fraction)
        _tuple = tuple([v if (v == "" or v > 0) else "" for v in _tuple])
        _hours, _minutes, _seconds, _seconds_fraction = _tuple
    result = f"{_hours}:{_minutes}:{_seconds}:{_seconds_fraction}"
    return result


# def zz_test_index_match_timeframe(data: pd.DataFrame, timeframe: str):
#     for index_value, mapped_index_value in map(lambda x, y: (x, y), data.index, to_timeframe(data.index, timeframe)):
#         if index_value != mapped_index_value:
#             raise Exception(
#                 f'In Data({data.columns.names}) found Index({index_value}) not align with timeframe:{timeframe}/{mapped_index_value}\n'
#                 f'Indexes:{data.index.values}')


def dict_of_list(input_dict: dict[str, object]) -> dict[str, list[object]]:
    result = {k: [v] for k, v in input_dict.items()}
    return result


def shift_timeframe(timeframe, shifter):
    index = app_config.timeframes.index(timeframe)
    if type(shifter) == int:
        return app_config.timeframes[index + shifter]
    elif type(shifter) == str:
        if shifter not in app_config.timeframe_shifter.keys():
            raise Exception(f"Shifter expected be in [{app_config.timeframe_shifter.keys()}]")
        return app_config.timeframes[index + app_config.timeframe_shifter[shifter]]
    else:
        raise Exception(f"shifter expected be int or str got type({type(shifter)}) in {shifter}")


def trigger_timeframe(timeframe):
    if app_config.timeframes.index(timeframe) < -app_config.timeframe_shifter["trigger"]:
        raise Exception(f"{timeframe} has not a trigger time!")
    return shift_timeframe(timeframe, app_config.timeframe_shifter["trigger"])


def pattern_timeframe(timeframe):
    if app_config.timeframes.index(timeframe) < -app_config.timeframe_shifter["pattern"]:
        raise Exception(f"{timeframe} has not a pattern time!")
    return shift_timeframe(timeframe, app_config.timeframe_shifter["pattern"])


def anti_pattern_timeframe(timeframe):
    if (
        app_config.timeframes.index(timeframe)
        > len(app_config.timeframes) + app_config.timeframe_shifter["pattern"] - 1
    ):
        raise Exception(f"{timeframe} has not an anit-pattern time!")
    return shift_timeframe(timeframe, -app_config.timeframe_shifter["pattern"])


def anti_trigger_timeframe(timeframe):
    if (
        app_config.timeframes.index(timeframe)
        > len(app_config.timeframes) + app_config.timeframe_shifter["trigger"] - 1
    ):
        raise Exception(f"{timeframe} has not an anti-trigger time!")
    return shift_timeframe(timeframe, -app_config.timeframe_shifter["trigger"])


@dataclass
class FileInfoSet:
    symbol: str
    file_type: str
    date_range: str


def extract_file_info(file_name: str) -> FileInfoSet:
    pattern = re.compile(r"^((?P<symbol>[\w]+)\.)?(?P<file_type>[\w_]+)\.(?P<date_range>[\d\-\.T]+)\.zip$")
    match = pattern.match(file_name)
    if not match or len(match.groupdict()) < 3:
        raise Exception("Invalid filename format:" + file_name)
    data = match.groupdict()
    if "symbol" not in data.keys() or data["symbol"] is None:
        data["symbol"] = app_config.under_process_symbol
    return FileInfoSet(**data)


# @cache


def expand_date_range(
    date_range_str: str,
    time_delta: timedelta,
    mode: Literal["start", "end", "both"],
    limit_to_processing_period: bool = None,
) -> str:
    if limit_to_processing_period is None:
        limit_to_processing_period = app_config.limit_to_under_process_period
    start, end = date_range(date_range_str)
    if mode == "start":
        start = start - time_delta
    elif mode == "end":
        end = end + time_delta
    elif mode == "both":
        start = start - time_delta
        end = end + time_delta
    else:
        raise Exception(f"mode={mode} not implemented")
    if limit_to_processing_period:
        _, processing_period_end = date_range(app_config.processing_date_range)
        end = min(end, processing_period_end)
    if end < start:
        raise RuntimeError("end < start")
    return date_range_to_string(start=start, end=end)


# def nearest_match(needles: Axes, reference: Axes, direction: str, start=None, end=None, shift: int = 1) -> Axes:
def nearest_match(needles: Axes, reference: Axes, direction: Literal["left", "right"], shift: int = 1) -> Axes:
    """
    it will merge the indexes for both needles and referance and return. missing indexes in needles will be filled
    forward and missing indexes of reference will be filled backward.\n
    to find adjacent or nearest PREVIOUS row we can use as:\n
    mapped_list = shift_over(needles, reference, 'left')\n
    to find adjacent or nearest NEXT row we can use as:\n
    mapped_list = shift_over(needles, reference, 'right')

    :param shift:
    :param needles: needles list
    :param reference: reference list
    :param direction: should be either "left" or "right". use "left" to get the PREVIOUS reference for needles and
    use "right" to get the NEXT reference for needles
    :return: ...

    Example:\n
    reference.index	    1 2 3 5 10 20       \n
    needles.index    	1 2 3 6 9 15 20     \n

    shift_over(needles, reference, 'left'):
    mapped_list.index   1  2 3 5 6 9  10 15 20      \n
    forward(reference)  NA 1 2 3 5 5  5  10 10      \n
    backward(needles)   2  3 6 6 9 15 15 20 NA      \n
    return:
                        1  2 3 6 9 15 20 \n
                        NA 1 2 5 5 10 10

    shift_over(needles, reference, 'right'):
    mapped_list.index   1  2 3  5  6  9 10 15 20
    forward(needles)    NA 1 2  3  3  6  9  9 15
    backward(reference) 2  3 5 10 10 10 20 20 NA
    return:
                        1 2 3  6  9 15 20   \n
                        2 3 5 10 10 20 NA
    """
    # Todo: replace with pd.merge_asof
    direction = direction.lower()
    if direction == "left":
        forward = reference
        backward = needles
    elif direction == "right":
        forward = needles
        backward = reference
    else:
        raise Exception(f'direction:{direction} should be either "left" or "right".')
    if isinstance(needles, DatetimeIndex) and isinstance(reference, DatetimeIndex):
        df = pd.DataFrame(index=forward.append(backward).unique())
    else:
        union = []
        [union.append(i) for i in forward]
        [union.append(i) for i in backward]
        union = list(set(union))
        df = pd.DataFrame(index=union)
    df = df.sort_index()
    if direction == "left":
        df.loc[forward, "forward"] = forward
        if shift != 0:
            df["forward"] = df["forward"].ffill().shift(shift)
        else:
            df["forward"] = df["forward"].ffill()
    elif direction == "right":
        df.loc[backward, "backward"] = backward
        if shift != 0:
            df["backward"] = df["backward"].bfill().shift(-shift)
        else:
            df["backward"] = df["backward"].bfill()
    if direction == "left":
        return df.loc[needles, "forward"].to_list()
    elif direction == "right":
        return df.loc[needles, "backward"].to_list()


