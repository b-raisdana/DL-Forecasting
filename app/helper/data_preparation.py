from datetime import datetime, timedelta
from typing import cast

import pandas as pd
import pytz
from config import app_config
from domain.schemas.common.MultiTimeframe import MultiTimeframe, MultiTimeframe_Type
from helper.date_utils import date_range
from helper.logging.do_log import log_d, log_w
from helper.pandera import pandera_validate
from pandas import DatetimeIndex, Timestamp
from pandera import typing as pt


@pandera_validate(allow_pandas_dataframe=True)
def single_timeframe(
    multi_timeframe_data: pt.DataFrame[MultiTimeframe_Type],
    timeframe: str,
    keep_timeframe: bool = False,
    sort: bool = True,
    index_only: bool = False,
) -> pd.DataFrame | DatetimeIndex:
    if "timeframe" not in multi_timeframe_data.index.names:
        raise Exception(
            f'multi_timeframe_data expected to have "timeframe" in indexes:[{multi_timeframe_data.index.names}]'
        )
    if timeframe not in app_config.timeframes:
        raise Exception(f"timeframe:{timeframe} is not in supported timeframes:{app_config.timeframes}")
    if multi_timeframe_data.index.names.index("timeframe") != 0:
        raise AssertionError("multi_timeframe_data.index.names.index('timeframe') != 0")
    if index_only:
        return cast(DatetimeIndex, multi_timeframe_data.loc[pd.IndexSlice[timeframe, :]].index)
    try:
        single_timeframe_data: pd.DataFrame = cast(pd.DataFrame, multi_timeframe_data.loc[pd.IndexSlice[timeframe, :]])
    except KeyError:
        return pd.DataFrame()
    # if 'timeframe' in single_timeframe_data.index.names:
    #     if keep_timeframe:
    #         out = (single_timeframe_data.reset_index(level='timeframe'))
    #     else:
    #         out = (single_timeframe_data.droplevel('timeframe', axis='index'))
    # else:
    if keep_timeframe:
        single_timeframe_data["timeframe"] = timeframe
    if "timeframe" in [single_timeframe_data.index.name, *single_timeframe_data.index.names]:
        raise AssertionError(
            "'timeframe' == single_timeframe_data.index.name or 'timeframe' in single_timeframe_data.index.names"
        )
    # else:
    #     out = validate_no_timeframe(single_timeframe_data.droplevel('timeframe'))
    validate_no_timeframe(single_timeframe_data)
    if sort:
        return single_timeframe_data.sort_index(level="date")
    else:
        return single_timeframe_data


def to_timeframe(
    time: DatetimeIndex | datetime | Timestamp,
    timeframe: str,
    ignore_cached_times: bool = False,
    do_not_warn: bool = False,
) -> datetime | DatetimeIndex:
    """
    Round down the given datetime or DatetimeIndex to the nearest time based on the specified timeframe.

    This function adjusts a datetime or each datetime in a DatetimeIndex to align with the start of a
    specified timeframe, such as '1min', '5min', '1H', etc. It is particularly useful for aligning
    timestamps to regular intervals.

    Parameters:
        time (Union[DatetimeIndex, datetime, Timestamp]): The datetime or DatetimeIndex to be rounded.
        timeframe (str): The desired timeframe to round to (e.g., '1min', '5min', '1H', etc.).
        ignore_cached_times (bool): If True, bypasses checking the time against a global cache of valid times.

    Returns:
        Union[datetime, DatetimeIndex]: The rounded datetime or DatetimeIndex, where each datetime value
        is adjusted to the start of the nearest interval as specified by the timeframe.

    Raises:
        Exception: If time types are incompatible, or if rounding requirements are not met.
        :param do_not_warn: Set it true if you are sure about using this method.
    """
    if not do_not_warn:
        log_w("Try not to use to_timeframe and use pd.merge_asof(...) instead")

    def round_single_datetime(dt: datetime | Timestamp) -> datetime | Timestamp:
        """
        Function to round a single datetime
        :param dt:
        :return:
        """

        if pd.to_timedelta(timeframe) >= timedelta(days=7):
            rounded_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            day_of_week = dt.weekday()
            rounded_dt = rounded_dt - timedelta(days=day_of_week)
        else:
            rounded_timestamp = (dt.timestamp() // seconds_in_timeframe) * seconds_in_timeframe
            if isinstance(dt, datetime):
                rounded_dt = datetime.fromtimestamp(rounded_timestamp, tz=dt.tzinfo)
            else:  # isinstance(dt, Timestamp)
                rounded_dt = pd.Timestamp(rounded_timestamp * 10**9, tz=dt.tzinfo)
        return rounded_dt

    # Calculate the timedelta for the specified timeframe
    timeframe_timedelta = pd.to_timedelta(timeframe)
    seconds_in_timeframe = timeframe_timedelta.total_seconds()

    if pd.to_timedelta(timeframe) >= timedelta(minutes=30) and getattr(time, "tzinfo", None) is None:
        raise Exception("To round times to timeframes > 30 minutes, timezone is significant")
    rounded_time: datetime | DatetimeIndex
    if isinstance(time, (datetime, Timestamp)):
        rounded_time = round_single_datetime(time)
    elif isinstance(time, DatetimeIndex):
        rounded_time = pd.DatetimeIndex(time.to_series().apply(round_single_datetime))
    else:
        raise Exception(f"Invalid type of time: {type(time)}")

    if not ignore_cached_times:
        check_time_in_cache(rounded_time, timeframe)

    return rounded_time


def check_time_in_cache(time: DatetimeIndex | pd.Series | datetime | Timestamp, timeframe: str) -> None:  # type: ignore[explicit-any]
    cache_key = f"valid_times_{timeframe}"
    if cache_key not in app_config.GLOBAL_CACHE:
        raise Exception(f"{cache_key} not initialized in config.GLOBAL_CACHE")
    cache_set: set = set(app_config.GLOBAL_CACHE[cache_key])
    if isinstance(time, (DatetimeIndex, pd.Series)):
        if not time.isin(cache_set).all():
            raise Exception(f"Some times: {time} not found in config.GLOBAL_CACHE[valid_times_{timeframe}]!")
    elif time not in cache_set:
        raise Exception(f"time {time} not found in config.GLOBAL_CACHE[valid_times_{timeframe}]!")


# def zz_test_index_match_timeframe(data: ptd, timeframe: str):
#     for index_value, mapped_index_value in map(lambda x, y: (x, y), data.index, to_timeframe(data.index, timeframe)):
#         if index_value != mapped_index_value:
#             raise Exception(
#                 f'In Data({data.columns.names}) found Index({index_value}) not align with '
#                 f'timeframe:{timeframe}/{mapped_index_value}\n'
#                 f'Indexes:{data.index.values}')


@pandera_validate(allow_pandas_dataframe=True)
def validate_no_timeframe(data: pd.DataFrame) -> pd.DataFrame:
    if "timeframe" in data.index.names:
        raise Exception(f"timeframe found in Data(indexes:{data.index.names}, columns:{data.columns.names}")
    return data


@pandera_validate(allow_pandas_dataframe=True)
def times_tester(
    df: pd.DataFrame,
    date_range_str: str,
    timeframe: str,
    return_bool: bool = False,
    limit_to_under_process_period: bool = True,
    processing_date_range: str | None = None,
    exact_match: bool = False,
) -> bool | None:
    expected_times = set(
        times_in_date_range(date_range_str, timeframe, limit_to_under_process_period, processing_date_range)
    )
    if len(expected_times) == 0:
        return True
    actual_times = set(df.index) if len(df.index) > 0 else set()

    # Checking if all expected times are in the dataframe's index
    missing_times = expected_times - actual_times
    if missing_times:
        message = f"Some times in {date_range_str}@{timeframe} are missing in the DataFrame's index:" + ", ".join(
            [str(time) for time in missing_times]
        )
        if True:
            log_d(message)
            return True  # False
        else:
            raise ValueError(message)
    else:
        if exact_match:
            excess_times = actual_times - expected_times
            if not excess_times:
                return True
            else:
                message = (
                    f"Some times in {date_range_str}@{timeframe} are excessive in the DataFrame's index:"
                    + ", ".join([str(time) for time in excess_times])
                )
                if return_bool:
                    log_d(message)
                    return False
                else:
                    raise ValueError(message)
        else:
            return True


# def dict_of_list(input_dict: dict[str, object]) -> dict[str, list[object]]:
# result = {k: [v] for k, v in input_dict.items()}
# return result


@pandera_validate
def multi_timeframe_times_tester(
    multi_timeframe_df: pt.DataFrame[MultiTimeframe],
    date_range_str: str,
    return_bool: bool = False,
    ignore_processing_date_range: bool = True,
    processing_date_range: str | None = None,
) -> bool | None:
    result: bool | None = True
    for timeframe in app_config.timeframes:
        _timeframe_df = single_timeframe(multi_timeframe_df, timeframe)
        _result = times_tester(
            cast(pd.DataFrame, _timeframe_df),
            date_range_str,
            timeframe,
            return_bool,
            ignore_processing_date_range,
            processing_date_range,
        )
        if _result is None:
            result = None
        elif result is not None:
            result = result & _result
    return result


# def shift_timeframe(timeframe, shifter):
# index = app_config.timeframes.index(timeframe)
# if type(shifter) == int:
# return app_config.timeframes[index + shifter]
# elif type(shifter) == str:
# if shifter not in app_config.timeframe_shifter.keys():
# raise Exception(f"Shifter expected be in [{app_config.timeframe_shifter.keys()}]")
# return app_config.timeframes[index + app_config.timeframe_shifter[shifter]]
# else:
# raise Exception(f"shifter expected be int or str got type({type(shifter)}) in {shifter}")


# def trigger_timeframe(timeframe):
# if app_config.timeframes.index(timeframe) < -app_config.timeframe_shifter["trigger"]:
# raise Exception(f"{timeframe} has not a trigger time!")
# return shift_timeframe(timeframe, app_config.timeframe_shifter["trigger"])


# def pattern_timeframe(timeframe):
# if app_config.timeframes.index(timeframe) < -app_config.timeframe_shifter["pattern"]:
# raise Exception(f"{timeframe} has not a pattern time!")
# return shift_timeframe(timeframe, app_config.timeframe_shifter["pattern"])


# def anti_pattern_timeframe(timeframe):
# if (
# app_config.timeframes.index(timeframe)
# > len(app_config.timeframes) + app_config.timeframe_shifter["pattern"] - 1
# ):
# raise Exception(f"{timeframe} has not an anit-pattern time!")
# return shift_timeframe(timeframe, -app_config.timeframe_shifter["pattern"])


# def anti_trigger_timeframe(timeframe):
# if (
# app_config.timeframes.index(timeframe)
# > len(app_config.timeframes) + app_config.timeframe_shifter["trigger"] - 1
# ):
# raise Exception(f"{timeframe} has not an anti-trigger time!")
# return shift_timeframe(timeframe, -app_config.timeframe_shifter["trigger"])


def map_symbol(symbol: str, map_dictionary: dict[str, str]) -> str:
    upper_symbol = symbol.upper()
    if upper_symbol in map_dictionary.values():
        return symbol.upper()
    return map_dictionary[upper_symbol]


# @dataclass
# class FileInfoSet:
# symbol: str
# file_type: str
# date_range: str


# def extract_file_info(file_name: str) -> FileInfoSet:
# pattern = re.compile(r"^((?P<symbol>[\w]+)\.)?(?P<file_type>[\w_]+)\.(?P<date_range>[\d\-\.T]+)\.zip$")
# match = pattern.match(file_name)
# if not match or len(match.groupdict()) < 3:
# raise Exception("Invalid filename format:" + file_name)
# data = match.groupdict()
# if "symbol" not in data.keys() or data["symbol"] is None:
# data["symbol"] = app_config.under_process_symbol
# return FileInfoSet(**data)


# @cache
@pandera_validate(allow_pandas_dataframe=True)
def trim_to_date_range(date_range_str: str, df: pd.DataFrame, ignore_duplicate_index: bool = False) -> pd.DataFrame:
    start, end = date_range(date_range_str)
    date_indexes = df.index.get_level_values(level="date")
    df = df[(date_indexes >= start) & (date_indexes <= end)]
    duplicate_indices = df.index[df.index.duplicated()].unique()
    if not ignore_duplicate_index and len(duplicate_indices) != 0:
        raise ValueError("len(duplicate_indices) != 0")
    # else:
    #     if len(duplicate_indices) > 0:
    #         log(f"Found duplicate indices:" + str(duplicate_indices))
    return df


# def expand_date_range(
# date_range_str: str,
# time_delta: timedelta,
# mode: Literal["start", "end", "both"],
# limit_to_processing_period: bool = None,
# ) -> str:
# if limit_to_processing_period is None:
# limit_to_processing_period = app_config.limit_to_under_process_period
# start, end = date_range(date_range_str)
# if mode == "start":
# start = start - time_delta
# elif mode == "end":
# end = end + time_delta
# elif mode == "both":
# start = start - time_delta
# end = end + time_delta
# else:
# raise Exception(f"mode={mode} not implemented")
# if limit_to_processing_period:
# _, processing_period_end = date_range(app_config.processing_date_range)
# end = min(end, processing_period_end)
# if end < start:
# raise RuntimeError("end < start")
# return date_range_to_string(start=start, end=end)


def after_under_process_date(date_range_str: str) -> bool:
    start, _ = date_range(date_range_str)
    _, end = date_range(app_config.processing_date_range)
    return start > end


def times_in_date_range(
    date_range_str: str,
    timeframe: str,
    ignore_out_of_process_period: bool = True,
    processing_date_range: str | None = None,
) -> DatetimeIndex:
    start, end = date_range(date_range_str)
    if ignore_out_of_process_period:
        if processing_date_range is None:
            processing_date_range = app_config.processing_date_range
        under_process_scope_start, under_process_scope_end = date_range(processing_date_range)
        end = min(end, under_process_scope_end)
        start = max(start, under_process_scope_start)
    in_timeframe_start_date = to_timeframe(start, timeframe, ignore_cached_times=True, do_not_warn=True)
    if (
        isinstance(start, datetime)
        and isinstance(in_timeframe_start_date, datetime)
        and in_timeframe_start_date < start
    ):
        in_timeframe_start_date += pd.to_timedelta(timeframe)
    if (
        isinstance(start, DatetimeIndex)
        and isinstance(in_timeframe_start_date, DatetimeIndex)
        and (in_timeframe_start_date < start).any()
    ):
        in_timeframe_start_date = in_timeframe_start_date + pd.to_timedelta(timeframe)
    if start < end:
        if timeframe == "1W":
            frequency = "W-MON"
        elif timeframe == "M":
            frequency = "MS"
        else:
            frequency = timeframe
        return pd.date_range(start=in_timeframe_start_date, end=end, freq=frequency)  # type: ignore[arg-type]
    return pd.DatetimeIndex([], tz=pytz.utc)


# def nearest_match(needles: Axes, reference: Axes, direction: str, start=None, end=None, shift: int = 1) -> Axes:
# def nearest_match(needles: Axes, reference: Axes, direction: Literal["left", "right"], shift: int = 1) -> Axes:
# """
# it will merge the indexes for both needles and referance and return. missing indexes in needles will be filled
# forward and missing indexes of reference will be filled backward.\n
# to find adjacent or nearest PREVIOUS row we can use as:\n
# mapped_list = shift_over(needles, reference, 'left')\n
# to find adjacent or nearest NEXT row we can use as:\n
# mapped_list = shift_over(needles, reference, 'right')

# :param shift:
# :param needles: needles list
# :param reference: reference list
# :param direction: should be either "left" or "right". use "left" to get the PREVIOUS reference for needles and
# use "right" to get the NEXT reference for needles
# :return: ...

# Example:\n
# reference.index	    1 2 3 5 10 20       \n
# needles.index    	1 2 3 6 9 15 20     \n

# shift_over(needles, reference, 'left'):
# mapped_list.index   1  2 3 5 6 9  10 15 20      \n
# forward(reference)  NA 1 2 3 5 5  5  10 10      \n
# backward(needles)   2  3 6 6 9 15 15 20 NA      \n
# return:
# 1  2 3 6 9 15 20 \n
# NA 1 2 5 5 10 10

# shift_over(needles, reference, 'right'):
# mapped_list.index   1  2 3  5  6  9 10 15 20
# forward(needles)    NA 1 2  3  3  6  9  9 15
# backward(reference) 2  3 5 10 10 10 20 20 NA
# return:
# 1 2 3  6  9 15 20   \n
# 2 3 5 10 10 20 NA
# """
# # Todo: replace with pd.merge_asof
# direction = direction.lower()
# if direction == "left":
# forward = reference
# backward = needles
# elif direction == "right":
# forward = needles
# backward = reference
# else:
# raise Exception(f'direction:{direction} should be either "left" or "right".')
# if isinstance(needles, DatetimeIndex) and isinstance(reference, DatetimeIndex):
# df = ptd(index=forward.append(backward).unique())
# else:
# union = []
# [union.append(i) for i in forward]
# [union.append(i) for i in backward]
# union = list(set(union))
# df = ptd(index=union)
# df = df.sort_index()
# if direction == "left":
# df.loc[forward, "forward"] = forward
# if shift != 0:
# df["forward"] = df["forward"].ffill().shift(shift)
# else:
# df["forward"] = df["forward"].ffill()
# elif direction == "right":
# df.loc[backward, "backward"] = backward
# if shift != 0:
# df["backward"] = df["backward"].bfill().shift(-shift)
# else:
# df["backward"] = df["backward"].bfill()
# if direction == "left":
# return df.loc[needles, "forward"].to_list()
# elif direction == "right":
# return df.loc[needles, "backward"].to_list()


@pandera_validate(allow_pandas_dataframe=True)
def concat(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if not left.empty and not left.isna().all().all():
        if not right.empty and not right.isna().all().all():
            if left.isna().all(axis=0).any():
                pass
            if right.isna().all(axis=0).any():
                pass
            right_na_columns = right.dtypes[right.isna().all()]
            # right_na_column_dtypes = right.dtypes[right_na_columns]
            left_na_columns = left.dtypes[left.isna().all()]
            # left_na_column_dtypes = left.dtypes[left_na_columns]
            left = pd.concat([left.dropna(axis=1, how="all"), right.dropna(axis=1, how="all")])
            for column, d_type in left_na_columns.items():
                if column not in right.columns:
                    left[column] = pd.Series(dtype=d_type)
                else:
                    left[column] = right[column]
            for column, d_type in right_na_columns.items():
                if column not in left.columns:
                    left[column] = pd.Series(dtype=d_type)
    else:
        left = right.copy() if not right.empty and not right.isna().all().all() else pd.DataFrame()
    return left
