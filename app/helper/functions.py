import sys
from datetime import datetime, timedelta, tzinfo
from typing import TypeVar

import numpy as np
import pandera
import pytz


def _children_size(obj: object, seen: set[int]) -> int:
    if isinstance(obj, dict):
        return sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(get_size(i, seen) for i in obj)
    if hasattr(obj, "__dict__"):
        return get_size(obj.__dict__, seen)
    return 0


def get_size(obj: object, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # جلوگیری از شمارش دوباره
    seen.add(obj_id)
    try:
        size = sys.getsizeof(obj)
    except TypeError:
        return 0  # اگر `sys.getsizeof` پشتیبانی نکند
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    size += _children_size(obj, seen)
    if obj in [None, Ellipsis, NotImplemented]:
        return sys.getsizeof(obj)
    return size


# # Initialize colorama
# init(autoreset=True)

Pandera_DFM_Type = TypeVar("Pandera_DFM_Type", bound=pandera.DataFrameModel)


def date_range(date_range_str: str) -> tuple[datetime, datetime]:
    start_date_string, end_date_string = date_range_str.split("T")
    start_date = datetime.strptime(start_date_string, "%y-%m-%d.%H-%M")
    # if start_date.tzinfo is None:
    start_date = start_date.replace(tzinfo=pytz.utc)
    end_date = datetime.strptime(end_date_string, "%y-%m-%d.%H-%M")
    # if end_date.tzinfo is None:
    end_date = end_date.replace(tzinfo=pytz.utc)
    return start_date, end_date


def date_range_to_string(end: datetime | None = None, days: float = 60, start: datetime | None = None) -> str:
    if end is None:
        end = today_morning() if start is None else start + timedelta(days=days) - timedelta(minutes=1)
    if start is None:
        start = end - timedelta(days=days) + timedelta(minutes=1)
    return f"{start.strftime('%y-%m-%d.%H-%M')}T{end.strftime('%y-%m-%d.%H-%M')}"


def today_morning(tz: tzinfo = pytz.utc) -> datetime:
    return morning(datetime.now(tz)) - timedelta(minutes=1)


def morning(date_time: datetime, tz: tzinfo = pytz.utc) -> datetime:
    # return tz.localize(datetime.combine(date_time.date(), time(0, 0)), is_dst=None)
    # if date_time.tzinfo is None or date_time.tzinfo.utcoffset(date_time) is None:
    #     date_time = tz.localize(date_time, is_dst=None)
    return date_time.replace(hour=0, minute=0, second=0)
