import datetime
from datetime import datetime, timedelta
from typing import Tuple, TypeVar

import pandera
import pytz
from colorama import init

from .br_py.profiling import profile_it
from .br_py.logging import log_i, log_e, log_w, log_d, log_it

# Initialize colorama
init(autoreset=True)

Pandera_DFM_Type = TypeVar('Pandera_DFM_Type', bound=pandera.DataFrameModel)


def date_range(date_range_str: str) -> Tuple[datetime, datetime]:
    start_date_string, end_date_string = date_range_str.split('T')
    start_date = datetime.strptime(start_date_string, '%y-%m-%d.%H-%M')
    # if start_date.tzinfo is None:
    start_date = start_date.replace(tzinfo=pytz.utc)
    end_date = datetime.strptime(end_date_string, '%y-%m-%d.%H-%M')
    # if end_date.tzinfo is None:
    end_date = end_date.replace(tzinfo=pytz.utc)
    return start_date, end_date


def date_range_to_string(end: datetime = None, days: float = 60, start: datetime = None) -> str:
    if end is None:
        if start is None:
            end = today_morning()
        elif days is not None:
            end = start + timedelta(days=days) - timedelta(minutes=1)
    if start is None:
        start = end - timedelta(days=days) + timedelta(minutes=1)
        return f'{start.strftime("%y-%m-%d.%H-%M")}T' \
               f'{end.strftime("%y-%m-%d.%H-%M")}'
    else:
        return f'{start.strftime("%y-%m-%d.%H-%M")}T' \
               f'{end.strftime("%y-%m-%d.%H-%M")}'


def today_morning(tz=pytz.utc) -> datetime:
    return morning(datetime.now(tz)) - timedelta(minutes=1)


def morning(date_time: datetime, tz=pytz.utc):
    # return tz.localize(datetime.combine(date_time.date(), time(0, 0)), is_dst=None)
    # if date_time.tzinfo is None or date_time.tzinfo.utcoffset(date_time) is None:
    #     date_time = tz.localize(date_time, is_dst=None)
    return date_time.replace(hour=0, minute=0, second=0)
