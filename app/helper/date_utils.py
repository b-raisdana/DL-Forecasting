from datetime import datetime, timedelta, tzinfo

import pytz


def date_range(date_range_str: str) -> tuple[datetime, datetime]:
    start_date_string, end_date_string = date_range_str.split("T")
    start_date = datetime.strptime(start_date_string, "%y-%m-%d.%H-%M")
    start_date = start_date.replace(tzinfo=pytz.utc)
    end_date = datetime.strptime(end_date_string, "%y-%m-%d.%H-%M")
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
    return date_time.replace(hour=0, minute=0, second=0)
