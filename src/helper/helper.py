import datetime
import functools
import os.path
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Tuple, TypeVar

import numpy as np
import pandas as pd
import pandera
import pytz

from Config import Config, config


class LogSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


Pandera_DFM_Type = TypeVar('Pandera_DFM_Type', bound=pandera.DataFrameModel)


class bcolors(Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


path_of_logs = config.path_of_logs  # 'logs'

__severity_color_map = {
    LogSeverity.INFO: bcolors.OKGREEN,
    LogSeverity.WARNING: bcolors.WARNING,
    LogSeverity.ERROR: bcolors.FAIL,
    LogSeverity.DEBUG: bcolors.OKGREEN,
}

log_file_handler = open(os.path.join(path_of_logs, 'runtime.log'), 'w')


def log_d(message: str, stack_trace: bool = False):
    log(message, LogSeverity.DEBUG, stack_trace)


def log_w(message: str, stack_trace: bool = True):
    log(message, LogSeverity.WARNING, stack_trace)


def log_e(message: str, stack_trace: bool = True):
    log(message, LogSeverity.ERROR, stack_trace)


def log(message: str, severity: LogSeverity = LogSeverity.INFO, stack_trace: bool = True):
    """
    Log a message with an optional severity level and stack trace.

    Args:
        message (str): The message to be logged.
        severity (LogSeverity, optional): The severity level of the log message. Defaults to LogSeverity.WARNING.
        stack_trace (bool, optional): Whether to include a stack trace in the log message. Defaults to True.

    Returns:
        None
    """
    severity_color = __severity_color_map[severity].value
    time_color = bcolors.OKBLUE.value
    print(f'{severity_color}{severity.value}@{time_color}{datetime.now().strftime("%m-%d.%H:%M:%S")}:'
          f'{severity_color}{message}')
    log_file_handler.write(f'{severity.value}@{datetime.now().strftime("%m-%d.%H:%M:%S")}:{message}\n')
    if stack_trace:
        stack = traceback.extract_stack(limit=2 + 1)[:-1]  # Remove the last item
        traceback.print_list(stack)


log_d('...Starting')


def profile_it(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start_time = time.time()
        function_parameters = get_function_parameters(args, kwargs)
        log_d(f"{func.__name__}({function_parameters}) started", stack_trace=False)

        # try:
        result = func(*args, **kwargs)
        # except OSError as e:
        #     log_e(f"Current directory is {os.path.abspath(os.path.curdir)}", stack_trace=False)
        #     log_e(f"Error in {func.__name__}({function_parameters}): {str(e)}", stack_trace=True)
        #     raise e
        # except Exception as e:
        #     log(f"Error in {func.__name__}({function_parameters}): {str(e)}", stack_trace=True)
        #     raise

        end_time = time.time()
        execution_time = end_time - start_time
        execution_tim_color = bcolors.OKBLUE if execution_time < 0.01 \
            else bcolors.OKGREEN if execution_time < 0.1 \
            else bcolors.WARNING if execution_time < 1 \
            else bcolors.FAIL
        log(f"{func.__name__}({function_parameters}) "
            f"executed in {execution_tim_color.value}{execution_time:.3f} seconds", stack_trace=False)
        return result

    return _measure_time


def get_function_parameters(args, kwargs):
    parameters = [
                     f'{len(arg)}*{arg.columns}' if isinstance(arg, pd.DataFrame)
                     else f'list{np.array(arg).shape}' if isinstance(arg, list)
                     else str(arg)
                     for arg in args
                 ] + [
                     f'{k}:{len(kwargs[k])}*{kwargs[k].columns}' if isinstance(kwargs[k], pd.DataFrame)
                     else f'{k}:list...' if isinstance(kwargs[k], list)
                     else f'{k}:{kwargs[k]}'
                     for k in kwargs.keys()
                 ]
    return ", ".join(parameters)


# Define a mapping from Pandera data types to pandas data types
pandera_to_pandas_type_map = {
    pandera.Float: float,
    pandera.Int: int,
    pandera.String: str,
    pandera.BOOL: bool,
    # Add more data types as needed
}


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
