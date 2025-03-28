import logging
import sys
import traceback
from pathlib import Path

from colorama import Fore
from colorama import init as colorama_init
from loguru import logger

from .ray_id import get_ray_id

__severity_color_map = {
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.DEBUG: Fore.CYAN,
}
__root_path = None
__log_format = "{time:YYYY-MM-DD HH:mm:ss.SS} | {level} | {message}"
__log_to_std_out_level = logging.DEBUG
__log_to_file_level = 0
__min_log_level = __log_to_std_out_level
# the default logger for when init_logger was not called!
logger.add(
    sys.stdout,
    format=__log_format,
    colorize=True,
    level=__log_to_std_out_level,
)

__all__ = ['log_e', 'log_w', 'log_i', 'log_d']


def init_logger(
    path_of_logs: str,
    log_to_std_out_level: int,
    log_to_file_level: int,
    root_path: str,
    file_log_rotation_size: str, file_log_retention_duration: str,
):
    global __root_path, __severity_color_map, __root_path, __log_format,\
        __log_to_std_out_level, __log_to_file_level, __min_log_level

    __root_path = root_path
    colorama_init(autoreset=True)
    log_file_path = Path(path_of_logs) / "runtime.log"
    __log_to_std_out_level = log_to_std_out_level
    __log_to_file_level = log_to_file_level
    __min_log_level = min(__log_to_std_out_level, __log_to_file_level)
    logger.remove()  # Removes all default handlers

    # Console logger configuration
    logger.add(
        sys.stdout,
        format=__log_format,
        colorize=True,
        level=__log_to_std_out_level,
    )

    # File logger configuration
    logger.add(
        log_file_path,
        # Rotate the log file when it reaches 100 MB. Tested on rotation="1 KB"
        rotation=file_log_rotation_size,
        # Retain logs for 30 days. Tested on retention="1 minute"
        retention=file_log_retention_duration,
        # Ensures logging happens asynchronously
        enqueue=True,
        format=__log_format,
        level=log_to_file_level,
    )


def root_path():
    global __root_path
    if __root_path is None:
        __root_path = Path(__file__)
        try:
            for i in range(4):
                __root_path = __root_path.parent
        # raise RuntimeError("root_path is not defined! call init_logger first.")
        except (NameError, FileNotFoundError) as e:
            print(
                f"[WARNING] Unable to find parent for {__root_path}. "
                f"Calling init_logger will enable extended "
                f"features and resolve this warning."
            )
            pass
    return __root_path


# noinspection DuplicatedCode
def get_caller_info(stack_offset: int = 0):
    """
    Extracts the caller's file, function name, and line number from the stack trace.

    Args:
        stack_offset (int): Number of additional stack frames to skip.

    Returns:
        tuple: (relative file path, function name, line number)
    """
    try:
        stack = traceback.extract_stack()
        caller = stack[-(3 + stack_offset)]  # Todo: check if 3 is fine.
        file_path = Path(caller.filename)
        path_of_root = root_path()
        if path_of_root != "" and file_path.is_relative_to(root_path()):
            relative_path = file_path.relative_to(root_path())
        else:
            relative_path = file_path
        return relative_path.as_posix().replace("/", ".").replace(".py", ""), caller.name, caller.lineno
    except (IndexError, AttributeError):
        return "UNKNOWN", "UNKNOWN", 0


def log_d(message: str, stack_limit: int = 0, stack_offset: int = 0):
    """
        Logs a debug message.

        Args:
            message (str): The message to log.
            stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
            stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
        """
    log(message, logging.DEBUG, stack_limit, stack_offset + 1)


def log_w(message: str, stack_limit: int = 0, stack_offset: int = 0):
    """
        Logs a warning message.

        Args:
            message (str): The message to log.
            stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
            stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
        """
    log(message, logging.WARNING, stack_limit, stack_offset + 1)


def log_i(message: str, stack_limit: int = 0, stack_offset: int = 0):
    """
        Logs an informational message.

        Args:
            message (str): The message to log.
            stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
            stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
        """
    log(message, logging.INFO, stack_limit, stack_offset + 1)


def log_e(message: str, stack_limit: int = 0, stack_offset: int = 0):
    """
        Logs an error message.

        Args:
            message (str): The message to log.
            stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
            stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
        """
    log(message, logging.ERROR, stack_limit, stack_offset + 1)


def log(message: str, severity: logging, stack_limit: int = 0, stack_offset: int = 0):
    """
    Log a message with severity and optional stack trace.

    Args:
        message (str): The message to log.
        severity (int): The severity level of the log.
        stack_limit (int): Number of stack trace levels to include.
        stack_offset (int): Number of additional stack frames to skip.
    """
    if __min_log_level > severity:
        return
    try:
        file, function_name, line = get_caller_info(stack_offset)
        # Generate stack trace if requested
        stack_trace = ""
        if stack_limit > 0:
            stack = traceback.format_stack()[:-(stack_offset + 1)][-(stack_limit):]
            stack_trace = "\n" + "".join(stack)
        # Apply color to the message based on severity
        color = __severity_color_map.get(severity, Fore.WHITE)
        id_of_ray = get_ray_id()
        logger.log(
            severity,
            f"{color}{file}:{function_name}:{line} - {message}{stack_trace} (ray:{id_of_ray})"
        )
    except Exception as e:
        logger.exception(f"Failed to log message: {message} | Error: {str(e)}")
