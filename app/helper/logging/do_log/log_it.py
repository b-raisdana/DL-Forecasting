import logging
import sys
import traceback
from pathlib import Path
from types import FrameType

from colorama import Fore
from loguru import logger

from .ray_id import get_ray_id

__severity_color_map = {
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.DEBUG: Fore.CYAN,
}
__root_path: Path | None = None
__log_format = "{time:YYYY-MM-DD HH:mm:ss.SS} | {level} | {name}:{function}:{line} - {message}"
__log_to_std_out_level = logging.DEBUG
__log_to_file_level = 0
__min_log_level = __log_to_std_out_level

__all__ = ["log_e", "log_w", "log_i", "log_d"]

# Standard levels only (excludes loguru-specific TRACE/SUCCESS): highest threshold <= a given
# numeric severity wins, so an ad-hoc int (e.g. a raw stdlib level not exactly 10/20/30/40/50)
# still renders as a named level instead of loguru's fallback "Level N".
_NAMED_LEVEL_THRESHOLDS = (
    (logging.CRITICAL, "CRITICAL"),
    (logging.ERROR, "ERROR"),
    (logging.WARNING, "WARNING"),
    (logging.INFO, "INFO"),
    (logging.DEBUG, "DEBUG"),
)


def _nearest_level_name(severity: int) -> str:
    for threshold, name in _NAMED_LEVEL_THRESHOLDS:
        if severity >= threshold:
            return name
    return "DEBUG"


# def init_logger(
# path_of_logs: str,
# log_to_std_out_level: int,
# log_to_file_level: int,
# root_path: str,
# file_log_rotation_size: str,
# file_log_retention_duration: str,
# ) -> None:
# global \
# __root_path, \
# __severity_color_map, \
# __root_path, \
# __log_format, \
# __log_to_std_out_level, \
# __log_to_file_level, \
# __min_log_level

# __root_path = Path(root_path)
# colorama_init(autoreset=True)
# log_file_path = Path(path_of_logs) / "runtime.log"
# __log_to_std_out_level = log_to_std_out_level
# __log_to_file_level = log_to_file_level
# __min_log_level = min(__log_to_std_out_level, __log_to_file_level)
# logger.remove()  # Removes all default handlers

# # Console logger configuration
# logger.add(
# sys.stdout,
# format=__log_format,
# colorize=True,
# level=__log_to_std_out_level,
# )

# # File logger configuration
# logger.add(
# log_file_path,
# # Rotate the log file when it reaches 100 MB. Tested on rotation="1 KB"
# rotation=file_log_rotation_size,
# # Retain logs for 30 days. Tested on retention="1 minute"
# retention=file_log_retention_duration,
# # Ensures logging happens asynchronously
# enqueue=True,
# format=__log_format,
# level=log_to_file_level,
# )

# _intercept_stdlib_logging()


class InterceptHandler(logging.Handler):
    """Redirects records from the stdlib `logging` module into loguru, so code (ours or a
    third-party library's) that calls `logging.getLogger(...)` still ends up in the same
    console/file sinks as log_i/log_e/etc., instead of bypassing them via logging's own default
    stderr handler."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = _nearest_level_name(record.levelno)

        frame: FrameType | None
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _intercept_stdlib_logging() -> None:
    # No explicit `level=`: the root logger keeps its normal WARNING default, so third-party
    # libraries' own DEBUG/INFO chatter (e.g. TensorFlow's internal tracing) stays suppressed
    # exactly as it would without this intercept - only WARNING+ (ours or a library's) gets
    # routed into our sinks. Our own log_i/log_d/etc. bypass stdlib entirely and are unaffected.
    logging.basicConfig(handlers=[InterceptHandler()], force=True)


def root_path(root_distance: int = 5) -> Path:
    global __root_path
    if __root_path is None:
        __root_path = Path(__file__)
        try:
            for _i in range(root_distance):
                __root_path = __root_path.parent
        # raise RuntimeError("root_path is not defined! call init_logger first.")
        except (NameError, FileNotFoundError):
            print(
                f"[WARNING] Unable to find parent for {__root_path}. "
                f"Calling init_logger will enable extended "
                f"features and resolve this warning."
            )
    return __root_path


def _init_default_logger() -> None:
    """The default logger for when init_logger was not called: stdout plus a daily-rotated
    file under <root>/logs, so runs that never call init_logger still leave a log on disk, and
    stdlib `logging` calls are captured into the same sinks."""
    log_dir = root_path() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        format=__log_format,
        colorize=True,
        level=__log_to_std_out_level,
    )
    logger.add(
        log_dir / "runtime.log",
        format=__log_format,
        level=__log_to_file_level,
        rotation="00:00",
        retention="30 days",
        enqueue=True,
    )
    _intercept_stdlib_logging()


_init_default_logger()


def log_d(message: str, stack_limit: int = 0, stack_offset: int = 0) -> None:
    """
    Logs a debug message.

    Args:
        message (str): The message to log.
        stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
        stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
    """
    log(message, logging.DEBUG, stack_limit, stack_offset + 1)


def log_w(message: str, stack_limit: int = 0, stack_offset: int = 0) -> None:
    """
    Logs a warning message.

    Args:
        message (str): The message to log.
        stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
        stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
    """
    log(message, logging.WARNING, stack_limit, stack_offset + 1)


def log_i(message: str, stack_limit: int = 0, stack_offset: int = 0) -> None:
    """
    Logs an informational message.

    Args:
        message (str): The message to log.
        stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
        stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
    """
    log(message, logging.INFO, stack_limit, stack_offset + 1)


def log_e(message: str, stack_limit: int = 0, stack_offset: int = 0) -> None:
    """
    Logs an error message.

    Args:
        message (str): The message to log.
        stack_limit (int, optional): Number of stack trace levels to include. Defaults to 0.
        stack_offset (int, optional): Number of additional stack frames to skip. Defaults to 0.
    """
    log(message, logging.ERROR, stack_limit, stack_offset + 1)


def log(message: str, severity: int, stack_limit: int = 0, stack_offset: int = 0) -> None:
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
        # Generate stack trace if requested
        stack_trace = ""
        if stack_limit > 0:
            stack = traceback.format_stack()[: -(stack_offset + 1)][-(stack_limit):]
            stack_trace = "\n" + "".join(stack)
        # Apply color to the message based on severity
        color = __severity_color_map.get(severity, Fore.WHITE)
        id_of_ray = get_ray_id()
        # depth skips this frame plus stack_offset wrapper frames (e.g. log_i/log_w/...), so
        # {name}/{function}/{line} in the format resolve to the original call site.
        logger.opt(depth=stack_offset + 1).log(
            _nearest_level_name(severity), f"{color}{message}{stack_trace} (ray:{id_of_ray})"
        )
    except Exception as e:
        logger.exception(f"Failed to log message: {message} | Error: {str(e)}")
