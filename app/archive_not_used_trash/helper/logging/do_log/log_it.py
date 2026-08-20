import logging
import sys
import traceback
from pathlib import Path
from types import FrameType

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


def init_logger(
    path_of_logs: str,
    log_to_std_out_level: int,
    log_to_file_level: int,
    root_path: str,
    file_log_rotation_size: str,
    file_log_retention_duration: str,
) -> None:
    global \
        __root_path, \
        __severity_color_map, \
        __root_path, \
        __log_format, \
        __log_to_std_out_level, \
        __log_to_file_level, \
        __min_log_level

    __root_path = Path(root_path)
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

    _intercept_stdlib_logging()


# duplicated from app/helper/logging/do_log/log_it.py (still live there; a dead function here depends on it)
def _intercept_stdlib_logging() -> None:
    # No explicit `level=`: the root logger keeps its normal WARNING default, so third-party
    # libraries' own DEBUG/INFO chatter (e.g. TensorFlow's internal tracing) stays suppressed
    # exactly as it would without this intercept - only WARNING+ (ours or a library's) gets
    # routed into our sinks. Our own log_i/log_d/etc. bypass stdlib entirely and are unaffected.
    logging.basicConfig(handlers=[InterceptHandler()], force=True)


# duplicated from app/helper/logging/do_log/log_it.py (still live there; a dead function here depends on it)
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


# duplicated from app/helper/logging/do_log/log_it.py (still live there; a dead function here depends on it)
def _nearest_level_name(severity: int) -> str:
    for threshold, name in _NAMED_LEVEL_THRESHOLDS:
        if severity >= threshold:
            return name
    return "DEBUG"
