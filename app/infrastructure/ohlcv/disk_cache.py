import os
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import get_args, get_type_hints
from zipfile import BadZipFile

import pandas as pd
import pytz
from config import app_config
from helper.data_preparation import after_under_process_date
from helper.functions import Pandera_DFM_Type, date_range, morning
from helper.logging.do_log import log_w
from helper.schema_casting import cast_and_validate
from infrastructure.ohlcv.fragmented_data import symbol_data_path

"""
Everything this repo needs to persist/read a (data_frame_type, date_range_str) artifact as a
feather/ZSTD (or legacy CSV-zip) file and never recompute or re-fetch it twice within its validity
scope — the disk-level half of the cache-or-generate skill. `cache_on_disk` is the primitive new
artifact types should use; `read_file`/`write_data_file` are its building blocks for callers (like
domain.schemas.common.ExtendedDf.read_file) that need a class-bound variant instead of a decorator.
"""


# @cache
def datarange_is_not_cachable(date_range_str):
    _, end = date_range(date_range_str)
    if end > morning(datetime.utcnow().replace(tzinfo=pytz.UTC)):
        return True
    return False


# In-process memo in front of read_file()'s disk read, keyed on the exact args that determine the file's
# content. Bounded LRU so a long multi-quarter/multi-symbol run doesn't grow this unboundedly. Never used
# for a datarange_is_not_cachable() range — those touch the live/incomplete present and must always be
# re-read fresh, per read_file()'s existing delete-after-use behavior for such ranges. See the
# cache-or-generate skill.
_READ_FILE_CACHE_MAX_ENTRIES = 32
_ReadFileCacheKey = tuple[str, str, str, int | None, int | None]
_read_file_cache: "OrderedDict[_ReadFileCacheKey, pd.DataFrame]" = OrderedDict()


def _read_file_cache_get(cache_key: _ReadFileCacheKey) -> pd.DataFrame | None:
    df = _read_file_cache.get(cache_key)
    if df is None:
        return None
    _read_file_cache.move_to_end(cache_key)
    return df.copy()


def _read_file_cache_put(cache_key: _ReadFileCacheKey, df: pd.DataFrame) -> None:
    _read_file_cache[cache_key] = df.copy()
    _read_file_cache.move_to_end(cache_key)
    while len(_read_file_cache) > _READ_FILE_CACHE_MAX_ENTRIES:
        _read_file_cache.popitem(last=False)


def _feather_file_path(data_frame_type: str, date_range_str: str, file_path: str) -> str:
    return os.path.join(file_path, f"{data_frame_type}.{date_range_str}.feather")


def _csv_zip_file_path(data_frame_type: str, date_range_str: str, file_path: str) -> str:
    return os.path.join(file_path, f"{data_frame_type}.{date_range_str}.zip")


def write_data_file(df: pd.DataFrame, data_frame_type: str, date_range_str: str, file_path: str) -> None:
    """
    Persist df as the primary feather/ZSTD artifact for (data_frame_type, date_range_str) — the
    write-side counterpart to _read_raw_data_file()'s feather-first read. Every generator writes
    through here instead of hand-rolling a CSV-zip write, so newly generated data lands as
    feather/ZSTD from the first write, with no CSV-zip round-trip.
    """
    feather_file_path = _feather_file_path(data_frame_type, date_range_str, file_path)
    df.reset_index().to_feather(feather_file_path, compression="zstd")


def remove_data_file(data_frame_type: str, date_range_str: str, file_path: str) -> None:
    """
    Delete whichever on-disk format (feather/ZSTD, primary, or legacy CSV-zip) currently backs this
    (data_frame_type, date_range_str) — used for date ranges datarange_is_not_cachable() flags as
    touching the live/incomplete present, which must never be left on disk between reads.
    """
    try:
        os.remove(_feather_file_path(data_frame_type, date_range_str, file_path))
    except FileNotFoundError:
        os.remove(_csv_zip_file_path(data_frame_type, date_range_str, file_path))


def _slice_rows(df: pd.DataFrame, n_rows: int | None, skip_rows: int | None) -> pd.DataFrame:
    if not skip_rows and n_rows is None:
        return df
    start = skip_rows or 0
    end = None if n_rows is None else start + n_rows
    return df.iloc[start:end].reset_index(drop=True)


def _migrate_csv_zip_to_feather(df: pd.DataFrame, feather_file_path: str, csv_zip_file_path: str) -> None:
    """Best-effort backup of a freshly-read whole CSV-zip file to feather/ZSTD; the legacy CSV-zip is
    only removed once the feather write has succeeded."""
    try:
        df.to_feather(feather_file_path, compression="zstd")
    except Exception as e:
        log_w(f"Failed to back up {csv_zip_file_path} to feather/ZSTD ({feather_file_path}): {e}")
        return
    os.remove(csv_zip_file_path)


def _read_raw_data_file(
    data_frame_type: str, date_range_str: str, file_path: str, n_rows: int, skip_rows: int
) -> pd.DataFrame:
    """
    Read the flat (index not yet set) data file for (data_frame_type, date_range_str). Feather/ZSTD is
    the primary on-disk format; the legacy CSV-zip format is still readable for files that predate the
    feather migration. A whole-file CSV-zip read (no skip_rows/n_rows slicing) is transparently migrated
    to feather/ZSTD and the old CSV-zip removed, so each file only ever pays the CSV-parsing cost once.
    """
    feather_file_path = _feather_file_path(data_frame_type, date_range_str, file_path)
    if os.path.exists(feather_file_path):
        df = pd.read_feather(feather_file_path)
        return _slice_rows(df, n_rows, skip_rows)

    csv_zip_file_path = _csv_zip_file_path(data_frame_type, date_range_str, file_path)
    try:
        df = pd.read_csv(csv_zip_file_path, sep=",", header=0, skiprows=skip_rows, nrows=n_rows)
    except BadZipFile:
        os.remove(csv_zip_file_path)
        raise Exception(f"{csv_zip_file_path} is not a zip file!")

    if skip_rows is None and n_rows is None:
        _migrate_csv_zip_to_feather(df, feather_file_path, csv_zip_file_path)
    return df


def read_without_index(data_frame_type, date_range_str, file_path, n_rows, skip_rows):
    return _read_raw_data_file(data_frame_type, date_range_str, file_path, n_rows, skip_rows)


def read_by_date(data_frame_type, date_range_str, file_path, n_rows, skip_rows):
    df = _read_raw_data_file(data_frame_type, date_range_str, file_path, n_rows, skip_rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    # Convert the 'date' index to UTC if it's timezone-unaware
    if len(df) > 0:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    return df


def read_with_timeframe(
    data_frame_type: str, date_range_str: str, file_path: str, n_rows: int, skip_rows: int
) -> pd.DataFrame:
    """
    Read data from a compressed CSV file, adjusting the index based on the data frame type.

    This function reads data from a compressed CSV file based on the specified data frame type and date range.
    It adjusts the index of the resulting DataFrame according to the data frame type. If the data frame type
    includes 'multi_timeframe', it sets the index with both 'timeframe' and 'date' levels and swaps them.
    The 'date' index is assumed to be UTC.

    Parameters:
        data_frame_type (str): The type of data frame being read, such as 'ohlcv', 'ohlcva', or 'multi_timeframe_ohlcva'.
        date_range_str (str): The date range string used to generate the file name.
        file_path (str): The path to the directory containing the data file.
        n_rows (int): The maximum number of rows to read from the CSV file.
        skip_rows (int): The number of rows to skip at the beginning of the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the read data with adjusted index.

    Example:
        # Read OHLC data with adjusted index
        ohlcv_data = read_with_timeframe('ohlcv', '21-07-01.00-00T21-07-02', '/path/to/data/', n_rows=1000, skip_rows=0)
    """
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    df = read_by_date(data_frame_type, date_range_str, file_path, n_rows, skip_rows)

    if "multi_timeframe" in data_frame_type:
        df.set_index("timeframe", append=True, inplace=True)
        df = df.swaplevel()

    return df


def read_file(
    date_range_str: str,
    data_frame_type: str,
    generator: Callable,
    caster_model: type[Pandera_DFM_Type],
    skip_rows=None,
    n_rows=None,
    file_path: str = None,
    zero_size_allowed: None | bool = None,
    generator_params: dict = {},
):
    """
    Read data from a file and return a DataFrame. If the file does not exist or the DataFrame does not
    match the expected columns, the generator function is used to build and persist the DataFrame.

    This function reads data from a file with the specified `data_frame_type` and `date_range_str` parameters.
    If the file exists and the DataFrame columns match the expected columns defined in the configuration, the
    function returns the DataFrame. Otherwise, `generator` is called to build the DataFrame; this single
    wrapper (not the generator) persists it via write_data_file() and hands back the in-memory result
    directly, so a cache miss never pays for a redundant re-read of the file it just wrote (see
    cache-or-generate skill).

    An in-process LRU memo sits in front of the disk read (see cache-or-generate skill) for
    datarange_is_not_cachable()==False ranges, keyed on (data_frame_type, date_range_str, file_path,
    skip_rows, n_rows) — the exact inputs that determine the file's content.

    Parameters:
        date_range_str (str): The date range string used to construct the filename.
        data_frame_type (str): The type of DataFrame to read/write, e.g., 'ohlcva', 'multi_timeframe_ohlcva', etc.
        generator (Callable): Builds and returns the DataFrame for date_range_str; must not write it itself.
        skip_rows (Optional[int]): The number of rows to skip while reading the file.
        n_rows (Optional[int]): The maximum number of rows to read from the file.
        file_path (str, optional): The path to the directory containing the data files.

    Returns:
        pd.DataFrame: The DataFrame read from the file or generated by the generator function.

    Raises:
        Exception: If the expected columns are not defined in the configuration or if the generated DataFrame
                   does not match the expected columns.

    Example:
        # Assuming you have a generator function 'generate_ohlcva' returning a DataFrame
        df = read_file(date_range_str='17-10-06.00-00T17-10-06', data_frame_type='ohlcva',
                       generator=generate_ohlcva)

    Note:
        This function first attempts to read the file based on the provided parameters. If the file is not found
        or the DataFrame does not match the expected columns, the generator function is called to build the
        DataFrame, which is then persisted and returned as-is (no re-read).
        :param zero_size_allowed:
        :param file_path:
        :param n_rows:
        :param skip_rows:
        :param date_range_str:
        :param data_frame_type:
        :param generator:
        :param caster_model:
    """
    if file_path is None:
        file_path = symbol_data_path()
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    cachable = not datarange_is_not_cachable(date_range_str)
    cache_key = (data_frame_type, date_range_str, file_path, skip_rows, n_rows) if cachable else None
    if cache_key is not None:
        cached_df = _read_file_cache_get(cache_key)
        if cached_df is not None:
            return cached_df

    df = None
    try:
        df = read_with_timeframe(data_frame_type, date_range_str, file_path, n_rows, skip_rows)
    except FileNotFoundError:
        pass
    if zero_size_allowed is None:
        zero_size_allowed = after_under_process_date(date_range_str)
    if df is None or not cast_and_validate(df, caster_model, return_bool=True, zero_size_allowed=zero_size_allowed):
        if skip_rows or n_rows is not None:
            # generator() returns the full, already-indexed range; _slice_rows() is only meaningful
            # against the flat, not-yet-indexed frame _read_raw_data_file() produces, so combining a
            # cache miss with a partial read isn't supported (no caller does this today).
            raise Exception(
                "read_file() cannot generate on a cache miss when skip_rows/n_rows is set; "
                "pre-populate the cache with a full read first."
            )
        df = generator(date_range_str, **generator_params)
        write_data_file(df, data_frame_type, date_range_str, file_path)
    if zero_size_allowed and len(df) == 0:
        raise Exception("Zero size data!")
    df = caster_model.validate(df)

    if not cachable:
        remove_data_file(data_frame_type, date_range_str, file_path)
    elif cache_key is not None:
        _read_file_cache_put(cache_key, df)
    return df


def _resolve_caster_model(generator: Callable) -> type[Pandera_DFM_Type]:
    """Pull the pandera model read_file() should validate/write against straight from generator's own
    return-type annotation, so cache_on_disk() callers don't repeat it. Accepts either a bare model class
    (`-> OHLCV`) or a pandera DataFrame-typed annotation (`-> pt.DataFrame[OHLCV]`), unwrapping the latter
    via its type args."""
    return_hint = get_type_hints(generator).get("return")
    if return_hint is None:
        raise Exception(
            f"cache_on_disk() requires {generator.__name__}() to have a return type annotation "
            f"(the pandera model to validate/write against) — none found."
        )
    type_args = get_args(return_hint)
    return type_args[0] if type_args else return_hint


def cache_on_disk(
    file_name_prefix: str,
    after_read: Callable[[pd.DataFrame], None] | None = None,
    skip_rows=None,
    n_rows=None,
    zero_size_allowed: None | bool = None,
):
    """
    Decorate a `generator(date_range_str, file_path=None, **kwargs) -> DataFrame` function so calling it
    directly *is* the read-or-generate-and-cache entry point for this artifact type — this repo no longer
    needs a separate `read_multi_timeframe_X()` wrapper next to `generate_multi_timeframe_X()`; call the
    decorated function itself (conventionally named `get_...`, see cache-or-generate skill). caster_model
    is inferred from `generator`'s own return-type annotation via `_resolve_caster_model()`.

    `after_read`, if given, runs on the result of every call — a disk/memo hit as well as a fresh
    generation — for side effects that must reflect what was actually read, not just fresh generation
    (e.g. `cache_times()` populating `app_config.GLOBAL_CACHE` for `to_timeframe()` validation).
    """

    def decorator(generator: Callable) -> Callable:
        caster_model = _resolve_caster_model(generator)

        @wraps(generator)
        def wrapper(date_range_str: str = None, file_path: str = None, **generator_params):
            result = read_file(
                date_range_str,
                file_name_prefix,
                generator,
                caster_model,
                skip_rows=skip_rows,
                n_rows=n_rows,
                file_path=file_path,
                zero_size_allowed=zero_size_allowed,
                generator_params=generator_params,
            )
            if after_read is not None:
                after_read(result)
            return result

        return wrapper

    return decorator
