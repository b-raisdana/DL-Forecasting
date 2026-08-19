import contextlib
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from typing import TypedDict, cast, get_args, get_type_hints
from zipfile import BadZipFile

import pandas as pd
import pandera
import pytz
from config import app_config
from helper.data_preparation import after_under_process_date
from helper.functions import Pandera_DFM_Type, date_range, morning
from helper.logging.do_log import log_i, log_w
from helper.schema_casting import cast_and_validate
from infrastructure.datastore_engine.disk_cache_layout import DATASET_DB as DATASET_DB
from infrastructure.datastore_engine.disk_cache_layout import CachableDataset as CachableDataset
from infrastructure.datastore_engine.disk_cache_layout import FilePathArg as FilePathArg
from infrastructure.datastore_engine.disk_cache_layout import _csv_zip_file_path as _csv_zip_file_path
from infrastructure.datastore_engine.disk_cache_layout import _data_frame_type_dir as _data_frame_type_dir
from infrastructure.datastore_engine.disk_cache_layout import _data_frame_type_of as _data_frame_type_of
from infrastructure.datastore_engine.disk_cache_layout import _disallowed_nan_columns as _disallowed_nan_columns
from infrastructure.datastore_engine.disk_cache_layout import _feather_file_path as _feather_file_path
from infrastructure.datastore_engine.disk_cache_layout import _Generator as _Generator
from infrastructure.datastore_engine.disk_cache_layout import _legacy_file_pattern as _legacy_file_pattern
from infrastructure.datastore_engine.disk_cache_layout import _parquet_file_path as _parquet_file_path
from infrastructure.datastore_engine.disk_cache_layout import _schema_nullable_columns as _schema_nullable_columns
from infrastructure.datastore_engine.disk_cache_layout import _window_freq as _window_freq
from infrastructure.datastore_engine.disk_cache_layout import add_timeframe_index as add_timeframe_index
from infrastructure.datastore_engine.disk_cache_layout import (
    cleanup_redundant_cache_files as cleanup_redundant_cache_files,
)
from infrastructure.datastore_engine.disk_cache_layout import index_by_date as index_by_date
from infrastructure.datastore_engine.disk_cache_layout import symbol_data_path as symbol_data_path
from infrastructure.datastore_engine.parquet_housekeeping import _migrate_to_parquet as _migrate_to_parquet

"""
Everything this repo needs to persist/read a (data_frame_type, date_range_str) artifact as a
Parquet/ZSTD (or legacy Feather/ZSTD/CSV-zip) file and never recompute or re-fetch it twice within its
validity scope — the disk-level half of the cache-or-generate skill. `cache_on_disk` is the primitive
new artifact types should use; `read_file`/`write_data_file` are its building blocks for callers (like
domain.schemas.common.ExtendedDf.read_file) that need a class-bound variant instead of a decorator.

Calendar-windowed caching (`cache_on_disk(..., windowed=True)` / `read_file_windowed()`), the legacy-
file reuse it does on a miss, and the disk-generation-rate monitor are documented in detail in
infrastructure/ohlcv/README.md (its examples are OHLCV/OHLCVA-specific; this module itself is
generic) — read that first if you're touching any of the `_window_*`/`*_cache_generation*`/
`cleanup_redundant_cache_files` functions below.
"""


# What cache_on_disk()'s decorator hands back: unlike _Generator, this always returns a pd.DataFrame
# at runtime (read_file()/read_file_windowed() do), so callers of a decorated `get_...` function get
# an accurate type back instead of _Generator's deliberately-loose `object`.
_Wrapper = Callable[..., pd.DataFrame]  # type: ignore[explicit-any]


# @cache
def datarange_is_not_cachable(date_range_str: str) -> bool:
    _, end = date_range(date_range_str)
    return end > morning(datetime.utcnow().replace(tzinfo=pytz.UTC))


# In-process memo in front of read_file()'s disk read, keyed on the exact args that determine the file's
# content. Bounded LRU so a long multi-quarter/multi-symbol run doesn't grow this unboundedly. Never used
# for a datarange_is_not_cachable() range — those touch the live/incomplete present and must always be
# re-read fresh, per read_file()'s existing delete-after-use behavior for such ranges. See the
# cache-or-generate skill.
_READ_FILE_CACHE_MAX_ENTRIES = 32
_ReadFileCacheKey = tuple[str, str, FilePathArg, int | None, int | None]
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


# Per-data_frame_type disk-generation-rate monitor (see README.md § monitoring). In-process only —
# resets on restart, same tradeoff as _read_file_cache above. Evaluated lazily on write, at most once
# per app_config.cache_generation_monitor_interval_minutes per prefix, so it never re-evaluates sooner
# than that regardless of how many files a given prefix writes in between.
class _CacheGenerationState(TypedDict):
    window_start: datetime
    bytes: int


_cache_generation_state: dict[str, _CacheGenerationState] = {}


def _record_cache_generation(file_name_prefix: str, written_bytes: int) -> None:
    now = datetime.now(pytz.UTC)
    state = _cache_generation_state.setdefault(file_name_prefix, {"window_start": now, "bytes": 0})
    state["bytes"] = state["bytes"] + written_bytes
    interval = timedelta(minutes=app_config.cache_generation_monitor_interval_minutes)
    elapsed = now - state["window_start"]
    if elapsed < interval:
        return
    daily_rate_bytes = state["bytes"] / (elapsed.total_seconds() / 86400)
    threshold = app_config.cache_generation_warn_bytes_per_day
    if daily_rate_bytes > threshold:
        log_w(
            f"disk_cache: '{file_name_prefix}' cache files generated at ~{daily_rate_bytes / 1e9:.2f} "
            f"GB/24h (measured over the last {elapsed}), above the {threshold / 1e9:.2f} GB/24h threshold. "
            f"Possible cache-window misconfiguration or redundant regeneration — see README.md § monitoring."
        )
    state["window_start"] = now
    state["bytes"] = 0


def write_data_file(
    df: pd.DataFrame,
    data_frame_type: str,
    date_range_str: str,
    file_path: FilePathArg,
    nan_allowed_columns: frozenset[str] | None = None,
) -> None:
    """
    Persist df as the primary Parquet/ZSTD artifact for (data_frame_type, date_range_str) — the
    write-side counterpart to _read_raw_data_file()'s parquet-first read. Every generator writes
    through here instead of hand-rolling a CSV-zip write, so newly generated data lands as
    Parquet/ZSTD from the first write, with no CSV-zip/feather round-trip. Lands under
    _data_frame_type_dir(data_frame_type, file_path), not flat in file_path.

    nan_allowed_columns, when given (not None), gates a write-time guard: any other column
    containing NaN raises rather than silently caching it. None (the default, used by callers outside
    the cache_on_disk path — e.g. legacy-file seeding/migration) skips the guard entirely. The
    cache_on_disk-driven generator path always passes a resolved set — see read_file().
    """
    if nan_allowed_columns is not None:
        offending = _disallowed_nan_columns(df, nan_allowed_columns)
        if offending:
            raise Exception(
                f"Refusing to cache NaN in column(s) {offending} for '{data_frame_type}' {date_range_str} — "
                f"declare them nullable in the pandera schema, or pass nan_allowed_columns=[...] to "
                f"cache_on_disk(), if NaN is genuinely expected there."
            )
    parquet_file_path = _parquet_file_path(data_frame_type, date_range_str, file_path)
    df.reset_index().to_parquet(parquet_file_path, compression="zstd")
    log_i(f"wrote Parquet/ZSTD cache file: {parquet_file_path.resolve()}")
    _record_cache_generation(data_frame_type, parquet_file_path.stat().st_size)


def remove_data_file(data_frame_type: str, date_range_str: str, file_path: FilePathArg) -> None:
    """
    Delete whichever on-disk format (Parquet/ZSTD primary, or legacy Feather/ZSTD or CSV-zip)
    currently backs this (data_frame_type, date_range_str) — used for date ranges
    datarange_is_not_cachable() flags as touching the live/incomplete present, which must never be
    left on disk between reads.
    """
    try:
        _parquet_file_path(data_frame_type, date_range_str, file_path).unlink()
    except FileNotFoundError:
        try:
            _feather_file_path(data_frame_type, date_range_str, file_path).unlink()
        except FileNotFoundError:
            _csv_zip_file_path(data_frame_type, date_range_str, file_path).unlink()


def _slice_rows(df: pd.DataFrame, n_rows: int | None, skip_rows: int | None) -> pd.DataFrame:
    if not skip_rows and n_rows is None:
        return df
    start = skip_rows or 0
    end = None if n_rows is None else start + n_rows
    return df.iloc[start:end].reset_index(drop=True)


def _read_raw_data_file(
    data_frame_type: str, date_range_str: str, file_path: FilePathArg, n_rows: int | None, skip_rows: int | None
) -> pd.DataFrame:
    """
    Read the flat (index not yet set) data file for (data_frame_type, date_range_str). Parquet/ZSTD is
    the primary on-disk format; the legacy Feather/ZSTD and CSV-zip formats are still readable for
    files that predate the Parquet migration. A whole-file legacy read (no skip_rows/n_rows slicing)
    is transparently migrated to Parquet/ZSTD (CSV-zip migrates straight to Parquet, skipping the
    Feather tier) and the old file removed, so each file only ever pays the feather/CSV-parsing cost
    once.
    """
    parquet_file_path = _parquet_file_path(data_frame_type, date_range_str, file_path)
    if parquet_file_path.exists():
        df = pd.read_parquet(parquet_file_path)
        return _slice_rows(df, n_rows, skip_rows)

    feather_file_path = _feather_file_path(data_frame_type, date_range_str, file_path)
    if feather_file_path.exists():
        df = pd.read_feather(feather_file_path)
        if skip_rows is None and n_rows is None:
            _migrate_to_parquet(df, parquet_file_path, feather_file_path, flatten=True)
        return _slice_rows(df, n_rows, skip_rows)

    csv_zip_file_path = _csv_zip_file_path(data_frame_type, date_range_str, file_path)
    try:
        df = pd.read_csv(csv_zip_file_path, sep=",", header=0, skiprows=skip_rows, nrows=n_rows)
    except BadZipFile as err:
        csv_zip_file_path.unlink()
        raise Exception(f"{csv_zip_file_path} is not a zip file!") from err

    if skip_rows is None and n_rows is None:
        _migrate_to_parquet(df, parquet_file_path, csv_zip_file_path, flatten=False)
    return df


def read_without_index(
    data_frame_type: str, date_range_str: str, file_path: FilePathArg, n_rows: int | None, skip_rows: int | None
) -> pd.DataFrame:
    return _read_raw_data_file(data_frame_type, date_range_str, file_path, n_rows, skip_rows)


def read_by_date(
    data_frame_type: str, date_range_str: str, file_path: FilePathArg, n_rows: int | None, skip_rows: int | None
) -> pd.DataFrame:
    df = _read_raw_data_file(data_frame_type, date_range_str, file_path, n_rows, skip_rows)
    return index_by_date(df)


def read_with_timeframe(
    data_frame_type: str, date_range_str: str, file_path: FilePathArg, n_rows: int | None, skip_rows: int | None
) -> pd.DataFrame:
    """
    Read data from a compressed CSV file, adjusting the index based on the data frame type.

    This function reads data from a compressed CSV file based on the specified data frame type and date range.
    It adjusts the index of the resulting DataFrame according to the data frame type. If the data frame type
    includes 'multi_timeframe', it sets the index with both 'timeframe' and 'date' levels and swaps them.
    The 'date' index is assumed to be UTC.

    Parameters:
        data_frame_type: Data-frame type, e.g. `ohlcv`, `ohlcva`, or `multi_timeframe_ohlcva`.
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
    return add_timeframe_index(df, data_frame_type)


def read_file(
    date_range_str: str | None,
    data_frame_type: str,
    generator: _Generator,
    caster_model: type[Pandera_DFM_Type],
    skip_rows: int | None = None,
    n_rows: int | None = None,
    file_path: str | None = None,
    zero_size_allowed: None | bool = None,
    generator_params: dict[str, object] | None = None,
    nan_allowed_columns: frozenset[str] | None = None,
) -> pd.DataFrame:
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

    A freshly-generated df is never cached with an unexpected NaN in it: the columns allowed to
    contain NaN are caster_model's own pandera nullable=True columns, unioned with
    nan_allowed_columns (extra columns to allow beyond what the schema already declares nullable —
    see cache_on_disk(nan_allowed_columns=...)). Any other column containing NaN raises rather than
    silently caching it.

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
    if generator_params is None:
        generator_params = {}
    # file_path stays str | None for the public contract; resolved_file_path is what every downstream
    # call actually uses, widened to also allow DATASET_DB (the data_frame_type-first dataset_db_root()
    # resolution — see disk_cache_layout.DatasetDbSentinel) — the default every real caller gets today.
    resolved_file_path: FilePathArg = file_path if file_path is not None else DATASET_DB
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    cachable = not datarange_is_not_cachable(date_range_str)
    cache_key = (data_frame_type, date_range_str, resolved_file_path, skip_rows, n_rows) if cachable else None
    if cache_key is not None:
        cached_df = _read_file_cache_get(cache_key)
        if cached_df is not None:
            return cached_df

    df = None
    with contextlib.suppress(FileNotFoundError):
        df = read_with_timeframe(data_frame_type, date_range_str, resolved_file_path, n_rows, skip_rows)
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
        df = cast(pd.DataFrame, generator(date_range_str, **generator_params))
        write_data_file(
            df,
            data_frame_type,
            date_range_str,
            resolved_file_path,
            nan_allowed_columns=_schema_nullable_columns(caster_model) | frozenset(nan_allowed_columns or ()),
        )
    if zero_size_allowed and len(df) == 0:
        raise Exception("Zero size data!")
    df = caster_model.validate(df)

    if not cachable:
        remove_data_file(data_frame_type, date_range_str, resolved_file_path)
    elif cache_key is not None:
        _read_file_cache_put(cache_key, df)
    return df


def _resolve_caster_model(generator: _Generator) -> type[Pandera_DFM_Type]:
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
    return cast("type[Pandera_DFM_Type]", type_args[0] if type_args else return_hint)


def cache_on_disk(
    dataset: CachableDataset,
    after_read: Callable[[pd.DataFrame], None] | None = None,
    skip_rows: int | None = None,
    n_rows: int | None = None,
    windowed: bool = False,
) -> Callable[[_Generator], _Wrapper]:
    """
    Decorate a `generator(date_range_str, file_path=None, **kwargs) -> DataFrame` function so calling it
    directly *is* the read-or-generate-and-cache entry point for this artifact type — this repo no longer
    needs a separate `read_multi_timeframe_X()` wrapper next to `generate_multi_timeframe_X()`; call the
    decorated function itself (conventionally named `get_...`, see cache-or-generate skill).

    `dataset` is a CachableDataset declared with just `dataset_folder_name` (plus, rarely,
    `nan_allowed_columns`/`zero_size_allowed` — see CachableDataset's own docstring); its
    `generator`/`caster_model` fields stay unset at the decorator site — `caster_model` is inferred
    from `generator`'s own return-type annotation via `_resolve_caster_model()` unless the dataset
    already sets one. Keep a reference to `dataset` (e.g. a module-level `OHLCV_DATASET`) so other
    call sites (find_cache_gaps(), cleanup_redundant_cache_files(), ...) can reuse it instead of
    repeating the data_frame_type string.

    `after_read`, if given, runs on the result of every call — a disk/memo hit as well as a fresh
    generation — for side effects that must reflect what was actually read, not just fresh generation
    (e.g. `cache_times()` populating `app_config.GLOBAL_CACHE` for `to_timeframe()` validation).

    `windowed=True` makes `generator` a per-window generator (see README.md § windowing): the decorated
    function accepts an arbitrary `date_range_str`, decomposes it into whole calendar windows sized per
    `app_config.cache_window_freq_overrides[dataset.dataset_folder_name]` (default
    `app_config.default_cache_window_freq`), and calls `generator` once per window via
    `read_file_windowed()` instead of once for the whole range. Not compatible with `skip_rows`/`n_rows`
    (same restriction `read_file()` already has on a cache miss).
    """

    def decorator(generator: _Generator) -> _Wrapper:
        caster_model: type[pandera.DataFrameModel] = dataset.caster_model or _resolve_caster_model(generator)

        @wraps(generator)
        def wrapper(
            date_range_str: str | None = None, file_path: str | None = None, **generator_params: object
        ) -> pd.DataFrame:
            if windowed:
                if skip_rows or n_rows is not None:
                    raise Exception("cache_on_disk(windowed=True) does not support skip_rows/n_rows.")
                # Deferred import: disk_cache_windowed.py imports read_file()/write_data_file()/etc.
                # from this module at load time, so importing back here at module level would cycle —
                # deferring to call-time (mirrors disk_cache_gaps.py's one-way dependency the other way).
                from infrastructure.datastore_engine.disk_cache_windowed import read_file_windowed

                result = read_file_windowed(
                    date_range_str,
                    dataset.dataset_folder_name,
                    generator,
                    caster_model,
                    file_path=file_path,
                    zero_size_allowed=dataset.zero_size_allowed,
                    generator_params=generator_params,
                    nan_allowed_columns=dataset.nan_allowed_columns,
                )
            else:
                result = read_file(
                    date_range_str,
                    dataset.dataset_folder_name,
                    generator,
                    caster_model,
                    skip_rows=skip_rows,
                    n_rows=n_rows,
                    file_path=file_path,
                    zero_size_allowed=dataset.zero_size_allowed,
                    generator_params=generator_params,
                    nan_allowed_columns=dataset.nan_allowed_columns,
                )
            if after_read is not None:
                after_read(result)
            return result

        return wrapper

    return decorator
