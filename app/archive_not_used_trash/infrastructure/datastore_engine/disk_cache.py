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
from infrastructure.datastore_engine.parquet_housekeeping import flag_repair_required as flag_repair_required

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


# In-process memo in front of read_file()'s disk read, keyed on the exact args that determine the file's
# content. Bounded LRU so a long multi-quarter/multi-symbol run doesn't grow this unboundedly. Never used
# for a datarange_is_not_cachable() range — those touch the live/incomplete present and must always be
# re-read fresh, per read_file()'s existing delete-after-use behavior for such ranges. See the
# cache-or-generate skill.
_READ_FILE_CACHE_MAX_ENTRIES = 32
_ReadFileCacheKey = tuple[str, str, FilePathArg, int | None, int | None]
_read_file_cache: "OrderedDict[_ReadFileCacheKey, pd.DataFrame]" = OrderedDict()


# Per-data_frame_type disk-generation-rate monitor (see README.md § monitoring). In-process only —
# resets on restart, same tradeoff as _read_file_cache above. Evaluated lazily on write, at most once
# per app_config.cache_generation_monitor_interval_minutes per prefix, so it never re-evaluates sooner
# than that regardless of how many files a given prefix writes in between.


# unused here: read_without_index()/_read_raw_data_file() below don't reference it, only the live
# _record_cache_generation() (stayed in app/infrastructure/datastore_engine/disk_cache.py) does.
# _cache_generation_state: dict[str, _CacheGenerationState] = {}


def read_without_index(
    data_frame_type: str, date_range_str: str, file_path: FilePathArg, n_rows: int | None, skip_rows: int | None
) -> pd.DataFrame:
    return _read_raw_data_file(data_frame_type, date_range_str, file_path, n_rows, skip_rows)


# duplicated from app/infrastructure/datastore_engine/disk_cache.py (still live there; a dead function here depends on it)
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


# duplicated from app/infrastructure/datastore_engine/disk_cache.py (still live there; a dead function here depends on it)
def _slice_rows(df: pd.DataFrame, n_rows: int | None, skip_rows: int | None) -> pd.DataFrame:
    if not skip_rows and n_rows is None:
        return df
    start = skip_rows or 0
    end = None if n_rows is None else start + n_rows
    return df.iloc[start:end].reset_index(drop=True)
