import json
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import pyarrow.parquet as pq
from domain.datastore_engine.parquet_normalization import flatten_index_to_columns
from helper.logging.do_log import log_i, log_w
from infrastructure.datastore_engine.disk_cache_layout import dataset_db_root

"""
CSV/ZIP- and Feather-to-Parquet conversion for dataset_db, plus repair of already-migrated Parquet
files — the datastore_engine sub-service application.datastore_engine.parquet_housekeeping orchestrates.
Two jobs, both format/shape-only (no data recomputation):

1. Legacy conversion (`find_legacy_files`/`convert_legacy_file`): the same on-touch migration
   disk_cache.py's `_read_raw_data_file()` already runs lazily on a whole-file read (`_migrate_to_parquet()`,
   moved here from disk_cache_layout.py per project-decisions skill § code layers "Splitting an oversized
   file" — disk_cache.py imports it from here now, not disk_cache_layout.py, to avoid a disk_cache_layout
   <->parquet_housekeeping cycle), exposed as an explicit batch pass over whatever legacy files remain
   instead of waiting for a read.

2. Parquet repair (`find_parquet_files`/`repair_parquet_file`): fixes a pre-existing bug where a large
   fraction of already-migrated Parquet files under data/dataset_db/ were written with `date` (and, for
   multi_timeframe_* types, `timeframe`) still set as the pandas index rather than flattened to plain
   columns — `_migrate_to_parquet(flatten=True)` used to `df.to_parquet(...)` the df exactly as read back
   from a legacy Feather file, and old Feather files predating today's flat-write convention had
   already been written with `date` as their own index, so read_feather() restored that index and the
   migration just carried the bad shape into Parquet. That shape is silently accepted by pandas
   (`pd.read_parquet()` just restores the persisted index) but crashes
   disk_cache_layout.index_by_date()/read_by_date() with KeyError: 'date' the first time anything tries
   to read it. `_migrate_to_parquet(flatten=True)` below now flattens before writing, so this can't recur
   going forward; `repair_parquet_file()` fixes files already on disk.
"""


def find_legacy_files(root: Path | None = None) -> list[Path]:
    """Every legacy Feather/ZSTD or CSV-zip cache file still on disk under dataset_db_root() (or
    `root`, for tests) — every file under there is a disk_cache-managed artifact (see
    data/dataset_db/README.md), so a plain extension glob is enough; no filename-pattern parsing
    needed."""
    root = root if root is not None else dataset_db_root()
    if not root.is_dir():
        return []
    return [*root.rglob("*.feather"), *root.rglob("*.zip")]


def find_parquet_files(root: Path | None = None) -> list[Path]:
    """Every on-disk Parquet cache file under dataset_db_root() (or `root`, for tests)."""
    root = root if root is not None else dataset_db_root()
    if not root.is_dir():
        return []
    return list(root.rglob("*.parquet"))


def _migrate_to_parquet(df: pd.DataFrame, parquet_file_path: Path, source_file_path: Path, *, flatten: bool) -> None:
    """Best-effort backup of a freshly-read whole legacy cache file (Feather/ZSTD or CSV-zip) to
    Parquet/ZSTD; the old file is only removed once the parquet write has succeeded. `flatten=True`
    (Feather) restores the standard flat/default-index shape first (see module docstring).
    `flatten=False` (CSV-zip) skips that call outright rather than relying on
    flatten_index_to_columns()'s own no-op path — pd.read_csv() never sets a custom index, so the call
    could never do anything there (see project-decisions skill § no-no-op-calls)."""
    try:
        (flatten_index_to_columns(df) if flatten else df).to_parquet(parquet_file_path, compression="zstd")
    except Exception as e:
        log_w(f"Failed to back up {source_file_path} to Parquet/ZSTD ({parquet_file_path}): {e}")
        return
    log_i(f"wrote Parquet/ZSTD cache file: {parquet_file_path.resolve()}")
    source_file_path.unlink()


def convert_legacy_file(path: Path) -> Path | None:
    """Convert one legacy cache file at `path` (Feather/ZSTD or CSV-zip, identified by extension) to
    Parquet/ZSTD in place, reusing the exact same migrate helpers disk_cache.py's on-touch read path
    calls — so a batch run and a lazy single-file read behave identically. Returns the new Parquet path
    on success, or None if the source file was missing/already converted/unreadable (logged, not
    raised, so one bad file never aborts a batch — see application.datastore_engine.parquet_housekeeping.
    run_datastore_conversion())."""
    parquet_path = path.with_suffix(".parquet")
    if path.suffix == ".feather":
        try:
            df = pd.read_feather(path)
        except Exception as e:
            log_w(f"parquet_housekeeping: failed reading legacy Feather file {path}: {e}")
            return None
        _migrate_to_parquet(df, parquet_path, path, flatten=True)
    elif path.suffix == ".zip":
        try:
            df = pd.read_csv(path, sep=",", header=0)
        except BadZipFile as e:
            log_w(f"parquet_housekeeping: {path} is not a valid zip file: {e}")
            return None
        _migrate_to_parquet(df, parquet_path, path, flatten=False)
    else:
        return None
    return parquet_path if parquet_path.exists() else None


def _persisted_index_columns(path: Path) -> list[str]:
    """Named pandas index level(s) a Parquet file's own metadata records, read from the file's schema
    only — no row data touched, so this is cheap enough to run over every file in dataset_db_root() on
    every scan. A default RangeIndex records a `{"kind": "range", ...}` dict here, not a string; a real
    index level (the bug this module repairs) records its column name as a plain string."""
    try:
        metadata = pq.read_schema(path).metadata or {}
    except Exception as e:
        raise Exception(f"Could not read Parquet schema for {path}: {e}") from e
    raw_pandas_metadata = metadata.get(b"pandas")
    if not raw_pandas_metadata:
        return []
    index_columns = json.loads(raw_pandas_metadata).get("index_columns", [])
    return [column for column in index_columns if isinstance(column, str)]


def parquet_file_has_index_bug(path: Path) -> bool:
    """True when `path` was written with a real column set as the pandas index (see module docstring)
    instead of the flat/default-RangeIndex shape every dataset_db cache file should have."""
    return bool(_persisted_index_columns(path))


def repair_parquet_file(path: Path, dry_run: bool = False) -> bool:
    """Flatten `path` back to the standard flat/default-index shape if it has the index-as-column bug
    (see module docstring); a no-op, returning False, for a file that's already fine. Repair is atomic:
    the fixed frame is written to a sibling temp file, verified to have actually cleared the persisted
    index metadata, and only then swapped in over the original via `Path.replace()` — so a crash mid-
    write, or a repair that somehow didn't fix the shape, never leaves `path` in a half-written or still-
    broken state. `dry_run=True` reports whether a file needs repair without writing anything.
    """
    if not parquet_file_has_index_bug(path):
        return False
    if dry_run:
        return True

    df = pd.read_parquet(path)
    fixed_df = flatten_index_to_columns(df)
    tmp_path = path.with_name(path.name + ".repair.tmp")
    fixed_df.to_parquet(tmp_path, compression="zstd")
    if parquet_file_has_index_bug(tmp_path):
        tmp_path.unlink()
        raise Exception(f"Repair of {path} did not clear its persisted index metadata")
    tmp_path.replace(path)
    log_i(f"parquet_housekeeping: repaired date-index Parquet file {path}")
    return True
