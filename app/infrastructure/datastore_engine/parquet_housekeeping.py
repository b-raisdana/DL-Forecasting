import itertools
import json
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import pyarrow.parquet as pq
from config import app_config
from domain.datastore_engine.parquet_normalization import flatten_index_to_columns
from helper.date_utils import date_range, date_range_to_string
from helper.logging.do_log import log_i, log_w
from helper.pandera import pandera_validate
from infrastructure.datastore_engine.disk_cache_layout import (
    _legacy_file_pattern,
    _window_date_range_strs,
    _window_freq,
    dataset_db_root,
)

"""
CSV/ZIP- and Feather-to-Parquet conversion, repair of already-migrated Parquet files, and compaction of
small per-window Parquet files into larger chunks for dataset_db — the datastore_engine sub-service
application.datastore_engine.parquet_housekeeping orchestrates as 3 separate jobs, all format/shape-only
(no data recomputation):

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
   going forward; `repair_parquet_file()` fixes files already on disk. Since a full scan is expensive at
   this repo's scale (17k+ files), the repair pass is trigger-gated rather than unconditional: a sentinel
   file (`flag_repair_required()`/`repair_required()`/`clear_repair_flag()`) is set by the one place that
   already detects the bug live — infrastructure.datastore_engine.disk_cache.read_by_date(), the instant
   a fresh single-file `pd.read_parquet()` read hits the KeyError: 'date' this bug causes (DuckDB's own
   read_parquet() is immune — verified it returns `date` as a plain column regardless of a file's pandas
   index metadata, so the batched windowed-read path never sees this particular failure) — and the
   application-layer repair job only runs its full scan when that flag (or an explicit `--force`) says to.

3. Compaction (`find_merge_batches`/`merge_batch`): merges small, contiguous, single-calendar-window
   Parquet files into larger ~`app_config.parquet_target_chunk_size_mb`-sized files, to cut the on-disk
   file count. Only exact single-window files are merge candidates (never a file already spanning
   multiple windows, whether a pre-windowing legacy range or a previous compaction's own output) and
   only strictly contiguous runs (no gap between one file's end window and the next's start) — merging
   across a gap would make the merged file's nominal span cover a period it doesn't actually hold data
   for, which would make disk_cache_windowed._find_covering_file()'s containment check wrongly report
   that gap as covered. disk_cache_windowed.py's read path was changed alongside this job so a merged
   file is read straight from where it sits (via DuckDB) instead of being silently re-sliced back into
   per-window tiles the first time each window in it is read again, which would otherwise undo the
   compaction — see `_covering_parquet_path()`'s docstring there.
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


@pandera_validate(allow_pandas_dataframe=True)
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


def _repair_required_sentinel(root: Path) -> Path:
    return root / ".parquet_repair_required"


def flag_repair_required(root: Path | None = None) -> None:
    """Mark that a Parquet index-repair pass is needed (see module docstring § 2). Called by
    infrastructure.datastore_engine.disk_cache's `read_by_date()` when a fresh single-file
    `pd.read_parquet()` read hits the KeyError this bug causes — the one place that already detects it
    live, on real reads, for free. A plain sentinel file (`Path.touch()`, atomic, no read-modify-write
    race) rather than a per-file registry: once triggered, `repair_parquet_file()`'s own scan already
    finds every affected file correctly, so there's nothing to gain from also tracking which exact file
    caused the trigger."""
    root = root if root is not None else dataset_db_root()
    root.mkdir(parents=True, exist_ok=True)
    _repair_required_sentinel(root).touch()


def repair_required(root: Path | None = None) -> bool:
    """Whether flag_repair_required() has been called since the last clear_repair_flag()."""
    root = root if root is not None else dataset_db_root()
    return _repair_required_sentinel(root).exists()


def clear_repair_flag(root: Path | None = None) -> None:
    root = root if root is not None else dataset_db_root()
    _repair_required_sentinel(root).unlink(missing_ok=True)


def _leaf_type_dirs(root: Path) -> dict[Path, str]:
    """Every directory under root holding at least one Parquet file, mapped to its data_frame_type —
    the root-relative path's first segment, per dataset_db's data_frame_type-first layout (see
    disk_cache_layout.py module docstring)."""
    dirs: dict[Path, str] = {}
    if not root.is_dir():
        return dirs
    for entry in root.rglob("*.parquet"):
        parent = entry.parent
        if parent not in dirs:
            dirs[parent] = parent.relative_to(root).parts[0]
    return dirs


def _split_run_by_size(
    data_frame_type: str, run: list[str], range_to_path: dict[str, Path], target_bytes: int
) -> list[tuple[str, list[Path]]]:
    """Greedily accumulate one contiguous run of window range strings into batches whose summed on-disk
    size crosses target_bytes, flushing each as soon as it does (on-disk ZSTD size of the sources is a
    reasonable proxy for the merged file's size — same data, same compressor). A trailing batch of just
    1 file is dropped — nothing to merge; likewise a single file already >= target_bytes on its own."""
    batches: list[tuple[str, list[Path]]] = []
    current: list[Path] = []
    current_size = 0
    for window_range in run:
        path = range_to_path[window_range]
        current.append(path)
        current_size += path.stat().st_size
        if current_size >= target_bytes:
            if len(current) > 1:
                batches.append((data_frame_type, current))
            current, current_size = [], 0
    if len(current) > 1:
        batches.append((data_frame_type, current))
    return batches


def _contiguous_merge_batches(type_dir: Path, data_frame_type: str, target_bytes: int) -> list[tuple[str, list[Path]]]:
    """Every merge-worthy batch within one (data_frame_type, market, trading_pair, exchange) directory —
    see module docstring § 3 for what counts as a merge candidate."""
    pattern = _legacy_file_pattern(data_frame_type)
    range_to_path: dict[str, Path] = {}
    for entry in type_dir.iterdir():
        match = pattern.match(entry.name)
        if match and match.group("ext") == "parquet":
            range_to_path[match.group("range")] = entry
    if not range_to_path:
        return []

    spans = {range_str: date_range(range_str) for range_str in range_to_path}
    overall_start = min(start for start, _end in spans.values())
    overall_end = max(end for _start, end in spans.values())
    window_freq = _window_freq(data_frame_type)
    window_ranges = _window_date_range_strs(date_range_to_string(start=overall_start, end=overall_end), window_freq)
    own_file_indices = [i for i, window_range in enumerate(window_ranges) if window_range in range_to_path]

    batches: list[tuple[str, list[Path]]] = []
    for _, group in itertools.groupby(enumerate(own_file_indices), lambda pair: pair[1] - pair[0]):
        run = [window_ranges[index] for _, index in group]
        batches.extend(_split_run_by_size(data_frame_type, run, range_to_path, target_bytes))
    return batches


def find_merge_batches(root: Path | None = None) -> list[tuple[str, list[Path]]]:
    """Every (data_frame_type, files) batch under dataset_db_root() (or `root`, for tests) worth merging
    into one ~app_config.parquet_target_chunk_size_mb file — see module docstring § 3. "Contiguous" is
    computed over the data_frame_type's own window index sequence (mirrors
    disk_cache_gaps.find_cache_gaps()'s groupby-on-window-indices for missing windows, here grouping
    present own-file windows instead), not by comparing raw start/end datetimes — a window's own end
    (e.g. 23:59) and the next window's start (00:00 the next day) are never exactly equal, so raw
    datetime adjacency would never match."""
    root = root if root is not None else dataset_db_root()
    target_bytes = app_config.parquet_target_chunk_size_mb * 1024 * 1024
    batches: list[tuple[str, list[Path]]] = []
    for type_dir, data_frame_type in _leaf_type_dirs(root).items():
        batches.extend(_contiguous_merge_batches(type_dir, data_frame_type, target_bytes))
    return batches


def merge_batch(data_frame_type: str, files: list[Path]) -> Path | None:
    """
    Merge one contiguous batch of single-window Parquet files (from find_merge_batches(), already in
    chronological order) into one new file spanning the whole batch. Verified — merged row count matches
    the sum of inputs, and the merged file doesn't carry the index-as-column bug — before the small
    source files are deleted, same "verify success before deleting source" pattern as
    `_migrate_to_parquet()`/`repair_parquet_file()` above. Returns the merged file's path on success, or
    None on any failure (logged, sources untouched, so one bad batch never aborts a run — see
    application.datastore_engine.parquet_housekeeping.run_compaction()).
    """
    pattern = _legacy_file_pattern(data_frame_type)
    first_match = pattern.match(files[0].name)
    last_match = pattern.match(files[-1].name)
    if first_match is None or last_match is None:
        log_w(f"parquet_housekeeping: batch file name doesn't match {data_frame_type} pattern, skipping merge")
        return None
    start, _ = date_range(first_match.group("range"))
    _, end = date_range(last_match.group("range"))
    merged_path = files[0].with_name(f"{data_frame_type}.{date_range_to_string(start=start, end=end)}.parquet")
    if merged_path.exists():
        log_w(f"parquet_housekeeping: {merged_path} already exists, skipping merge of {len(files)} file(s)")
        return None

    try:
        frames = [pd.read_parquet(path) for path in files]
    except Exception as e:
        log_w(f"parquet_housekeeping: failed reading batch to merge into {merged_path}: {e}")
        return None
    expected_rows = sum(len(frame) for frame in frames)
    merged_df = flatten_index_to_columns(pd.concat(frames, ignore_index=True))
    try:
        merged_df.to_parquet(merged_path, compression="zstd")
    except Exception as e:
        log_w(f"parquet_housekeeping: failed writing merged file {merged_path}: {e}")
        return None
    if len(merged_df) != expected_rows or parquet_file_has_index_bug(merged_path):
        merged_path.unlink(missing_ok=True)
        log_w(f"parquet_housekeeping: merge of {len(files)} file(s) into {merged_path} failed verification")
        return None

    for path in files:
        path.unlink()
    log_i(f"parquet_housekeeping: merged {len(files)} file(s) into {merged_path.resolve()}")
    return merged_path
