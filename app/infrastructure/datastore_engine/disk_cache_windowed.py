from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pytz
from config import app_config
from helper.data_preparation import trim_to_date_range
from helper.functions import Pandera_DFM_Type, date_range
from helper.logging.do_log import log_w
from helper.pandera import pandera_transform
from helper.schema_casting import apply_as_type, empty_df
from infrastructure.datastore_engine.disk_cache import (
    DATASET_DB,
    FilePathArg,
    _csv_zip_file_path,
    _data_frame_type_dir,
    _feather_file_path,
    _Generator,
    _legacy_file_pattern,
    _parquet_file_path,
    _window_freq,
    datarange_is_not_cachable,
    read_file,
    read_with_timeframe,
)
from infrastructure.datastore_engine.disk_cache_layout import _window_date_range_strs as _window_date_range_strs
from infrastructure.datastore_engine.duckdb_reader import read_parquet_files

"""
Calendar-windowed counterpart to disk_cache.read_file() — split out from disk_cache.py itself
(project-decisions skill § code layers, "Splitting an oversized file") once it crossed the 500-line
soft cap, mirroring disk_cache_gaps.py's/disk_cache_layout.py's earlier splits for the same reason.
Depends on disk_cache.py (read_file, write_data_file, read_with_timeframe — the single-file
primitives every window is built from); disk_cache.py's own cache_on_disk() wrapper imports
read_file_windowed back from here via a deferred (in-function) import to avoid a module-load-time
cycle, the same shape disk_cache_gaps.py already has in the other direction.

See README.md § windowing for the full design; docs/infrastructure.md § DuckDB and
data/dataset_db/README.md for why the already-cached-windows path batches through
infrastructure.duckdb_reader instead of one pd.read_parquet() call per window.
"""


def _find_covering_file(
    data_frame_type: str, window_start: datetime, window_end: datetime, file_path: FilePathArg
) -> tuple[str, str] | None:
    """
    Smallest on-disk (range, ext) for data_frame_type whose own date range fully contains
    [window_start, window_end], if any. "Smallest" minimizes the read cost of the reuse in
    _covering_parquet_path() — a window is usually covered by many nested legacy ranges.
    """
    pattern = _legacy_file_pattern(data_frame_type)
    type_dir = _data_frame_type_dir(data_frame_type, file_path)
    if not type_dir.is_dir():
        return None
    best: tuple[str, str, datetime, datetime] | None = None
    for entry in type_dir.iterdir():
        match = pattern.match(entry.name)
        if not match:
            continue
        candidate_range = match.group("range")
        candidate_start, candidate_end = date_range(candidate_range)
        if (
            candidate_start <= window_start
            and candidate_end >= window_end
            and (best is None or (candidate_end - candidate_start) < (best[3] - best[2]))
        ):
            best = (candidate_range, match.group("ext"), candidate_start, candidate_end)
    if best is None:
        return None
    return best[0], best[1]


def _covering_parquet_path(data_frame_type: str, window_date_range_str: str, file_path: FilePathArg) -> Path | None:
    """
    If window_date_range_str's own canonical file is missing, look for an existing file that fully
    covers it — either a parquet_housekeeping-compacted multi-window file, or a (typically
    pre-windowing, arbitrary-range) legacy file in any format (README.md § backward compatibility). If
    found, returns the covering file's own Parquet path (migrating a Feather/CSV-zip covering file to
    Parquet first, as an on-touch side effect of read_with_timeframe() — same migration
    _read_raw_data_file() already does on any whole-file read) — read_file_windowed()'s caller reads
    straight from that path via DuckDB.

    Deliberately never writes a separate, smaller per-window slice of the covering file (unlike this
    function's earlier behavior): read_file_windowed() may resolve several windows to the very same
    covering file within one call, and once an earlier window's own tile existed while a later window
    still pointed at the (now-Parquet) covering file, the same rows got read twice — once from each
    file — in the batched DuckDB query. Returning only the covering path, uniformly for every window
    that resolves to it, keeps exactly one file backing that whole span. This also means a compacted
    file is never silently re-fragmented back into small per-window tiles the first time each window in
    it is read again, which would otherwise undo compaction.

    Returns None if window_date_range_str already has its own canonical file, or if no covering file
    exists at all. Never touches the covering file's own date range/identity either way — a separate,
    explicit cleanup_redundant_cache_files() pass removes files that become fully redundant.
    """
    if (
        _parquet_file_path(data_frame_type, window_date_range_str, file_path).exists()
        or _feather_file_path(data_frame_type, window_date_range_str, file_path).exists()
        or _csv_zip_file_path(data_frame_type, window_date_range_str, file_path).exists()
    ):
        return None
    window_start, window_end = date_range(window_date_range_str)
    covering = _find_covering_file(data_frame_type, window_start, window_end, file_path)
    if covering is None:
        return None
    covering_range_str, ext = covering
    if ext != "parquet":
        try:
            read_with_timeframe(data_frame_type, covering_range_str, file_path, n_rows=None, skip_rows=None)
        except Exception as e:
            log_w(
                f"disk_cache: failed reading legacy cache file {data_frame_type}.{covering_range_str} to "
                f"resolve window {window_date_range_str}: {e}"
            )
            return None
    return _parquet_file_path(data_frame_type, covering_range_str, file_path)


@pandera_transform
def read_file_windowed(
    date_range_str: str | None,
    data_frame_type: str,
    generator: _Generator,
    caster_model: type[Pandera_DFM_Type],
    file_path: str | None = None,
    zero_size_allowed: None | bool = None,
    generator_params: dict[str, object] | None = None,
    nan_allowed_columns: frozenset[str] | None = None,
) -> pd.DataFrame:
    """
    Windowed counterpart to read_file() — see README.md § windowing for the full design. Decomposes
    date_range_str into whole calendar windows (_window_date_range_strs(), sized per
    app_config.cache_window_freq_overrides/default_cache_window_freq), fetches/generates each window
    through the ordinary read_file() single-file path (so memoization, legacy CSV-zip migration, and
    validation are all unchanged and shared with non-windowed callers), stitches the windows back
    together, and trims to exactly date_range_str.

    A window whose entire span is still in the future is never fetched (generator() would just
    fail/return nothing useful for it) — it contributes an empty frame instead. A window missing its
    own canonical file but fully covered by an existing file is resolved via _covering_parquet_path()
    first: whatever covers it (compacted or legacy, Parquet or Feather/CSV-zip on-touch-migrated to
    Parquet) is read straight from where it sits — see that function's docstring for why it no longer
    writes a separate per-window slice.

    Windows whose canonical Parquet file is already on disk, or that resolve to a covering file above,
    are batched through infrastructure.duckdb_reader's single multi-file query instead of one
    read_file() call each — see _read_cached_windows_via_duckdb(); a covering file shared by several
    windows is only queried once, not once per window. Windows needing generation or a
    future/not-yet-cachable span still go through read_file() individually, unchanged.
    """
    if generator_params is None:
        generator_params = {}
    resolved_file_path: FilePathArg = file_path if file_path is not None else DATASET_DB
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    window_freq = _window_freq(data_frame_type)
    window_ranges = _window_date_range_strs(date_range_str, window_freq)
    now = datetime.now(pytz.UTC)

    window_dfs: list[pd.DataFrame] = []
    duckdb_batch_paths: list[Path] = []
    duckdb_batch_paths_seen: set[Path] = set()
    duckdb_batch_ranges: list[str] = []
    for window_range in window_ranges:
        window_start, _window_end = date_range(window_range)
        if window_start > now:
            window_dfs.append(cast(pd.DataFrame, empty_df(caster_model)))
            continue
        cachable = not datarange_is_not_cachable(window_range)
        batch_path: Path | None = None
        if cachable:
            batch_path = _covering_parquet_path(data_frame_type, window_range, resolved_file_path)
            if batch_path is None:
                own_path = _parquet_file_path(data_frame_type, window_range, resolved_file_path)
                if own_path.exists():
                    batch_path = own_path
        if batch_path is not None:
            if batch_path not in duckdb_batch_paths_seen:
                duckdb_batch_paths.append(batch_path)
                duckdb_batch_paths_seen.add(batch_path)
            duckdb_batch_ranges.append(window_range)
            continue
        window_dfs.append(
            read_file(
                window_range,
                data_frame_type,
                generator,
                caster_model,
                file_path=file_path,
                zero_size_allowed=zero_size_allowed,
                generator_params=generator_params,
                nan_allowed_columns=nan_allowed_columns,
            )
        )
    if duckdb_batch_paths:
        window_dfs.extend(
            _read_cached_windows_via_duckdb(
                duckdb_batch_paths,
                duckdb_batch_ranges,
                data_frame_type,
                caster_model,
                file_path=file_path,
                zero_size_allowed=zero_size_allowed,
                generator=generator,
                generator_params=generator_params,
                nan_allowed_columns=nan_allowed_columns,
            )
        )
    df: pd.DataFrame = pd.concat(window_dfs)
    df = df.sort_index(level="date")
    return trim_to_date_range(date_range_str, df)


@pandera_transform
def _read_cached_windows_via_duckdb(
    paths: list[Path],
    window_ranges: list[str],
    data_frame_type: str,
    caster_model: type[Pandera_DFM_Type],
    file_path: str | None,
    zero_size_allowed: None | bool,
    generator: _Generator,
    generator_params: dict[str, object],
    nan_allowed_columns: frozenset[str] | None,
) -> list[pd.DataFrame]:
    """
    Batches `paths` (already-cached window files for the same data_frame_type) through
    duckdb_reader.read_parquet_files() in one query, validated once via caster_model — cheaper than
    read_file_windowed()'s previous per-window read+validate loop, and skips this batch's individual
    windows' entries in disk_cache._read_file_cache (documented tradeoff: DuckDB reads straight from
    row-group-pruned Parquet, comparable cost to a memo hit; the memo is only 32 entries).

    On any failure (corrupted file, stale on-disk schema not matching caster_model — rare, since every
    file was itself validated at write time) falls back to the ordinary per-window read_file() path for
    just this batch, preserving read_file()'s existing self-healing-via-regeneration guarantee rather
    than silently propagating a batch-wide error for one bad window. Not the detection point for the
    date-as-index bug (see module docstring § 2's repair job) — verified DuckDB's own read_parquet()
    returns `date` as a plain column regardless of a source file's pandas index metadata, so a batch
    with that bug still reads fine here; disk_cache.read_by_date()'s pd.read_parquet() + index_by_date()
    path is where it actually surfaces and gets flagged.
    """
    overall_start, _ = date_range(window_ranges[0])
    _, overall_end = date_range(window_ranges[-1])
    try:
        batched_df = read_parquet_files(paths, data_frame_type, overall_start, overall_end)
        # apply_as_type() coerces dtype/index (esp. the `date` index — DuckDB's TIMESTAMPTZ export
        # follows the connection's local session timezone, not always UTC, and pandas's own inferred
        # resolution needn't match caster_model's 'ns') before validating.
        batched_df = apply_as_type(batched_df, caster_model)
        batched_df = caster_model.validate(batched_df)
    except Exception as e:
        log_w(
            f"disk_cache: DuckDB batched read for '{data_frame_type}' over {len(paths)} window(s) failed "
            f"({e}); falling back to per-window reads."
        )
        return [
            read_file(
                window_range,
                data_frame_type,
                generator,
                caster_model,
                file_path=file_path,
                zero_size_allowed=zero_size_allowed,
                generator_params=generator_params,
                nan_allowed_columns=nan_allowed_columns,
            )
            for window_range in window_ranges
        ]
    return [batched_df]
