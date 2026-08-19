from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from config import app_config
from helper.data_preparation import trim_to_date_range
from helper.functions import Pandera_DFM_Type, date_range, date_range_to_string
from helper.logging.do_log import log_w
from helper.schema_casting import empty_df
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
    write_data_file,
)
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


def _window_date_range_strs(date_range_str: str, window_freq: str) -> list[str]:
    """
    Decompose date_range_str into the full-span, calendar-aligned window_freq periods it overlaps
    (e.g. every whole calendar day/month it touches) — always whole windows, never clipped to
    date_range_str's own start/end, so the same window file is reused verbatim across differently
    bounded requests instead of writing a fragment. read_file_windowed() trims the merged result back
    down to date_range_str afterwards. See README.md § windowing.
    """
    start, end = date_range(date_range_str)
    periods = pd.period_range(start=start, end=end, freq=window_freq)
    return [
        date_range_to_string(
            start=period.start_time.tz_localize(pytz.UTC),
            end=period.end_time.floor("min").tz_localize(pytz.UTC),
        )
        for period in periods
    ]


def _find_covering_file(
    data_frame_type: str, window_start: datetime, window_end: datetime, file_path: FilePathArg
) -> tuple[str, str] | None:
    """
    Smallest on-disk (range, ext) for data_frame_type whose own date range fully contains
    [window_start, window_end], if any. "Smallest" minimizes the read cost of the reuse in
    _seed_window_from_legacy_file() — a window is usually covered by many nested legacy ranges.
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


def _seed_window_from_legacy_file(data_frame_type: str, window_date_range_str: str, file_path: FilePathArg) -> None:
    """
    Backward-compatibility path for the windowing migration (README.md § backward compatibility): if
    window_date_range_str's own canonical file is missing but an existing (typically pre-windowing,
    arbitrary-range) file for data_frame_type fully covers it, slice that file down to the window and
    write it out as the window's canonical file — so the caller's read_file() call right after this
    finds it and never calls generator(). Does not touch the legacy file itself; a separate,
    explicit cleanup_redundant_cache_files() pass removes files that become fully redundant.
    """
    if (
        _parquet_file_path(data_frame_type, window_date_range_str, file_path).exists()
        or _feather_file_path(data_frame_type, window_date_range_str, file_path).exists()
        or _csv_zip_file_path(data_frame_type, window_date_range_str, file_path).exists()
    ):
        return
    window_start, window_end = date_range(window_date_range_str)
    covering = _find_covering_file(data_frame_type, window_start, window_end, file_path)
    if covering is None:
        return
    covering_range_str, _ext = covering
    try:
        covering_df = read_with_timeframe(data_frame_type, covering_range_str, file_path, n_rows=None, skip_rows=None)
    except Exception as e:
        log_w(
            f"disk_cache: failed reading legacy cache file {data_frame_type}.{covering_range_str} to "
            f"seed window {window_date_range_str}: {e}"
        )
        return
    window_df = trim_to_date_range(window_date_range_str, covering_df)
    write_data_file(window_df, data_frame_type, window_date_range_str, file_path)


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
    own canonical file but fully covered by an existing legacy file is seeded from that file first;
    see _seed_window_from_legacy_file().

    Windows whose canonical Parquet file is already on disk (the common case for repeated reads over
    already-fully-cached historical data) are batched through infrastructure.duckdb_reader's single
    multi-file query instead of one read_file() call each — see _read_cached_windows_via_duckdb().
    Windows needing generation, legacy-seeding, or a future/not-yet-cachable span still go through
    read_file() individually, unchanged.
    """
    if generator_params is None:
        generator_params = {}
    resolved_file_path: FilePathArg = file_path if file_path is not None else DATASET_DB
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    window_freq = _window_freq(data_frame_type)
    window_ranges = _window_date_range_strs(date_range_str, window_freq)
    now = datetime.now(pytz.UTC)

    window_dfs = []
    duckdb_batch_paths: list[Path] = []
    duckdb_batch_ranges: list[str] = []
    for window_range in window_ranges:
        window_start, _window_end = date_range(window_range)
        if window_start > now:
            window_dfs.append(empty_df(caster_model))
            continue
        cachable = not datarange_is_not_cachable(window_range)
        if cachable:
            _seed_window_from_legacy_file(data_frame_type, window_range, resolved_file_path)
        parquet_path = _parquet_file_path(data_frame_type, window_range, resolved_file_path)
        if cachable and parquet_path.exists():
            duckdb_batch_paths.append(parquet_path)
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
    df = pd.concat(window_dfs)
    df = df.sort_index(level="date")
    return trim_to_date_range(date_range_str, df)


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
    than silently propagating a batch-wide error for one bad window.
    """
    overall_start, _ = date_range(window_ranges[0])
    _, overall_end = date_range(window_ranges[-1])
    try:
        batched_df = read_parquet_files(paths, data_frame_type, overall_start, overall_end)
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
