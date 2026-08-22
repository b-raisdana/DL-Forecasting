import itertools
from datetime import datetime

from helper.date_utils import date_range, date_range_to_string
from infrastructure.datastore_engine.disk_cache import (
    DATASET_DB,
    CachableDataset,
    FilePathArg,
    _csv_zip_file_path,
    _data_frame_type_dir,
    _data_frame_type_of,
    _feather_file_path,
    _legacy_file_pattern,
    _parquet_file_path,
    _window_freq,
    datarange_is_not_cachable,
)
from infrastructure.datastore_engine.disk_cache_windowed import _find_covering_file, _window_date_range_strs

"""
Gap/overlap discovery over infrastructure.disk_cache's generic windowed (data_frame_type,
date_range_str) cache — split out from disk_cache.py itself (project-decisions skill § code layers,
"Splitting an oversized file") once it crossed the 500-line soft cap, rather than growing it further.
A companion module, not a separate cache: it reads disk_cache.py's own file-naming/windowing
internals to answer "what's missing"/"what already overlaps", it never writes.
"""


def _window_present(data_frame_type: str, window_date_range_str: str, file_path: FilePathArg) -> bool:
    """A window counts as present if it has its own canonical file (parquet, or legacy feather/zip) or
    an existing legacy file fully contains it (same containment check
    disk_cache_windowed._covering_parquet_path() uses to backfill from)."""
    if (
        _parquet_file_path(data_frame_type, window_date_range_str, file_path).exists()
        or _feather_file_path(data_frame_type, window_date_range_str, file_path).exists()
        or _csv_zip_file_path(data_frame_type, window_date_range_str, file_path).exists()
    ):
        return True
    window_start, window_end = date_range(window_date_range_str)
    return _find_covering_file(data_frame_type, window_start, window_end, file_path) is not None


def find_cache_gaps(
    dataset: CachableDataset | str,
    date_range_str: str,
    file_path: str | None = None,
    window_freq: str | None = None,
) -> list[str]:
    """
    Every window_freq-granularity gap in date_range_str for data_frame_type: windows with neither
    their own canonical file nor legacy-file coverage on disk (see _window_present()). A window that's
    still open (datarange_is_not_cachable() — e.g. today) is never reported, since it's never expected
    to be cached (see Cache-or-generate skill). Contiguous missing windows are merged into one
    date_range_str per run, so e.g. Aug 1/5/6/7/9 present yields two gaps: Aug 2-4 and Aug 8.

    `dataset` accepts either a CachableDataset (reusing the same object cache_on_disk() was declared
    with, now archived) or a bare data_frame_type string.
    """
    data_frame_type = _data_frame_type_of(dataset)
    resolved_file_path: FilePathArg = file_path if file_path is not None else DATASET_DB
    if window_freq is None:
        window_freq = _window_freq(data_frame_type)
    window_ranges = _window_date_range_strs(date_range_str, window_freq)

    missing_indices = [
        i
        for i, window_range in enumerate(window_ranges)
        if not datarange_is_not_cachable(window_range)
        and not _window_present(data_frame_type, window_range, resolved_file_path)
    ]

    gaps = []
    for _, group in itertools.groupby(enumerate(missing_indices), lambda pair: pair[1] - pair[0]):
        run = [index for _, index in group]
        gap_start, _ = date_range(window_ranges[run[0]])
        _, gap_end = date_range(window_ranges[run[-1]])
        gaps.append(date_range_to_string(start=gap_start, end=gap_end))
    return gaps


def find_overlapping_cache_files(
    dataset: CachableDataset | str, date_range_str: str, file_path: str | None = None
) -> list[tuple[str, str]]:
    """
    Every on-disk (range, ext) for data_frame_type whose own date range merely overlaps
    date_range_str at all — broader than disk_cache._find_covering_file()'s full-containment check,
    since this answers "what existing cached data could migration reuse", not "what already fully
    satisfies one window". Sorted by start.

    `dataset` accepts either a CachableDataset (reusing the same object cache_on_disk() was declared
    with, now archived) or a bare data_frame_type string.
    """
    data_frame_type = _data_frame_type_of(dataset)
    resolved_file_path: FilePathArg = file_path if file_path is not None else DATASET_DB
    query_start, query_end = date_range(date_range_str)
    pattern = _legacy_file_pattern(data_frame_type)
    type_dir = _data_frame_type_dir(data_frame_type, resolved_file_path)
    if not type_dir.is_dir():
        return []

    overlaps: list[tuple[str, str, datetime]] = []
    for entry in type_dir.iterdir():
        match = pattern.match(entry.name)
        if not match:
            continue
        candidate_range = match.group("range")
        candidate_start, candidate_end = date_range(candidate_range)
        if candidate_start < query_end and candidate_end > query_start:
            overlaps.append((candidate_range, match.group("ext"), candidate_start))
    overlaps.sort(key=lambda entry: entry[2])
    return [(range_str, ext) for range_str, ext, _start in overlaps]
