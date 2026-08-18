import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from config import app_config
from helper.functions import Pandera_DFM_Type, date_range
from helper.logging.do_log import log_i, log_w

"""
On-disk layout, cleanup, and write-time hygiene helpers for disk_cache.py's generic
(data_frame_type, date_range_str) cache — split out from disk_cache.py itself (project-decisions
skill § code layers, "Splitting an oversized file") once it crossed the 500-line soft cap, mirroring
disk_cache_gaps.py's earlier split for the same reason. Covers: the per-symbol base directory
(symbol_data_path()) and per-data_frame_type cache subdirectory (with on-touch migration of
pre-existing flat-layout files into it) that every cache path is built from; the legacy filename
pattern shared by windowing/cleanup/gap-discovery; the file-path builders for each on-disk format and
the on-touch format-migration helpers (Parquet is primary, Feather/ZSTD and CSV-zip are legacy
fallbacks); the redundant-file cleanup pass; and the NaN-column bookkeeping behind
write_data_file()'s write-time guard.
"""


def symbol_data_path(
    path_of_data: str | None = None,
    exchange: str | None = None,
    market: str | None = None,
    trading_pair: str | None = None,
) -> str:
    if path_of_data is None:
        path_of_data = str(app_config.path_of_data)
    if exchange is None:
        exchange = app_config.under_process_exchange
    if market is None:
        market = app_config.under_process_market
    if trading_pair is None:
        trading_pair = app_config.under_process_symbol
    return str(Path(path_of_data) / exchange / market / trading_pair)


def _window_freq(data_frame_type: str) -> str:
    return app_config.cache_window_freq_overrides.get(data_frame_type, app_config.default_cache_window_freq)


_DATE_RANGE_STR_RE = r"\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}"


def _legacy_file_pattern(data_frame_type: str) -> "re.Pattern[str]":
    return re.compile(rf"^{re.escape(data_frame_type)}\.(?P<range>{_DATE_RANGE_STR_RE})\.(?P<ext>parquet|feather|zip)$")


_data_frame_type_dirs_ensured: set[tuple[str, str]] = set()


def _data_frame_type_dir(data_frame_type: str, file_path: str) -> Path:
    """
    Per-artifact-type cache subdirectory: file_path/<data_frame_type>/ — e.g. a dedicated `ohlcv/`
    folder next to `rolling_mean_std_multi_timeframe_ohlcv/`, instead of every type's files mixed
    flat in file_path. Always ensured to exist (creates the full chain, including file_path itself).

    The first time this directory is resolved for a given (file_path, data_frame_type) pair in this
    process, any pre-existing flat-layout files for that type still sitting directly in file_path
    (from before this per-type split) are moved in — self-healing on-touch migration, same policy as
    disk_cache.py's Feather/CSV-zip -> Parquet migration, just for location instead of format.
    Memoized per process (like disk_cache._read_file_cache) so file_path is only scanned once per
    type, not on every read/write.
    """
    type_dir = Path(file_path) / data_frame_type
    type_dir.mkdir(parents=True, exist_ok=True)
    key = (file_path, data_frame_type)
    if key not in _data_frame_type_dirs_ensured:
        _migrate_flat_files_into_type_dir(data_frame_type, file_path, type_dir)
        _data_frame_type_dirs_ensured.add(key)
    return type_dir


def _migrate_flat_files_into_type_dir(data_frame_type: str, file_path: str, type_dir: Path) -> None:
    pattern = _legacy_file_pattern(data_frame_type)
    flat_dir = Path(file_path)
    if not flat_dir.is_dir():
        return
    for entry in flat_dir.iterdir():
        if not pattern.match(entry.name) or not entry.is_file():
            continue
        dst = type_dir / entry.name
        if dst.exists():
            continue
        entry.rename(dst)
        log_i(f"disk_cache: moved {entry.resolve()} into per-type cache dir {type_dir.resolve()}")


def _parquet_file_path(data_frame_type: str, date_range_str: str, file_path: str) -> Path:
    return _data_frame_type_dir(data_frame_type, file_path) / f"{data_frame_type}.{date_range_str}.parquet"


def _feather_file_path(data_frame_type: str, date_range_str: str, file_path: str) -> Path:
    return _data_frame_type_dir(data_frame_type, file_path) / f"{data_frame_type}.{date_range_str}.feather"


def _csv_zip_file_path(data_frame_type: str, date_range_str: str, file_path: str) -> Path:
    return _data_frame_type_dir(data_frame_type, file_path) / f"{data_frame_type}.{date_range_str}.zip"


def _migrate_feather_to_parquet(df: pd.DataFrame, parquet_file_path: Path, feather_file_path: Path) -> None:
    """Best-effort backup of a freshly-read whole legacy Feather/ZSTD file to Parquet/ZSTD; the old
    feather file is only removed once the parquet write has succeeded."""
    try:
        df.to_parquet(parquet_file_path, compression="zstd")
    except Exception as e:
        log_w(f"Failed to back up {feather_file_path} to Parquet/ZSTD ({parquet_file_path}): {e}")
        return
    log_i(f"wrote Parquet/ZSTD cache file: {parquet_file_path.resolve()}")
    feather_file_path.unlink()


def _migrate_csv_zip_to_parquet(df: pd.DataFrame, parquet_file_path: Path, csv_zip_file_path: Path) -> None:
    """Best-effort backup of a freshly-read whole CSV-zip file straight to Parquet/ZSTD (skipping the
    Feather tier, since Parquet is the current primary format); the legacy CSV-zip is only removed
    once the parquet write has succeeded."""
    try:
        df.to_parquet(parquet_file_path, compression="zstd")
    except Exception as e:
        log_w(f"Failed to back up {csv_zip_file_path} to Parquet/ZSTD ({parquet_file_path}): {e}")
        return
    log_i(f"wrote Parquet/ZSTD cache file: {parquet_file_path.resolve()}")
    csv_zip_file_path.unlink()


def _disallowed_nan_columns(df: pd.DataFrame, nan_allowed_columns: frozenset[str]) -> list[str]:
    return [col for col in df.columns if col not in nan_allowed_columns and df[col].isna().any()]


def _schema_nullable_columns(caster_model: type[Pandera_DFM_Type]) -> frozenset[str]:
    """Columns caster_model's own pandera schema already declares nullable=True — the existing,
    reviewed statement of which columns legitimately hold NaN for this artifact type. This is the
    default NaN allowance for disk_cache.write_data_file()'s write-time guard (see
    disk_cache.read_file()); it's also exactly what caster_model.validate() a few lines later in
    read_file() would already reject if violated, so the guard never rejects a df that would
    otherwise pass validation."""
    try:
        schema = caster_model.to_schema()
    except AttributeError:
        return frozenset()
    return frozenset(name for name, column in schema.columns.items() if column.nullable)


def cleanup_redundant_cache_files(
    data_frame_type: str, file_path: str | None = None, window_freq: str | None = None, dry_run: bool = False
) -> list[tuple[str, int]]:
    """
    Delete every on-disk cache file for data_frame_type whose whole date-range span is already fully,
    exactly reconstructable from OTHER cache files for the same data_frame_type at window_freq
    granularity (defaults to this prefix's configured window). See README.md § cleanup. Typical
    target: a pre-windowing file that duplicates the canonical window tiles now sitting underneath it
    (e.g. a multi-year "full range" file where every day in it also has its own daily tile). A file is
    never deleted on the strength of its own span alone — every constituent window period must be
    backed by a DIFFERENT file, so a genuine window tile is never mistaken for redundant.

    Returns the (path, size_in_bytes) pairs deleted (or, if dry_run, that would be deleted).
    """
    if file_path is None:
        file_path = symbol_data_path()
    if window_freq is None:
        window_freq = _window_freq(data_frame_type)
    type_dir = _data_frame_type_dir(data_frame_type, file_path)
    pattern = _legacy_file_pattern(data_frame_type)
    if not type_dir.is_dir():
        return []

    files: list[tuple[Path, datetime, datetime]] = []
    exact_index: dict[tuple[datetime, datetime], list[Path]] = {}
    for entry in type_dir.iterdir():
        match = pattern.match(entry.name)
        if not match:
            continue
        start, end = date_range(match.group("range"))
        files.append((entry, start, end))
        exact_index.setdefault((start, end), []).append(entry)

    removed: list[tuple[str, int]] = []
    for entry, start, end in files:
        periods = pd.period_range(start=start, end=end, freq=window_freq)
        fully_covered_by_others = True
        for period in periods:
            key = (period.start_time.tz_localize(pytz.UTC), period.end_time.floor("min").tz_localize(pytz.UTC))
            other_files = [f for f in exact_index.get(key, []) if f != entry]
            if not other_files:
                fully_covered_by_others = False
                break
        if not fully_covered_by_others:
            continue
        size = entry.stat().st_size
        if not dry_run:
            entry.unlink()
        removed.append((str(entry), size))
    return removed
