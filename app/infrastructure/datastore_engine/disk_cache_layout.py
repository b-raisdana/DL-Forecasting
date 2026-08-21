import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pandera
import pytz
from config import app_config
from helper.functions import Pandera_DFM_Type, date_range, date_range_to_string
from helper.logging.do_log import log_i
from helper.pandera import pandera_validate

"""
On-disk layout, cleanup, and write-time hygiene helpers for disk_cache.py's generic
(data_frame_type, date_range_str) cache — split out from disk_cache.py itself (project-decisions
skill § code layers, "Splitting an oversized file") once it crossed the 500-line soft cap, mirroring
disk_cache_gaps.py's earlier split for the same reason. Covers: the per-symbol base directory
(symbol_data_path(), used only by non-cache per-symbol consumers now — see below) and
per-data_frame_type cache subdirectory (with on-touch migration of pre-existing flat-layout files
into it) that every cache path is built from; the legacy filename pattern shared by
windowing/cleanup/gap-discovery; the file-path builders for each on-disk format (Parquet is primary,
Feather/ZSTD and CSV-zip are legacy fallbacks — the format-migration helpers themselves live in
infrastructure/datastore_engine/parquet_housekeeping.py, to avoid a cycle between the two modules); the
redundant-file cleanup pass; and the NaN-column bookkeeping behind write_data_file()'s write-time
guard.

Two distinct roots live under path_of_data:
- symbol_data_path() (exchange/market/trading_pair) — non-cache, per-symbol output: backtesting
  vault/orders/signals CSVs (ExtendedStrategy.py), plots (presentation/shared/plotter.py). Not
  touched by the dataset_db migration below; not managed by disk_cache's Parquet cache at all.
- dataset_db_root() (data_frame_type/market/trading_pair/exchange) — every (data_frame_type,
  date_range_str) artifact disk_cache.py's cache_on_disk/read_file/read_file_windowed family
  manages, data_frame_type-first so a single glob (e.g. dataset_db/multi_timeframe_ohlcv/**/*.parquet)
  scans one type across every symbol/market/exchange at once — the partitioning
  infrastructure/duckdb_reader.py's batched reads rely on. See data/dataset_db/README.md.
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


class DatasetDbSentinel:
    """Marker `file_path` value meaning "resolve under dataset_db_root(), data_frame_type-first"
    instead of treating `file_path` as the opaque caller-supplied root `_data_frame_type_dir()`
    normally nests `data_frame_type` under. Only the real disk-cache call sites (disk_cache.py's
    read_file()/read_file_windowed(), disk_cache_gaps.py, ExtendedDf.py, AtrMovementPivots.py) pass
    this instead of a real path string; every other caller — including the full
    tests/unit/infrastructure/test_disk_cache*.py suite, which passes a bare tmp_path — keeps hitting
    the original opaque-root branch unchanged. Kept as a dedicated type (not a magic string/None) so
    it can't be produced by accident."""


DATASET_DB = DatasetDbSentinel()
FilePathArg = str | DatasetDbSentinel


# `generator(date_range_str, **kwargs) -> DataFrame` callables read_file()/read_file_windowed()/
# cache_on_disk() accept — genuinely arbitrary per artifact type (extra kwargs like
# `base_timeframe`, `symbols`, ...), so a precise Protocol would reject real generators; the ignore
# is for Callable's `...`, itself an explicit Any under disallow_any_explicit. Return type is
# `object`, not `pd.DataFrame`: a generator's return annotation is legitimately either a
# pt.DataFrame[Model] or a bare Model class read only by disk_cache._resolve_caster_model() (see its
# docstring), never actually returned as a Model instance — read_file()/read_file_windowed() cast
# the real call result to pd.DataFrame themselves once generator() has run.
_Generator = Callable[..., object]  # type: ignore[explicit-any]


@dataclass(frozen=True)
class CachableDataset:
    """
    Everything disk_cache.py's cache_on_disk()/find_cache_gaps()/find_overlapping_cache_files()/
    cleanup_redundant_cache_files() need to know about one (data_frame_type) cache artifact, bundled
    once at the artifact's definition site instead of repeating the same data_frame_type string (and,
    at the cache_on_disk() decorator site, generator/caster_model/nan_allowed_columns/zero_size_allowed)
    at every call site — e.g. `application.market_data.fetch_market_data` reuses
    `domain.ohlcv.ohlcv.OHLCV_DATASET` instead of a bare "ohlcv" literal.

    generator/caster_model are optional: leave them unset when declaring the dataset for a
    cache_on_disk() decorator site — the decorator fills generator in from the function it decorates
    and infers caster_model from that function's own return-type annotation (via
    disk_cache._resolve_caster_model()) unless overridden here. nan_allowed_columns/zero_size_allowed
    default to cache_on_disk()'s own defaults (no extra NaN allowance beyond the schema's nullable
    columns; zero_size_allowed inferred per date_range_str).
    """

    dataset_folder_name: str
    generator: _Generator | None = None
    # type[pandera.DataFrameModel], not type[Pandera_DFM_Type]: that TypeVar is only meaningful bound
    # to a generic function/class call (as disk_cache._resolve_caster_model()'s return type is), not as
    # a plain dataclass field annotation — mypy flags it "unbound" there.
    caster_model: type[pandera.DataFrameModel] | None = None
    nan_allowed_columns: frozenset[str] = field(default_factory=frozenset)
    zero_size_allowed: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nan_allowed_columns", frozenset(self.nan_allowed_columns))


def _data_frame_type_of(dataset: "CachableDataset | str") -> str:
    """Resolve either a CachableDataset or a bare data_frame_type string (still accepted by
    find_cache_gaps()/find_overlapping_cache_files()/cleanup_redundant_cache_files() for callers that
    have no CachableDataset to reuse, e.g. a one-off ad-hoc artifact type) to its data_frame_type
    string."""
    return dataset.dataset_folder_name if isinstance(dataset, CachableDataset) else dataset


def dataset_db_root(path_of_data: str | None = None) -> Path:
    if path_of_data is None:
        path_of_data = str(app_config.path_of_data)
    return Path(path_of_data) / "dataset_db"


_dataset_db_dirs_ensured: set[tuple[str, str, str, str]] = set()


def _dataset_db_type_dir(
    data_frame_type: str,
    exchange: str | None = None,
    market: str | None = None,
    trading_pair: str | None = None,
) -> Path:
    """
    data_frame_type-first cache directory: dataset_db_root()/<data_frame_type>/<market>/
    <trading_pair>/<exchange>/. On-touch migration counterpart to _migrate_flat_files_into_type_dir(),
    one tier up: the first time a given (data_frame_type, market, trading_pair, exchange) directory is
    resolved in this process, any files still sitting at the old symbol-first location
    (symbol_data_path()/<data_frame_type>/) are moved in.
    """
    if exchange is None:
        exchange = app_config.under_process_exchange
    if market is None:
        market = app_config.under_process_market
    if trading_pair is None:
        trading_pair = app_config.under_process_symbol
    type_dir = dataset_db_root() / data_frame_type / market / trading_pair / exchange
    type_dir.mkdir(parents=True, exist_ok=True)
    key = (data_frame_type, market, trading_pair, exchange)
    if key not in _dataset_db_dirs_ensured:
        _migrate_symbol_first_dir_into_dataset_db(data_frame_type, exchange, market, trading_pair, type_dir)
        _dataset_db_dirs_ensured.add(key)
    return type_dir


def _migrate_symbol_first_dir_into_dataset_db(
    data_frame_type: str, exchange: str, market: str, trading_pair: str, type_dir: Path
) -> None:
    old_type_dir = Path(symbol_data_path(exchange=exchange, market=market, trading_pair=trading_pair)) / data_frame_type
    if not old_type_dir.is_dir():
        return
    for entry in old_type_dir.iterdir():
        if not entry.is_file():
            continue
        dst = type_dir / entry.name
        if dst.exists():
            continue
        entry.rename(dst)
        log_i(f"disk_cache: moved {entry.resolve()} into dataset_db cache dir {type_dir.resolve()}")


@pandera_validate(allow_pandas_dataframe=True)
def index_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Shared tail of disk_cache.read_by_date() and infrastructure.duckdb_reader's batched read: parse
    the on-disk `date` column (every cache file carries one — write_data_file() writes
    df.reset_index().to_parquet(...)), set it as the index, UTC-localize if naive. Kept here (not in
    disk_cache.py) so duckdb_reader.py can reuse it without an import cycle."""
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    date_index = cast("pd.DatetimeIndex", df.index)
    if len(df) > 0 and date_index.tz is None:
        df.index = date_index.tz_localize("UTC")
    return df


@pandera_validate(allow_pandas_dataframe=True)
def add_timeframe_index(df: pd.DataFrame, data_frame_type: str) -> pd.DataFrame:
    """Shared tail of disk_cache.read_with_timeframe() and duckdb_reader's batched read: for
    multi_timeframe_* types, promote the on-disk `timeframe` column to an outer index level alongside
    `date` (index_by_date() must already have run)."""
    if "multi_timeframe" in data_frame_type:
        df.set_index("timeframe", append=True, inplace=True)
        df = df.swaplevel()
    return df


def _window_freq(data_frame_type: str) -> str:
    return app_config.cache_window_freq_overrides.get(data_frame_type, app_config.default_cache_window_freq)


def _window_date_range_strs(date_range_str: str, window_freq: str) -> list[str]:
    """
    Decompose date_range_str into the full-span, calendar-aligned window_freq periods it overlaps
    (e.g. every whole calendar day/month it touches) — always whole windows, never clipped to
    date_range_str's own start/end, so the same window file is reused verbatim across differently
    bounded requests instead of writing a fragment. disk_cache_windowed.read_file_windowed() trims the
    merged result back down to date_range_str afterwards. Moved here from disk_cache_windowed.py
    (project-decisions skill § code layers, "Splitting an oversized file") so
    infrastructure.datastore_engine.parquet_housekeeping's compaction pass can reuse it without a
    disk_cache_windowed<->parquet_housekeeping cycle (parquet_housekeeping already sits underneath
    disk_cache_windowed by way of disk_cache.py). See README.md § windowing for the full design.
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


_DATE_RANGE_STR_RE = r"\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}"


def _legacy_file_pattern(data_frame_type: str) -> "re.Pattern[str]":
    return re.compile(rf"^{re.escape(data_frame_type)}\.(?P<range>{_DATE_RANGE_STR_RE})\.(?P<ext>parquet|feather|zip)$")


_data_frame_type_dirs_ensured: set[tuple[str, str]] = set()


def _data_frame_type_dir(data_frame_type: str, file_path: FilePathArg) -> Path:
    """
    Per-artifact-type cache subdirectory. Two modes, selected by file_path's type:
    - file_path is DATASET_DB: delegates to _dataset_db_type_dir() — data_frame_type-first, under
      dataset_db_root(). This is what every real disk-cache call site uses (see DatasetDbSentinel).
    - file_path is a plain str (the original, still-generic contract every unit test and any future
      non-symbol caller relies on): file_path/<data_frame_type>/ — e.g. a dedicated `ohlcv/` folder
      next to `rolling_mean_std_multi_timeframe_ohlcv/`, instead of every type's files mixed flat in
      file_path. Always ensured to exist (creates the full chain, including file_path itself).

      The first time this directory is resolved for a given (file_path, data_frame_type) pair in this
      process, any pre-existing flat-layout files for that type still sitting directly in file_path
      (from before this per-type split) are moved in — self-healing on-touch migration, same policy as
      disk_cache.py's Feather/CSV-zip -> Parquet migration, just for location instead of format.
      Memoized per process (like disk_cache._read_file_cache) so file_path is only scanned once per
      type, not on every read/write.
    """
    if isinstance(file_path, DatasetDbSentinel):
        return _dataset_db_type_dir(data_frame_type)
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


def _parquet_file_path(data_frame_type: str, date_range_str: str, file_path: FilePathArg) -> Path:
    return _data_frame_type_dir(data_frame_type, file_path) / f"{data_frame_type}.{date_range_str}.parquet"


def _feather_file_path(data_frame_type: str, date_range_str: str, file_path: FilePathArg) -> Path:
    return _data_frame_type_dir(data_frame_type, file_path) / f"{data_frame_type}.{date_range_str}.feather"


def _csv_zip_file_path(data_frame_type: str, date_range_str: str, file_path: FilePathArg) -> Path:
    return _data_frame_type_dir(data_frame_type, file_path) / f"{data_frame_type}.{date_range_str}.zip"


@pandera_validate(allow_pandas_dataframe=True)
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
    dataset: CachableDataset | str,
    file_path: str | None = None,
    window_freq: str | None = None,
    dry_run: bool = False,
) -> list[tuple[str, int]]:
    """
    Delete every on-disk cache file for data_frame_type whose whole date-range span is already fully,
    exactly reconstructable from OTHER cache files for the same data_frame_type at window_freq
    granularity (defaults to this prefix's configured window). See README.md § cleanup. Typical
    target: a pre-windowing file that duplicates the canonical window tiles now sitting underneath it
    (e.g. a multi-year "full range" file where every day in it also has its own daily tile). A file is
    never deleted on the strength of its own span alone — every constituent window period must be
    backed by a DIFFERENT file, so a genuine window tile is never mistaken for redundant.

    `dataset` accepts either a CachableDataset (reusing the same object cache_on_disk() was declared
    with) or a bare data_frame_type string — see _data_frame_type_of().

    Returns the (path, size_in_bytes) pairs deleted (or, if dry_run, that would be deleted).
    """
    data_frame_type = _data_frame_type_of(dataset)
    resolved_file_path: FilePathArg = file_path if file_path is not None else DATASET_DB
    if window_freq is None:
        window_freq = _window_freq(data_frame_type)
    type_dir = _data_frame_type_dir(data_frame_type, resolved_file_path)
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
