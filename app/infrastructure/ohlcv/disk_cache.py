import os
import re
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from typing import get_args, get_type_hints
from zipfile import BadZipFile

import pandas as pd
import pytz
from config import app_config
from helper.data_preparation import after_under_process_date, trim_to_date_range
from helper.functions import Pandera_DFM_Type, date_range, date_range_to_string, morning
from helper.logging.do_log import log_w
from helper.schema_casting import cast_and_validate, empty_df
from infrastructure.ohlcv.fragmented_data import symbol_data_path

"""
Everything this repo needs to persist/read a (data_frame_type, date_range_str) artifact as a
feather/ZSTD (or legacy CSV-zip) file and never recompute or re-fetch it twice within its validity
scope — the disk-level half of the cache-or-generate skill. `cache_on_disk` is the primitive new
artifact types should use; `read_file`/`write_data_file` are its building blocks for callers (like
domain.schemas.common.ExtendedDf.read_file) that need a class-bound variant instead of a decorator.

Calendar-windowed caching (`cache_on_disk(..., windowed=True)` / `read_file_windowed()`), the legacy-
file reuse it does on a miss, and the disk-generation-rate monitor are documented in detail in
README.md next to this file — read that first if you're touching any of the `_window_*`/
`*_cache_generation*`/`cleanup_redundant_cache_files` functions below.
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


# Per-data_frame_type disk-generation-rate monitor (see README.md § monitoring). In-process only —
# resets on restart, same tradeoff as _read_file_cache above. Evaluated lazily on write, at most once
# per app_config.cache_generation_monitor_interval_minutes per prefix, so it never re-evaluates sooner
# than that regardless of how many files a given prefix writes in between.
_cache_generation_state: dict[str, dict[str, object]] = {}


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
    _record_cache_generation(data_frame_type, os.path.getsize(feather_file_path))


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


_DATE_RANGE_STR_RE = r"\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}"


def _legacy_file_pattern(data_frame_type: str) -> "re.Pattern":
    return re.compile(rf"^{re.escape(data_frame_type)}\.(?P<range>{_DATE_RANGE_STR_RE})\.(?P<ext>feather|zip)$")


def _window_freq(data_frame_type: str) -> str:
    return app_config.cache_window_freq_overrides.get(data_frame_type, app_config.default_cache_window_freq)


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
    data_frame_type: str, window_start: datetime, window_end: datetime, file_path: str
) -> tuple[str, str] | None:
    """
    Smallest on-disk (range, ext) for data_frame_type whose own date range fully contains
    [window_start, window_end], if any. "Smallest" minimizes the read cost of the reuse in
    _seed_window_from_legacy_file() — a window is usually covered by many nested legacy ranges.
    """
    pattern = _legacy_file_pattern(data_frame_type)
    try:
        entries = os.listdir(file_path)
    except FileNotFoundError:
        return None
    best: tuple[str, str, datetime, datetime] | None = None
    for entry in entries:
        match = pattern.match(entry)
        if not match:
            continue
        candidate_range = match.group("range")
        candidate_start, candidate_end = date_range(candidate_range)
        if candidate_start <= window_start and candidate_end >= window_end:
            if best is None or (candidate_end - candidate_start) < (best[3] - best[2]):
                best = (candidate_range, match.group("ext"), candidate_start, candidate_end)
    if best is None:
        return None
    return best[0], best[1]


def _seed_window_from_legacy_file(data_frame_type: str, window_date_range_str: str, file_path: str) -> None:
    """
    Backward-compatibility path for the windowing migration (README.md § backward compatibility): if
    window_date_range_str's own canonical file is missing but an existing (typically pre-windowing,
    arbitrary-range) file for data_frame_type fully covers it, slice that file down to the window and
    write it out as the window's canonical file — so the caller's read_file() call right after this
    finds it and never calls generator(). Does not touch the legacy file itself; a separate,
    explicit cleanup_redundant_cache_files() pass removes files that become fully redundant.
    """
    if os.path.exists(_feather_file_path(data_frame_type, window_date_range_str, file_path)) or os.path.exists(
        _csv_zip_file_path(data_frame_type, window_date_range_str, file_path)
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
    date_range_str: str,
    data_frame_type: str,
    generator: Callable,
    caster_model: type[Pandera_DFM_Type],
    file_path: str = None,
    zero_size_allowed: None | bool = None,
    generator_params: dict = {},
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
    """
    if file_path is None:
        file_path = symbol_data_path()
    if date_range_str is None:
        date_range_str = app_config.processing_date_range
    window_freq = _window_freq(data_frame_type)
    window_ranges = _window_date_range_strs(date_range_str, window_freq)
    now = datetime.now(pytz.UTC)

    window_dfs = []
    for window_range in window_ranges:
        window_start, _window_end = date_range(window_range)
        if window_start > now:
            window_dfs.append(empty_df(caster_model))
            continue
        if not datarange_is_not_cachable(window_range):
            _seed_window_from_legacy_file(data_frame_type, window_range, file_path)
        window_dfs.append(
            read_file(
                window_range,
                data_frame_type,
                generator,
                caster_model,
                file_path=file_path,
                zero_size_allowed=zero_size_allowed,
                generator_params=generator_params,
            )
        )
    df = pd.concat(window_dfs)
    df = df.sort_index(level="date")
    return trim_to_date_range(date_range_str, df)


def cleanup_redundant_cache_files(
    data_frame_type: str, file_path: str = None, window_freq: str | None = None, dry_run: bool = False
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
    pattern = _legacy_file_pattern(data_frame_type)
    try:
        entries = os.listdir(file_path)
    except FileNotFoundError:
        return []

    files: list[tuple[str, datetime, datetime]] = []
    exact_index: dict[tuple[datetime, datetime], list[str]] = {}
    for entry in entries:
        match = pattern.match(entry)
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
        full_path = os.path.join(file_path, entry)
        size = os.path.getsize(full_path)
        if not dry_run:
            os.remove(full_path)
        removed.append((full_path, size))
    return removed


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
    windowed: bool = False,
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

    `windowed=True` makes `generator` a per-window generator (see README.md § windowing): the decorated
    function accepts an arbitrary `date_range_str`, decomposes it into whole calendar windows sized per
    `app_config.cache_window_freq_overrides[file_name_prefix]` (default `app_config.default_cache_window_freq`),
    and calls `generator` once per window via `read_file_windowed()` instead of once for the whole range.
    Not compatible with `skip_rows`/`n_rows` (same restriction `read_file()` already has on a cache miss).
    """

    def decorator(generator: Callable) -> Callable:
        caster_model = _resolve_caster_model(generator)

        @wraps(generator)
        def wrapper(date_range_str: str = None, file_path: str = None, **generator_params):
            if windowed:
                if skip_rows or n_rows is not None:
                    raise Exception("cache_on_disk(windowed=True) does not support skip_rows/n_rows.")
                result = read_file_windowed(
                    date_range_str,
                    file_name_prefix,
                    generator,
                    caster_model,
                    file_path=file_path,
                    zero_size_allowed=zero_size_allowed,
                    generator_params=generator_params,
                )
            else:
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
