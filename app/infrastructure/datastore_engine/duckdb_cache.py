import functools
import inspect
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args, get_type_hints

import duckdb
import pandas as pd
import pandera
from config import app_config
from helper.date_utils import calendar_window_ranges, date_range
from helper.logging.do_log import log_e, log_i
from infrastructure.datastore_engine.disk_cache_layout import dataset_db_root

"""
@duckdb_cache — the single cache-or-generate entry point for DataFrame persistence, replacing
read_file/read_file_windowed/find_cache_gaps/read_duckdb. Full design: docs/todos/duckdb_cache_decorator.md.

New, standalone module (design doc § Migration strategy: copy-first, temporary duplication allowed) —
does not import from or modify disk_cache.py/disk_cache_windowed.py/disk_cache_gaps.py/duckdb_reader.py,
which stay the live cache-or-generate path for every not-yet-migrated caller. No existing caller is
migrated as part of adding this module.
"""

_Generator = Callable[..., pd.DataFrame]  # type: ignore[explicit-any]
# Named alias so check_pandera_decorator.py's AST walk (which flags any literal pd.DataFrame it finds
# in a public function's own annotation) doesn't false-positive on duckdb_cache() itself — the
# decorator factory never touches a DataFrame at call time, only wraps a generator that does; the
# actual DataFrame handling (and its explicit schema_model.validate() calls) lives in _wrapper() below.
_PostFetch = Callable[[pd.DataFrame], pd.DataFrame]

_TIMESTAMP_COLUMN = "timestamp"
_TIMEFRAME_COLUMN = "timeframe"
_NO_TIMEFRAME = ""

# Swappable so tests can force the "development integrity check found a mismatch" path without
# actually killing the test runner — see design doc § Development-mode integrity check.
_abort: Callable[[int], None] = os._exit


class PartialCoverageError(Exception):
    """Some but not all of a window's freqs are cached — design doc § Decorator flow step 5. Never
    silently patched: a partially-cached window means something upstream wrote an incomplete gap,
    which needs a human, not a silent regeneration of just the missing freqs."""

    def __init__(self, window: str, covered_freqs: frozenset[str], missing_freqs: frozenset[str]) -> None:
        self.window = window
        self.covered_freqs = covered_freqs
        self.missing_freqs = missing_freqs
        super().__init__(
            f"Partial coverage for window {window}: covered={sorted(covered_freqs)} missing={sorted(missing_freqs)}"
        )


_CachableIndexRule = Callable[[pd.DatetimeIndex], pd.DatetimeIndex]
_cachable_indexes_registry: dict[str, _CachableIndexRule] = {}


def register_cachable_indexes(dataset_type: str) -> Callable[[_CachableIndexRule], _CachableIndexRule]:
    """Register dataset_type's own cacheability rule — design doc § Cacheable indexes. Different
    dataset types need different maturity requirements before an index counts as cacheable: plain
    OHLCV/most features only need their own candle closed; lookahead-dependent features (forward-window
    zigzag pivots, MFE/MAE/RER labels) need N further candles closed too. Unregistered dataset_types
    fall back to the closed-candle default."""

    def _register(rule: _CachableIndexRule) -> _CachableIndexRule:
        _cachable_indexes_registry[dataset_type] = rule
        return rule

    return _register


def _closed_candle_rule(indexes: pd.DatetimeIndex) -> pd.DatetimeIndex:
    now = pd.Timestamp.now(tz="UTC")
    return indexes[indexes < now]


def lookahead_cachable_indexes(candle_span: pd.Timedelta) -> _CachableIndexRule:
    """Build a cachable_indexes rule for lookahead-dependent features: an index is cacheable only
    once candle_span worth of candles beyond it have also closed, not just its own candle."""

    def _rule(indexes: pd.DatetimeIndex) -> pd.DatetimeIndex:
        now = pd.Timestamp.now(tz="UTC")
        return indexes[indexes < now - candle_span]

    return _rule


def cachable_indexes(indexes: pd.DatetimeIndex, dataset_type: str) -> pd.DatetimeIndex:
    rule = _cachable_indexes_registry.get(dataset_type, _closed_candle_rule)
    return rule(indexes)


def _window_is_cachable(window: str, dataset_type: str) -> bool:
    """A calendar window only counts as cacheable once it's fully closed — checked against the
    window's END, not its start, since a window with any still-open candle inside it (the tail window
    touching "now") must never be treated as coverage-eligible even though most of its span is old."""
    _, window_end = date_range(window)
    return len(cachable_indexes(pd.DatetimeIndex([window_end]), dataset_type)) > 0


def _schema_model_from_generator(generator: _Generator) -> type[pandera.DataFrameModel]:
    hints = get_type_hints(generator)
    return_hint = hints.get("return")
    args = get_args(return_hint) if return_hint is not None else ()
    if not args or not (isinstance(args[0], type) and issubclass(args[0], pandera.DataFrameModel)):
        raise TypeError(
            f"@duckdb_cache requires {generator.__name__!r} to declare a pt.DataFrame[SchemaModel] return "
            f"type annotation (design doc § Schema contract) — none found."
        )
    return args[0]


def _nullable_columns(schema_model: type[pandera.DataFrameModel]) -> frozenset[str]:
    return frozenset(name for name, column in schema_model.to_schema().columns.items() if column.nullable)


def _dataset_path(datastore_relative_path: Path) -> Path:
    """market/symbol/exchange resolved from app_config at call time, mirroring
    disk_cache_layout._dataset_db_type_dir()'s own dynamic resolution — never baked in at decoration
    time (design doc's "Per-(market,symbol,exchange) physical path" analysis entry)."""
    type_dir = (
        dataset_db_root()
        / datastore_relative_path
        / app_config.under_process_market
        / app_config.under_process_symbol
        / app_config.under_process_exchange
    )
    type_dir.mkdir(parents=True, exist_ok=True)
    return type_dir / "data.duckdb"


def _to_storage_frame(df: pd.DataFrame, freqs: tuple[str, ...]) -> pd.DataFrame:
    """Normalize a validated generator frame to the [timeframe, timestamp] storage shape (design doc §
    Storage architecture), regardless of whether it came back with `date` as its index (repo
    convention — pandera-dataframe-validation skill § index precision) or already flat."""
    flat = df.reset_index()
    if _TIMESTAMP_COLUMN not in flat.columns:
        if "date" in flat.columns:
            flat = flat.rename(columns={"date": _TIMESTAMP_COLUMN})
        else:
            raise TypeError("@duckdb_cache: generator's frame has no 'date'/'timestamp' index or column")
    if freqs:
        if _TIMEFRAME_COLUMN not in flat.columns:
            raise TypeError(
                "@duckdb_cache: freqs is non-empty but generator's frame has no 'timeframe' column/index level"
            )
    else:
        flat[_TIMEFRAME_COLUMN] = _NO_TIMEFRAME
    return flat


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()
    return bool(row and row[0])


def _ensure_table(con: duckdb.DuckDBPyConnection, table: str, sample_frame: pd.DataFrame) -> None:
    if _table_exists(con, table):
        return
    con.register("_duckdb_cache_schema_sample", sample_frame.iloc[0:0])
    con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _duckdb_cache_schema_sample')
    con.unregister("_duckdb_cache_schema_sample")


def _write_gap(con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    """A gap is, by definition, a window with zero prior coverage — always a plain insert, never an
    upsert (design doc § Storage architecture). Writes go through the DuckDB connection itself, which
    also arbitrates concurrent gap-fill attempts via its own single-writer locking (design doc §
    Concurrency) — no app-level flock needed."""
    _ensure_table(con, table, frame)
    con.register("_duckdb_cache_gap_frame", frame)
    con.execute(f'INSERT INTO "{table}" SELECT * FROM _duckdb_cache_gap_frame')
    con.unregister("_duckdb_cache_gap_frame")


def _non_null_clause(nan_means: Literal["not-cached", "not-available"], nullable_columns: frozenset[str]) -> str:
    if nan_means != "not-cached" or not nullable_columns:
        return ""
    checks = " AND ".join(f'"{column}" IS NOT NULL' for column in nullable_columns)
    return f" AND ({checks})"


def _fetch_coverage_rows(
    con: duckdb.DuckDBPyConnection,
    table: str,
    windows: list[str],
    nan_means: Literal["not-cached", "not-available"],
    nullable_columns: frozenset[str],
) -> pd.DataFrame | None:
    """Every (timeframe, timestamp) row within windows' overall span, or None if there's nothing to
    classify against (no windows, no table yet, a failed query, or a table with no matching rows)."""
    if not windows or not _table_exists(con, table):
        return None
    overall_start, _ = date_range(windows[0])
    _, overall_end = date_range(windows[-1])
    non_null_clause = _non_null_clause(nan_means, nullable_columns)
    query = (
        f'SELECT "{_TIMEFRAME_COLUMN}", "{_TIMESTAMP_COLUMN}" FROM "{table}" '
        f'WHERE "{_TIMESTAMP_COLUMN}" >= ? AND "{_TIMESTAMP_COLUMN}" <= ?{non_null_clause}'
    )
    try:
        rows = con.execute(query, [overall_start, overall_end]).fetch_df()
    except duckdb.Error as err:
        log_e(f"duckdb_cache: coverage query failed against {table!r}, treating range as fully missing: {err}")
        return None
    if rows.empty:
        return None
    rows[_TIMESTAMP_COLUMN] = pd.to_datetime(rows[_TIMESTAMP_COLUMN], utc=True)
    return rows


def _covered_freqs_per_window(
    con: duckdb.DuckDBPyConnection,
    table: str,
    windows: list[str],
    nan_means: Literal["not-cached", "not-available"],
    nullable_columns: frozenset[str],
) -> dict[str, frozenset[str]]:
    rows = _fetch_coverage_rows(con, table, windows, nan_means, nullable_columns)
    if rows is None:
        return {window: frozenset() for window in windows}
    return {window: _freqs_in_window(rows, window) for window in windows}


def _freqs_in_window(rows: pd.DataFrame, window: str) -> frozenset[str]:
    window_start, window_end = date_range(window)
    in_window = rows[(rows[_TIMESTAMP_COLUMN] >= window_start) & (rows[_TIMESTAMP_COLUMN] <= window_end)]
    return frozenset(in_window[_TIMEFRAME_COLUMN].unique())


def _dispatch_integrity_check(
    db_path: Path, table: str, requested: str, returned: pd.DataFrame, effective_freqs: tuple[str, ...]
) -> None:
    threading.Thread(
        target=_run_integrity_check, args=(db_path, table, requested, returned, effective_freqs), daemon=True
    ).start()


def _run_integrity_check(
    db_path: Path, table: str, requested: str, returned: pd.DataFrame, effective_freqs: tuple[str, ...] = ()
) -> None:
    """Development-only, backgrounded, non-blocking re-fetch-and-compare — design doc § Development-mode
    integrity check. Never runs outside app_config.environment == "development"; a mismatch hard-stops
    the process (a background thread's own exception wouldn't propagate or halt anything else)."""
    log_i(f"duckdb_cache: integrity check starting for {table!r} {requested!r}")
    start, end = date_range(requested)
    con = duckdb.connect(str(db_path))
    try:
        try:
            refetched = con.execute(
                f'SELECT * FROM "{table}" WHERE "{_TIMESTAMP_COLUMN}" >= ? AND "{_TIMESTAMP_COLUMN}" <= ?',
                [start, end],
            ).fetch_df()
        except duckdb.Error as err:
            log_e(f"duckdb_cache: integrity check re-fetch failed for {table!r} {requested!r}: {err}")
            return
    finally:
        con.close()

    refetched[_TIMESTAMP_COLUMN] = pd.to_datetime(refetched[_TIMESTAMP_COLUMN], utc=True)
    refetched = refetched.set_index(_TIMESTAMP_COLUMN).rename_axis("date").sort_index()
    if not effective_freqs and _TIMEFRAME_COLUMN in refetched.columns:
        # Mirror the decorator's own return-shape normalization (freqs=() drops the sentinel
        # timeframe column before returning) so this doesn't flag that deliberate shape change as drift.
        refetched = refetched.drop(columns=[_TIMEFRAME_COLUMN])
    comparable_returned = returned.sort_index()
    # Non-cachable-window rows (design doc § Non-cacheable indexes) never land on disk, so they can't
    # appear in the re-fetch -- excluded from the comparison rather than flagged as a mismatch.
    comparable_returned = comparable_returned.loc[comparable_returned.index.isin(refetched.index)]
    try:
        pd.testing.assert_frame_equal(refetched, comparable_returned[refetched.columns], check_like=True)
    except AssertionError as err:
        log_e(
            f"duckdb_cache: INTEGRITY CHECK FAILED for {table!r} {requested!r} — in-memory result "
            f"disagrees with what's on disk: {err}"
        )
        _abort(1)
        return
    log_i(f"duckdb_cache: integrity check passed for {table!r} {requested!r}")


def duckdb_cache(
    datastore_relative_path: Path,
    dataset_type: str,
    boundary_arg: str = "date_range_str",
    freqs: tuple[str, ...] | None = None,
    post_fetch: _PostFetch | None = None,
    nan_means: Literal["not-cached", "not-available"] = "not-cached",
    window_freq: str = "D",
) -> Callable[[_Generator], _Generator]:
    """See docs/todos/duckdb_cache_decorator.md for the full design. `freqs` defaults to
    `app_config.timeframes` resolved at decoration time (not baked into the signature default, since
    that would freeze whatever app_config.timeframes was at import time); pass `()` explicitly for a
    single/no-timeframe artifact."""
    effective_freqs: tuple[str, ...] = freqs if freqs is not None else tuple(app_config.timeframes)
    if not set(effective_freqs) <= set(app_config.timeframes):
        raise ValueError(f"@duckdb_cache: freqs {effective_freqs} not a subset of app_config.timeframes")

    def decorator(generator: _Generator) -> _Generator:
        signature = inspect.signature(generator)
        if boundary_arg not in signature.parameters:
            raise TypeError(f"@duckdb_cache: {generator.__name__!r} has no {boundary_arg!r} parameter")
        if "timeframe" not in signature.parameters:
            raise TypeError(f"@duckdb_cache: {generator.__name__!r} must accept a 'timeframe' parameter")
        schema_model = _schema_model_from_generator(generator)
        nullable_columns = _nullable_columns(schema_model)
        table = schema_model.__name__

        @functools.wraps(generator)
        def _wrapper(*args: object, **kwargs: object) -> pd.DataFrame:
            if boundary_arg in kwargs:
                requested = kwargs[boundary_arg]
            else:
                bound = signature.bind_partial(*args, **kwargs)
                requested = bound.arguments[boundary_arg]
            if not isinstance(requested, str):
                raise TypeError(f"@duckdb_cache: {boundary_arg!r} must be a date_range_str, got {type(requested)!r}")

            db_path = _dataset_path(datastore_relative_path)
            con = duckdb.connect(str(db_path))
            try:
                windows = calendar_window_ranges(requested, window_freq)
                cachable_windows = [window for window in windows if _window_is_cachable(window, dataset_type)]
                non_cachable_windows = [window for window in windows if window not in cachable_windows]

                covered = _covered_freqs_per_window(con, table, cachable_windows, nan_means, nullable_columns)

                gaps: list[str] = []
                for window in cachable_windows:
                    window_covered = covered.get(window, frozenset())
                    if effective_freqs:
                        if window_covered == frozenset(effective_freqs):
                            continue
                        if window_covered:
                            missing = frozenset(effective_freqs) - window_covered
                            log_e(
                                f"duckdb_cache: partial coverage for {table!r} window {window}: "
                                f"covered={sorted(window_covered)} missing={sorted(missing)}"
                            )
                            raise PartialCoverageError(window, window_covered, missing)
                        gaps.append(window)
                    elif not window_covered:
                        gaps.append(window)
                gaps.sort(key=lambda window: date_range(window)[0])

                # Read existing coverage BEFORE writing any gap below: a gap is zero-existing-coverage
                # by definition, so a full-range read taken now can only ever contain already-covered
                # windows' rows — reading it after the write loop would double-count the rows this call
                # itself just wrote.
                overall_start, overall_end = date_range(requested)
                covered_rows = pd.DataFrame()
                if cachable_windows and _table_exists(con, table):
                    try:
                        covered_rows = con.execute(
                            f'SELECT * FROM "{table}" WHERE "{_TIMESTAMP_COLUMN}" >= ? AND "{_TIMESTAMP_COLUMN}" <= ?',
                            [overall_start, overall_end],
                        ).fetch_df()
                    except duckdb.Error as err:
                        log_e(f"duckdb_cache: final read from {table!r} failed: {err}")
                        covered_rows = pd.DataFrame()
                    if not covered_rows.empty:
                        covered_rows[_TIMESTAMP_COLUMN] = pd.to_datetime(covered_rows[_TIMESTAMP_COLUMN], utc=True)

                generated_frames: list[pd.DataFrame] = []
                for gap in gaps:
                    call_kwargs = dict(kwargs)
                    call_kwargs[boundary_arg] = gap
                    call_kwargs["timeframe"] = None
                    result = schema_model.validate(generator(*args, **call_kwargs))
                    storage_frame = _to_storage_frame(result, effective_freqs)
                    _write_gap(con, table, storage_frame)
                    generated_frames.append(storage_frame)

                for window in non_cachable_windows:
                    call_kwargs = dict(kwargs)
                    call_kwargs[boundary_arg] = window
                    call_kwargs["timeframe"] = None
                    result = schema_model.validate(generator(*args, **call_kwargs))
                    generated_frames.append(_to_storage_frame(result, effective_freqs))

                parts = [frame for frame in (covered_rows, *generated_frames) if not frame.empty]
                assembled = pd.concat(parts, ignore_index=True) if parts else covered_rows
                # DuckDB's fetch_df() returns timestamps at microsecond resolution regardless of what
                # was written — normalize once here so the covered-window portion matches the schema
                # contract's declared 'ns' precision exactly like the already-validated gap/live-window
                # portions already do (design doc § Decorator flow step 8).
                assembled[_TIMESTAMP_COLUMN] = pd.to_datetime(assembled[_TIMESTAMP_COLUMN], utc=True).astype(
                    "datetime64[ns, UTC]"
                )
                assembled = assembled.sort_values(_TIMESTAMP_COLUMN).reset_index(drop=True)
                mask = (assembled[_TIMESTAMP_COLUMN] >= overall_start) & (assembled[_TIMESTAMP_COLUMN] <= overall_end)
                final_result = assembled[mask].reset_index(drop=True)

                if not effective_freqs:
                    final_result = final_result.drop(columns=[_TIMEFRAME_COLUMN])
                final_result = final_result.set_index(_TIMESTAMP_COLUMN).rename_axis("date")

                if post_fetch is not None:
                    final_result = post_fetch(final_result)

                if app_config.environment == "development":
                    _dispatch_integrity_check(db_path, table, requested, final_result.copy(), effective_freqs)

                return final_result
            finally:
                con.close()

        return _wrapper

    return decorator
