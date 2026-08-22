---
name: duckdb-cache-persistence
description: Use when adding or modifying any DataFrame persist/cache-or-generate code — the one required mechanism is the @duckdb_cache decorator, not read_file/read_file_windowed/find_cache_gaps/read_duckdb (being retired) or hand-rolled read/write-parquet logic.
metadata:
  status: pending-implementation
---

# DataFrame persistence via @duckdb_cache

Trigger: persisting a generated/fetched DataFrame to disk so it isn't recomputed twice, or reading such an artifact back.

**Status**: design phase — see [docs/todos/duckdb_cache_decorator.md](../../../docs/todos/duckdb_cache_decorator.md) for the full design and migration plan. Until `duckdb_cache` actually lands, new code still uses `read_file`/`read_file_windowed` (`infrastructure/datastore_engine/disk_cache.py`/`disk_cache_windowed.py`) per the `project-decisions` skill's cache-or-generate section. This skill documents the target end-state so the convention is already written down before the first caller migrates — don't treat it as live until the design doc's migration lands and this status line is removed.

## Once implemented

`@duckdb_cache(...)` is the **only** cache-or-generate mechanism for DataFrame persistence in this repo — it replaces `read_file`, `read_file_windowed`, `find_cache_gaps`, and `read_duckdb` outright, not alongside them (project-decisions § cache-or-generate: "one cache per artifact type"). Do not:

- Add a new `read_file_windowed(...)`-style call site — decorate the generator with `@duckdb_cache(...)` instead and call it directly.
- Hand-roll a new Parquet read/write path for a persisted artifact — `duckdb_cache` owns gap detection, generation dispatch, casting/validation, and the on-disk write.
- Reach into `archive_not_used_trash/` for any of the retired functions' old bodies — they're frozen reference only (project-decisions § "Check archive_not_used_trash").

See the design doc's "Decorator flow" section for the exact contract (`boundary_arg`, `freqs`, `nan_means`, `post_fetch`, the `timeframe=None` sentinel meaning "all freqs in one call"). Storage stays Parquet/ZSTD — DuckDB is the query engine over it, not a separate native store.

## Migrating an existing caller

Order from the design doc's "Migration strategy": start with a generator that has no market/symbol fan-out beyond what `dataset_db_root()` already resolves (e.g. `get_base_timeframe_ohlcv` in `infrastructure/ohlcv/ohlcv.py`) before the long tail. Copy-first: build the decorated generator as new code, keep the old `read_file_windowed(...)` call site working until the new one is verified, then remove the old call site. Once a retired function has zero remaining callers, move it into `archive_not_used_trash/` — don't delete it outright, and don't move it there while anything still imports it.
