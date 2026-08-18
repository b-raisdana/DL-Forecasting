# Feather → Parquet migration plan

First (and, for now, only) step of a broader data-pipeline review — everything else considered
(catalog, DuckDB, data-quality extensions, ClickHouse, PyArrow dataset API, versioned datasets,
Spark/Cassandra/Polars/Zarr) is deferred; none of it is needed to take this step, and several of
those options only pay off once Parquet is actually the on-disk format. Not re-derived here.

- [Feather → Parquet migration plan](#feather--parquet-migration-plan)
  - [why](#why)
  - [how — mirrors the existing CSV-zip → Feather migration exactly](#how--mirrors-the-existing-csv-zip--feather-migration-exactly)
  - [distinguishing data_frame_types under Parquet](#distinguishing-data_frame_types-under-parquet)
  - [multi-broker handling](#multi-broker-handling)
  - [where — on-touch, incremental, no sweep](#where--on-touch-incremental-no-sweep)
  - [testing](#testing)

## why

- `pyarrow` (already a dependency) makes `to_parquet`/`read_parquet` drop-in replacements for
  `to_feather`/`read_feather` — same `zstd` compression option, near-zero migration cost.
- Parquet carries row-group min/max stats Feather lacks, enabling range-query pruning any future
  query engine gets for free once files are Parquet. Feather was chosen originally for same-process
  zero-copy handoff, not on-disk range queries — which is what this cache actually needs.
- Prerequisite for everything else in the broader review (catalog, DuckDB, PyArrow dataset API) —
  the reason it's the first step, not one option among several.

## how — mirrors the existing CSV-zip → Feather migration exactly

`infrastructure/disk_cache.py` already has this exact pattern for a prior format swap; reuse it
verbatim, one tier further:

- `_parquet_file_path()` alongside the existing `_feather_file_path()`/`_csv_zip_file_path()`.
- `write_data_file()`: writes `.parquet` (zstd) instead of `.feather` (zstd) — one-line change
  (`df.reset_index().to_parquet(path, compression="zstd")` instead of `.to_feather(...)`).
- `_read_raw_data_file()`: three-tier fallback instead of two — parquet → feather → legacy CSV-zip.
  A whole-file feather read with no `.parquet` present migrates to parquet the same way today's
  CSV-zip→feather migration works (`_migrate_csv_zip_to_feather` → add a symmetric
  `_migrate_feather_to_parquet`), old feather file removed only once the parquet write succeeds.
- `remove_data_file()`: try parquet, then feather, then CSV-zip (extends its existing
  `except FileNotFoundError` chain by one tier).

## distinguishing data_frame_types under Parquet

Already solved by the existing layout — no new mechanism needed, one regex extension required:

- **Directory-level**: `_data_frame_type_dir(data_frame_type, file_path)` already gives every
  `data_frame_type` its own subdirectory (`file_path/<data_frame_type>/`) — a glob against that
  directory can never cross types.
- **Filename-level**: `_legacy_file_pattern(data_frame_type)` already embeds the literal
  `data_frame_type` string in its regex
  (`^{re.escape(data_frame_type)}\.(?P<range>...)\.(?P<ext>feather|zip)$`). The one required change:
  extend the `ext` alternation to `feather|zip|parquet` so gap-finding, cleanup, and legacy-file-reuse
  (`disk_cache_gaps.py`, `cleanup_redundant_cache_files`) recognize the new extension — same role
  Feather already plays there today.

## multi-broker handling

Already solved by the existing layout — no change needed. `symbol_data_path(path_of_data, exchange,
market, trading_pair)` puts `exchange` (broker) as the first path segment under `path_of_data`, so
different brokers (`ccxt_client.py`'s `SUPPORTED_BROKERS`: `kucoin`, `binance`) never share a
directory tree regardless of format — Parquet inherits this for free, same as Feather does today.
Non-goal: representing "the same date range fetched from two different brokers as alternative/
reconciled sources" isn't a modeled concept anywhere today and isn't needed for this migration —
out of scope unless it becomes a real requirement later.

## where — on-touch, incremental, no sweep

Same discipline as the existing feather/ZSTD-migration-on-touch rule: a file already being edited for
another reason gets its write call swapped to Parquet as part of that edit; files not otherwise
touched keep reading via the existing feather/CSV-zip fallback chain until naturally rewritten. No
standalone repo-wide conversion pass.

Primary touch point: `infrastructure/disk_cache.py` (`_feather_file_path`/`write_data_file`/
`_read_raw_data_file`/`remove_data_file`) plus `disk_cache_layout.py`'s `_legacy_file_pattern` — every
one of the ~20+ callers goes through these two files, so the format swap needs exactly these places
changed, not one change per caller.

## testing

- Unit test mirroring `tests/unit/infrastructure/test_disk_cache.py`'s existing feather-write
  assertions, for the new parquet write path.
- Migration test for `_migrate_feather_to_parquet`: an existing on-disk `.feather` file (no
  `.parquet` present) is read correctly and migrated to `.parquet` on a whole-file read, old feather
  file removed only after the parquet write succeeds. No equivalent CSV-zip→feather migration test
  exists today to mirror exactly — write this one from scratch, asserting the same shape
  `_migrate_csv_zip_to_feather` implements.
- Existing full `read_file()`/`cache_on_disk()`/windowing test suite must stay green — the format
  swap must not change any DataFrame contents/dtypes on roundtrip.
