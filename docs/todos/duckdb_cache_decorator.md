# `@duckdb_cache` decorator — design spec

Generic `@duckdb_cache(...)` decorator: the single cache-or-generate entry point for DataFrame persistence.

## Workload

Training system, not online/live-fetch: random historical date → fetch surrounding history → compute derived factors. Requests skew toward old, closed data, scattered across a long history. Caching compounds across training runs as coverage trends toward "almost everything already cached."

- **`window_freq` granularity**: wider windows raise the odds a new random date is already fully covered by a nearby earlier sample — fewer, larger gaps vs. many small scattered ones.
- **Coverage-check frequency**: every random-date sample triggers a coverage lookup.

## Storage architecture

Single dataset per `datastore_relative_path`, indexed by `[timeframe, timestamp]`. No per-freq file or directory split. All freqs for a given artifact live in one logical dataset.

Physical layout is a true unpartitioned single dataset — not Hive-partitioned by `timeframe`. The dataset may still span multiple physical files (e.g. one file per gap-fill write, or compaction-driven splits), but file boundaries carry no semantic meaning — `timeframe` is never used as a partition key. Coverage queries always filter/group by the `[timeframe, timestamp]` index columns against the dataset as a whole, never by routing to a specific file based on `timeframe`.

- Writes go through a DuckDB connection, not raw pandas Parquet writes (see Concurrency).
- **Coverage/gap detection**: one query against `datastore_relative_path`'s dataset, filtered/grouped by `timeframe` within the requested window.
- **Generation**: a gap is a window with zero existing coverage across any `timeframe` — no partial-row overwrite case; persisting a gap's result is a new write, never an upsert.
- **Partial coverage** (some but not all `freqs` present for a window) raises. `covered_freqs`/`missing_freqs` come directly from `SELECT DISTINCT timeframe ... WHERE timestamp IN window`.

## Schema contract

The schema contract is inferred from the generator's return-type annotation — not passed as an explicit decorator kwarg.

Validation/casting uses pandera's own API directly (`DataFrameModel.validate(df)` / `pt.DataFrame[Model]`) — not the legacy casting helpers (`apply_as_type`, `Pandera_DFM_Type`). Anything in the old casting path that isn't just a direct `.validate()` call moves to `archive_not_used_trash/`.

Validation depth is environment-gated:
- `app_config.environment == "development"`: full `.validate()` (lazy, all checks) on every generated/re-fetched frame — same gate as the development-mode integrity check.
- Outside development: coercion/dtype cast only (`coerce=True`, no full constraint/lazy validation); the covered-window re-fetch (step 8) skips validation entirely, since that data already passed full validation once at write time (step 7).
- Applies throughout the flow: step 7 (per-gap validate/cast) and step 8 (covered-window portion) both follow this gate.

A `nan_means="not-available"` generator whose own schema grows a new column after data is already cached will misread old rows' NaN in that column as confirmed-valid rather than not-yet-computed. `"not-available"` is only used for OHLCV, whose columns are fixed — this is handled manually if it ever arises, not designed around.

## Decorator flow

```python
@duckdb_cache(
    datastore_relative_path: Path,
    dataset_type: str,                      # keys into cachable_indexes(indexes, dataset_type)
    boundary_arg: str = "date_range_str",   # raise if not in generator signature
    freqs: tuple[str, ...] = tuple(app_config.timeframes),  # () for single/no-timeframe artifact
    post_fetch: Callable[[SchemaContract], SchemaContract] | None = None,
    nan_means: Literal["not-cached", "not-available"] = "not-cached",
    window_freq: str = "D",
)
def generator(*, timeframe: str | None, date_range_str: str, ...) -> SchemaContract: ...
```

1. **Decoration time**: `boundary_arg in inspect.signature(generator).parameters`; `set(freqs) <= set(app_config.timeframes)`; generator's return-type annotation must be present and resolve to the schema contract type, else raise.
2. **Call time**: resolve dataset location dynamically from `datastore_relative_path` plus runtime context (market/symbol/exchange), never baked in at decoration time.
3. Normalize the incoming boundary to `(start, end)`.
4. Build the expected `[window × freq]` grid over the boundary at `window_freq` granularity × `freqs`. Indexes `cachable_indexes(indexes, dataset_type)` excludes are removed from the coverage/gap grid entirely (see Non-cacheable indexes) but not skipped from generation.
5. One batched DuckDB query against the `[timeframe, timestamp]`-indexed dataset classifies coverage for the whole grid at once:
   - all `freqs` covered → not a gap
   - none covered → gap, queued for generation
   - some but not all → raise `PartialCoverageError(window, covered_freqs, missing_freqs)`, logged before raising
   - `nan_means` disambiguates NaN only within the schema contract's own nullable columns: `"not-cached"` → NaN is a gap; `"not-available"` → NaN is confirmed, valid, non-gap
   - A read/parse failure on existing data is treated as absent (regenerate), not a hard error
6. Gaps are processed in ascending chronological order; each gap is written and durable independently, so an interrupted run resumes cleanly.
7. Per gap (fully-missing, already-closed window):
   - `results = generator(*args, timeframe=None, **{boundary_arg: gap})` — `timeframe=None` means "produce every freq in `freqs` for this window in one call"
   - validate/cast `results` against the schema contract
   - write via the DuckDB connection (never an upsert — gaps are, by definition, zero prior coverage)
7b. Non-cacheable indexes (see Cacheable indexes), every call, regardless of gaps: call `generator(..., timeframe=None, boundary_arg=non_cacheable_span)` unconditionally for the excluded span, validate the same way, never write it to disk, defensively delete any stray data covering those indexes.
8. Assemble the result in RAM. Step 5's query already reads actual rows for covered windows; step 7 produced validated frames for gaps; step 7b (if applicable) produced a validated frame for the non-cacheable span. Validate/cast the covered-window portion specifically (DuckDB's export timezone/resolution can drift from the schema contract's declared dtype) before concatenating with the already-validated gap/non-cacheable portions; sort; trim to requested boundary.
9. Apply `post_fetch(final_result)` if given, dispatch the development-mode integrity check without waiting on it, return `final_result` immediately.
10. Development-mode only, backgrounded, non-blocking: re-fetch the same boundary via a fresh query through the same DuckDB connection, validate, compare against the frame already returned in step 9. Never runs in production.

## Development-mode integrity check

- **Gate**: `app_config.environment == "development"` only.
- **Dispatch**: background thread, no `join()`.
- **Throttling**: none at the app level. The verification query goes through the same DuckDB connection as everything else; concurrent queries queue naturally at the connection.
- **Comparison**: `pd.testing.assert_frame_equal`, or a cheap shape/dtype/row-count check first, full comparison only on mismatch. Also compares per-column NaN counts/positions between the two frames — a shape/dtype/row-count match can still hide a NaN-distribution drift (e.g. `nan_means` misclassification, a column silently going all-NaN).
- **Logging**: `datastore_relative_path`, boundary, `freqs`, window(s) at start; success line on match; specific differing rows/columns/dtypes on mismatch.
- **On mismatch**: hard-stop the process (`os._exit(1)` after logging) — a background thread's own exception won't propagate or halt the main thread.
- **Testing this mechanism**: inject the abort call (module-level, swappable) so a forced mismatch can be asserted without killing the test runner.

## Cacheable indexes

Whether an index is eligible for caching at all is determined by `cachable_indexes(indexes, dataset_type) -> Index`, a pluggable per-dataset-type function — not a single hardcoded live-candle rule. Different dataset types have different maturity requirements before a candle's index counts as cacheable:

- **Plain OHLCV / most feature types**: an index is cacheable once its own candle is closed. The still-open current candle is excluded.
- **Lookahead-dependent features** (e.g. forward-window zigzag pivots, MFE/MAE/RER labels needing a confirmed forward horizon): an index is cacheable only once N additional candles beyond it have also closed — the exclusion zone trails further behind "now" than a single candle.

`cachable_indexes` is looked up by `dataset_type` and applied wherever the grid or a requested boundary needs to be filtered for eligibility — grid construction (step 4) and the non-cacheable-window handling (step 7b) both call it rather than assuming a single fixed live-candle cutoff.

## Non-cacheable indexes (live and near-live)

Indexes `cachable_indexes` excludes are generated and returned on every call that touches them — never counted as coverage, never persisted, always regenerated in full. Bounded cost (the excluded tail, not the dataset). Incremental refresh for this tail, if a generator needs it, is internal to that generator; it still returns a plain frame to `duckdb_cache`.

## Concurrency

Writes are issued through a DuckDB connection — the connection itself performs the write (`COPY ... TO ...` / native table write), not raw pandas Parquet writes. DuckDB's native single-writer locking on the connection arbitrates concurrent gap-fill attempts. No app-level `flock` or double-checked locking.

## New skills

- **DataFrame persistence via `duckdb_cache`** — the one way to persist/cache-or-generate a DataFrame in this repo, once implemented.
- **SQLAlchemy + DuckDB ORM** — scoped to relational metadata only (e.g. an optional future coverage ledger), not bulk time-series rows.

## Testing

- unit: coverage classification (all/none/partial, both `nan_means` modes) against a small on-disk fixture + in-memory DuckDB connection.
- integration: full round-trip — gap detection, idempotent re-run, `post_fetch`, `PartialCoverageError`, live-window never persisted.
- regression: added once a concrete bug surfaces.

## Open items

None outstanding.

## File bloat / stale data

Excluded from this design — handled by a separate mechanism outside `duckdb_cache`.

- Compaction merges the small files produced by successive gap-fill writes into fewer, larger files. Since the dataset is unpartitioned, this is a plain file-count/size operation with no `timeframe`-aware repartitioning logic — compaction preserves the `[timeframe, timestamp]` index correctly across the merge.
- Compaction and cleanup mechanisms apply to whatever files accumulate under this layout.
- Manual regeneration of an already-cached window (generator bugfix, corrected source data) requires deleting the old data first — there is no automatic staleness/versioning detection.

## Non-goals

- Reading/migrating legacy cached data.
- Multi-writer distributed locking beyond DuckDB's own connection-level arbitration.
- ClickHouse or any client-server backing.
- File compaction and stale/versioned-data cleanup (see File bloat / stale data).
- Coverage ledger — a separate metadata table tracking which `(dataset, timeframe, window)` combinations are already generated, to avoid scanning the real dataset for coverage checks.
- Cache invalidation on generator-logic change.
