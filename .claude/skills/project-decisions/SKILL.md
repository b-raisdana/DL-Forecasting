---
name: project-decisions
description: Use for any of — adding a "fetch/compute and persist" function (cache-or-generate); placing a new module/file under app/ (code-layers); adding network I/O or CPU-heavy fan-out (concurrency-and-blocking); before hand-rolling a new algorithm/transform (lib-first); deciding what test type a change needs (test-strategy); writing/reviewing pandas/numpy code (vectorized-pandas-numpy); or touching a file that reads/writes the OHLCV/indicator/label disk cache (feather/ZSTD migration-on-touch). One skill, several independent sections — read only the one(s) that match.
---

# Project decisions

- [Cache or generate](#cache-or-generate)
- [Code layers](#code-layers)
- [Concurrency & blocking](#concurrency--blocking)
- [Lib-first](#lib-first)
- [Test strategy](#test-strategy)
- [Vectorized pandas/numpy](#vectorized-pandasnumpy)
- [Feather/ZSTD migration on touch](#featherzstd-migration-on-touch)

## Cache or generate

Trigger: adding a "fetch or compute this data" function, or a function called repeatedly with the same effective inputs.

**Disk-level** (persists across restarts): `read_file()` in `helper/data_preparation.py` (`ExtendedDf.read_file()` is the pandera-bound variant). Reads feather/ZSTD first, falls back to legacy CSV-zip (auto-migrates to feather/ZSTD and deletes the zip on a full read), else calls `generator(date_range_str)` and re-reads. New persisted artifacts: reuse `read_file()` with a `data_frame_type` + `generator` — don't invent a new naming/read/write scheme. Generators write via `write_data_file(df, data_frame_type, date_range_str, file_path)` (feather/ZSTD), never `to_csv(...zip...)`. An LRU memo (~32 entries) sits in front of the disk read, keyed on `(data_frame_type, date_range_str, file_path, skip_rows, n_rows)`; skipped for `datarange_is_not_cachable()` ranges (touch the live/incomplete present — never memoize, never leave cached on disk).

**In-memory** (derived from an already-in-RAM object, doesn't need to survive the process): cache on `df.attrs[private_key]`, not a dict — `pd.DataFrame` is unhashable. Self-bounded by the owning object's lifetime. Example: `_cached_training_frames()` in `training_datasets.py`, avoiding ~100x/quarter recompute of indicators/labels in the dataset-generator producer loops.

Rules:
- Key on exactly what the computation depends on — no more, no less.
- Always return a copy on a cache hit, never the cached object (mutation must not corrupt the cache).
- Never cache/memoize a `datarange_is_not_cachable()` range.
- Bound disk-level caches (fixed-size LRU, evict-oldest); in-memory caches self-bound via object lifetime.
- One cache per artifact type — don't add a second read-or-generate branch for data already cached elsewhere (Repository design pattern).

## Code layers

Trigger: adding a module/file under `app/`, or reviewing a layer boundary. DDD-flavored, not MVC — offline pipeline, no request/response cycle. Dependency arrows point inward: Presentation → Application → Infrastructure/Domain, and Infrastructure → Domain schemas only. Domain depends on nothing in-house.

1. **Presentation** (`presentation/*`) — plotting, entrypoints, notebooks.
2. **Application** (`application/dataset_generation`, `application/preprocessing`, `application/model_implementations`, `application/optimization`, `application/backtesting`, `application/live_trading`) — orchestrates Domain + Infrastructure for one job (dataset gen, training, prediction, optimization, backtesting).
3. **Infrastructure** (`infrastructure/ohlcv`, `infrastructure/market_data_fetch`, `infrastructure/model_artifacts`, `infrastructure/order_execution`, `config/Config.py`) — I/O/framework adapters only: exchange fetch, disk cache/persistence, GPU/TF setup, config, logging. **No calculation logic** — `infrastructure/ohlcv/atr.py` is the pattern to copy: the disk-cached repository generator (`get_multi_timeframe_ohlcva`, `@cache_on_disk`-decorated) lives here because it does I/O; the indicator math it calls (`insert_atr`) lives in `domain/technical_analysis/`, imported in.
4. **Domain** (`domain/technical_analysis`, `domain/price_action`, `domain/order`, `domain/schemas`) — pure algorithms (DataFrame→DataFrame, no I/O) + PanderaDFM schemas.
   - `domain/technical_analysis/` — generic indicator math (ATR, RMA, …).
   - `domain/price_action/` — price-action/market-structure pattern detection (peak/valley, bull/bear/side, base-pattern, pivots).
   - `domain/schemas/` — PanderaDFM value-object schemas.

Placing new code:
- Pure transform, no I/O → Domain (`domain/technical_analysis/*` for indicator math, `domain/price_action/*` for pattern detection) or a new PanderaDFM schema under `domain/schemas/`.
- Orchestrates multiple domain steps + I/O for one job → Application (e.g. `application/backtesting/BasePatternStrategy.py`, `application/dataset_generation/*`).
- Direct disk/exchange/GPU/config/logging → Infrastructure (`infrastructure/market_data_fetch/*`, `infrastructure/ohlcv/*`, `config/Config.py`); new persisted-artifact types go through a repository (`get`/`save`), not inline I/O.
- Plot/print/CLI → Presentation (`presentation/*`).

DDD map: value objects = PanderaDFM schemas (no persistent identity); domain services = the TA transform chain (PeakValley → BullBearSide → BasePattern → ATR/pivots → ftc); application services = orchestrators; bounded contexts = *price action* (`domain/price_action`, `application/backtesting`) and *forecasting* (`application/dataset_generation`, `application/model_implementations`), sharing the OHLCV/PanderaDFM kernel.

Rules:
- Never import Presentation/Application from Domain. Current known violation: several `domain/price_action/*` modules (`BasePattern.py`, `BullBearSide.py`, `BullBearSidePivot.py`, `PeakValleyPivots.py`, `AtrMovementPivots.py`) import `get_multi_timeframe_ohlcva`/`read_multi_timeframe_ohlcva` straight from `infrastructure/ohlcv/atr.py` (a Domain→Infrastructure dependency, not just schemas) — don't copy this pattern into new code; a proper fix passes the OHLCVA frame in rather than having Domain pull it.
- Don't fork a second copy of a module for a new pipeline — extend/move the existing one instead of duplicating.
- Pass config/state explicitly into constructors/functions, never pulled from `app_config` mid-function.

Splitting an oversized file (~500+ lines, by responsibility not raw count): extract one cohesive cluster per commit (few/no external callers first — check via grep), re-export moved names (`from new_module import name as name`) so callers keep working unchanged in that commit, log remaining clusters as follow-up instead of a full-file rewrite in one pass.

## Concurrency & blocking

Trigger: adding network/exchange I/O or CPU-heavy fan-out (indicators across symbols/timeframes, dataset gen, backtesting sweeps). Pick the cheapest primitive for the actual bottleneck — don't add threads/processes reflexively. Single-process offline pipeline (no services) — concurrency means in-process asyncio/thread-pool/process-pool, never standing up separate services.

- **I/O-bound** (CCXT fetch, disk read/write): don't block a loop over symbols/timeframes with a sync wait — `ccxt.async_support` + `asyncio.gather`, or `ThreadPoolExecutor` if a full asyncio rewrite isn't worth it yet (threads release the GIL while blocked on I/O).
- **CPU-bound** (indicator computation, dataset gen, training, backtesting sweeps): threads don't help (GIL). Vectorize first (see vectorized-pandas-numpy below) — usually removes the need for parallelism entirely; only if still a measured bottleneck, split independent units (symbols/timeframes/folds/Optuna trials) across `ProcessPoolExecutor`/`joblib.Parallel`.

Rules:
- No network/disk call inside a tight loop over more than a handful of independent items without a concurrency wrapper.
- Keep data fetch/prep and GPU/TF-compute phases separable — don't block on I/O while holding GPU/TF resources.
- Config/shared state passed explicitly (also what makes a function process-pool-safe — can't parallelize across processes if it reaches into shared mutable globals).
- Downcast dtypes once shape is finalized (`float32`, `category` for repeated strings); stream/chunk large reads (`chunksize=`, parquet row groups) instead of materializing a full multi-year file for a slice; prefer generators over fully-materialized lists for once-consumed values.

## Lib-first

Trigger: before implementing any new non-trivial algorithm/transform/indicator/concurrency primitive.

Order of attack:
1. Does pandas/numpy already express it (`rolling`, `groupby`, `where`/`select`, `merge_asof`)?
2. Does an existing dependency cover it? Already in use: `pandas-ta`, `scipy`, `pandera`, `optuna`, `ccxt`.
3. Is there a well-maintained third-party lib scoped to exactly this (table below)?
4. Only hand-write the residual logic no library covers.

Candidates (none adopted yet — evaluate once profiling shows a real bottleneck, not preemptively):

| Library | Reach for it when |
| --- | --- |
| `numexpr` | wide numeric expressions over large DataFrames, skip pandas' intermediate temporaries |
| `bottleneck` | C nanops/rolling stats once pandas' own rolling is the confirmed bottleneck (commented out in `requirements.txt`) |
| `numba` | the rare loop that truly can't be expressed as pandas/numpy ops |
| `polars` | dataset-gen rewrite, if pandas is confirmed the bottleneck at current scale |
| `pyarrow` | already a dep — feather/parquet I/O, zero-copy interop |
| `dask.dataframe` | dataset gen once data stops fitting in RAM |
| `ta-lib` | C-backed `pandas-ta` alternative if indicator computation is a measured bottleneck |
| `joblib` | CPU fan-out across symbols/timeframes/folds, after vectorization is exhausted |
| `concurrent.futures` (stdlib) | default thread/process pool — prefer over `joblib` unless its memory-mapping/caching is actually needed |
| `asyncio` + `ccxt.async_support` | multi-symbol/timeframe fetch loops (currently sequential in `fetch_ohlcv.py`) |
| `vectorbt` | planned backtesting replacement, not yet integrated |

Hand-rolling is justified for: domain-specific market-structure logic with no generic lib equivalent (peak/valley detection, bull/bear/side classification, base-pattern detection); or a lib that's unmaintained, has no vectorized path, or pulls a disproportionate dependency tree. Note the reason in the commit ("no vectorized lib does X, hand-rolled because Y") so the next pass doesn't re-search from scratch.

Anti-pattern: reimplementing rolling-window stats, JSON/CSV/parquet parsing, retry/backoff, or async HTTP plumbing that `pandas`/`pyarrow`/`ccxt`/stdlib already provide; a bespoke thread/process pool instead of `concurrent.futures`/`joblib`.

## Test strategy

Trigger: deciding what test type a change needs, or reviewing test coverage. This repo: offline data/ML pipeline (pandas → TensorFlow → `backtrader`), no HTTP API/UI/other-team's-service caller — several categories below don't apply here.

| change is... | write a... | marker |
| --- | --- | --- |
| pure function, no I/O (indicator/label/scaling math) | unit test, synthetic in-memory fixture | `unit` |
| touching legacy code with no independent spec | characterization test — pin *today's actual* output, never hand-derive | `characterization` |
| wiring modules together (dataset assembly, schema-gated repository read) | integration test | `integration` |
| fixing a bug / protecting an invariant that broke before | regression test, named after the invariant, not the ticket | `regression` |
| broad "does it still work" check, safe every commit | smoke test | `smoke` |
| full fetch→dataset→train→predict→strategy chain | e2e test, real/pinned data, not run every commit | `e2e` |
| a vectorization/throughput claim | perf test, explicit budget | `perf` |

Characterization discipline: run the real function against the fixture, capture what it *actually* outputs today, assert that — never hand-compute an expected value. It's a refactor safety net, not a spec-conformance check (that's a regression test against a written spec, once one exists).

Not applicable here: CDC (no separately-deployed consumer/provider pair), synthetic/shadow monitoring (nothing deployed/serving), UI testing (plotly usage is diagnostic, not product UI), fault injection (no live broker/network calls yet).

Reviewing a PR: ask what type the change actually needs per the table — not "does it have *a* test." (Mechanics — directory layout, naming, fixture policy, marker config — are in the separate `pytest` skill.)

## Vectorized pandas/numpy

Trigger: writing/reviewing any pandas/numpy code (dataset gen, indicator/label computation, scaling/normalization, OHLCV processing). A loop over rows/samples is a bug magnet and 10-1000x slower than the vectorized equivalent.

Red flags to eliminate: `for i in range(len(df))`, `df.iterrows()`, `df.apply(..., axis=1)`; `while remained > 0: ... .append(...)` sample-generation loops that recompute one slice at a time (see `train_data_of_mt_n_profit` in `training_datasets.py`) — prefer precomputing all boundaries at once (`np.random.randint(size=n)`) and slicing in one vectorized pass; repeated `np.array(df[cols])` conversion inside a hot loop — convert once, slice after; growing a list of DataFrames/arrays then `pd.concat`/`np.array` at the end, when boundaries could've been precomputed; scalar-by-scalar column assignment in a loop over columns.

Preferred patterns: boolean masking (`df.loc[mask, col] = value`); `np.where(cond, a, b)` / `np.select([...], [...], default=c)`; broadcasting across the whole array/DataFrame; `.groupby(...).transform(...)` / `.rolling(...).agg(...)`; batch `.loc`/`pd.IndexSlice` or `numpy.lib.stride_tricks.sliding_window_view` for windowed extraction at many offsets; `.to_numpy()` (not bare `.values`) for hot-path ndarray drops.

A loop is legitimate for: truly sequential/stateful logic where step *n* depends on step *n-1*'s *computed result* (vectorize the per-step body, the outer loop may stay); one-time setup/config code, not hot paths.

Review checklist: any row/element/sample loop replaceable by a mask, `np.where`/`np.select`, groupby/rolling, or a fully-vectorized index computation? Any repeated array conversion of the same columns to hoist out of a loop? Any column-by-column loop collapsible to one vectorized expression over the selected columns? Does the vectorized version's shape/dtype match what downstream code expects (upcasting, index alignment) before replacing the loop?

## Feather/ZSTD migration on touch

Trigger: about to edit a file that reads or writes the OHLCV/indicator/label disk cache — the `read_file()`/`data_frame_type` family (see [Cache or generate](#cache-or-generate)).

Feather/ZSTD (`write_data_file()`/`read_file()`-native) is the primary on-disk format; legacy CSV-zip (`to_csv(..., compression='zip')`) is the old format — still readable, no longer written. Migration is **incremental, on-touch only** — never a repo-wide sweep in one change.

Rule: if a file you're already modifying for another reason still writes via `to_csv(os.path.join(file_path, f'{name}.{date_range_str}.zip'), compression='zip')`, replace that call with `write_data_file(df, '<name>', date_range_str, file_path)` (import from `helper.data_preparation`) as part of the same edit. If the file isn't otherwise being touched, leave it as-is — the read-side (`_read_raw_data_file()` in `helper/data_preparation.py`) already auto-converts any lingering CSV-zip to feather/ZSTD and deletes the zip the first time it's read, so untouched files self-heal on next read regardless.

Don't: proactively grep the whole repo for remaining `.zip` writers and convert them all in one pass — that's the "change whole project at once" this rule exists to avoid.
