# performance, concurrency & resource usage

Standing rules for all new code in this repo, not a one-off optimization pass. Enforced day to day by
the skills in [performance-skills.md](performance-skills.md).

- [performance, concurrency \& resource usage](#performance-concurrency--resource-usage)
  - [principles](#principles)
  - [decision flow: I/O-bound or CPU-bound?](#decision-flow-io-bound-or-cpu-bound)
  - [library-first checklist](#library-first-checklist)
  - [candidate libraries: pandas/numpy complements \& alternatives](#candidate-libraries-pandasnumpy-complements--alternatives)
  - [resource-usage rules](#resource-usage-rules)
  - [scope note](#scope-note)

## principles

1. **Production-grade speed/resource use by default** — not deferred to a later optimization pass.
2. **Concurrency-compatible** — new code shouldn't assume it's the only thing running; avoid designs
   that can't later run under asyncio/thread-pool/process-pool without a rewrite (e.g. mutating shared
   global state instead of passing it explicitly — already required by [infrastructure.md §
   Dependency injection](infrastructure.md#dependency-injection)).
3. **Least blocking** — I/O (exchange fetch, disk read/write) shouldn't block a thread that could be
   doing other work while it waits.
4. **Least resource usage** — minimize memory footprint: right-sized dtypes, no needless copies,
   stream/chunk instead of materializing everything.
5. **Library-first** — research an existing, well-maintained library before hand-writing an algorithm.
   Already the stated rule in [infrastructure.md § general guide](infrastructure.md#general-guide)
   ("prefered to do via well-known libs instead of self-implementation (research first / then
   implement)"); this doc operationalizes it for performance-shaped problems.

## decision flow: I/O-bound or CPU-bound?

- **I/O-bound** (CCXT fetch in `data_processing/fetch_ohlcv.py`, disk read/write, any network call) —
  the bottleneck is waiting, not computing. Don't block the caller with a synchronous wait in a loop
  over multiple symbols/timeframes:
  - `ccxt.async_support` + `asyncio.gather` for concurrent fetches, or
  - `concurrent.futures.ThreadPoolExecutor` when the surrounding code is sync and a full asyncio
    rewrite isn't worth it yet — threads release the GIL while blocked on I/O.
- **CPU-bound** (indicator computation over a large panel, dataset generation, training, backtesting
  sweeps) — threads don't help (GIL). Order of attack:
  1. Vectorize first — a single pandas/numpy call over the whole panel already runs in C and releases
     the GIL; see the `vectorized-pandas-numpy` skill. This alone usually removes the need for
     parallelism.
  2. Only if still measured as a bottleneck: split independent units (symbols, timeframes, folds,
     Optuna trials) across `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel`.

## library-first checklist

Before writing new logic:

1. Does pandas/numpy already express this as a vectorized primitive (`rolling`, `groupby`,
   `where`/`select`, `merge_asof`, ...)?
2. Does an existing project dependency already cover it? `requirements.txt` already has `pandas-ta`
   (indicators), `scipy` (stats), `pandera` (validation), `optuna` (search), `ccxt` (exchange data) —
   a new need often overlaps one of these.
3. Is there a well-maintained third-party library scoped to exactly this problem? Search before
   implementing — see the candidate table below for ones already evaluated for this repo's shape of
   problem.
4. Only write custom code for the residual logic no library covers, and note *why* no library fit
   (in the commit/PR) so the next pass doesn't re-search from scratch.

## candidate libraries: pandas/numpy complements & alternatives

All accelerate by pushing work into a C/C++/Rust layer instead of Python bytecode — evaluate before
reaching for a hand-rolled loop or a bespoke parallel wrapper. None are adopted yet; each is a
candidate to reach for once profiling shows a real bottleneck, not a default.

| Library | Category | What it buys you | Reach for it when |
| --- | --- | --- | --- |
| `numexpr` | vectorized-expression accelerator | evaluates multi-op numeric expressions in C, skips intermediate array allocations; optional pandas backend for `.eval()`/`.query()` | wide arithmetic expressions over large DataFrames where pandas allocates many temporaries |
| `bottleneck` | numpy/pandas nanops accelerator | C-implemented `nanmean`/`nanstd`/rolling ops; already an optional pandas dep, commented out in `requirements.txt` | rolling/window stats on large panels once pandas' own rolling is confirmed as the bottleneck — uncomment to activate, pandas auto-detects it |
| `numba` | JIT compiler | `@njit`/`@vectorize`/`@guvectorize` compile numeric Python to machine code; `prange` for trivially parallel loops | the rare loop that can't be expressed as pandas/numpy ops at all (see "when a loop is legitimate" in `vectorized-pandas-numpy` skill) |
| `polars` | Rust DataFrame engine | multi-threaded by default, lazy query optimization, generally faster than pandas at scale; API close enough for a targeted rewrite | dataset-generation stage, if pandas is confirmed the bottleneck at current data volume — evaluate before rewriting, not preemptively |
| `pyarrow` | columnar format (C++) | already a dependency; `pd.read_parquet` / `dtype_backend="pyarrow"` for smaller memory footprint + faster I/O than CSV, zero-copy interop | any cached OHLCV/indicator artifact currently stored as CSV |
| `dask.dataframe` | parallel/out-of-core pandas | pandas-like API, chunks work across threads/processes/cluster, spills to disk when data exceeds RAM | dataset generation once the full-panel OHLCV data stops fitting in memory |
| `ta-lib` | C technical-indicator library | already referenced (commented out) in `requirements.txt` as a `pandas-ta` alternative | if `pandas-ta` indicator computation becomes a measured bottleneck — same indicators, C-backed |
| `joblib` | parallel helper | `Parallel(n_jobs=-1)(delayed(fn)(x) for x in xs)` — simpler ergonomics than raw `multiprocessing`, reuses worker pools | CPU-bound fan-out across symbols/timeframes/folds, after vectorization is exhausted |
| `concurrent.futures` (stdlib) | thread/process pool | `ThreadPoolExecutor` for I/O-bound fan-out, `ProcessPoolExecutor` for CPU-bound fan-out | default choice — no new dependency; prefer over `joblib` unless its memory-mapping/caching is actually needed |
| `asyncio` + `ccxt.async_support` | async I/O | every CCXT exchange client ships an async variant; non-blocking concurrent fetches without threads | multi-symbol/multi-timeframe fetch loops — currently sequential in `fetch_ohlcv.py` per repo scan |
| `vectorbt` | C/numba-backed backtesting | already flagged "planned, not yet integrated" in [infrastructure.md](infrastructure.md#vectorbt--not-yet-integrated) | once wired (owned by `evaluation-metrics.md` todo) — replaces any hand-rolled backtest loop |

## resource-usage rules

- Downcast dtypes once shape is finalized (`float32` vs `float64`, `category` for repeated strings) —
  don't leave pandas defaults.
- Avoid `df.copy()` fan-out; prefer in-place `.loc` assignment or chained ops without intermediate
  copies (see `vectorized-pandas-numpy` skill).
- Stream/chunk large reads (`chunksize=`, parquet row groups) instead of materializing a full
  multi-year OHLCV file for a slice — the repository layer's job ([infrastructure.md § Repository
  design pattern](infrastructure.md#repository-design-pattern)).
- Prefer generators over fully-materialized Python lists for values only consumed once downstream.

## scope note

This repo is a single-process offline pipeline (SOA explicitly rejected — see
[architecture-layers.md](architecture-layers.md)). "Concurrency-compatible" here means asyncio/
thread-pool/process-pool primitives *within* that process for I/O and CPU fan-out, not standing up
separate services.
