---
name: lib-first
description: Use before implementing any new non-trivial algorithm, data transform, indicator, or concurrency primitive in this repo. Forces a "does a well-maintained library already do this" check before hand-writing code — keeps code size down and inherits the library's C/Rust-level performance instead of a slower/buggier hand-rolled version. Trigger before writing new logic, not as a retroactive review.
---

# Lib-first

Goal: least hand-written code, most delegated to well-maintained libraries — smaller surface to
maintain, and free performance (the libraries behind pandas/numpy/scipy/etc. are C/C++/Rust under the
hood).

This operationalizes the standing rule in [infrastructure.md § general
guide](../../../docs/infrastructure.md#general-guide): "prefered to do via well-known libs instead of
self-implementation (research first / then implement)."

## before writing new logic

1. Does pandas/numpy already express this as a vectorized primitive (`rolling`, `groupby`,
   `where`/`select`, `merge_asof`, ...)? → see [vectorized-pandas-numpy](../vectorized-pandas-numpy/SKILL.md).
2. Does an existing project dependency already cover it? Check `requirements.txt` first —
   `pandas-ta` (indicators), `scipy` (stats), `pandera` (validation), `optuna` (search), `ccxt`
   (exchange data) are already in use; a new need often overlaps one of these.
3. Is there a well-maintained third-party library scoped to exactly this problem? Search before
   implementing — see the candidate table below for ones already evaluated for this repo's shape of
   problem.
4. Only write custom code for the residual logic no library covers — keep it minimal, not a
   reimplementation of what the library almost gave you.

## candidate libraries: pandas/numpy complements & alternatives

All accelerate by pushing work into a C/C++/Rust layer instead of Python bytecode — evaluate before
reaching for a hand-rolled loop or a bespoke parallel wrapper. None are adopted yet; each is a
candidate to reach for once profiling shows a real bottleneck, not a default.

| Library | Category | What it buys you | Reach for it when |
| --- | --- | --- | --- |
| `numexpr` | vectorized-expression accelerator | evaluates multi-op numeric expressions in C, skips intermediate array allocations; optional pandas backend for `.eval()`/`.query()` | wide arithmetic expressions over large DataFrames where pandas allocates many temporaries |
| `bottleneck` | numpy/pandas nanops accelerator | C-implemented `nanmean`/`nanstd`/rolling ops; already an optional pandas dep, commented out in `requirements.txt` | rolling/window stats on large panels once pandas' own rolling is confirmed as the bottleneck — uncomment to activate, pandas auto-detects it |
| `numba` | JIT compiler | `@njit`/`@vectorize`/`@guvectorize` compile numeric Python to machine code; `prange` for trivially parallel loops | the rare loop that can't be expressed as pandas/numpy ops at all (see "when a loop is legitimate" in [vectorized-pandas-numpy](../vectorized-pandas-numpy/SKILL.md)) |
| `polars` | Rust DataFrame engine | multi-threaded by default, lazy query optimization, generally faster than pandas at scale; API close enough for a targeted rewrite | dataset-generation stage, if pandas is confirmed the bottleneck at current data volume — evaluate before rewriting, not preemptively |
| `pyarrow` | columnar format (C++) | already a dependency; `pd.read_parquet` / `dtype_backend="pyarrow"` for smaller memory footprint + faster I/O than CSV, zero-copy interop | any cached OHLCV/indicator artifact currently stored as CSV |
| `dask.dataframe` | parallel/out-of-core pandas | pandas-like API, chunks work across threads/processes/cluster, spills to disk when data exceeds RAM | dataset generation once the full-panel OHLCV data stops fitting in memory |
| `ta-lib` | C technical-indicator library | already referenced (commented out) in `requirements.txt` as a `pandas-ta` alternative | if `pandas-ta` indicator computation becomes a measured bottleneck — same indicators, C-backed |
| `joblib` | parallel helper | `Parallel(n_jobs=-1)(delayed(fn)(x) for x in xs)` — simpler ergonomics than raw `multiprocessing`, reuses worker pools | CPU-bound fan-out across symbols/timeframes/folds, after vectorization is exhausted |
| `concurrent.futures` (stdlib) | thread/process pool | `ThreadPoolExecutor` for I/O-bound fan-out, `ProcessPoolExecutor` for CPU-bound fan-out | default choice — no new dependency; prefer over `joblib` unless its memory-mapping/caching is actually needed |
| `asyncio` + `ccxt.async_support` | async I/O | every CCXT exchange client ships an async variant; non-blocking concurrent fetches without threads | multi-symbol/multi-timeframe fetch loops — currently sequential in `fetch_ohlcv.py` per repo scan |
| `vectorbt` | C/numba-backed backtesting | already flagged "planned, not yet integrated" in [infrastructure.md](../../../docs/infrastructure.md#vectorbt--not-yet-integrated) | once wired (owned by `evaluation-metrics.md` todo) — replaces any hand-rolled backtest loop |

## when hand-rolling is justified

- Domain-specific market-structure logic with no generic library equivalent (peak/valley detection,
  bull/bear/side classification, base-pattern detection — this repo's actual Domain layer).
- The library exists but is unmaintained, has no vectorized path (forces the exact Python loop
  [vectorized-pandas-numpy](../vectorized-pandas-numpy/SKILL.md) warns against), or pulls in a
  dependency tree disproportionate to the problem.
- Note the reason in the commit/PR ("no vectorized lib does X, hand-rolled because Y") so the next
  pass doesn't re-search from scratch.

## anti-patterns

- Reimplementing rolling-window statistics, JSON/CSV/parquet parsing, retry/backoff, or async HTTP
  plumbing that `pandas`/`pyarrow`/`ccxt`/stdlib already provide.
- Adding a bespoke thread/process pool wrapper instead of `concurrent.futures`/`joblib` — see
  [concurrency-and-blocking](../concurrency-and-blocking/SKILL.md).
