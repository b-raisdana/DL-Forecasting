---
name: concurrency-and-blocking
description: Use when adding code that does network/exchange I/O (CCXT calls, file reads/writes) or CPU-heavy fan-out (indicator computation across symbols/timeframes, dataset generation, backtesting sweeps). Picks the right concurrency primitive so new code doesn't block its caller and stays compatible with running under asyncio/thread-pool/process-pool later. Trigger before writing a new I/O call or a loop over independent symbols/timeframes/folds.
---

# Concurrency & blocking

Goal: new code should never assume it's the only thing running, and should pick the cheapest
concurrency primitive that fits the actual bottleneck — not add threads/processes reflexively.

## principles

1. **Production-grade speed/resource use by default** — not deferred to a later optimization pass.
2. **Concurrency-compatible** — new code shouldn't assume it's the only thing running; avoid designs
   that can't later run under asyncio/thread-pool/process-pool without a rewrite (e.g. mutating shared
   global state instead of passing it explicitly — already required by [infrastructure.md §
   Dependency injection](../../../docs/infrastructure.md#dependency-injection)).
3. **Least blocking** — I/O (exchange fetch, disk read/write) shouldn't block a thread that could be
   doing other work while it waits.
4. **Least resource usage** — minimize memory footprint: right-sized dtypes, no needless copies,
   stream/chunk instead of materializing everything.
5. **Library-first** — research an existing, well-maintained library before hand-writing an algorithm;
   see [lib-first](../lib-first/SKILL.md) for the checklist and candidate-library table.

## first question: I/O-bound or CPU-bound?

- **I/O-bound** (CCXT fetch in `data_processing/fetch_ohlcv.py`, disk read/write, any network call) —
  the bottleneck is waiting, not computing. Don't block the caller with a synchronous wait in a loop
  over multiple symbols/timeframes:
  - `ccxt.async_support` + `asyncio.gather` for concurrent fetches across symbols/timeframes, or
  - `concurrent.futures.ThreadPoolExecutor` when the surrounding code is sync and a full asyncio
    rewrite isn't worth it yet — threads release the GIL while blocked on I/O.
- **CPU-bound** (indicator computation over a large panel, dataset generation, training, backtesting
  sweeps) — threads don't help (GIL). Order of attack:
  1. Vectorize first — a single pandas/numpy call over the whole panel already runs in C and releases
     the GIL; see [vectorized-pandas-numpy](../vectorized-pandas-numpy/SKILL.md). This alone usually
     removes the need for parallelism.
  2. Only if still measured as a bottleneck: split independent units (symbols, timeframes, folds,
     Optuna trials) across `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel`.

## least-blocking rules

- Never put a network/disk call inside a tight loop without a concurrency wrapper if the loop runs
  over more than a handful of independent items (symbols × timeframes fetch loops are the recurring
  shape in this repo).
- Don't hold a lock or block on I/O inside code that also holds GPU/TF resources — keep data
  fetch/prep and model-compute phases separable so one can be made concurrent without touching the
  other.
- Config/shared state must be passed explicitly (already required by [infrastructure.md § Dependency
  injection](../../../docs/infrastructure.md#dependency-injection)) — this is also what makes a
  function safe to run under a process pool; anything reaching into shared mutable global state can't
  be parallelized across processes.

## least resource usage

- Downcast dtypes once shape is finalized (`float32` vs `float64`, `category` for repeated strings) —
  don't leave pandas defaults.
- Stream/chunk large reads (`chunksize=`, parquet row groups) instead of materializing a full
  multi-year OHLCV file for a slice — the repository layer's job ([infrastructure.md § Repository
  design pattern](../../../docs/infrastructure.md#repository-design-pattern)).
- Prefer generators over fully-materialized Python lists for values only consumed once downstream.

## scope note

This repo is a single-process offline pipeline (SOA rejected — see the `code-layers` skill). Concurrency
here means asyncio/
thread-pool/process-pool primitives *within* that process for I/O and CPU fan-out, not standing up
separate services.
