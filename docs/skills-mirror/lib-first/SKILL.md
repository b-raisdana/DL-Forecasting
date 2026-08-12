---
name: lib-first
description: Use before implementing any new non-trivial algorithm, data transform, indicator, or concurrency primitive in this repo. Forces a "does a well-maintained library already do this" check before hand-writing code — keeps code size down and inherits the library's C/Rust-level performance instead of a slower/buggier hand-rolled version. Trigger before writing new logic, not as a retroactive review.
---

# Lib-first

Goal: least hand-written code, most delegated to well-maintained libraries — smaller surface to
maintain, and free performance (the libraries behind pandas/numpy/scipy/etc. are C/C++/Rust under the
hood). Full rationale and evaluated candidates:
[performance-and-concurrency.md](../../../docs/performance-and-concurrency.md).

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
   implementing — see [performance-and-concurrency.md § candidate
   libraries](../../../docs/performance-and-concurrency.md#candidate-libraries-pandasnumpy-complements--alternatives)
   for ones already evaluated for this repo.
4. Only write custom code for the residual logic no library covers — keep it minimal, not a
   reimplementation of what the library almost gave you.

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
