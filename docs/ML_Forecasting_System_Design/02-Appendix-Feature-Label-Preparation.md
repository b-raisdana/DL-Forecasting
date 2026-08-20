# Appendix — Feature & Label Preparation

Computation and caching strategy for features and labels consumed by the trainer. Covers what is computed per sample versus precomputed and cached, cache lifecycle, concurrency, and storage design.

## What is computed per sample vs. precomputed

- **Per sample (recomputed fresh for each anchor, never cached):** `normal_close → rel_normal_close`, `rer`, `MFE / MAE`. Source values may be backfilled or revised without touching a cache.
- **Precomputed (once over full data history, per timeframe, then queried per sample):** all other features and labels. Computation happens in bulk; individual sample queries are served from the disk-backed datastore.

## Incremental append-only caching

Caching is incremental and immutable — no updates, only appends. If two anchors are 1 candle apart, only the non-overlapping candle(s) are newly computed; the rest is served from cache.

- Querying Aug 1–5 computes and caches all timeframes.
- A later query for Aug 2–7 fetches Aug 2–5 from cache and computes/appends only Aug 5–7.

## Anchor rounding

- Anchors are chosen at the current time, rounded to 5-minute boundaries.
- Anchor rounding does not affect 1h candles, which are UTC-fixed and unaffected by anchor rounding.

## Cache invalidation

No invalidation by mutation. The dataframe/dataset-type name encodes its definition or version; changing feature logic means choosing a new name, which starts from a fresh one-time full-history computation. Old-named tables are orphaned and ignored.

## Concurrency

Data is append-only and never rewritten, so there is no update-conflict risk. DuckDB has a single-writer limitation: concurrent `build_dataset` writers should be serialized or restricted to read-only to avoid lock contention. A reader that arrives while a write is in progress should wait for the write block to finish before reading.

## Missing or incomplete data

Fill with NaN rather than dropping or erroring, to simplify adoption. NaN-containing selected samples are filtered or dropped later at feed time, not at cache time.

## Trainer → build_dataset flow

- **trainer** requests a dataset for a given anchor range and timeframe set.
- **build_dataset** checks the cache for coverage; computes any missing candle ranges, appends them immutably, then returns the assembled samples.
- A second **build_dataset** pass for a partially overlapping range hits cache for the overlap and appends only the new range.

Each hand-off passes range, timeframe list, and feature-group identity; responsibility for cache lookup, incremental computation, and NaN handling sits at the `build_dataset` layer.

## Storage design

Group logically related features and labels into as few DuckDB dataframe-types or tables as possible, to keep the folder structure clean and queries efficient.

Recommended naming convention for dataframe-types: `feature-group + timeframe + version/definition-hash`. The version/definition-hash component is what changes when logic changes, triggering a fresh table rather than an in-place mutation.

## Out of scope

- Partitioning strategy.
- Full feature and label inventory table (revisit later).
