---
name: cache-or-generate
description: Use when adding code that produces a dataset/table derivable from other data (fetched OHLCV, computed indicators, computed labels, peak/valley detection) — read a cached copy if a valid one exists, otherwise generate it once and reuse. Trigger before writing a new "fetch or compute this data" function, and when a function is observed being called repeatedly with the same effective inputs.
---

# Cache-or-generate

Goal: never recompute or re-fetch the same data twice within its validity scope. This is the standing
shape of every artifact-producing function in this repo (`read_file` in `helper/data_preparation.py` is
the canonical instance) — reuse it instead of writing a new ad hoc "if exists, read; else, build" branch
per caller, per
[infrastructure.md § Repository design pattern](../../../docs/infrastructure.md#repository-design-pattern)
("if two places implement their own read-or-fetch-and-cache logic for the same kind of data, that's the
signal to introduce one").

## the two shapes in this repo

**Disk-level (persisted artifact, survives process restart)** — `read_file()`
(`helper/data_preparation.py`; `ExtendedDf.read_file()` in `PanderaDFM/ExtendedDf.py` is the same
primitive for pandera-model-bound readers): try `read_with_timeframe()` from a
`{data_frame_type}.{date_range_str}.zip`/`.feather` file; on `FileNotFoundError` or a pandera-validation
miss, call the passed-in `generator(date_range_str)`, which **builds and returns the DataFrame — it must
not write it itself**. `read_file()` persists that return value via `write_data_file()` and hands it back
directly, with no re-read of the file it just wrote. Every OHLCV/peak-valley reader in
`data_processing/ohlcv.py` and `Model/TechnicalAnalysis/PeakValley.py` is a thin wrapper around this one
primitive — new persisted artifacts (a new indicator table, a new label table) should call `read_file()`
with a `data_frame_type` name and a `generator`, not invent a new zip-naming/read/write scheme, and the
generator should end with `return df`, never a `write_data_file()` call of its own.
`read_file()` also carries an in-process LRU memo (bounded, `~32` entries) in front of the disk read, so
repeat calls with identical `(data_frame_type, date_range_str, file_path, skip_rows, n_rows)` skip the
disk round-trip entirely — this matters because the data cache lives on a slow `drvfs`/9p mount in this
repo's dev setup (see [infrastructure.md § environments](../../../docs/infrastructure.md#environments)).
The memo is **skipped** for date ranges `datarange_is_not_cachable()` flags as touching the live/incomplete
present — those must always re-fetch fresh, never memoize.

**In-memory (derived from an object already in RAM, doesn't need to survive the process)** — e.g. the
per-timeframe indicator + label frames built in
`ai_modelling/dataset_generator/training_datasets.py`'s `_cached_training_frames()`. `pd.DataFrame` is
explicitly unhashable (`NDFrame.__hash__ = None`), so it can't be a plain dict key and a
`weakref.WeakKeyDictionary` can't hold it either — hold the cache **on the object itself** via
`df.attrs[some_private_key]` instead. `.attrs` is pandas' own sanctioned slot for attaching arbitrary
metadata to a DataFrame; it's garbage-collected for free with the DataFrame that owns it (no separate
registry to prune, no `id()`-reuse risk), and since you only ever read/write it on the *exact* object
instance you were handed (never a value derived by a transform that may or may not propagate `.attrs`),
propagation quirks don't apply.

## when to reach for which

- Needs to survive across process restarts / be shared across worker processes reading the same range →
  disk-level, `read_file()`.
- Purely derived from a DataFrame already in memory, recomputed identically every time that same object
  is reused (the actual gap found: dataset-generator producer loops — `npz_batch.py`, `ram_batch.py`,
  `stream_loader.py` — load `mt_ohlcv` once per quarter, then call `train_data_of_mt_n_profit()` up to
  100× against that same object; without a cache, indicators and the rolling-window label computation in
  `profit_loss_adder.py` were recomputed from scratch on every one of those 100 calls) → in-memory,
  `.attrs`-keyed.

## rules

- **Key on everything the computation actually depends on**, nothing more/less — e.g.
  `_cached_training_frames()` keys on `(structure_tf, label_tf, forecast_trigger_bars)` because those are
  exactly the prep-phase's free variables; `batch_size`/`dataset_batches`/`verbose` don't affect prep and
  must not be in the key (would fragment the cache for no reason) or left out incorrectly (would return
  stale data for a genuinely different input).
- **Always return a copy from a hit**, never the cached object itself — a DataFrame is mutable, and a
  caller mutating what it thinks is its own private slice must not corrupt the cache for the next reader.
  This is real: `add_long_n_short_profit()` mutates its `ohlc` argument's columns in place, and downstream
  slicing isn't guaranteed copy-on-write pre-pandas-2.x.
- **Never cache a "not yet settled" range.** Disk-level: `datarange_is_not_cachable()` already gates this
  in `read_file()` — a date range touching today keeps changing as new candles close, so memoizing it
  would serve stale data even within one process's lifetime. Any new cache layer over live-updating data
  needs the equivalent bypass.
- **Bound it.** Disk-level uses a fixed-size LRU (`OrderedDict` + evict-oldest) so a long multi-quarter
  training run doesn't grow the memo unboundedly. In-memory `.attrs`-keyed caches self-bound by the
  owning object's lifetime — once the producer loop moves to the next quarter's `mt_ohlcv`, the old one
  (and its `.attrs` cache) is garbage-collected with it.
- **One cache per artifact type, not one per caller.** If you're about to write a second
  "read-if-exists-else-generate" branch for a kind of data this repo already caches somewhere, that's the
  [Repository design pattern](../../../docs/infrastructure.md#repository-design-pattern)'s signal to
  extend the existing primitive instead.
