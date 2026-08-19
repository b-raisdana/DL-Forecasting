# Data pipeline upgrade plan

Open work only. Shipped work (Feather → Parquet migration, `dataset_db` repartitioning + DuckDB batched-window reads, `BasePattern`/`rolling_mean_std` converted to `windowed=True`) isn't re-documented here. Also still deferred, not covered below: catalog/data-quality extensions, ClickHouse (parked, see `docs/infrastructure.md` § ClickHouse), PyArrow dataset API, versioned datasets, Spark/Cassandra/Polars/Zarr.

- [Unify on `windowed=True`, remove the `windowed=False` path (blocked)](#unify-on-windowedtrue-remove-the-windowedfalse-path-blocked)
  - [why the `if windowed:` branch exists](#why-the-if-windowed-branch-exists)
  - [could a generic dispatch just replace every remaining `windowed=False` site?](#could-a-generic-dispatch-just-replace-every-remaining-windowedfalse-site)
  - [blocked sites](#blocked-sites)
  - [how to unblock](#how-to-unblock)
  - [testing](#testing)

## Unify on `windowed=True`, remove the `windowed=False` path (blocked)

### why the `if windowed:` branch exists

`cache_on_disk(dataset, ..., windowed: bool = False)` dispatches its decorated `generator` between two calling conventions (`infrastructure/datastore_engine/disk_cache.py`'s `wrapper()`, the `if windowed:` branch): `read_file()` calls `generator` once for the whole requested `date_range_str`; `read_file_windowed()` decomposes the range into whole calendar windows (`_window_freq()`-sized, "D" for OHLCV/OHLCVA, "M" default elsewhere) and calls `generator` once per window, stitching + trimming the result. `disk_cache_gaps.py`'s gap/overlap discovery already treats every `data_frame_type` as implicitly windowed at `_window_freq()` granularity regardless of the decorator's flag — the on-disk layout is dataset-agnostic; only `cache_on_disk()`'s write-side dispatch still special-cases which generators get windowed calls. The `if windowed:` branch is what's left standing between "every dataset is windowed" and today's two-call-convention reality.

### could a generic dispatch just replace every remaining `windowed=False` site?

Proposal on the table: teach `cache_on_disk()` itself (not each generator) to (1) tell multi-timeframe datasets apart via `caster_model` — feasible, every multi-timeframe schema already inherits a shared `domain.schemas.common.MultiTimeframe` pandera base (`MultiTimeframeOHLCV(OHLCV, MultiTimeframe)`, `MultiTimeframeOHLCVA(OHLCVA, MultiTimeframe)`, …), so `issubclass(caster_model, MultiTimeframe)` is a real, reliable signal, not a guess; (2) build the expected window index for `date_range_str`; (3) find which windows are missing; (4) generate the missing ones and merge with what's already on the datastore; (5) trim to `date_range_str` and return.

Steps 2, 3, and most of 4 are **not new work** — `read_file_windowed()`/`_window_date_range_strs()` and `disk_cache_gaps.find_cache_gaps()` already do exactly this, generically, for any `data_frame_type`. Auto-detecting multi-timeframe-ness (1) via `caster_model` instead of hand-listing `cache_window_freq_overrides` entries is a legitimate, low-risk simplification on top of that. None of this was ever the obstacle.

The obstacle is entirely inside step 4 — "generate results for missing windows" — for any generator whose correctness depends on data *outside* its own window. Two independent axes matter: **bounded vs. unbounded** (is there a fixed, config-derivable amount of extra context that's always enough?), and **backward vs. forward** (does the extra context lie before the window, after it, or both?). Verified against the actual `shift`/`merge_asof` calls (see the table below): `PeakValley` and both `BullBearSide` functions need *unbounded context in both directions* — not backward-only, despite "next"/"previous" naming suggesting otherwise (e.g. `PeakValley.calculate_strength()` takes the min of a left-scan *and* a right-scan distance); `BullBearSidePivot`'s own mechanism (`previous_trend()`) is backward-only, though it consumes both-direction results from the other two.

- **Unbounded backward — a *fixed* buffer can't bound this, but an *adaptive* one can.** `BullBearSidePivot.previous_trend()` walks backward one trend-hop at a time (`trends[movement_end_time == this.movement_start_time]`) until it finds a match — arbitrary-length in raw bars, since a trend can span many windows. A fixed multiplier buffer is never *provably* sufficient (silently wrong, not an error, when too small) — but since this data is historical and DuckDB batching already makes reading many already-cached window tiles cheap, an *adaptively expanding* backward query (read one more window, check whether the chain resolved, repeat until it does or the dataset's own start is reached) genuinely solves this: no bound needs to be guessed in advance, and it self-terminates on success. Same expand-then-trim shape `get_multi_timeframe_ohlcva` already uses for ATR, just with the expansion size decided at run time instead of fixed in config.
- **Unbounded forward — the same adaptive-widening trick works, but it's not free.** `BullBearSide.get_multi_timeframe_candle_trend`'s `merge_asof(direction="forward")` ("next top") and `get_multi_timeframe_bull_bear_side_trends`'s `movement_end_time` expansion (`merge_asof(direction="forward")`) both need data from *after* the window ends. Expanding forward adaptively (keep pulling in the next window, check whether the lookup resolved, repeat) solves this for any range safely in the past — but unlike backward expansion, the needed future windows are often *not yet cached*, so "expand the query" usually means triggering fresh generation of window N+1 (fetch + full derived-dataset computation), not a cheap DuckDB read — resolving window N can cascade into generating several windows past what was actually requested. The one case this never resolves: a window whose forward dependency reaches into the live/incomplete present, where the needed future data simply doesn't exist yet — not new or windowing-specific (today's whole-range approach has the identical edge case at the end of whatever was requested), and `datarange_is_not_cachable()`/`after_under_process_date()` already keep such a window from being cached as final, the same way they already keep live OHLCV data from being cached today.

Net: adaptive (rather than fixed) expansion — read/generate further in whichever direction is needed until the lookup actually resolves — genuinely closes both the unbounded-backward and unbounded-forward cases for historical ranges. What it doesn't remove is the design cost: (a) a window's generator must be allowed to recursively pull in as many neighboring windows as needed — backward via cheap DuckDB reads, forward via full generation — before that window's own file is finalized, replacing today's one-call-per-window model; (b) a window's cache write must be deferred until its dependencies actually resolve (or the request legitimately hits the live boundary, in which case it's never cached as final) — "a written window file is permanent" is a load-bearing assumption throughout the current windowed-caching design (`_seed_window_from_legacy_file()`, DuckDB batching) that no longer holds unconditionally; and (c) tail-of-request cost can exceed what was actually requested (bounded by how far the real answer lives — the same cost the current whole-range approach already pays implicitly, just now visible as extra generation calls instead of one larger single-shot computation).

### blocked sites

| site | dataset | direction | mechanism |
|---|---|---|---|
| `PeakValley.get_multi_timeframe_peaks_n_valleys` | `multi_timeframe_peaks_n_valleys` | both | `calculate_strength()` = min of `left_distance`/`right_distance` — nearest opposite top scanned in both directions (`PeakValley.py:33-37`) |
| `PeakValleyPivots.get_multi_timeframe_major_times_top_pivots` | `multi_timeframe_major_times_top_pivots` | both (inherited) | consumes PeakValley's `strength` column directly |
| `BullBearSide.get_multi_timeframe_bull_bear_side_trends` | `multi_timeframe_bull_bear_side_trends` | both | `movement_end_time` expansion via `merge_asof(direction="forward")`, `movement_start_time` via `merge_asof(direction="backward")` (`BullBearSide.py:153-187`) |
| `BullBearSide.get_multi_timeframe_candle_trend` | `multi_timeframe_candle_trend` | both | nearest prior/next top attached via `merge_asof(direction="backward")` and `merge_asof(direction="forward")` (`BullBearSide.py:306-310`) |
| `BullBearSidePivot.get_multi_timeframe_bull_bear_side_pivots` | `multi_timeframe_bull_bear_side_pivots` | backward (own); both (inherited) | `previous_trend()`: single-hop backward lookup, `trends[movement_end_time == this.movement_start_time]` (`BullBearSide.py:596`) — inputs above are both-direction |

`AtrMovementPivots.py` isn't a live `@cache_on_disk` site (uses the older `ExtendedDf.read_file()` API directly) and is reached only from two commented-out lines in `presentation/ohlcv/main.py` — dead code today, but would face the same blocked verdict if ever revived, since it depends on the same shift/merge_asof patterns.

### how to unblock

Per dataset, decided independently — not a single repo-wide fix:

- Implement adaptive backward/forward window expansion (previous section) inside each generator, with the caching-model changes it requires (deferred writes until dependencies resolve). Real fix, matches the correctness the current whole-range (`windowed=False`) path already gives — but needs the cache-invalidation/deferred-write mechanism built first, which doesn't exist today.
- Or: accept an explicit, documented, accepted-risk fixed lookback/lookahead cap — a heuristic, silently imprecise beyond the cap, needing product sign-off on what "close enough" means for a trading signal. Faster to build, weaker guarantee.

Once every site is resolved (or explicitly exempted), remove the `windowed: bool` parameter from `cache_on_disk()` and the `if windowed:`/`else` dispatch in `wrapper()` (`infrastructure/datastore_engine/disk_cache.py:432-462`), leaving `read_file_windowed()` as the only entry point — `read_file()` itself stays as its per-window primitive (also still used directly by `ExtendedDf.read_file()` and other non-decorator callers).

### testing

- Per resolved site: an equivalence test comparing the new windowed output against today's `windowed=False` output for the same multi-window `date_range_str`, byte-for-byte on values/dtypes/index.
- Once `windowed=False` is fully removed: delete the now-dead `else` branch's coverage, keep only the windowed-path tests.
