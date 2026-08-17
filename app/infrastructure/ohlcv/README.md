# disk_cache windowing

Design notes for the calendar-windowed caching added to `disk_cache.py`
(`cache_on_disk(..., windowed=True)` / `read_file_windowed()`), the legacy-file reuse it does on a
miss, and the disk-generation-rate monitor. `disk_cache.py`'s module docstring points here.

## Why

Every `get_X(date_range_str)` entry point (`get_base_timeframe_ohlcv`, `get_multi_timeframe_ohlcv`,
`get_multi_timeframe_ohlcva`) is called with an arbitrary, caller-chosen range. Before this change,
each of the three hand-rolled its own day-by-day loop (`read_daily_*` + a plain `while` over dates)
*and* was independently `@cache_on_disk`-decorated at the outer, arbitrary-range level — so every
distinct requested range wrote its own full-range file, on top of the daily tiles underneath it. As
`processing_date_range`'s end date advanced over time, each new run's full-range file re-duplicated
almost everything the previous one already had. On the live cache this had grown to **~8.1GB**, of
which **~4.94GB was exact, provable duplication** (see § cleanup) — BTCUSDT alone accounted for 96%
of it, some full-range files spanning 2.5+ years and entirely containing the daily tiles beneath them.

## Windowing

`cache_on_disk(file_name_prefix, windowed=True)` decorates a **per-window** generator: it still
receives a `date_range_str`, but `read_file_windowed()` guarantees that range is always one whole,
calendar-aligned window before the generator ever sees it. The decorated function itself still
accepts an arbitrary range — `read_file_windowed()` is what decomposes it:

1. `_window_date_range_strs()` splits the requested range into every whole calendar window it
   overlaps, sized by `_window_freq(file_name_prefix)` (`app_config.cache_window_freq_overrides`,
   falling back to `app_config.default_cache_window_freq`, default `"M"` — calendar month). Windows
   are **never clipped** to the caller's own start/end — a request for `14:30 → 05:00` two days later
   still generates/reads two whole-day files, so the same window file is reused verbatim across every
   caller that happens to overlap it, instead of each caller writing its own fragment.
2. Each window is fetched/generated through the ordinary single-file `read_file()` — memoization,
   legacy CSV-zip migration, and `caster_model` validation are all shared with non-windowed callers,
   not reimplemented.
3. A window whose entire span is still in the future contributes an `empty_df(caster_model)` instead
   of being fetched — mirrors the old `read_daily_*` "day hasn't happened yet" guard.
4. The per-window results are concatenated, sorted, and `trim_to_date_range()`'d back down to exactly
   the caller's original range.

Current per-prefix windows (`app_config.cache_window_freq_overrides`): `ohlcv`,
`multi_timeframe_ohlcv`, `multi_timeframe_ohlcva` are all `"D"` (calendar day) — this preserves every
existing daily tile on disk exactly as-is; nothing needed migrating. New prefixes default to `"M"`
unless given their own override.

Nesting is intentional and cheap: `get_multi_timeframe_ohlcva`'s generator body calls
`get_multi_timeframe_ohlcv()` (also windowed) for ATR lookback context — when both are configured to
the same window size, a single-window call degrades to a one-window no-op split; when the outer
caller passes a wider range, the inner call fetches the extra lookback days from their own (likely
already-cached) window tiles.

## Backward compatibility

A window can be missing its own canonical file while still being fully reconstructable from an
existing (typically pre-windowing) file that happens to cover it — e.g. one of the old giant
full-range files described above. `_seed_window_from_legacy_file()` runs before each cachable
window's `read_file()` call: if the window's canonical file isn't there yet, it looks for the
*smallest* on-disk file for that `data_frame_type` whose own range fully contains the window
(`_find_covering_file()`), reads it via the existing `read_with_timeframe()`, trims it down to the
window, and writes it out as the window's canonical file — so the `read_file()` call right after
finds it and never invokes `generator()`. It never touches the legacy file itself; that's a separate,
explicit step.

## Cleanup

`cleanup_redundant_cache_files(data_frame_type, file_path=None, window_freq=None, dry_run=False)`
deletes every on-disk file for `data_frame_type` whose whole date-range span is *exactly*
reconstructable from **other** files for the same prefix at window-freq granularity: it decomposes
each file's own span into the same whole-calendar-window periods `read_file_windowed()` would use,
and only deletes the file if every one of those periods is backed by a **different** file already on
disk. A file can never be considered redundant on the strength of its own span alone — a genuine
single-window tile's only "period" is itself, so it's never a candidate. This is the same
"other-witness-required" check whether the source file was day-boundary-aligned or not (it also
catches partial-start ranges like `23:44 → next day 23:59`, which fully overlap the same whole-day
tiles), so no separate handling was needed for any particular file-naming quirk found on disk.

Run against the live cache (2026-08-17, all 6 symbols, `ohlcv`/`multi_timeframe_ohlcv`/
`multi_timeframe_ohlcva`): **6,899 files removed, 4.937GB reclaimed** (Kucoin cache: 8.1GB → 3.5GB).
BTCUSDT alone: 6,115 files / 4.38GB. Safe to re-run any time — a no-op once nothing is redundant.

## Monitoring

`write_data_file()` reports every write's byte count to `_record_cache_generation(file_name_prefix,
bytes)`. Per prefix, bytes accumulate in `_cache_generation_state` until at least
`app_config.cache_generation_monitor_interval_minutes` (default 30) have elapsed since the last
evaluation — so a prefix's rate is checked, and possibly warned on, **at most once per interval**,
regardless of how many files it writes in between. On evaluation, accumulated bytes are extrapolated
to a 24h rate; if that exceeds `app_config.cache_generation_warn_bytes_per_day` (default 1GB), a
`log_w()` warning fires naming the prefix and the measured rate. State is in-process only (a plain
dict, same tradeoff as the `_read_file_cache` LRU above it) — it resets on restart, so it catches
runaway generation *within* a run, not across separate script/notebook invocations.
