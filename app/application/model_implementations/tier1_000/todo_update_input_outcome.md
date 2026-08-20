# Tier-1_000 Model Implementation - required modifications

Keep model_implementation folder minimal. Place technical-analysis, pre-processing, and shared logic elsewhere so model architecture review stays focused.

## maximized cacheable once per horizon calculations

- Per-sample calc reduced to subtracting `atr_normal_close` of NOW from price-driven columns, giving all training sets a common price base (zero at NOW).
- Everything else is cacheable: queried, windowed-filled, calculated once per candle regardless of sample count.

## Unified datastore

- New folder: `data/dataset_db(unified_no_nan)`
- One row per datetime/timeframe (unique index)
- Fill OHLCVA first as origin of truth, backed up in current folder
- Generators query this store; if generator-relevant columns are NAN, generator fills them
- No future-dependent columns, no ad-hoc columns (e.g. `is_peak`)
- Fill missing timestamps with NAN only after broker confirms no candles exist for that timestamp
- For >1min TF: if 1m candle missing, use remaining 1m candles to fill the higher TF timestamp; only if zero 1m candles exist, mark higher TF as NAN
- If OHLCV present for a timestamp, all other Unified datastore columns must be calculable; otherwise log error and raise

## Endpoints for pre-computing required features

- Endpoint returns NOW time and generates one complete training set
- Option to amend future candles and MFE/RER selected extremums for visual chart validation

### read_multi_timeframe_ohlcv

[CACHE: disk/unified_no_nan and disk/multi_timeframe_ohlcv, windowed]

- `get_multi_timeframe_ohlcv(date_range_str)`
  - `get_base_timeframe_ohlcv(date_range_str)` [CACHE: disk, windowed]
    - `fetch_ohlcv_by_range(broker, date_range_str, base_timeframe)`
      - `fetch_ohlcv(broker, symbol, timeframe, start, number_of_ticks, params)`
      - `ccxt exchange.fetch_ohlcv(...)` [NETWORK I/O]
    - `build_base_timeframe_ohlcv(raw_ohlcv, date_range_str, base_timeframe)`
      - `pd.DataFrame + cast_and_validate(OHLCV)`
  - `aggregate_multi_timeframe_ohlcv(ohlcv, date_range_str)`
    - `pd.Grouper` resample to 15min/1h/4h/1D/1W + concat

### read_atr_relative_ohlc

[CACHE: disk/multi_timeframe_unified_no_nan, windowed]

- Uses: multi_timeframe_ohlcv (volume excluded from cache to reduce file count)
- Add ATR column
  - [1W] `ta.atr(length=32)` overrides default `ta.atr(length=255)`
- `add_relative_candle_columns(ohlc)` [CACHE: per-branch, per-run]
  - `norm_close = close / atr`
  - `rel_norm_high_close = (high - close) / atr`
  - `rel_norm_close_low = (close - low) / atr`
  - `rel_norm_open_gap = (open - prev_close) / atr`
  - `norm_candle_height = (high - low) / atr`

### read_extremums

[CACHE: disk/multi_timeframe_extremums, windowed]

- Logic already implemented (`peaks_n_valleys`)
- Add `is_final` column
  - If extremum is 4H, ~4H before it may become 1D/1W/1M extremum later → `is_final = false`
  - Note: strength only increases over time; once `is_final` becomes true, underlying calculations do not need recomputation.

### read_higher_extrema_distance

**Objective:** For every 15m candle, find the eligible extremum with minimum absolute price distance from candle close, without per-candle queries or Cartesian merges.

1. **Pre-sort extrema once** - Sort all extrema by `price`; keep fields aligned.
2. **Define candle price ranges** - `low = close - target_ATR_range`, `high = close + target_ATR_range`.
3. **Vectorized range discovery** - `np.searchsorted(ext_price, low, "left")` and `np.searchsorted(ext_price, high, "right")`.
4. **Calculate candidate workload** - `candidate_count = right - left`; true cost is `sum(candidate_count)`, not `candles × extrema`.
5. **Price-sort unresolved candles** - Sort by `close`; apply same ordering to `low`, `high`, `left`, `right`, and indices.
6. **Bound candidate materialization** - `MAX_CANDIDATE_PAIRS = 100_000_000`; batch by candidate workload, not candle count.
7. **Materialize one batch** - Expand extrema ranges for that batch into `(candle_idx, extremum_idx)` pairs; compute `distance = abs(extremum_price - candle_close)`.
8. **Select nearest extremum** - Vectorized `argmin` per candle; release temp arrays before next batch.
9. **Repeat** - Remove resolved candles, recalculate ranges/workload, repeat 3-8.

**Memory target:** `O(candles + extrema + MAX_CANDIDATE_PAIRS)` instead of `O(candles × extrema)`.

### read_action_rer_mfe_labels

[CACHE: disk/multi_timeframe_unified_no_nan, windowed]

**1. Extremum event streams**

Use existing vectorized extremum engine to generate:

- 15m extrema
- 1H extrema
- 4H+ extrema (normalized into one stream)

Treat timeframe as extremum strength, not independent identity:

- 15m → 15m only
- 1H → 1H + 15m
- 4H+ → 4H + 1H + 15m

Each event: `time, price, direction (peak|valley), strength (15m|1H|4H+)`

Do not duplicate one physical extremum across lower timeframes.

**2. Build six candidate columns for every 15m candle**

For complete 15m epoch, perform six vectorized loose/as-of future merges between 15m OHLC and extremum streams:

- `peak_15m`, `peak_1h`, `peak_4h+`
- `valley_15m`, `valley_1h`, `valley_4h+`

Each merge selects nearest qualifying future extremum.

**3. Candidate validity / normalization**

Vectorized rules:

- Reject extrema after `NOW + 4H`
- Peak strictly above current candle high
- Valley strictly below current candle low
- Stronger same-direction extremum supersedes later weaker one → invalidate weaker
- Same physical extremum across multiple TF slots → preserve strongest timeframe
- If required slot has no valid extremum within 4H horizon → use favorable extremum of 4H window as fallback
- Optionally retain `is_fallback` for validation/debugging
- Fill six final candidate positions with remaining valid extrema ordered chronologically

**4. Best long/short entries**

Vectorized:

- `best_long_entry`
- `best_short_entry`

**5. Candidate OM / MFE / MAE / RER**

All six candidates calculated vectorized.

Long candidate:

- `reward = candidate_peak_price - best_long_entry`
- `adverse = worst low between entry and candidate extremum`
- `OM = (reward - trading_fees) / adverse`

Short candidate:

- `reward = best_short_entry - candidate_valley_price`
- `adverse = worst high between entry and candidate extremum`
- `OM = (reward - trading_fees) / adverse`

Adverse extreme is direction-specific and occurs before candidate extremum. No `abs()` to hide direction.

Output:

- `OM_1 ... OM_6`, `MFE_1 ... MFE_6`, `MAE_1 ... MAE_6`, `RER_1 ... RER_6`

**6. Final selection**

Per 15m candle:

1. Apply minimum OM/quality rules
2. Select valid candidate with maximum OM per documented tie-breaking/weight rules
3. Return:

- `action_head`
- `MFE`
- `MAE`
- `RER`
- `selected_extremum_time`
- `selected_extremum_price`
- `selected_extremum_type`
- `selected_extremum_strength`

**7. Cache boundary**

Cache everything within disk limits. Batch update/replace columns/values and migrate if base value changes.

**Target pipeline**

```text
OHLCV
  ↓
existing vectorized extremum engine
  ├── 15m extrema
  ├── 1H extrema
  └── ≥4H extrema → one 4H+ stream
          ↓
six vectorized loose future merges
          ↓
six candidate extrema per 15m candle
          ↓
validity + deduplication + 4H fallback
          ↓
chronological candidate normalization
          ↓
best long/short entries
          ↓
6× vectorized MFE / MAE / OM / RER
          ↓
vectorized best-candidate selection
```

## End-to-End Data Flow

Entry point: `build_dataset(symbol, date_range_str, n_samples)` in `datafeeder_input3_outcome1.py:155`.

```text
while (len(concatenated_sample) >= n_samples):
  n_remained_samples = n_samples - len(concatenated_sample)
  concatenated_sample = concat
    build_dataset(symbol, date_range_str, n_remained_samples):
      - read_multi_timeframe_ohlcv (no need to read whole date-range)
      - selected_now_times = randomly n_samples in logically reasonable sub-range of date_range
      - for each elected_time:
        - for each timeframe:
          - calculate date range of sample candle in timeframe
          - read_atr_relative_ohlc(timeframe, timeframe_date_range)
          - read_higher_extrema_distance(timeframe, timeframe_date_range)
      - read_mfe_mae_om_labels(selected_now_times)
      - merge and combine
      - find and drop NANed gaps
      - return
```

## Explanations

Resolved elsewhere in the project:

- NOW time / deterministic selection: 5min NOW candle with 240min future horizon, deterministic entry rules. Deep dive: `docs/ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md`
- Trading fees: single overhead rate `FEE_RATE = 0.001`; safe-side multiplier in config. Deep dive: `app/application/dataset_generation/mfe_mae_om_labels.py:18`; `app/config/Config.py:148`
- `is_final` population: `False` when extremum strength is limited by boundary and insufficient data exists to know if it can acquire higher strength; set `True` when `bull_bear_side` is populated. Deep dive: `app/archive_not_used_trash/domain/price_action/BullBearSide.py:43,85`
- Empty candidate handling: missing valid extremum within 4H horizon falls back to favorable extremum of the 4H window. Deep dive: `read_action_rer_mfe_labels` step 3
- Data validation schemas: PanderaDFM schemas in `domain/schemas/`, input validation at top and output validation after assembly. Deep dive: `.kilo/skills/pandera-dataframe-validation/SKILL.md`
- Cache window/overlap/invalidation: window freq from config; overlap resolved by sample-uniqueness weighting; invalidation by choosing a new versioned dataset name. Deep dive: `app/config/Config.py:108-116`; `docs/ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md:197-204`; `docs/ML_Forecasting_System_Design/02-Appendix-Feature-Label-Preparation.md:26`
- Concurrency: I/O-bound uses asyncio/thread pool; CPU-bound uses process pool after vectorization. Deep dive: `.kilo/skills/project-decisions/SKILL.md:66-77`
- Backward migration: handle added/removed/renamed columns, type changes, and incompatible schemas; define merge vs new-version criteria. Deep dive: `app/application/datastore_engine/planing.md:45`; `docs/todos/data_pipeline_upgrade_plan.md:7`
- Broker gap-acknowledgment: two direct broker asks per gap; still empty after both → confirmed-unavailable, marked with empty cache file, logged, not raised. Deep dive: `app/application/market_data/fetch_market_data.py:48-74`
- Reproducibility seed: deterministic RNG seeded from symbol hash. Deep dive: `app/presentation/dataset_generation/now_review_notebook.py:92`
- `n_remained_samples` update logic: decremented after each `build_dataset` concatenation; stale-after-selection problem noted below. Deep dive: End-to-End Data Flow
- `action_head` mapping: 3-class softmax (long/short/none) with OM gate and tie-breaking by higher OM. Deep dive: `docs/ML_Forecasting_System_Design/designsets/PROMPT.md:328-333`; `app/application/model_implementations/tier1_000/README.md:160-164`; `app/application/model_implementations/tier1_000/model.py:352-363`

- **"no future dependent columns" rule does not conflict with MFE/RER/MAE labels** — "Future dependent" means a feature may lose credibility by receiving data feed from the real future. MFE/RER/MAE labels are calculated based on future relative to the selected NOW-moment for sampling and will not change or lose their credibility in the future.

- **1m→higher TF gap-fill is safe with partial data** — We need at least one candle inside any timeframe to build it. Missing some fractions is not a problem.

- **`selected_now_times` reproducibility** — Accept a seed number as input, default to `42`.

- **"2tf higher extremum" definition** — A 2tf higher extremum means extrema located two timeframes higher and above. Example: for a 5m candle, 2tf higher extrema are the extrema in 1H and higher (1H, 4H, …), i.e. skipping the immediate 15m higher timeframe.

- **"time distance extremum finding" definition** — This mode finds extrema that have already been passed by the NOW candle (timestamp strictly before the NOW candle) and selects the one with the least time distance to the NOW candle. It is distinct from price-distance finding: the ranking criterion is elapsed time, not absolute price gap.

- `is_final` transitioning from false to true : If an extremum starts as `is_final = false` and later transitions to true, its strength can only increase and never decrease, so underlying price-distance and candidate-selection calculations remain valid without recomputation; record this invariant explicitly to prevent future confusion.

- When the broker responds without errors but returns no data (down-time, no-trade), the system must treat this as confirmed-unavailable rather than retrying indefinitely; see `app/application/market_data/fetch_market_data.py:48-74` for the existing two-ask pattern and empty-cache-file convention.

- `build_dataset` recursively calls itself, Add a `max_retries` guard (e.g., 3 attempts); on exhaustion, raise an error and log via `log_e` instead of recursing indefinitely.

- `n_remained_samples`: After randomly selecting `selected_now_times`, recalculate the remaining sample budget from the actual number of valid times returned, not the original `n_remained_samples` estimate.

- Use `get_oldest_available_timestamp(app_config.under_process_exchange.lower(), symbol)` from `app/infrastructure/market_data_fetch/ccxt_client.py:153-209` to bound backward fills to the exchange's actual data availability.
