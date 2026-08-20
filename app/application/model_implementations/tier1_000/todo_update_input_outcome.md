# Tier-1_000 Model Implementation - required modifications

try to have as few as possible code in the model_implementation folder.
if the code is related to technical-analysis, pre-processing, or any other locations plac them there to be able to share among different models and focous on model architecture review with least code.

## maximized cacheable once per horizon calculations

update all documents the only thing remains for per sample calculation is subtracting atr_normal_close of NOW from all price driven columns to let all training datasets have a common price base (zero at NOW)
every thing else is cacheable, queried to fetch if already cached, windows based filled and calculated just once for each candle. even if this candles participate in thusands of sample.

## Unified datastore

use a new folder under data/dataset_db(unified_no_nan)
only one row for each datetime/timeframe (as unique index)
first we fill OHLCVA

- OHLCVA as the origin of data and source of truce backed up in current folder too.
- every generator queries this; if generator-relevant columns has any NAN value, it tries to fill them.
- do not put future dependent columns in this datastore
- do not put ad-hoc data in this datastore (like is peak or valley)
- fill missing timestamps for all timeframes with NAN values after making sure the range has been queried from broker and broker acknowledge there no candles for the timestamp.
- for higher than 1minutes if a 1minutes cnadle is missing, we wil use the rest to fill the timestamp in the timeframe. just if there is not even one candle the candle in higher timestamp will be considered for NAN filling
- if we have ohlcv for a timestamp we should be able to calculate all other columns in Unified datastore other wise log_err and raise exception.

## adding endpoints for pre-computing required features

endpoint to get NOW time and generate a single complete training set

- option to amend future candles and time of MFE, rer selected extremums to let us visually validate operation on chart.

### read_multi_timeframe_ohlcv

[CACHE: disk/unified_no_nan and disk/multi_timeframe_ohlcv, windowed]
get_multi_timeframe_ohlcv(date_range_str)
│ ├─ get_base_timeframe_ohlcv(date_range_str) [CACHE: disk, windowed]
│ │ ├─ fetch_ohlcv_by_range(broker, date_range_str, base_timeframe)
│ │ │ └─ fetch_ohlcv(broker, symbol, timeframe, start, number_of_ticks, params)
│ │ │ └─ ccxt exchange.fetch_ohlcv(...) [NETWORK I/O]
│ │ └─ build_base_timeframe_ohlcv(raw_ohlcv, date_range_str, base_timeframe)
│ │ └─ pd.DataFrame + cast_and_validate(OHLCV)
│ └─ aggregate_multi_timeframe_ohlcv(ohlcv, date_range_str)
│ └─ pd.Grouper resample to 15min/1h/4h/1D/1W + concat

### read_atr_relative_ohlc

[CACHE: disk/multi_timeframe_unified_no_nan, windowed]
│ │ ├─ uses: multi_timeframe_ohlcv (v is extra but i want to not to cache a lot of files just for volume.)
│ │ ├─ add atr column
│ │ │ └─ [1W] ta.atr(length=32) override default ta.atr(length=255)
│ │ ├─ add_relative_candle_columns(ohlc) [CACHE: per-branch, per-run]
│ │ │ ├─ norm_close = close / atr
│ │ │ ├─ rel_norm_high_close = (high - close) / atr
│ │ │ ├─ rel_norm_close_low = (close - low) / atr
│ │ │ ├─ rel_norm_open_gap = (open - prev_close) / atr
│ │ │ └─ norm_candle_height = (high - low) / atr

### read_extremums

[CACHE: disk/multi_timeframe_extremums, windowed]
logic already implemented and peaks_n_valleys
make sure has a is_final column to know if the extremum level is final?

- if extremum is for 4H just about 4H before it may become a 1D or 1W or 1M extremum later is_final = false.

### read_higher_extrema_distance

**Objective:** For every 15m candle, find the eligible extremum whose price has the minimum absolute distance from the candle close, using the complete epoch OHLC + extrema DataFrames without per-candle queries or uncontrolled Cartesian merges.

1. **Pre-sort extrema once**
   - Sort all extrema globally by `price`.
   - Keep all extremum fields aligned with the sorted order.

2. **Define candle price ranges**
   - For each unresolved 15m candle:
     - `low = close - target_ATR_range`
     - `high = close + target_ATR_range`

   - These define the eligible extremum-price interval.

3. **Vectorized range discovery**
   - Use `np.searchsorted()` against the sorted extremum prices:
     - `left = searchsorted(ext_price, low, "left")`
     - `right = searchsorted(ext_price, high, "right")`

   - No per-candle querying.

4. **Calculate actual candidate workload**
   - `candidate_count = right - left`
   - Never use `candles × total_extrema` as the workload estimate.
   - The true expansion cost is `sum(candidate_count)`.

5. **Price-sort unresolved candles**
   - Sort candles by `close` price.
   - Apply the same ordering to `close`, `low`, `high`, `left`, `right`, and candle indices.
   - This groups similar price ranges and improves batching.

6. **Bound candidate materialization**
   - Set `MAX_CANDIDATE_PAIRS = 100_000_000`.
   - Build batches whose total `candidate_count` does not exceed the threshold.
   - Split by **candidate workload**, not simply by candle count.
   - Never materialize the full epoch Cartesian product.

7. **Materialize one batch**
   - Expand only the extrema ranges belonging to that batch into `(candle_idx, extremum_idx)` pairs.
   - Calculate:
     - `distance = abs(extremum_price - candle_close)`

8. **Select nearest extremum**
   - Perform a fully vectorized minimum/`argmin` per candle.
   - Store the selected extremum and its required metadata.
   - Release the temporary candidate arrays/DataFrame before processing the next batch.

9. **Repeat**
   - Remove resolved candles from the unresolved set.
   - Recalculate ranges/workload for the remaining candles.
   - Repeat steps 3–8 until all candles are resolved or have no eligible extremum.

10. **Preferred optimization**
    - If the only criterion is minimum `abs(extremum_price - close)` and the ATR range is merely an eligibility boundary, **do not materialize ranges at all**.
    - Since extrema are price-sorted, the nearest eligible extremum can only be the immediate extremum below or above `close`.
    - Use vectorized `searchsorted(close)` to obtain both neighbors, compare their distances, and validate the winner against `[low, high]`.
    - This eliminates the merge, candidate expansion, and 100M batching entirely.

**Memory target:**

`O(candles + extrema + MAX_CANDIDATE_PAIRS)`

instead of:

`O(candles × extrema)`

### read_action_rer_mfe_labels

[CACHE: disk/multi_timeframe_unified_no_nan, windowed]

**1. Extremum event streams**

Use the existing vectorized extremum formulas to generate(get_extremums):

- 15m extrema
- 1H extrema
- 4H+ extrema

Treat timeframe as extremum strength, not independent event identity:

- 15m → 15m only
- 1H → 1H + 15m
- 4H+ → 4H + 1H + 15m

Normalize all ≥4H extrema into one `4H+` event stream and process it once.

Each event contains at least:

```text
time, price, direction (peak|valley), strength (15m|1H|4H+)
```

Do not duplicate one physical extremum merely because it qualifies at multiple lower timeframes.

**2. Build six candidate columns for every 15m candle**

For the complete 15m epoch, perform six vectorized loose/as-of future merges between the 15m OHLC DataFrame and the extremum streams:

```text
peak_15m
peak_1h
peak_4h+
valley_15m
valley_1h
valley_4h+
```

Each merge selects the nearest qualifying future extremum.

**3. Candidate validity / normalization**

Vectorized rules:

- Reject extrema after `NOW + 4H`.
- Peak must be strictly above current candle high.
- Valley must be strictly below current candle low.
- If a stronger same-direction extremum supersedes a later weaker extremum, invalidate the weaker candidate.
- If multiple timeframe slots refer to the same physical extremum, preserve the strongest timeframe rather than treating them as separate events.
- If a required slot has no valid extremum inside the 4H horizon, use the favorable extremum of the 4H window as fallback.
- Optionally retain `is_fallback` for validation/debugging.
- Fill the six final candidate positions with remaining valid extrema ordered chronologically.

**4. Best long/short entries**

Calculate the best permissible entries according to the project's existing documented entry rules:

```text
best_long_entry
best_short_entry
```

Keep them vectorized and reusable across candidate extrema.

**5. Candidate OM / MFE / MAE / RER**

Calculate all six candidates vectorized.

For a long candidate:

```text
reward  = candidate_peak_price - best_long_entry
adverse = worst low between entry and candidate extremum
OM      = (reward - trading_fees) / adverse
```

For a short candidate:

```text
reward  = best_short_entry - candidate_valley_price
adverse = worst high between entry and candidate extremum
OM      = (reward - trading_fees) / adverse
```

The adverse extreme must be direction-specific and occur before the candidate extremum.

Do not use `abs()` to hide direction.

Produce vectorized:

```text
OM_1 ... OM_6
MFE_1 ... MFE_6
MAE_1 ... MAE_6
RER_1 ... RER_6
```

**6. Final selection**

For each 15m candle:

1. Apply minimum OM/quality rules.
2. Select the valid candidate with maximum OM according to the documented tie-breaking/weight rules.
3. Return:

```text
action_head
MFE
MAE
RER
selected_extremum_time
selected_extremum_price
selected_extremum_type
selected_extremum_strength
```

**7. Cache boundary**

Cache every thing we do not have disk limitation!
we will batch update/replace columns/values and migrate if base value has been changed.

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

- while (len(concatenated_sample) >= n_samples):
  - n_remained_samples = n_samples - len(concatenated_sample)
  - concat
    - build_dataset(symbol, date_range_str, n_remained_samples):
      - do nto need to read the whole date-range read_multi_timeframe_ohlcv
      - selected_now_times = randomly n_samples in logically reasonable sub-range of date_range
      - for each elected_time:
        - for each timeframe:
          - calculate date range of sample candle in the timeframe
          - read_atr_relative_ohlc(timeframe, timeframe_date_range)
          - read_higher_extrema_distance(timeframe, timeframe_date_range)
      - read_mfe_mae_om_labels(selected_now_times)
      - merge and combine
      - find and drop NANed gaps
      - return
