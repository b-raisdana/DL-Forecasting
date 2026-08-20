# Tier-1_000 Model Implementation - required modifications

try to have as few as possible code in the model_implementation folder.
if the code is related to technical-analysis, pre-processing, or any other locations plac them there to be able to share among different models and focous on model architecture review with least code.

## Unified datastore

use a new folder under data/dataset_db(unified_no_nan)
only one row for each datetime/timeframe (as unique index)
first we fill OHLCVA

- OHLCVA as the origin of data and source of truce backed up in current folder too.
  every generator queries this; if generator-relevant columns has any NAN value, it tries to fill them.
  do not put future dependent columns in this datastore
  do not put ad-hoc data in this datastore (like is peak or valley)

## adding endpoints for pre-computing required features

endpoint to get NOW time and generate a single complete training set
- option to amend future candles and time of MFE, rer selected extremums to let us visually validate operation on chart.

### get_multi_timeframe_ohlcv

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

### get_atr_relative_ohlc

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

### get_extremums

[CACHE: disk/multi_timeframe_extremums, windowed]
logic already implemented and peaks_n_valleys
make sure has a is_final column to know if the extremum level is final?

- if extremum is for 4H just about 4H before it may become a 1D or 1W or 1M extremum later is_final = false.

### get_action_rer_mfe_labels

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

Entry point: `build_dataset(symbol, date_range_str)` in `datafeeder_input3_outcome1.py:155`.

```
build_dataset(symbol, date_range_str)
│
├─ get_multi_timeframe_ohlcv(date_range_str)                          [CACHE: disk, windowed]
│  └─ [wait for generator / share logic with the endpoint to fill gaps]
│
├─ single_timeframe(mt_ohlcv, "5min") → base_ohlc
├─ single_timeframe(mt_ohlcv, "15min") → fifteen_min_ohlc
│
├─ add_mfe_mae_om_labels(base_ohlc, fifteen_min_ohlc)                  [CACHE: per-run]
│  ├─ _atr_255_15min_floor(five_min_index, fifteen_min_ohlc)
│  │  └─ ta.atr(high, low, close, length=255) + merge_asof backward
│  ├─ sliding_window_view(high[1:], HORIZON_BARS) / sliding_window_view(low[1:], HORIZON_BARS)
│  ├─ _direction_excursions(entry_long, high_windows, low_windows, True, atr_floor)   → mfe_long, mae_long
│  ├─ _direction_excursions(entry_short, low_windows, high_windows, False, atr_floor) → mfe_short, mae_short
│  └─ om_long = mfe_long/mae_long, om_short = mfe_short/mae_short
│     ├─ qualifies_long = om_long > 5.0
│     ├─ qualifies_short = om_short > 5.0
│     ├─ action = np.select([long/short/both], default="none")
│     ├─ mfe = chosen direction's mfe
│     └─ rer = clip(mae / (mfe - mae + eps), 0, 1/4)
│
├─ FOR EACH tf_name IN BRANCH_TIMEFRAMES:
│  │
│  ├─ read_atr_relative_ohlc(single_timeframe(mt_ohlcv, tf_name), tf_name) [CACHE: disk/atr_relative_ohlc, windowed]
│  │  ├[the generator:]
│  │  ├─ uses: multi_timeframe_ohlcv (v is extra but i want to not to cache a lot of files just for volume.)
│  │  ├─ add atr column
│  │  │  └─ [1W] ta.atr(length=32) override default ta.atr(length=255)
│  │  ├─ add_relative_candle_columns(ohlc)                             [CACHE: per-branch, per-run]
│  │  │  ├─ rel_close = close / atr
│  │  │  ├─ rel_high_close = (high - close) / atr
│  │  │  ├─ rel_close_low = (close - low) / atr
│  │  │  ├─ open_gap = (open - prev_close) / atr
│  │  │  └─ rel_candle_height = (high - low) / atr
│  ├─ read_log_sma_normal_volume(single_timeframe(mt_ohlcv, tf_name), tf_name) [CACHE: disk/log_sma_noraml_volume, windowed]
│  │  ├[the generator:]
│  │  └─ add_log_sma_volume_feature_column(ohlc, length=256 or 20)
│  │     └─ ta.sma(volume, length) → log_volume_sma_ratio
│  │
│  ├─ _last_closed_position(anchor_index, branch_index, tf_minutes, 5.0)
│  │  └─ shift = (branch_tf - 5min); merge_asof(shifted_anchors, branch_positions, backward)
│  │
│  ├─ build_branch_extremum(features_by_tf[tf_name])                  [CACHE: per-branch, per-run]
│  │  └─ compute_true_extremum(ohlc)                                   [O(n) monotonic stack]
│  │     ├─ _reach_minutes(high, times_ns, direction="right", sense="peak")
│  │     ├─ _reach_minutes(high, times_ns, direction="left", sense="peak")
│  │     ├─ _reach_minutes(low, times_ns, direction="right", sense="valley")
│  │     ├─ _reach_minutes(low, times_ns, direction="left", sense="valley")
│  │     ├─ true_peak_reach = min(left_peak, right_peak)
│  │     ├─ true_valley_reach = min(left_valley, right_valley)
│  │     ├─ true_extremum_tf_minutes = floor_to_tf_ladder(max(peak, valley))
│  │     └─ extremum_sign = +1 (peak wins ties), -1, or 0
│  │
│  ├─ _source_for(plus2_tf) → plus2_source (BranchExtremum)
│  ├─ _source_for(plus3_tf) → plus3_source (BranchExtremum)
│  │
│  ├─ align_source_atr(feat_df.index, plus2_source, plus2_native_minutes)   [CACHE: per-branch, per-source]
│  │  └─ shift = plus2_tf_minutes; merge_asof(shifted, source_positions, backward)
│  ├─ align_source_atr(feat_df.index, plus3_source, plus3_native_minutes)   [CACHE: per-branch, per-source]
│  │
│  ├─ compute_extremum_weight(branch, gather_idx, anchor_time_ns, tf_minutes)
│  │  └─ weight = sign * log1p(observed/tf_minutes) * min(1, age/observed)
│  │
│  └─ compute_higher_extremum_distance(
│        windowed_close, windowed_time_ns, anchor_time_ns,
│        plus2_source, plus2_tf_minutes, plus2_atr_windowed,
│        plus3_source, plus3_tf_minutes, plus3_atr_windowed)
│     ├─ _threshold_filtered_pool(plus2_source, sign=+1, plus2_tf_minutes)  → peak_pool
│     ├─ _threshold_filtered_pool(plus2_source, sign=-1, plus2_tf_minutes)  → valley_pool
│     ├─ _threshold_filtered_pool(plus3_source, sign=+1, plus3_tf_minutes)  → peak_pool
│     ├─ _threshold_filtered_pool(plus3_source, sign=-1, plus3_tf_minutes)  → valley_pool
│     └─ FOR EACH anchor a:
│        ├─ _nearest_and_last(close[a], time[a], peak_pool, cutoff_ns)    → price_peak, time_peak
│        ├─ _nearest_and_last(close[a], time[a], valley_pool, cutoff_ns)  → price_valley, time_valley
│        ├─ price_normal = price_diff / atr_aligned
│        └─ time_normal = log1p(elapsed_minutes / target_tf_minutes)
│
├─ auxiliary_features = concat([last_candle_by_tf[0], ..., last_candle_by_tf[5]], axis=1)
│
├─ NaN scrub:
│  └─ clean = ~np.isnan(windows).any(axis=(1,2)) & ~np.isnan(aux).any(axis=1)
│
└─ DatasetBundle(
     branch_windows={tf: windows[clean]},
     auxiliary_features=aux[clean],
     mfe=labels["mfe"][clean],
     rer=labels["rer"][clean],
     action=labels[["action_long","action_short","action_none"]][clean],
     anchor_index=anchor_index[clean])
```

### Cacheable Results

| Method / Step               | Scope                                     | Cache Mechanism                                                | Notes                                                                                                                                                                          |
| --------------------------- | ----------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `get_base_timeframe_ohlcv`  | `(date_range_str, base_timeframe)`        | `@cache_on_disk(OHLCV_DATASET, windowed=True)`                 | Disk-level Parquet/ZSTD; LRU in-process read cache (32 entries). Network fetch only on cold miss.                                                                              |
| `get_multi_timeframe_ohlcv` | `date_range_str`                          | `@cache_on_disk(MULTI_TIMEFRAME_OHLCV_DATASET, windowed=True)` | Depends on `get_base_timeframe_ohlcv`; post-read hook `cache_times` populates `GLOBAL_CACHE["valid_times_{tf}"]`.                                                              |
| `build_branch_extremum`     | Per branch (6× per `build_dataset`)       | In-memory, recomputed each `build_dataset()` call              | Step A (`compute_true_extremum`) is O(n) monotonic-stack. Reused for that branch's own `extremum_weight` and as source pool for lower-TF branches' `higher_extremum_distance`. |
| `align_source_atr`          | Per (encoding_branch, source_branch) pair | In-memory, computed once per branch over full series           | Windowed with the same `gather_idx` as all other per-branch channels; not recomputed per anchor.                                                                               |
| `_atr_255_15min_floor`      | Per `add_mfe_mae_om_labels` call          | In-memory                                                      | Forward-filled ATR(255) from completed 15min candles onto 5min anchor index.                                                                                                   |
| `_branch_features`          | Per branch (6× per `build_dataset`)       | In-memory, recomputed each call                                | Adds ATR + 5 static ratio columns + volume features. Side-effect: populates `ohlc['atr']` for downstream `build_branch_extremum`.                                              |

**Cache key insight:** The two disk-cached functions (`get_base_timeframe_ohlcv`, `get_multi_timeframe_ohlcv`) are the only results that survive across process restarts. All other expensive transforms (`build_branch_extremum`, `align_source_atr`, `_atr_255_15min_floor`) are recomputed each `build_dataset()` invocation even though their inputs are derived from the same cached `mt_ohlcv`.

## Outcome Features

Set 1 produces three targets (`mfe_mae_om_labels.py`):

- `action`: 3-class one-hot (long/short/none). A direction qualifies if its `om = mfe / mae > OM_GATE (5.0)`. If both qualify, the higher OM wins. Neither qualifies → none (`mfe_mae_om_labels.py:107-114`).
- `mfe`: Maximum favorable excursion from entry (best high for long, best low for short) within `HORIZON_BARS = 48` (240 min) (`mfe_mae_om_labels.py:62-65`). Entry price is the first bar of the horizon (`mfe_mae_om_labels.py:100-101`).
- `rer`: `mae / (mfe - mae)`, clipped to `(0, 1/4)` (`mfe_mae_om_labels.py:120`). `mae` is the worst adverse excursion before the best bar, floored at `ATR(255, 15min)` (`mfe_mae_om_labels.py:76`).

Loss: categorical crossentropy for action; Gaussian NLL (`gaussian_nll_loss` in `model.py:397`) for mfe/rer mean+std pairs trained jointly.

## Architecture Backbone

### Six parallel branches

`TimeframeBranchEncoder` per timeframe (`model.py:147`): `Dense(channels)` → `ModernTCNBlock × depth` → `LSTM × layers` (all `return_sequences=True`).

### ModernTCN

`ModernTCNBlock` (`model.py:124`): pre-norm → causal depthwise conv (kernel=7, left-padded) → dropout → pre-norm → ConvFFN (4x expansion, GELU) → residual. Depth=8, channels=192.

### LSTM

4 unidirectional layers, hidden=384, `return_sequences=True` (`model.py:167`).

### Perceiver fusion

`PerceiverFusion` (`model.py:220`): learnable timeframe identity embeddings (shape `(1,1,hidden_units)`) broadcast-added to each branch's sequence; all 6 sequences concatenated; `Perceiver_latent_tokens=96` learnable latents cross-attend to the concatenated sequence via `MultiHeadAttention` (heads=8, key_dim=768/8) for 8 layers, each with pre-norm residual + FFN. No self-attention among latents — that role is filled by downstream GQA.

### GQA

`GQAEncoder` (`model.py:289`): 8 layers of hand-rolled GroupedQueryAttention (heads=8, kv_heads=2, d_model=768=`Perceiver_latent_dim`) + FFN, pre-norm residual. Hand-rolled because Keras `MultiHeadAttention` lacks `kv_heads < heads` (`model.py:182`).

### Pooling

`last_token` by default (`model.py:321`): takes the final latent token.

### Prediction head

`PredictionHead` (`model.py:331`): MLP trunk (depth=4, width=512, GELU, dropout=0.1) on `Concatenate([pooled, auxiliary_features])`. Three output heads:

- `action`: Dense(3, softmax, float32)
- `mfe_params`: Dense(2, softplus, float32) → [mean, std]
- `rer_params`: Dense(2, [sigmoid, softplus], float32) → [mean, std]

### Training

AdamW (lr=3e-4, weight_decay=1e-4, clipnorm=1.0), cosine scheduler, mixed_float16, batch_size=8 (reduced from spec's 128 due to 8GB VRAM OOM profiling — `model.py:93`).

## Resource Usage (Measured)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 5560 MB VRAM.

| Metric                                             | Value                                                           |
| -------------------------------------------------- | --------------------------------------------------------------- |
| Model parameters                                   | 143.8M                                                          |
| batch_size=8 GPU peak (forward+backward)           | 4521.6 MB (81.3% of VRAM)                                       |
| batch_size=8 stability                             | OOM on 2nd gradient step — BFC allocator fragmentation near cap |
| batch_size=4 GPU peak                              | 4727.9 MB                                                       |
| batch_size=4 stability                             | 5+ steps stable                                                 |
| Steady-state step time (batch=4, after XLA warmup) | ~6–10s                                                          |
| Throughput (batch=4)                               | ~0.3–0.7 samples/sec                                            |

The spec's own transient_activation_memory estimate identified the Perceiver's cross-attention over the concatenated 1536-token multi-tf sequence as the dominant scalable term. The measured peak confirms this: activation memory, not parameter count, binds VRAM on this hardware.

## Hyperparameter Optimization Judgment

Optimization is required before broad search.

- batch_size=8 is **unstable** on this hardware (OOM on step 2). This blocks normal training. The highest-leverage fix is gradient checkpointing on the Perceiver cross-attention layers, trading ~20-30% compute for ~500-800 MB activation savings and restoring batch_size=8 stability.
- Perceiver_latent_tokens (96) and Perceiver_cross_attention_layers (8) are the next best knobs. Reducing either cuts the dominant activation term directly. A token reduction to 48 or layer reduction to 4 should fit comfortably under 5560 MB at batch_size=8.
- Backbone widths (ModernTCN_channels=192, LSTM_hidden_units=384) are the main parameter drivers. If activation memory remains tight after Perceiver changes, reducing these toward base (128/256) saves ~30-40% parameters with modest accuracy trade-off.
- GQA heads/layers (8/8) and MLP width/depth (512/4) are lower priority — GQA operates on already-compressed 96 tokens, so its footprint is small; MLP width is held constant across candidates by design.
- Learning rate, weight decay, scheduler, and optimizer are spec-resolved at reasonable defaults. They cannot be judged without actual training loss curves; leave them fixed until the model trains stably.

Recommended sequence:

1. Add gradient checkpointing to Perceiver cross-attention.
2. If still unstable, reduce Perceiver_latent_tokens or Perceiver_cross_attention_layers.
3. If still unstable, reduce backbone widths.
4. Only after stable multi-step training at batch_size≥8, run a focused Optuna search over the remaining searchable parameters.
