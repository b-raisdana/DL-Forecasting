# TODO — input data / channels preparation

Closing the gap between [input-features.md](../input-features.md) (the candle feature schema + feature
screening spec) and what actually feeds the model today. Current focus topic — see
[master-todo.md](master-todo.md).

- [TODO — input data / channels preparation](#todo--input-data--channels-preparation)
  - [todo](#todo)
  - [done (reference)](#done-reference)
  - [appendix: current implementation status](#appendix-current-implementation-status)
    - [audit result (step 1)](#audit-result-step-1)
    - [what actually feeds the model today](#what-actually-feeds-the-model-today)
    - [peak/valley detection exists, but isn't wired in](#peakvalley-detection-exists-but-isnt-wired-in)
    - [peak/valley reuse decision (step 4)](#peakvalley-reuse-decision-step-4)
    - [volume data-source confirmation (step 3)](#volume-data-source-confirmation-step-3)
    - [data-quality / CCXT feed gaps](#data-quality--ccxt-feed-gaps)

## todo

Ordered so each step is small and independently testable. Steps marked **(decision)** need a one-line
confirmation before implementing since they affect input shape (and therefore every downstream model
candidate in [model-architecture.md](model-architecture.md)); everything else is a direct fix against
the already-written spec in [input-features.md](../input-features.md). Steps 1-4 (audit, relative-HLC,
volume/ATR, peak/valley reuse decision) are done — see [done (reference)](#done-reference).

1. Implement the **no-lookahead cap** on peak/valley confirmation: "is a top? which tf" must be capped
   by elapsed time to the anchor candle (a candle 4mo before anchor can confirm 4M peak only if it
   stayed max to anchor; 1wk before anchor caps at 1W) — per
   [input-features.md § candle feature schema](../input-features.md#candle-feature-schema). This is the
   part of the schema most likely to silently leak future information if implemented naively; write the
   causality test in step 2 against this specific field first. Builds on the peak/valley reuse decision
   in [done (reference)](#done-reference) (raw extrema reimplemented, confirmation logic still to write
   here).
2. **Add the no-lookahead regression test** for this pipeline stage: perturbing FUTURE-slice data must
   never change a computed peak/valley/top-distance feature at or before the anchor candle. Companion to
   the label-side version of this test in
   [training-data-labels.md](training-data-labels.md#todo) — same discipline, different pipeline stage.
3. Implement the **multi-tf top-distance fields** (2 and 3 higher tfs' top time/price distances: signed
   time offset, top volume strength, abs-time, natural/normal price distance, abs-price) per the
   tf-ordered-list (5min/15min/1H/4H/1D/1W as actual input series; 1M/4M/1Y confirmation-only). Depends
   on steps 1-2 landing first.
4. Implement **nearest-top/nearest-valley distance** fields (`nearest_top_distance`,
   `nearest_top_tf`, nearest top volume strength, and the valley equivalents) — the
   `min(distance/ATR(peak's tf))` aggregation across all tfs, per the schema.
5. **(decision) `distance from anchor candle (minutes)`** and **timeframe-in-minutes** fields — per the
   schema, `timeframe-in-minutes` is a config option gated on architecture (included for flat/shared-encoder
   archs, excluded for per-tf-branch archs). Confirm which architecture branch
   [model-architecture.md](model-architecture.md) is building toward before wiring this field in or out.
   a. [next-step] according to the scope of this doc these features should be avialble. the decision is responciblity of optimization.
6. Once steps 1-5 land, **run the ablation pass**: permutation importance against the new full set,
   against the priority-ordered suspicion queue already named in the spec (candle-height/ATR flagged as
   the most likely redundant-but-cheap field). Needs a trained model to permute against — sequence
   this after a first Stage-1 candidate exists in
   [model-architecture.md](model-architecture.md), not before.
7. **Run the candidate-feature MI/GBM screen** against the candidate pool (OBV — already implemented,
   confirm it's actually screened not just present; ICHIMUKU — same; MACD, ADX, VWAP, volatility-regime
   trio, session/time cyclical, structural counts — all unimplemented). Use
   `sklearn.mutual_info_classif/regression` per
   [input-features.md § candidate-feature screening](../input-features.md#candidate-feature-screening--method);
   small LightGBM/XGBoost only if MI is ambiguous.
8. **Reconcile the candidate pool with what's already in `classic_indicators.py`.** Code currently
   computes `cci`/`rsi`/`mfi`/`bbands` — none of which appear in
   [input-features.md § candidate feature pool](../input-features.md#candidate-feature-pool) (which
   instead names MACD and says "drop RSI as redundant with it"). Either the pool needs updating to
   reflect what's already screened-and-kept, or these existing indicators need to go through the same
   MI screen as new candidates before being trusted as load-bearing. Don't assume either way.
9. Implement **input/feature embedding** stage: linear/MLP projection to `d_model` (the shared default
   across Stage-1 candidates in [model-architecture.md](model-architecture.md)); defer PatchTST-style
   patch embedding and per-tf tf-id embedding until the architecture-branch decision in step 5 is made,
   since both are conditional on it.
10. **Data-quality pass on the CCXT feed** (see appendix): add gap detection, restated/adjusted-candle
    detection, and delisted-pair survivorship-bias handling for the "train on all other pairs" set,
    named but unbuilt. Natural home: a data-quality subsection under
    [infrastructure.md](infrastructure.md)'s repository pattern, or its own check module — not scoped to
    a single file here since it touches the fetch/cache layer, not just the feature schema.

## done (reference)

Convention: when a `todo` step is completed, move its bullet here (append at the end, keep its original
number) instead of rewriting it in place — renumber the remaining `todo` steps to close the gap. Keeps
the active list short without needing prose edits on every completion.

1. **Audit the gap precisely.** **Done 2026-08-12** — confirmed exhaustive, see
   [appendix § audit result](#audit-result-step-1). `training_x_columns` in
   [training_datasets.py:40](../../app/ai_modelling/dataset_generator/training_datasets.py#L40) is
   currently just `['open', 'high', 'low', 'close', 'volume'] + classic_indicator_columns()` — raw OHLCV
   plus `bbands`/`obv`/`cci`/`rsi`/`mfi`/`ichimoku` (see appendix). None of
   [input-features.md § candle feature schema](../input-features.md#candle-feature-schema) (relative-HLC,
   gap-from-close, candle-height/ATR, volume/ATR, peak/valley tf detection, multi-tf top-distance
   fields) is wired into the actual model input today.
2. Implement the **relative-HLC block**: `close/ATR`, `(high-close)/ATR`, `(close-low)/ATR`, absolute
   close, `gap = (open - prev_close)/ATR`, `candle_height/ATR`. Pure per-row vectorized pandas, no
   cross-candle state — the cheapest, most load-bearing part of the schema per
   [input-features.md § feature-set completeness](../input-features.md#feature-set-completeness--testing)
   ("OHLC/ATR ... = load-bearing, not suspect"). **Done 2026-08-12** — new
   [relative_candle.py](../../app/ai_modelling/dataset_generator/relative_candle.py)
   (`add_relative_candle_columns`/`relative_candle_columns`, the 5 new ratios; absolute `close` already
   existed), wired into `single_timeframe_n_indicators`/`training_x_columns` in
   [training_datasets.py](../../app/ai_modelling/dataset_generator/training_datasets.py). ATR now computes
   per-timeframe (structure/pattern/trigger/double), not just `trigger` — removed the now-redundant
   standalone `trigger` `atr` line. Input width per timeframe: 17 → 22 columns (`master_x_shape` in
   [base.py](../../app/ai_modelling/base.py) updated). New fields are already ATR-normalized, so
   `scale_slice` leaves them unscaled (same treatment as `rsi`/`mfi`) — no change needed there. Unit tests:
   [test_relative_candle.py](../../app/tests/unit/dataset_generator/test_relative_candle.py).
3. Implement **`volume/ATR(volume)`** — confirm the data source actually provides genuine per-candle
   volume first (the spec itself flags this as unconfirmed); if it doesn't, this step blocks and needs
   its own decision. **Done 2026-08-12** — confirmed not blocked, see
   [appendix § volume data-source confirmation](#volume-data-source-confirmation-step-3). New
   [volume_feature.py](../../app/ai_modelling/dataset_generator/volume_feature.py)
   (`add_volume_feature_columns`/`volume_feature_columns`; `volume_atr = volume / RMA(volume,
atr_timeperiod)` — Wilder's RMA is the only "ATR" concept volume has, since it has no H/L/C to derive a
   true-range from), wired into `single_timeframe_n_indicators`/`training_x_columns` in
   [training_datasets.py](../../app/ai_modelling/dataset_generator/training_datasets.py) alongside step
   2's relative-HLC block. Input width per timeframe: 22 → 23 columns (`master_x_shape` in
   [base.py](../../app/ai_modelling/base.py) updated). Already ~1-centered, so `scale_slice` leaves it
   unscaled (same treatment as the relative-HLC fields). Unit tests:
   [test_volume_feature.py](../../app/tests/unit/dataset_generator/test_volume_feature.py).
4. **(decision) Wire peak/valley detection into the feature pipeline.** **Decided 2026-08-12** — split
   reuse: raw local-extrema detection reimplemented fresh in `ai_modelling`, confirmation logic left for
   todo step 1, `PeakValley.py` itself not imported. See
   [appendix § peak/valley reuse decision](#peakvalley-reuse-decision-step-4).

## appendix: current implementation status

Verified against `app/` directly on 2026-08-12.

### audit result (step 1)

Grepped `training_x_columns`/`classic_indicator_columns`/`add_classic_indicators`/`train_data_of_mt_n_profit`
repo-wide. Findings:

- **Single live definition.** `training_x_columns` is defined in exactly one place
  ([training_datasets.py:40](../../app/ai_modelling/dataset_generator/training_datasets.py#L40)), inside
  `train_data_of_mt_n_profit()`. No other file defines a competing `x_columns`/`feature_columns`/
  `input_columns` list (checked via grep for those three names too).
- **All real consumers go through that one function.** `train_data_of_mt_n_profit` (from
  `ai_modelling.dataset_generator.training_datasets`) is imported and called by
  `cnn_lstm/prediction.py`, `dataset_generator/{npz_batch,ram_batch,zip_pkl_batch,stream_loader,
test_normalization}.py` — all consume its output, none redefine the column list. So the schema in
  step 1 is exhaustive; nothing already-wired would get duplicated by the steps below.
- **Dead/broken second copy found, not a real consumer.**
  [predicting/predictor.py](../../app/ai_modelling/predicting/predictor.py) imports
  `train_data_of_mt_n_profit` from `ai_modelling.training.training_batches` — but that module
  ([training_batches.py](../../app/ai_modelling/training/training_batches.py)) is entirely commented
  out and defines no such function; the same file also imports a nonexistent
  `ai_modelling.cnn_lstm.trining_datasets` (typo'd module, no such file) and calls an undefined
  `zz_train_data_of_mt_n_profit` in its `__main__` block. This file cannot run as-is — safe to ignore
  for this schema audit, but flagged here since it looks live at a glance. Not in scope to fix here.
- **Separate legacy pipeline, unrelated to `training_x_columns`.** `read_multi_timeframe_rolling_mean_std_ohlcv`
  (referenced by the broken `predictor.py`) has duplicate implementations at
  `app/ai_modelling/training_data/PreProcessing/encoding/rolling_mean_std.py` and
  `app/PreProcessing/encoding/rolling_mean_std.py`, plus a duplicate `app/predicting/predictor.py`
  alongside `app/ai_modelling/predicting/predictor.py`. Neither references `classic_indicator_columns`
  or `training_x_columns` — a different, older normalization approach, out of scope for this schema but
  worth a separate dead-code cleanup pass someday.

Conclusion: proceed with steps 2+ against `training_x_columns` in `training_datasets.py:40` as the sole
target — confirmed exhaustive.

### what actually feeds the model today

`training_x_columns` ([training_datasets.py:40](../../app/ai_modelling/dataset_generator/training_datasets.py#L40)):

```python
training_x_columns = ['open', 'high', 'low', 'close', 'volume'] + classic_indicator_columns()
```

`classic_indicator_columns()` ([classic_indicators.py:43-46](../../app/ai_modelling/dataset_generator/classic_indicators.py#L43-L46))
returns `['bbands_u', 'bbands_m', 'bbands_l', 'sc_obv', 'sc_cci', 'rsi', 'mfi', 'ichi_conv', 'ichi_base',
'ichi_lead_a', 'ichi_lead_b', 'ichi_lag']`, computed by `add_classic_indicators()`:

- `sc_obv` = OBV, rolling-288-window z-scored (`10 * (obv - mean) / (3 * std)`)
- `sc_cci` = CCI (not in the spec's candidate pool at all), same rolling z-score treatment
- `rsi` = raw RSI — the spec's candidate pool says "drop RSI as redundant with MACD," MACD isn't
  implemented, RSI still is
- `mfi` = raw Money Flow Index
- `bbands_u`/`bbands_m`/`bbands_l` = Bollinger Bands (upper/middle/lower), raw price units
- `ichi_conv`/`ichi_base`/`ichi_lead_a`/`ichi_lead_b`/`ichi_lag` = Ichimoku components

None of [input-features.md § candle feature schema](../input-features.md#candle-feature-schema)'s
relative-HLC, gap, candle-height/ATR, volume/ATR, peak/valley-tf, or multi-tf top-distance fields exist
in this list. The model today trains on raw OHLCV + this fixed technical-indicator set only.

### peak/valley detection exists, but isn't wired in

`app/Model/TechnicalAnalysis/PeakValley.py` (plus `AtrMovementPivots.py`, `PivotsHelper.py`,
`SupportResistance.py`, `BullBearSide.py`, `RBD.py`) implement peak/valley/pivot detection elsewhere in
the codebase — likely for the `BullBearSide`/base-pattern strategy machinery (see
[training-data-labels.md § secondary mechanism](training-data-labels.md#secondary-unrelated-mechanism-livebacktest-bracket-orders)),
not for ML feature generation. Nothing in `classic_indicator_columns()` or `training_x_columns`
consumes it. Reuse-vs-parallel-implementation was decided in the peak/valley reuse decision below (done
step 4); the causal-cap wrapper itself is todo step 1.

### peak/valley reuse decision (step 4)

Read `PeakValley.py` line-by-level to decide reuse-vs-wrapper. Split decision, not all-or-nothing:

- **Reimplement raw extrema detection, don't import it.** `find_peaks_n_valleys()`'s core check
  (`high[i] > high[i-1] and high[i] > high[i+1]`, one-candle lookahead via `shift(-1)`) is a pure local
  3-candle comparison — cheap, correct, easy to reimplement in ~5 lines directly in
  `ai_modelling/dataset_generator/`. Not importing the actual function because the module it lives in
  pulls in `read_base_timeframe_ohlcv`, `symbol_data_path`, zip-file caching, and pandera multi-timeframe
  schemas — I/O and multi-tf machinery built for the `BullBearSide`/backtest pipeline, not for a
  single-timeframe df already in memory inside `single_timeframe_n_indicators()`
  ([training_datasets.py:24](../../app/ai_modelling/dataset_generator/training_datasets.py#L24)). No
  existing `ai_modelling` code imports from `Model.TechnicalAnalysis` (checked repo-wide); adding the
  first such import would couple training code to the strategy package for a 5-line function.
- **Do not reuse `calculate_strength`/`top_timeframe`/`insert_distance` for tf confirmation — write todo
  step 1 fresh instead.** `calculate_strength` sets `strength = min(left_distance, right_distance)`, where
  `right_distance` is the distance to the first _future_ crossing candle — i.e. tf confirmation as
  implemented here is symmetric past+future and bakes in unbounded lookahead relative to any anchor
  candle that would consume it as a feature. That's exactly the leak todo step 1 warns about, not a
  pre-built solution to reuse. It's also a whole-range batch computation (assumes the full date range
  up front), not the per-anchor-candle causal computation the schema needs.
- **Net effect:** todo step 1's causal-cap wrapper is genuinely new code, not glue around `PeakValley.py`. It
  reuses only the local-extrema geometry (reimplemented), then caps "confirmed tf" by elapsed time from
  the extremum to the anchor candle — never by looking at what happens after the anchor.

### volume data-source confirmation (step 3)

Traced the fetch/aggregation path to confirm `volume` is genuine per-candle exchange data, not
synthesized — the spec itself flagged this as unconfirmed, so this had to be checked before trusting
`volume/ATR(volume)`.

- **Base timeframe (`app_config.timeframes[0]`)**: `fetch_ohlcv()`
  ([fetch_ohlcv.py:72-104](../../app/data_processing/fetch_ohlcv.py#L72-L104)) calls
  `ccxt.kucoin().fetch_ohlcv(...)` directly — real per-candle volume from the exchange.
- **Higher timeframes**: `core_generate_multi_timeframe_ohlcv()`
  ([ohlcv.py:40-59](../../app/data_processing/ohlcv.py#L40-L59)) builds them by
  `.groupby(pd.Grouper(freq=frequency)).agg({..., 'volume': 'sum'})` — summed base-tf volume, not a
  separate per-timeframe fetch. This equals genuine per-candle volume for that timeframe as long as the
  base series has no gaps.
- No forward-fill/cumulative-sum/zero-fill transform found anywhere on `volume` in the fetch/aggregate
  path; the `OHLCV` PanderaDFM schema requires it non-nullable `float`.
- **One caveat, not a blocker**: the base-fetch retry loop
  ([fetch_ohlcv.py:86-104](../../app/data_processing/fetch_ohlcv.py#L86-L104)) has no gap-detection or
  backfill — after 20 retries on `RequestTimeout`/`NetworkError` it silently proceeds with whatever
  partial response it has, which would understate summed higher-tf volume for that window. Already
  covered by todo step 10 (unbuilt gap detection), not new scope here.

Conclusion: step 3 not blocked — `volume/ATR(volume)` implemented as specified.

### data-quality / CCXT feed gaps

Only coverage today is "windows built only from contiguous complete data; any gap-containing range
discarded entirely" ([model-architecture-planning.md § validation & train/test splitting](../model-architecture-planning.md#validation--traintest-splitting)).
Nothing addresses exchange downtime gaps, restated/adjusted candles, delisted-pair survivorship bias in
the "train on all other pairs" set, or per-exchange history-depth inconsistency — all real CCXT pain
points that quietly bias a multi-pair training set. Not yet built; see todo step 10.
