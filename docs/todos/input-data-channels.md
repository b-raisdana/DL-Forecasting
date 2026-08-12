# TODO — input data / channels preparation

Closing the gap between [input-features.md](../input-features.md) (the candle feature schema + feature
screening spec) and what actually feeds the model today. Current focus topic — see
[master-todo.md](master-todo.md).

- [TODO — input data / channels preparation](#todo--input-data--channels-preparation)
  - [todo](#todo)
  - [appendix: current implementation status](#appendix-current-implementation-status)
    - [what actually feeds the model today](#what-actually-feeds-the-model-today)
    - [peak/valley detection exists, but isn't wired in](#peakvalley-detection-exists-but-isnt-wired-in)
    - [data-quality / CCXT feed gaps](#data-quality--ccxt-feed-gaps)

## todo

Ordered so each step is small and independently testable. Steps marked **(decision)** need a one-line
confirmation before implementing since they affect input shape (and therefore every downstream model
candidate in [model-architecture.md](model-architecture.md)); everything else is a direct fix against
the already-written spec in [input-features.md](../input-features.md).

1. **Audit the gap precisely.** `training_x_columns` in
   [training_datasets.py:40](../../app/ai_modelling/dataset_generator/training_datasets.py#L40) is
   currently just `['open', 'high', 'low', 'close', 'volume'] + classic_indicator_columns()` — raw OHLCV
   plus `bbands`/`obv`/`cci`/`rsi`/`mfi`/`ichimoku` (see appendix). None of
   [input-features.md § candle feature schema](../input-features.md#candle-feature-schema) (relative-HLC,
   gap-from-close, candle-height/ATR, volume/ATR, peak/valley tf detection, multi-tf top-distance
   fields) is wired into the actual model input today. Confirm this list is exhaustive (grep
   `training_x_columns`/`classic_indicator_columns` for any other consumer) before starting the build
   below, so nothing already-wired gets duplicated.
2. Implement the **relative-HLC block**: `close/ATR`, `(high-close)/ATR`, `(close-low)/ATR`, absolute
   close, `gap = (open - prev_close)/ATR`, `candle_height/ATR`. Pure per-row vectorized pandas, no
   cross-candle state — the cheapest, most load-bearing part of the schema per
   [input-features.md § feature-set completeness](../input-features.md#feature-set-completeness--testing)
   ("OHLC/ATR ... = load-bearing, not suspect").
3. Implement **`volume/ATR(volume)`** — confirm the data source actually provides genuine per-candle
   volume first (the spec itself flags this as unconfirmed); if it doesn't, this step blocks and needs
   its own decision.
4. **(decision) Wire peak/valley detection into the feature pipeline.**
   `app/Model/TechnicalAnalysis/PeakValley.py` already implements peak/valley detection (see appendix)
   but nothing in `classic_indicator_columns()`/`training_x_columns` consumes it. Decide: reuse that
   module directly, or the schema needs a causal-cap wrapper around it (see step 5) that doesn't exist
   yet either way.
5. Implement the **no-lookahead cap** on peak/valley confirmation: "is a top? which tf" must be capped
   by elapsed time to the anchor candle (a candle 4mo before anchor can confirm 4M peak only if it
   stayed max to anchor; 1wk before anchor caps at 1W) — per
   [input-features.md § candle feature schema](../input-features.md#candle-feature-schema). This is the
   part of the schema most likely to silently leak future information if implemented naively; write the
   causality test in step 6 against this specific field first.
6. **Add the no-lookahead regression test** for this pipeline stage: perturbing FUTURE-slice data must
   never change a computed peak/valley/top-distance feature at or before the anchor candle. Companion to
   the label-side version of this test in
   [training-data-labels.md](training-data-labels.md#todo) — same discipline, different pipeline stage.
7. Implement the **multi-tf top-distance fields** (2 and 3 higher tfs' top time/price distances: signed
   time offset, top volume strength, abs-time, natural/normal price distance, abs-price) per the
   tf-ordered-list (5min/15min/1H/4H/1D/1W as actual input series; 1M/4M/1Y confirmation-only). Depends
   on steps 4-6 landing first.
8. Implement **nearest-top/nearest-valley distance** fields (`nearest_top_distance`,
   `nearest_top_tf`, nearest top volume strength, and the valley equivalents) — the
   `min(distance/ATR(peak's tf))` aggregation across all tfs, per the schema.
9. **(decision) `distance from anchor candle (minutes)`** and **timeframe-in-minutes** fields — per the
   schema, `timeframe-in-minutes` is a config option gated on architecture (included for flat/shared-encoder
   archs, excluded for per-tf-branch archs). Confirm which architecture branch
   [model-architecture.md](model-architecture.md) is building toward before wiring this field in or out.
10. Once steps 2-9 land, **run the ablation pass**: permutation importance against the new full set,
    against the priority-ordered suspicion queue already named in the spec (candle-height/ATR flagged as
    the most likely redundant-but-cheap field). Needs a trained model to permute against — sequence
    this after a first Stage-1 candidate exists in
    [model-architecture.md](model-architecture.md), not before.
11. **Run the candidate-feature MI/GBM screen** against the candidate pool (OBV — already implemented,
    confirm it's actually screened not just present; ICHIMUKU — same; MACD, ADX, VWAP, volatility-regime
    trio, session/time cyclical, structural counts — all unimplemented). Use
    `sklearn.mutual_info_classif/regression` per
    [input-features.md § candidate-feature screening](../input-features.md#candidate-feature-screening--method);
    small LightGBM/XGBoost only if MI is ambiguous.
12. **Reconcile the candidate pool with what's already in `classic_indicators.py`.** Code currently
    computes `cci`/`rsi`/`mfi`/`bbands` — none of which appear in
    [input-features.md § candidate feature pool](../input-features.md#candidate-feature-pool) (which
    instead names MACD and says "drop RSI as redundant with it"). Either the pool needs updating to
    reflect what's already screened-and-kept, or these existing indicators need to go through the same
    MI screen as new candidates before being trusted as load-bearing. Don't assume either way.
13. Implement **input/feature embedding** stage: linear/MLP projection to `d_model` (the shared default
    across Stage-1 candidates in [model-architecture.md](model-architecture.md)); defer PatchTST-style
    patch embedding and per-tf tf-id embedding until the architecture-branch decision in step 9 is made,
    since both are conditional on it.
14. **Data-quality pass on the CCXT feed** (see appendix): add gap detection, restated/adjusted-candle
    detection, and delisted-pair survivorship-bias handling for the "train on all other pairs" set,
    named but unbuilt. Natural home: a data-quality subsection under
    [infrastructure.md](infrastructure.md)'s repository pattern, or its own check module — not scoped to
    a single file here since it touches the fetch/cache layer, not just the feature schema.

## appendix: current implementation status

Verified against `app/` directly on 2026-08-12.

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
consumes it. Whether it's reusable as-is for the schema's causally-capped peak/valley feature (todo
steps 4-5) or needs a parallel implementation hasn't been checked at the line level — first task of
todo step 4.

### data-quality / CCXT feed gaps

Only coverage today is "windows built only from contiguous complete data; any gap-containing range
discarded entirely" ([model-architecture-planning.md § validation & train/test splitting](../model-architecture-planning.md#validation--traintest-splitting)).
Nothing addresses exchange downtime gaps, restated/adjusted candles, delisted-pair survivorship bias in
the "train on all other pairs" set, or per-exchange history-depth inconsistency — all real CCXT pain
points that quietly bias a multi-pair training set. Not yet built; see todo step 14.
