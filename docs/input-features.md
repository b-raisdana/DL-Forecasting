# Model Architecture Planning — Input Features & Embedding

Broken out from [AI Trading System — Planning Notes](model-architecture-planning.md#data-feed-design) for size. Covers the candle-level input feature schema, the feature-set completeness-testing workflow, and the input/feature-embedding stage of the model pipeline — see [top level architecture](model-architecture-planning.md#top-level-architecture) in the main doc for how the embedding stage fits into the rest of the pipeline.

- [Model Architecture Planning — Input Features & Embedding](#model-architecture-planning--input-features--embedding)
  - [candle feature schema](#candle-feature-schema)
  - [feature-set completeness — testing](#feature-set-completeness--testing)
    - [non-contributing-channel detection — method](#non-contributing-channel-detection--method)
    - [new-feature addition — workflow](#new-feature-addition--workflow)
    - [candidate feature pool](#candidate-feature-pool)
    - [candidate-feature screening — method](#candidate-feature-screening--method)
  - [input / feature embedding](#input--feature-embedding)
  - [glossary](#glossary)

## candle feature schema

each candle includes:

- relative-HLC:
  - C / ATR
  - (H - C) / ATR
  - (C - L) / ATR
- close: absolute close price without any change
- gap from last close (O - last-C) / ATR
- candle height / ATR
- volume / ATR(volume) — assumes the data source provides per-candle volume; confirm before relying on this field.
- is a top? which tf (minutes of highest confirmed tf; + peak / − valley)
  - no lookahead — confirmed peak/valley tf capped by elapsed time to anchor candle (see glossary). E.g. candle 4mo before anchor can confirm as 4M peak if it stayed max to anchor; candle 1wk before anchor capped at 1W. Causal by construction.
  - if not = 0
- timeframe in minutes — only for multi-tf shared architectures; omitted for per-timeframe-branch architectures (branch identity already encodes tf, field would be constant/redundant there). See [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion).
- 2 and 3 higher tfs tops time and price distances
  - tf ordered list: 5min, 15min, 1H, 4H, 1D, 1W, 1M, 4M, 1Y — note: this list is only used for peak/valley timeframe confirmation and the fields below; only the first 6 (5min–1W) are actual input series, per [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion).
    - eg. for 15mins, it is 4H and 1D
  - dont overlook future but we know about input duration
  - time: +/- ~ number of candles before/after the top
  - top volume strength
  - abs-time
  - natural-price-distance
  - normal-price-distance
  - abs-price
- distance from anchor candle (minutes)
- price distance / ATR from nearest previous top
  - nearest_top_distance = min(distance / ATR(in the tf of the peak) for tfs)
    - abs
    - natural price distance
    - normal price distance
  - nearest_top_tf = tf number of minutes of nearest peak
  - nearest top volume strength
  - the same for nearest_valley_distance, and nearest_valley_tf

## feature-set completeness — testing

1. ablation pass: permutation importance against the current set — flags features pulling no weight.
2. candidate screen: MI / GBM screen against the candidate pool — flags missing signal not yet captured.
3. set is provisional until both 2 and 3 have actually run at least once; re-run 2 whenever a new feature is promoted out of 3.

- priority order for step 2, ranked by suspicion (unverified guess, just for sequencing the ablation queue):
  1. candle height/ATR — deterministic fn of existing OHLC (high−low), redundant info though may speed small-model training.
  - OHLC/ATR, peak/valley signal, dist-from-anchor = load-bearing, not suspect.
- timeframe-in-minutes was also on this queue (near-zero-info under per-tf-branch archs, since branch id already encodes tf) — out of the ablation queue entirely instead: made a config option gated on architecture — included for flat/shared-encoder archs (disambiguates mixed timeframes, so _not_ low-info there), excluded for per-tf-branch archs (would be constant/redundant), per the feature schema above / [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion) in the main doc.
  Alt:
  - certify completeness by reasoning alone, no ablation — rejected, too easy to fool self
  - start with large feature set + prune — rejected, costlier than minimal+screened-additions

### non-contributing-channel detection — method

- permutation importance on trained model — shuffle feature in val set, measure KPI drop (cheaper, no retrain).
  Alt (will not be used unless permutation importance proves insufficient):
  - SHAP — more informative, expensive for seq models; deferred til shortlist exists
  - exhaustive 2^n subset search — infeasible
  - gradient saliency/integrated gradients — viable, deferred; better for "where in time" than "whether"

### new-feature addition — workflow

hypothesis-driven, not "more=better." Flow: state hypothesis → cheap MI screen (below) → full run only if signal → keep only if backtested KPI improves enough to justify cost.
Alt:

- speculative additions w/o hypothesis — rejected, causes redundant-channel risk
- fixed periodic review cadence — deferred, hypothesis-driven preferred while system small

### candidate feature pool

- OBV (volume / accumulation-distribution)
- ICHIMUKU (trend, support/resistance, momentum — all-in-one)
- MACD (keep as the momentum read; drop RSI as redundant with it)
- ADX (trend strength — distinguishes trending vs. ranging, which RSI/MACD/Ichimoku don't measure directly)
- VWAP (volume-weighted price benchmark — different volume angle than OBV; intraday fair-value reference)
- volatility/regime:
  - realized-vol ratio across timeframes: ratio of short-window vs. long-window realized volatility, flags whether volatility is expanding or contracting relative to its own recent history.
  - Bollinger-width squeeze: width of Bollinger Bands (upper − lower) relative to price, narrow width signals low-volatility compression that often precedes a breakout.
  - ATR-of-ATR: ATR computed on a series of ATR values (volatility of volatility), captures whether the volatility regime itself is stable or unstable.
- session/time: hour/day cyclical (sin/cos), session-open/overlap flags (Asia/EU/US).
- cross-symbol (once multi-symbol): BTC-dominance/BTC price as market-beta, rolling correlation/beta to BTC.
- structural: time-since-last-peak/valley per horizon, consecutive-same-direction-candle count.
- distance from other tf tops.

### candidate-feature screening — method

mutual information (`sklearn.mutual_info_classif/regression`) between candidate + each label head, vs current top features — no GPU needed. Near-zero MI → deprioritize. If ambiguous: small LightGBM/XGBoost w/ vs w/o candidate. Full DL run reserved for candidates passing both. This is the correct scoped use of a GBM here: cheap, tabular, point-in-time, no sequence structure needed — the question at this stage is "does this one feature carry signal at all," not "does the sequence pattern matter" (see [auxiliary tabular models (GBM-family)](model-architecture-planning.md#auxiliary-tabular-models-gbm-family) for the other roles GBMs play in this pipeline, and for scope limits).
Alt:

- Pearson/Spearman alone — rejected, misses nonlinear; kept as even-cheaper pre-check
- full training run per candidate — rejected, the expensive default this avoids
- Boruta/RFE around GBM screen — deferred, useful once larger candidate pool exists
- CatBoost instead of/alongside LightGBM/XGBoost for this screen — viable third option, near-identical API; see [modern GBM-family alternatives](model-architecture-planning.md#modern-gbm-family-alternatives) for why it may edge out the other two on this repo's likely feature mix (mixed-type, moderate-noise, some high-cardinality categoricals like tf-id)

## input / feature embedding

- linear/MLP projection of the per-candle feature vector → `d_model` — shared first step across all Stage-1 candidates.
- per-tf embedding — learned tf-id embedding for flat/shared-encoder archs; implicit via branch identity for per-tf-branch archs (see the timeframe-in-minutes resolution under [candle feature schema](#candle-feature-schema) above).
- **PatchTST-style patch embedding** — groups contiguous candles into patches before projecting, shortening the effective sequence length fed to attention. Directly relevant given the VRAM-cost note under [hardware constraints](model-architecture-candidate-sets.md#hardware-constraints) → "max feasible model size" (full attention over the concatenated multi-tf sequence is the dominant cost, not param count) — patching is a cheap lever on that cost. This is the patching _mechanism_ trained from scratch as part of the Stage-1 candidate, distinct from the pretrained PatchTST-based checkpoints covered (and excluded) under [excluded topics](model-architecture-planning.md#excluded-topics-broken-out-into-separate-files) → [TSFMs](timeseries-foundation-models-architecture-planning.md).
  Alt: raw per-candle projection, no patching — simpler, longer effective sequence into attention; current default.

## glossary

ATR, anchor candle, tf, tf-ordered-list, natural/normal price distance — see [model-architecture-planning.md § glossary](model-architecture-planning.md#glossary).
