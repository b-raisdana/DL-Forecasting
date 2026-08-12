# Data, Label & Feature Engineering

## Market data

Binance OHLCV
Historical depth: as long as available in binance API, the model trainer can fetch not-having data.
Candle synchronization: only closed candles will be used for trainig and prediction.
Missing candles: input series with missing candles are skipped.
Data integrity is reliable.

## Derived market data

Returns

### normalization strategy

- for all price based inputs use rolling /ATR scheme.
- **alternative schemes to test:**
  - log-return norm (scale-free)
  - rolling z-score
  - min-max per window (cheap, likely worse — loses cross-window vol comparability)
  - hybrid — ATR-norm price + separate raw log-return channel (position + velocity)
    Alt:
  - no normalization — rejected, non-stationary
  - min-max as primary — rejected, loses vol-regime comparability; kept only as test candidate
- **testing protocol** for a normalization change: same discipline as the [seed-count workflow](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) — ≥3 seeds/scheme, compare backtested-KPI distributions (not train loss), same train/validate split scheme (see "validation & train/test splitting").
  Alt: single-run train-loss comparison (rejected — exactly the noise risk flagged).

### candle feature schema

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
- timeframe in minutes — only for multi-tf shared architectures; omitted for per-timeframe-branch architectures (branch identity already encodes tf, field would be constant/redundant there). See [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion).
- 2 and 3 higher tfs tops time and price distances
  - tf ordered list: 5min, 15min, 1H, 4H, 1D, 1W, 1M, 4M, 1Y — note: this list is only used for peak/valley timeframe confirmation and the fields below; only the first 6 (5min–1W) are actual input series, per [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion).
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

### feature-set completeness — testing

1. ablation pass: permutation importance against the current set — flags features pulling no weight.
2. candidate screen: MI / GBM screen against the candidate pool — flags missing signal not yet captured.
3. set is provisional until both 2 and 3 have actually run at least once; re-run 2 whenever a new feature is promoted out of 3.

- priority order for step 2, ranked by suspicion (unverified guess, just for sequencing the ablation queue):
  1. candle height/ATR — deterministic fn of existing OHLC (high−low), redundant info though may speed small-model training.
  - OHLC/ATR, peak/valley signal, dist-from-anchor = load-bearing, not suspect.
- timeframe-in-minutes was also on this queue (near-zero-info under per-tf-branch archs, since branch id already encodes tf) — out of the ablation queue entirely instead: made a config option gated on architecture — included for flat/shared-encoder archs (disambiguates mixed timeframes, so _not_ low-info there), excluded for per-tf-branch archs (would be constant/redundant), per the feature schema above / [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) in the main doc.
  Alt:
  - certify completeness by reasoning alone, no ablation — rejected, too easy to fool self
  - start with large feature set + prune — rejected, costlier than minimal+screened-additions

#### non-contributing-channel detection — method

- permutation importance on trained model — shuffle feature in val set, measure KPI drop (cheaper, no retrain).
  Alt (will not be used unless permutation importance proves insufficient):
  - SHAP — more informative, expensive for seq models; deferred til shortlist exists
  - exhaustive 2^n subset search — infeasible
  - gradient saliency/integrated gradients — viable, deferred; better for "where in time" than "whether"

#### new-feature addition — workflow

hypothesis-driven, not "more=better." Flow: state hypothesis → cheap MI screen (below) → full run only if signal → keep only if backtested KPI improves enough to justify cost.
Alt:

- speculative additions w/o hypothesis — rejected, causes redundant-channel risk
- fixed periodic review cadence — deferred, hypothesis-driven preferred while system small

#### candidate feature pool

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

#### candidate-feature screening — method

mutual information (`sklearn.mutual_info_classif/regression`) between candidate + each label head, vs current top features — no GPU needed. Near-zero MI → deprioritize. If ambiguous: small LightGBM/XGBoost w/ vs w/o candidate. Full DL run reserved for candidates passing both. This is the correct scoped use of a GBM here: cheap, tabular, point-in-time, no sequence structure needed — the question at this stage is "does this one feature carry signal at all," not "does the sequence pattern matter" (see [auxiliary tabular models (GBM-family)](03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family) for the other roles GBMs play in this pipeline, and for scope limits).
Alt:

- Pearson/Spearman alone — rejected, misses nonlinear; kept as even-cheaper pre-check
- full training run per candidate — rejected, the expensive default this avoids
- Boruta/RFE around GBM screen — deferred, useful once larger candidate pool exists
- CatBoost instead of/alongside LightGBM/XGBoost for this screen — viable third option, near-identical API; see [modern GBM-family alternatives](03-Model & Architecture Engineering.md#modern-gbm-family-alternatives) for why it may edge out the other two on this repo's likely feature mix (mixed-type, moderate-noise, some high-cardinality categoricals like tf-id)

## Label design

Future return
Multi-horizon labels

### where can be a position?

- NOW candle timeframe: 5 minutes.
- for each NOW candle, use 4hours FUTURE knowledge (training-time only, not available at inference) to decide the candle's label: Long, Short, or None
- earliest possible entry: the 5-minute candle immediately following NOW (a position cannot open before NOW has closed; verify with 1-minute candles during backtesting)
- look forward boundary: 4H, exclusive of the boundary instant — e.g. NOW = 9:55-10:00 means the position must close before 14:00; the candle 13:55-14:00 is in-bounds, 14:00-14:05 is Timeout
- simulate opening a position at the targeted bid price (see targeting bid price) in each direction, applying trading overhead (see trading overhead): compute `MFE`, `SL`, `MAE`, `OM` per direction (see TP / MAE / OM labels)
- a direction is a valid label only if `OM > 1` (MFE exceeds MAE — some favorable edge over adverse risk); if neither direction clears it, the candle is labeled None
- tie-break when both directions clear `OM > 1`: pick whichever has the higher `OM`, zero the other direction's signal

### trading overhead

- trading fee rate `F` = 0.1% — no separate spread cost is modeled; a single targeted bid price is used for both Long and Short entries
- fees enter only on the risk side: `Risk = MAE × (1 + F) × V` (see [TP / MAE / OM labels](#tp--mae--om-labels)) — `MFE`/`OM` are raw price-derived, not fee-adjusted

### targeting bid price

- entry price `E` = the best price reachable within the 5-minute candle immediately following NOW's close
- limit-order target: the model commits open price + SL + TP together as one decision as soon as NOW closes; the best price in that next candle is the training label for the entry-price output
- at inference: if the predicted price isn't reached within that candle, the limit order never fills and no position opens

### risk factors

- SL = effective future adverse level = `max(worst adverse excursion before TP4, ATR floor distance)`, ATR floor = 1x `ATR(255, 15min)` (see glossary)
- `MAE = abs(E - SL)` — SL re-expressed as a distance from entry
- the ATR floor guards against a near-zero adverse move producing an unrealistically tight SL
- SL is a retrospective risk-sizing measure, not a live barrier: by construction it is never breached along the path to TP4. A live SL order is placed an epsilon farther out than this computed level.

### TP / MAE / OM labels

Per direction, computed from the FUTURE window (see [where can be a position?](#where-can-be-a-position)); favorable/adverse flips by direction — Long: favorable = higher price, adverse = lower price; Short: favorable = lower price, adverse = higher price.

- `MFE` = maximum favorable price excursion during the 4H horizon
- `TP4 = E ± MFE` — the MFE endpoint (`+` Long, `-` Short)
- `SL`, `MAE` = see [risk factors](#risk-factors)
- `Risk = MAE × (1 + F) × V` (`V` = position volume, `F` = fee rate — see [trading overhead](#trading-overhead))
- `OM` (Opportunity Multiple) = `MFE / MAE` — reward-to-risk ratio; drives the direction/None call in [where can be a position?](#where-can-be-a-position)

Execution levels (`TP1`-`TP4`) — discrete scale-out prices derived from `TP4`/`Risk`, for live order placement, not primary ML targets:

- `TP1_vol_ratio`, `TP1_risk_mult` — configurable params (fraction of `V` closed at TP1; risk multiplier sizing TP1's distance)
- `TP1_dist = Risk × TP1_risk_mult × TP1_vol_ratio`; `TP1 = E ± TP1_dist`
- `TP_step = abs(TP1 - TP4) / 3`; `TP2 = TP1 ± TP_step`; `TP3 = TP1 ± 2 × TP_step`; `TP4` = the MFE endpoint above

Rules:

- never use a future-derived value (`SL`, `MAE`, `MFE`, `TP4`, `OM`, or any `TP1`-`TP4` level) as an input feature — features carry only information available at entry
- always preserve raw `MAE`, `MFE`, `OM` even when only derived TP/Risk levels are consumed downstream
- each sample keeps its future horizon and trade direction fixed, so labels stay reproducible and path-dependent outcomes from different horizons/directions are never mixed
- supersedes the old three-way SL-hit/Timeout/TP outcome label: `TP4` is always the realized MFE endpoint (no TP-vs-Timeout ambiguity); SL-hit-vs-not still matters for live execution (see risk factors) but is no longer a separate ML label

### model output targets

What the model actually predicts, given the labels above (target/label definitions belong with the labeling spec; how the model represents/predicts them stays in [model-architecture-planning.md](03-Model & Architecture Engineering.md#model-architecture--selection)):

- action head = Long / Short / None (see [where can be a position?](#where-can-be-a-position))
- primary regression targets = `MAE`, `OM` (see [TP / MAE / OM labels](#tp--mae--om-labels))
- auxiliary regression target = `MFE`
- confidence metric = open gap, no input features carry confidence information today — see [error-rating-and-evaluation.md § confidence & calibration metrics](04-Experimentation, Evaluation & Optimization.md#confidence--calibration-metrics).

## Future-information handling

Overlapping labels

## Dataset construction

Sliding windows
Sequence generation
Sampling
Train/validation/test segmentation

### class imbalance handling

- test class-weighted vs focal loss for classification-style targets (peak/valley class, TP-hit-before-SL class), compare.
- multi-horizon ATR-distance to nearest peak/valley at fixed horizons (tf-ordered-list 4H-to-1Y) — turns single categorical "highest confirmed tf" feature into continuous features, sidesteps imbalance for that feature.

- **multi-horizon vs categorical peak/valley feature** feed both. Continuous ATR-distance features = primary (solve imbalance); keep categorical "highest confirmed tf" too as a cheap compact discrete summary — may capture something continuous version doesn't, esp. at low data volume. Confirm via ablation, don't assume; drop categorical only if ablation shows zero marginal contribution.
  Alt:
  - continuous-only, drop categorical now — rejected/deferred, no ablation evidence yet it's safe
  - categorical-only, skip multi-horizon — rejected, reintroduces the imbalance problem it solves
  - replace w/o ever testing — rejected, riskier than testing first, no evidence
- **class-weight-vs-focal test scope** applies to both peak/valley and TP-hit/drawdown targets, as two separate experiments (not one shared decision):
  - peak/valley target if kept as aux output
  - ~~TP-hit/SL-hit/Timeout target~~ — removed from spec (see [training-data.md § TP / MAE / OM labels](#tp--mae--om-labels)); `MAE`/`OM` are now continuous regression targets, not a categorical outcome, so this class-weight/focal question no longer applies here
  - Tune class weights/focal-gamma per target, not globally.
    Alt:
  - scope to peak/valley only — rejected, leaves TP/SL imbalance, likely worse, unaddressed
  - one blanket loss choice untested per-target — rejected, no evidence of transfer, gamma likely needs per-target tuning
  - resampling alternatives:
    - SMOTE-style — awkward for sequential windows, likely rejected
    - class-balanced batch sampling — deferred, complementary
    - inference-time cost-sensitive thresholding — relates to No-Trade threshold answer; complementary lever, not replacement
- **prevalence measurement — next action, not yet run:** actual prevalence (% candles peak/valley per horizon, % trades reaching each TP vs SL-hit vs Timeout) isn't known — measure empirically once the labeling pipeline exists, via a data-profiling script, before finalizing the class-weight/focal choice above.
  Alt: assume prevalence from market-structure intuition, no measurement (rejected — exactly what this step avoids).
- **cheap iteration proxy for the class-weight/focal choice** — a GBM on flattened features is a cheap place to iterate which weighting scheme/gamma looks promising before committing to a full DL retrain cycle, same cheap-proxy-before-expensive-run pattern already used for feature screening; see "auxiliary tabular models (GBM-family)".

## Temporal dataset splitting

Rolling windows
Expanding windows
Purged validation

### validation & train/test splitting

- windows built only from contiguous complete data; any gap-containing range discarded entirely.

- **split scheme — resolved (simplified):** train on all other trading pairs, validate on BTC/USDT. This is a cross-symbol (leave-one-symbol-out) split, not a temporal one — since train and validation are entirely different assets, there's no same-symbol window overlap to leak across, so the walk-forward-vs-embargo machinery isn't needed. Use the full BTC/USDT history as the validation set.
  Alt:
  - walk-forward / random-split-with-embargo within a single symbol's own history — previous approach, dropped as unnecessary complexity now that validation is cross-symbol
  - rotating leave-one-symbol-out across all pairs rather than always BTC/USDT — viable generalization check, deferred; BTC/USDT fixed as the validation symbol since it's the primary target market
- **final holdout — resolved:** reserve the most-recent contiguous block of BTC/USDT (≥ several weeks, enough trade outcomes at 4H scale); never touches training-pair selection or any tuning decision; used exactly once, after arch/hparams/normalization/threshold are locked in from the BTC/USDT validation split, for final reported KPIs. Materially worse holdout result than validation = overfitting-to-tuning signal → investigate, don't re-tune against it (would require a fresh holdout).
  Alt: no separate final holdout, reporting the BTC/USDT validation KPIs directly as final (rejected — still risks overfitting through repeated validation-set tuning).

## Glossary

- `E` / `V` / `F` = entry price / position volume / trading fee rate
- `MFE` (maximum favorable excursion) = best move _for_ the position from entry, over the horizon
- `MAE` (maximum adverse excursion) = worst move _against_ the position from entry to `SL` — not a pullback from an interim peak (that's "retracement")
- `OM` (Opportunity Multiple) = `MFE / MAE`, the reward-to-risk ratio
- `Risk` = fee-inflated `MAE` in position-volume terms, `MAE × (1 + F) × V`
- `TP1`-`TP4` = discrete execution scale-out levels between entry and the `MFE` endpoint (see [TP / MAE / OM labels](#tp--mae--om-labels))
- HISTORY / NOW / FUTURE = already-closed candles / the candle we're in / not-yet-started candles
- SL / TP = stop loss / take profit
- ATR = pandas-ta.ATR(256)
- anchor candle = last candle of a 256-candle window; the "as of" point for a prediction (training or live)
- tf-ordered-list = 5min, 15min, 1H, 4H, 1D, 1W, 1M, 4M, 1Y
- tf = timeframe
- natural price distance = signed distance from a top: + = price higher than the top, − = lower (not adjusted for peak vs. valley)
- normal price distance = natural price distance with sign flipped for valleys, so + always means "away from the top" for both peaks and valleys
- volume strength of tops = SUM(volume) / ATR(volume) of the 2-tf-lower candles (e.g. 4H top → 15min) within ±256 top-tf candles, restricted to candles whose [L,H] overlaps the top's price range (peak-high/valley-low ± 2-tf-lower ATR(256))
- Stage-1 = the current architecture-search phase; picks one whole model architecture from the candidate set (see "model architecture & selection")
- S1/S2/S3 = hyperparameter-profile labels per Stage-1 architecture candidate: depth-heavy / width-heavy / context-heavy (see [Stage-1 candidate sets](03-Model & Architecture Engineering.md#stage-1-candidate-sets))
- GBM = gradient boosting machine (LightGBM/XGBoost/CatBoost family) — see "auxiliary tabular models (GBM-family)"
