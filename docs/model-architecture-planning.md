# AI Trading System — Planning Notes

Goal: best AI method to predict price moves → optimal trade positions, max profit / min risk.

- [AI Trading System — Planning Notes](#ai-trading-system--planning-notes)
  - [definitions](#definitions)
    - [natural price distance](#natural-price-distance)
    - [normal price distance](#normal-price-distance)
    - [volume strength of tops](#volume-strength-of-tops)
  - [key questions](#key-questions)
    - [data feed design](#data-feed-design)
      - [candle feature schema](#candle-feature-schema)
      - [feature-set completeness — testing](#feature-set-completeness--testing)
        - [non-contributing-channel detection — method](#non-contributing-channel-detection--method)
        - [new-fature addition — workflow](#new-fature-addition--workflow)
        - [candidate feature pool](#candidate-feature-pool)
        - [candidate-feature screening — method](#candidate-feature-screening--method)
    - [normalization strategy](#normalization-strategy)
    - [model architecture \& selection](#model-architecture--selection)
  - [hardware constraints](#hardware-constraints)
  - [multi-timeframe fusion](#multi-timeframe-fusion)
  - [validation \& train/test splitting](#validation--traintest-splitting)
  - [model outputs \& targets](#model-outputs--targets)
    - [how TP1-4 / drawdown labels are built for training](#how-tp1-4--drawdown-labels-are-built-for-training)
    - [normalization](#normalization)
  - [optimization strategy](#optimization-strategy)
  - [evaluation \& error metrics](#evaluation--error-metrics)
    - [error metric vs. trading objective](#error-metric-vs-trading-objective)
  - [class imbalance handling](#class-imbalance-handling)
  - [experiment tracking (current priority)](#experiment-tracking-current-priority)
  - [deferred topics (not current concerns, placeholders)](#deferred-topics-not-current-concerns-placeholders)
  - [glossary](#glossary)


## definitions

ATR = pandas-ta.ATR(256)

### natural price distance

+/- ~ price is higher/lower than top

### normal price distance

+/- : price is [higher-for-peaks or lower-for-valley]/[vice-versa]

### volume strength of tops

- example for a 4H top:
  - some volume of 2 times lower (15min):
    - time: +/-256 original time candles
    - peak: L < H + original ATR

## key questions

### data feed design

#### candle feature schema

each candle includes:

- HLC:
  - C / ATR
  - (H - C) / ATR
  - (C - L) / ATR
- gap from last close (O - last-C) / ATR
- candle height / ATR
- volume / ATR(volume) — assumes the data source provides per-candle volume; confirm before relying on this field.
- is a top? which tf (minutes of highest confirmed tf; + peak / − valley)
  - no lookahead — confirmed peak/valley tf capped by elapsed time to **anchor candle** (last candle of window, the "as of" point; distinct from real "now" since each training window has its own anchor). E.g. candle 4mo before anchor can confirm as 4M peak if it stayed max to anchor; candle 1wk before anchor capped at 1W. Causal by construction.
  - if not = 0
- timeframe in minutes — only for multi-tf shared architectures; omitted for per-timeframe-branch architectures (branch identity already encodes tf, field would be constant/redundant there). See "multi-timeframe fusion."
- 2 and 3 higher tfs tops time and price distances
  - tf ordered list: 5min, 15min, 1H, 4H, 1D, 1W, 1M, 4M, 1Y — note: this list is only used for peak/valley timeframe confirmation and the fields below; only the first 6 (5min–1W) are actual input series, per "multi-timeframe fusion."
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

#### feature-set completeness — testing

1. ablation pass: permutation importance against the current set — flags features pulling no weight.
2. candidate screen: MI / GBM screen against the candidate pool — flags missing signal not yet captured.
3. set is provisional until both 2 and 3 have actually run at least once; re-run 2 whenever a new feature is promoted out of 3.

- priority order for step 2, ranked by suspicion (unverified guess, just for sequencing the ablation queue):
  1. candle height/ATR — deterministic fn of existing OHLC (high−low), redundant info though may speed small-model training.
  - OHLC/ATR, peak/valley signal, dist-from-anchor = load-bearing, not suspect.
- timeframe-in-minutes was also on this queue (near-zero-info under per-tf-branch archs, since branch id already encodes tf) — out of the ablation queue entirely instead: made a config option gated on architecture — included for flat/shared-encoder archs (disambiguates mixed timeframes, so _not_ low-info there), excluded for per-tf-branch archs (would be constant/redundant), per the feature schema above / "multi-timeframe fusion" below.
  Alt:
  - certify completeness by reasoning alone, no ablation — rejected, too easy to fool self
  - start with large feature set + prune — rejected, costlier than minimal+screened-additions

##### non-contributing-channel detection — method

- permutation importance on trained model — shuffle feature in val set, measure KPI drop (cheaper, no retrain).
  Alt (will not be used unless permutation insufitiency be proved):
  - SHAP — more informative, expensive for seq models; deferred til shortlist exists
  - exhaustive 2^n subset search — infeasible
  - gradient saliency/integrated gradients — viable, deferred; better for "where in time" than "whether"

##### new-fature addition — workflow

hypothesis-driven, not "more=better." Flow: state hypothesis → cheap MI screen (below) → full run only if signal → keep only if backtested KPI improves enough to justify cost.
Alt:

- speculative additions w/o hypothesis — rejected, causes redundant-channel risk
- fixed periodic review cadence — deferred, hypothesis-driven preferred while system small

##### candidate feature pool

- OBV (volume / accumulation-distribution)
- ICHIMUKU (trend, support/resistance, momentum — all-in-one)
- MACD (keep as the momentum read; drop RSI as redundant with it)
- ADX (trend strength — distinguishes trending vs. ranging, which RSI/MACD/Ichimoku don't measure directly)
- VWAP (volume-weighted price benchmark — different volume angle than OBV; intraday fair-value reference)
- volatility/regime: realized-vol ratio across timeframes, Bollinger-width squeeze, ATR-of-ATR.
- session/time: hour/day cyclical (sin/cos), session-open/overlap flags (Asia/EU/US).
- cross-symbol (once multi-symbol): BTC-dominance/BTC price as market-beta, rolling correlation/beta to BTC.
- structural: time-since-last-peak/valley per horizon, consecutive-same-direction-candle count.
- distance from other tf tops.

##### candidate-feature screening — method

mutual information (`sklearn.mutual_info_classif/regression`) between candidate + each label head, vs current top features — no GPU needed. Near-zero MI → deprioritize. If ambiguous: small LightGBM/XGBoost w/ vs w/o candidate. Full DL run reserved for candidates passing both.
Alt:

- Pearson/Spearman alone — rejected, misses nonlinear; kept as even-cheaper pre-check
- full training run per candidate — rejected, the expensive default this avoids
- Boruta/RFE around GBM screen — deferred, useful once larger candidate pool exists

### normalization strategy

- for all price based inputs use rolling /ATR scheme.
- **alternative schemes to test:** (a) log-return norm (scale-free); (b) rolling z-score; (c) min-max per window (cheap, likely worse — loses cross-window vol comparability); (d) hybrid — ATR-norm price + separate raw log-return channel (position + velocity).
  Alt:
  - no normalization — rejected, non-stationary
  - min-max as primary — rejected, loses vol-regime comparability; kept only as test candidate
- **testing protocol** for a normalization change: same discipline as the seed-count workflow below — ≥3 seeds/scheme, compare backtested-KPI distributions (not train loss), same train/validate split scheme (see "validation & train/test splitting").
  Alt: single-run train-loss comparison (rejected — exactly the noise risk flagged).

### model architecture & selection

- **architecture candidates:** (a) Transformer w/ per-tf embedding + cross-tf attention; (b) TCN — cheaper, dilated convs for multi-scale, good single-GPU baseline; (c) hybrid CNN→Transformer; (d) state-space (Mamba/S4) — cheap long-seq alt to attention; (e) LSTM/GRU — sanity-check floor.
  Alt:
  - pure MLP on flattened features — rejected as serious candidate, discards seq structure; trivial baseline only
  - 4 separate per-tf models + late ensembling — kept as cheap baseline, see fusion section
  - GNN over tf/symbol nodes — deferred, no evidence needed yet
- **hyperparam search-space bounds:** not fixed a priori — `profile_trial_cost()` measures real wall-clock/VRAM per arch+hparam combo on this card; `max_trials_for_budget()` derives trial cap. Search-space priors: seq len capped 256/tf; batch size s.t. largest arch/seq combo fits VRAM at batch≥8; hidden-dim/depth kept modest vs ~1yr data (small vs NLP-scale). Concrete bounds from profiler's first pass, not hand-picked.
  Alt:
  - fixed ranges from DL-literature defaults w/o profiling — rejected, wrong hardware/dataset scale
  - very wide ranges relying only on Hyperband — rejected as primary, wastes trials in OOM regions

- **cross-architecture fairness**
  - architecture = categorical param in one Optuna study (not N sweeps) → fairness enforced at study level:
    - (1) same train-pairs/BTC-USDT split every trial (see "validation & train/test splitting");
    - (2) one shared GPU-hour budget via `estimate_total_budget()`, not per-arch;
    - (3) Hyperband pruning arch-agnostic;
    - (4) min grace-period epochs before pruning (protects slow-converging archs); post-study sanity-check trial counts per arch, top-up budget if one is starved.
  - Alt:
    - separate sweeps w/ equalized budgets — rejected, old approach, wastes compute
    - fixed wall-clock per arch — rejected, same waste, time-boxed
    - compare only best trial per arch — rejected, too seed-sensitive

## hardware constraints

- RTX 4060 Laptop GPU, 8GB VRAM (8188 MiB per `nvidia-smi`), 64GB RAM, 2 SSD/HDD.

- **max feasible model size**
  - don't hand-calculate — `profile_trial_cost()`/`estimate_total_budget()`/`max_trials_for_budget()`
  - measure real wall-clock+VRAM on this exact card.
    - Rough prior only: 4tf×256candles×few scalars is a modest seq length; 8GB should fit small/med Transformer/TCN at batch 16–64 w/ mixed precision; VRAM more likely bound by hidden-dim/full-attention-over-concat-seq than seq length itself. If full cross-attention doesn't fit, caps toward cheaper fusion alternatives (per-tf encoders + light fusion) below.
  - Alt:
    - gradient checkpointing — fallback if needed, slows training
    - mixed precision AMP — near-free win, enable by default, not as fallback
    - gradient accumulation — fallback if batch-size-bound
    - cloud/rented GPU — rejected, conflicts w/ local-only decision; revisit only if hard-bottlenecked
    - model parallelism — n/a, single GPU

## multi-timeframe fusion

- each series 256 candles
- from anchor backards: 5min/15min/1H/4H/1D/1W tfs
- overlap: ≤1 higher-tf candle may overlap a lower-tf series.
- domain assumption: pattern meaning is scale-invariant across tf (15min long-pattern ≈ 1H long-pattern); combining tfs clarifies the "real truth" behind any one tf's pattern.
- **multi-tf combination approach:** per-tf encoders (small TCN/Transformer per series) → concat/pool → shared cross-tf fusion block (small Transformer over pooled reps, or concat+MLP as cheaper baseline). Lower effort than full cross-attention over the concatenated sequence; natural first arch to profile before the pricier full-attention option. Per the timeframe-in-minutes resolution above, this per-tf-branch design drops that field entirely (branch identity already tells the encoder the tf); it's only added back if the arch choice switches to the flat/shared-encoder option below.
  Alt:
  - 4 separate models + late ensemble — rejected as primary, loses cross-tf interaction; cheap baseline only
  - flat Transformer full self-attn over concat seq, no per-tf stage — most expensive, candidate only if profiling allows
  - hierarchical/wavelet decomposition — deferred, more complex, no evidence needed
- **long-window focus:** attention/state-space over fixed pooling is the standard approach, since 1yr+ of data is fed but target patterns can live anywhere from the last few candles to half the sequence, and the relevant window shifts case-to-case. Include as an arch candidate; compare vs recency-weighted-pooling baseline to confirm it earns its cost.
  Alt:
  - fixed recency weighting w/o learned attention — rejected as sole approach, can't adapt; kept as cheap baseline
  - manual windowing/hand-picked N — rejected, reintroduces the problem attention solves
- **pattern speed-invariance** (same pattern over 3 vs 30 candles) — a time-warping problem: (a) TCN multi-dilation captures multi-scale shape w/o explicit warp; (b) attention has no fixed receptive field either — test directly. Explicit DTW preprocessing kept as fallback/diagnostic if the architectural approach fails empirically (test: does model score known same-pattern-diff-speed examples similarly?).
  Alt:
  - DTW preprocessing as default — rejected, heavier engineering, hurts real-time variable-length inference; diagnostic/fallback only
  - volatility-based logical-candle resampling — deferred, hard to reconcile w/ existing windowing/multi-tf design
- **pattern scale-invariance** (same pattern, different price magnitude): largely already handled by the resolved ATR-relative normalization (scale expressed relative to volatility by construction). Open part is architectural: conv layers naturally somewhat scale-robust; attention has no strong scale bias — evaluate empirically. Treat as validated by construction; architecture comparison is the remaining lever, not a new normalization step.
  Alt: separate explicit "scale normalization" beyond ATR-relative (rejected — double-normalizes, unnecessary).
- **decision-anchor point** keep the primary Long/Short/No-Trade decision anchored at the anchor candle for now (simplest, matches current label design). Treat entry-timing (too-soon/too-late detection) as a secondary/future output — changes label design non-trivially, wait for anchor-based baseline first.
  Alt:
  - build entry-timing into v1 — rejected for now, adds complexity before baseline validated
  - separate downstream "timing" model post-Long/Short — viable future step, deferred
- **higher-tf "in progress" candles — decision:** use only completed candles — lowest tf 15min; 1H candles = 256×15min prior so most-recent is fully closed; same for higher tfs. Sidesteps partial-candle state and cross-tf boundary-alignment leakage by construction.

## validation & train/test splitting

- windows built only from contiguous complete data; any gap-containing range discarded entirely.

- **split scheme — resolved (simplified):** train on all other trading pairs, validate on BTC/USDT. This is a cross-symbol (leave-one-symbol-out) split, not a temporal one — since train and validation are entirely different assets, there's no same-symbol window overlap to leak across, so the walk-forward-vs-embargo machinery isn't needed. Use the full BTC/USDT history as the validation set.
  Alt:
  - walk-forward / random-split-with-embargo within a single symbol's own history — previous approach, dropped as unnecessary complexity now that validation is cross-symbol
  - rotating leave-one-symbol-out across all pairs rather than always BTC/USDT — viable generalization check, deferred; BTC/USDT fixed as the validation symbol since it's the primary target market
- **final holdout — resolved:** reserve the most-recent contiguous block of BTC/USDT (≥ several weeks, enough trade outcomes at 4H scale); never touches training-pair selection or any tuning decision; used exactly once, after arch/hparams/normalization/threshold are locked in from the BTC/USDT validation split, for final reported KPIs. Materially worse holdout result than validation = overfitting-to-tuning signal → investigate, don't re-tune against it (would require a fresh holdout).
  Alt: no separate final holdout, reporting the BTC/USDT validation KPIs directly as final (rejected — still risks overfitting through repeated validation-set tuning).

## model outputs & targets

- action: Long/Short/No-Trade
- SL-distance = drawdown-before-TP. TP1 will be calculated based on this to.
- TP4-distance = max forcasted profit.
- TP2-ratio = 0 to 1 scale between TP1 and TP4
- TP3-ratio = 0 to 1 scale between TP2 and TP4
- confidence factors
  - I need to deploy a model architecture which give me a configdenc metric aside with forcasted values.
  - I do not have input data for confidence.

### how TP1-4 / drawdown labels are built for training

Resolved spec moved to [training-data.md](training-data.md#tp1-4--drawdown-labels); rationale/alternatives kept here.

Within 4H timeout, labels built with hindsight (label-construction only, not a live feature):

- TP1 = break-even point (partial close → zero-loss + banked profit on remainder)
- TP4 = max gainable profit in 4H window (hindsight)
- TP2/TP3 = intermediate levels, local max-gainable-profit points before a max-drawdown pullback
- SL optimized per trade; TP1 defined relative to SL, so SL definition must come first
- **TP2/TP3 selection rule — resolved:** walk forward chronologically from TP1, take local maxima in time order (not size-sorted), each qualifying if followed by a drawdown pullback > threshold (e.g. fraction of ATR) before the next higher max. TP2 = first qualifying max after TP1, TP3 = next after TP2, TP4 = global max (may coincide w/ TP3's successor). Chronological order preferred over "two largest" so TP2<TP3<TP4 in time too, matching real sequential partial-exit execution. Fallback if <2 qualifying maxima: collapse/duplicate TP2/3 toward TP4 rather than leaving undefined (every example gets a complete label).
  Alt:
  - "two largest" regardless of time order — rejected as primary, could put TP3 before TP2 in time; kept as alt rule to test
  - "two most persistent" — viable, not default; harder to define precisely, test if primary rule unstable
  - null/masked TP2-3 when <2 maxima — rejected as default, extra loss-masking complexity vs simpler duplicate fallback
- required probabilistic outputs (P(TPn), P(drawdown)) vs the deterministic hindsight labels above are a separate modeling decision (quantile regression / calibrated per-level classifier / ensemble-MC spread) — belongs under "model architecture & selection."
- **no-breakeven edge case — resolved:** three-way outcome space, not forced TP/SL binary, covers the case where price never returns to breakeven before timeout: (a) SL literally hit → SL/loss; (b) neither breakeven nor SL hit by timeout → distinct **Timeout** label (no stop was actually triggered); (c) TP1+ reached → TP-tier labels. Keeps "stopped at defined max loss" distinguishable from "closed flat/small-loss at timeout untouched by SL."
  Alt:
  - force-label as SL anyway — rejected, conflates severities, distorts distribution
  - drop such examples — rejected, survivorship bias, loses legit hard examples
  - mark-to-market P&L as extra continuous regression target — deferred, second head before categorical version validated

### normalization

- (OHLC − anchor close) / ATR of same candle; anchor close = 0.
- each candle normalized by its _own_ rolling ATR (not anchor's ATR) in it's _own_ tf — standard vol-normalization, expresses move in "ATRs of that period's own regime," making calm- and volatile-period moves comparable.

## optimization strategy

- optimization = **one search** across (1) arch/model-combo choice + (2) each arch's hparams, not two disjoint phases.
- architecture = single categorical param inside same Optuna study as hparams (conditional sub-params per arch), not exhaustive — bad archs pruned early instead of full independent sweeps each. Impl: `app/ai_modelling/parameter_optimizser/optuna_optimizer.py`.
- Optuna TPE (sample-efficient, single-GPU budget) + Hyperband pruning.
- GA/NSGA-II for optional 2nd refinement stage.
- Pareto front across competing KPIs (e.g. Sortino vs max-DD).
- per-trial time measured not assumed.
  - runs real training steps per arch, measures wall-clock+peak-VRAM;
  - `estimate_total_budget()`/`max_trials_for_budget()` → projected total + trial-count cap before full study.
- `OptunaPruningCallback` reports val_loss/epoch, prunes Hyperband-unpromising or NaN/Inf trials.
- best-run selection KPI:
  - under "evaluation & error metrics" below — primary=expectancy, guardrail=max-DD, secondary=Sortino, once the backtest module is built. Until then `val_loss` remains the training-time proxy (`compute_fitness()`), explicitly interim not final.
    Alt: see full list under "evaluation & error metrics" (Sharpe/Calmar/profit-factor/win-rate/NSGA-II-Pareto) — not repeated here.

## evaluation & error metrics

- **error-rate measurement** no single one — measured per-head instead (quantile-loss/MAE for price levels, Brier/log-loss for probabilities, precision/recall/F1 for action). Final selection uses backtested trading KPIs, not these directly.
- **per-head loss candidates to test:** quantile/pinball vs MAE/MSE (price levels); Brier vs log-loss (calibration); cross-entropy vs focal vs class-weighted-CE (action, ties to imbalance section).
- **model-selection method:** per-head metrics = dev diagnostics only; final selection = backtested KPIs (expectancy primary, max-DD guardrail, Sortino secondary) on the BTC/USDT validation split, then the untouched final holdout.

### error metric vs. trading objective

- low statistical loss ≠ profitability — training-time signal only, not selection criterion.
- final selection = backtested KPIs: win rate, profit factor, expectancy/trade, Sharpe/Sortino, max-DD, Calmar — via actual simulated trades from TP/SL predictions.

- **primary KPI vs guardrails** primary = expectancy/trade (R-multiples or %) — reflects real profitability per opportunity, less sensitive to trade frequency than profit factor. Guardrail = max-DD (reject any config over acceptable DD/risk tolerance regardless of other numbers). Secondary ranking (among guardrail-passers) = Sortino (vs Sharpe — doesn't penalize upside vol, fits asymmetric TP strategy). Win rate/profit factor = diagnostics only (each gameable alone).
  Alt:
  - Sharpe as primary — rejected, penalizes wanted upside vol
  - Calmar as primary — viable, not chosen; kept adjacent to max-DD guardrail
  - profit factor as primary — rejected, ignores trade frequency/opportunity cost
  - NSGA-II multi-objective Pareto — the actual longer-term plan via `run_kpi_refinement()`, pending the backtest module; single-primary-KPI is the interim until then
- **per-head metric list** (regression + calibration + classification heads, not one blended number):
  - price levels (TP1-4, SL): quantile/pinball loss + MAE companion.
  - probabilities: Brier + log-loss + calibration curve/ECE.
  - action: precision/recall/F1 per class (macro-F1, imbalance-aware) + confusion matrix.
  - per-head metrics feed Optuna's scalar objective only as a weighted-sum interim proxy (matches existing val_loss use); real selection stays the backtested-KPI stage.
    Alt:
    - single blended loss only — rejected, already flagged insufficient
    - per-head multi-objective Optuna — more complex, deferred; single-GPU budget
    - AUC-ROC instead of F1 — viable companion/secondary diagnostic, not primary
- **seed-count workflow** min 3 seeds/config, 5 preferred if budget allows; paired stat test across matched folds (paired t-test / Wilcoxon) — require CI excluding zero, not eyeballed means. Reserve multi-seed re-run budget for top finalists post-search only (too expensive per-trial during search) — factor into `estimate_total_budget()`/`max_trials_for_budget()`.
  Alt:
  - single seed — rejected, can't separate signal/noise
  - bootstrap resampling of val set — cheaper, complementary, combinable w/ 3-seed approach
  - 10+ seeds per candidate during search — rejected, too expensive; finalists-only instead

## class imbalance handling

- test class-weighted vs focal loss for classification-style targets (peak/valley class, TP-hit-before-SL class), compare.
- multi-horizon ATR-distance to nearest peak/valley at fixed horizons (4H/1D/1W/1M/4M/1Y) — turns single categorical "highest confirmed tf" feature into continuous features, sidesteps imbalance for that feature.

- **multi-horizon vs categorical peak/valley feature** feed both. Continuous ATR-distance features = primary (solve imbalance); keep categorical "highest confirmed tf" too as a cheap compact discrete summary — may capture something continuous version doesn't, esp. at low data volume. Confirm via ablation, don't assume; drop categorical only if ablation shows zero marginal contribution.
  Alt:
  - continuous-only, drop categorical now — rejected/deferred, no ablation evidence yet it's safe
  - categorical-only, skip multi-horizon — rejected, reintroduces the imbalance problem it solves
  - replace w/o ever testing — rejected, riskier than testing first, no evidence
- **class-weight-vs-focal test scope** applies to both peak/valley and TP-hit/drawdown targets, as two separate experiments (not one shared decision): (a) peak/valley target if kept as aux output; (b) TP-hit/SL-hit/Timeout target (see 3-way label above), likely differently-shaped rarity (TP4 probably rarer than generic peak/valley). Tune class weights/focal-gamma per target, not globally.
  Alt:
  - scope to peak/valley only — rejected, leaves TP/SL imbalance, likely worse, unaddressed
  - one blanket loss choice untested per-target — rejected, no evidence of transfer, gamma likely needs per-target tuning
  - resampling alternatives:
    - SMOTE-style — awkward for sequential windows, likely rejected
    - class-balanced batch sampling — deferred, complementary
    - inference-time cost-sensitive thresholding — relates to No-Trade threshold answer; complementary lever, not replacement
- **prevalence measurement — next action, not yet run:** actual prevalence (% candles peak/valley per horizon, % trades reaching each TP vs SL-hit vs Timeout) isn't known — measure empirically once the labeling pipeline exists, via a data-profiling script, before finalizing the class-weight/focal choice above.
  Alt: assume prevalence from market-structure intuition, no measurement (rejected — exactly what this step avoids).

## experiment tracking (current priority)

- needed now: ad hoc file-naming (`- Copy (2).keras`, `.bak`, `.nan` in /data) won't scale, can't trace which run→which result.
- decide lightweight tracking: min = consistent naming/logging convention (config hash+date+key hparams); ideally a tool (MLflow/W&B/CSV-SQLite) logging config+dataset-version+metrics(loss+trading KPIs)+artifact path together.
- local-only (e.g. MLflow w/ local file backend).
  Alt:
  - W&B/cloud-hosted — rejected for now, conflicts w/ local-only; revisit if collaboration/remote-dashboard becomes a real need
  - bare CSV/SQLite log, no dedicated tool — viable fallback if MLflow local-server overhead isn't worth it
  - no formal tracking — rejected, explicitly doesn't scale, see above

## deferred topics (not current concerns, placeholders)

- **transaction costs/spread/slippage/latency**: matters for sub-4H scalping, not addressed now. Revisit before live/paper trading — cost-free backtest overstates real perf.
- **risk/position sizing beyond TP targets**: handled manually via existing procedure, not by model. No AI work needed now.
- **market regime robustness/retraining cadence**: not addressed now. Revisit once live a while — crypto regime shifts (trend/range/vol), untouched model can decay silently.

## glossary

- ATR = pandas-ta.ATR(256)
- anchor candle = last candle of a 256-candle window; the "as of" point for a prediction (training or live)
