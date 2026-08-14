# Experimentation, Evaluation & Optimization

## Measurement & Evaluation

**Open gap — generalization-testing methodology** ([99-Weakness Analysis.md § A20](99-Weakness%20Analysis.md), Imp 2): the [validation & train/test splitting](02-Data, Label & Feature Engineering.md#validation--traintest-splitting) 4-way scheme gives an aggregate empirical cross-symbol/regime signal per training run, but there's no formal methodology yet for testing whether learned patterns transfer (BTC→ETH, BTC→other symbols) vs. are market-specific. Computational-cost metrics (wall-clock/VRAM) are covered under [hyperparam search-space bounds](#hyperparam-search-space-bounds) / [optimization strategy](#optimization-strategy); ranking robustness across seeds/conditions is covered in [03 § cross-seed and cross-condition robustness](03-Model & Architecture Engineering.md#cross-seed-and-cross-condition-robustness); training stability is covered below.

### core principle: error metric ≠ trading objective

Low statistical loss does not imply profitability. Per-head statistical metrics are a **training-time signal only** — they drive gradient descent and Optuna's interim objective — never the final model-selection criterion. Final selection is always the backtested trading KPIs below, computed from actual simulated trades derived from TP/SL predictions. Every metric choice in this doc traces back to that split: dev diagnostic, or selection criterion — never both.

### per-head statistical metrics (dev diagnostics)

No single blended loss — measured per output head, since each has a different error shape:

| head                                     | candidates to test                                                                                                                                                                                                                                                 | notes                                                                                                                                                                                                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| price levels (TP, SL / MAE, OM, aux MFE) | **baseline (point):** quantile/pinball loss (primary) + MAE/MSE companion. **alternative (probabilistic `MFE`/`MAE`):** distributional NLL, moments added incrementally — Gaussian (mean+std) → skew-normal/Johnson-SU (+skew) → skew-t/Pearson-system (+kurtosis) | quantile loss reused by [uncertainty-native GBM variants](03-Model & Architecture Engineering.md#uncertainty-native-gbm-variants--confidence-metric-gap); probabilistic alt defined in [02 § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets) |
| probabilities / confidence               | Brier vs log-loss                                                                                                                                                                                                                                                  | see [confidence & calibration metrics](#confidence--calibration-metrics)                                                                                                                                                                                                         |
| action (Long/Short/None)                 | cross-entropy vs focal vs class-weighted-CE                                                                                                                                                                                                                        | tuning scope (which loss, per-target gamma) lives in [class imbalance handling](02-Data, Label & Feature Engineering.md#class-imbalance-handling)                                                                                                                                |

Per-head metrics feed Optuna's scalar objective only as a **weighted-sum interim proxy** (matches the existing `val_loss` use in `compute_fitness()`) until the backtest module exists — real selection always stays at the backtested-KPI stage below.
Alt: single blended loss — rejected, already flagged insufficient. Per-head multi-objective Optuna — more complex, deferred (single-GPU budget). AUC-ROC — viable companion/secondary diagnostic to F1, not primary.

### confidence & calibration metrics

**Open gap — confidence-metric mechanism** ([99-Weakness Analysis.md § A19](99-Weakness%20Analysis.md), Imp 2): a confidence metric is needed alongside every forecast, but no input feature carries confidence information today (see [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets)).

Calibration is measured the same way as any probability head above — Brier score, log-loss, and a **calibration curve / Expected Calibration Error (ECE)** to check predicted vs. observed hit-rate, not just point accuracy.

The preferred fix is a model that produces calibrated uncertainty as a mechanism of itself, not via extra engineered features. Candidate techniques (quantile GBM, NGBoost, quantile regression forests) and their priority order live in [model-architecture-planning.md § uncertainty-native GBM variants](03-Model & Architecture Engineering.md#uncertainty-native-gbm-variants--confidence-metric-gap), scored in [prioritization-framework.md § auxiliary tabular models](05-Prioritization Framework.md#auxiliary-tabular-models-gbm-family) — this doc owns _what_ to measure and why it matters, that section owns _which technique_ produces it.

A second candidate, on the primary DL head rather than the auxiliary GBM: probabilistic `MFE`/`MAE` (distribution-parameter regression, mean/std/skew/kurtosis added incrementally) — see [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets). If adopted, TP/SL probability/risk estimates derive directly from the fitted distribution feeding the existing TP/SL ladder mechanism, instead of needing a bolted-on separate confidence head.

### backtested trading KPIs (final selection)

Computed from actual simulated trades derived from TP/SL predictions, on Validation A and Validation B, then the untouched Final Test (see [model-architecture-planning.md § validation & train/test splitting](02-Data, Label & Feature Engineering.md#validation--traintest-splitting)).

- **primary — expectancy/trade** (R-multiples or %): real profitability per opportunity, less sensitive to trade frequency than profit factor.
- **guardrail — max-DD**: reject any config over acceptable drawdown/risk tolerance regardless of other numbers.
- **secondary (among guardrail-passers) — Sortino**: doesn't penalize upside vol, fits an asymmetric TP strategy, ranks finalists.
- **diagnostics only (each gameable alone)** — win rate, profit factor.
- **longer-term plan** — NSGA-II multi-objective Pareto front (e.g. Sortino vs max-DD) across competing KPIs via `run_kpi_refinement()`, once the backtest module exists; single-primary-KPI is the interim until then. Until the backtest module is built, `val_loss` remains the training-time proxy (`compute_fitness()`) — explicitly interim, not final.

Alt: Sharpe as primary — rejected, penalizes wanted upside vol. Calmar as primary — viable, not chosen; kept adjacent to the max-DD guardrail. Profit factor as primary — rejected, ignores trade frequency/opportunity cost.

### terminology: "MAE" means two different things here

- **Here (statistical):** Mean Absolute Error — a companion loss for price-level heads, alongside quantile/pinball loss.
- **In [training-data.md](02-Data, Label & Feature Engineering.md#glossary) and [todos/training-data-labels.md](../todos/02-training-data-labels.md#what-drawdown-actually-measures-mae-not-peak-retracement) (trading):** Maximum Adverse Excursion — the worst adverse price move from entry before the best-case exit, used to derive SL/labels. Unrelated to the statistical metric above; always read from context.

### glossary

KPI/metric terms specific to this doc; general project terms (ATR, tf-ordered-list, Stage-1, S1/S2/S3, GBM) live in [model-architecture-planning.md § glossary](02-Data, Label & Feature Engineering.md#glossary).

- expectancy/trade = average R-multiple (or %) gained per trade opportunity, including losers
- profit factor = gross profit / gross loss (diagnostic only — gameable by trade frequency)
- Sharpe = return / total volatility (penalizes upside vol — rejected as primary here)
- Sortino = return / downside volatility only (secondary KPI — fits asymmetric TP strategy)
- Calmar = return / max drawdown
- max-DD = maximum drawdown (guardrail, not a ranking KPI)
- Brier score = mean squared error of a predicted probability against the binary outcome
- ECE = Expected Calibration Error — gap between predicted confidence and observed hit-rate, bucketed
- pinball/quantile loss = asymmetric loss for a target quantile (e.g. 10th/50th/90th percentile TP)
- macro-F1 = F1 averaged per class, unweighted — imbalance-aware action-head diagnostic

## Experiment Design

Ablation/component-independence methodology is covered in [03 § component-independence testing](03-Model & Architecture Engineering.md#component-independence-testing). Feature/label leakage prevention is out of scope here — **open gap, tracked in 02-Data's domain** ([99-Weakness Analysis.md § A5](99-Weakness%20Analysis.md), Imp 1); this doc's own leakage-adjacent guard is the cross-symbol (not temporal) train/val split, see [training-data sampling strategy](#training-data-sampling-strategy).

### statistical validity of comparisons

Applies to every A/B decision in this project's docs (normalization scheme, activation function, architecture candidate, class-weight/focal choice) — one shared discipline so results aren't noise:

- **min 3 seeds/config**, 5 preferred if budget allows.
- **paired stat test** across matched folds (paired t-test / Wilcoxon) — require a confidence interval excluding zero, not eyeballed means. N here is seeds/folds (independent retrains), unaffected by same-symbol label overlap below.
- compare **backtested-KPI distributions**, never a single train-loss number.
- reserve multi-seed re-run budget for **top finalists post-search only** (too expensive per-trial during search) — factor into `estimate_total_budget()`/`max_trials_for_budget()`.
- **overlapping-sample correction, when a stat is computed directly over per-candle rows** (dev-diagnostic per-head losses, bootstrap resampling of the val set below) rather than over seeds/folds: same-symbol samples' label windows overlap (240-min horizon over 5-min candles — see [02 § overlapping labels](02-Data, Label & Feature Engineering.md#overlapping-labels)), so raw row count overstates independence. Use that section's per-sample uniqueness weights: weight rows by uniqueness before computing/bootstrapping the statistic, and report **effective N = Σ uniqueness**, never raw row count, alongside any such significance claim.

Alt: single seed — rejected, can't separate signal from noise. Bootstrap resampling of the val set — cheaper, complementary, combinable with the 3-seed approach; must apply the uniqueness correction above since it resamples raw rows. 10+ seeds per candidate during search — rejected, too expensive; finalists-only instead.

### model-selection pipeline

How the pieces above chain together, end to end:

1. **during search** — per-head losses (weighted-sum interim proxy, or plain `val_loss` pre-backtest-module) drive Optuna/Hyperband pruning. Dev diagnostics only.
2. **finalists** — re-run top configs across ≥3 seeds (see [statistical validity](#statistical-validity-of-comparisons)), compare backtested-KPI distributions: expectancy primary, max-DD guardrail, Sortino secondary.
3. **selection** — best finalist on _both_ [Validation A and Validation B](02-Data,%20Label%20&%20Feature%20Engineering.md#validation--traintest-splitting): must not regress on either vs. the current best, not a single blended score (see that section's selection rule). Two comparisons per candidate instead of one raises the stakes on the **still-open family-wise/backtest-overfitting correction** ([99-Weakness Analysis.md § A14](99-Weakness%20Analysis.md)/B5, Imp 1) — not yet solved by this split.
4. **final holdout** — run exactly once, on Final Test (BTC/USDT, temporal-only — see split doc), after everything (arch/hparams/normalization/threshold) is locked in. A materially worse holdout result than Validation B is an overfitting-to-tuning signal → investigate, don't re-tune against it (that would require a fresh holdout).

### experiment tracking (current priority)

- needed now: ad hoc file-naming (`- Copy (2).keras`, `.bak`, `.nan` in /data) won't scale, can't trace which run→which result.
- decide lightweight tracking: min = consistent naming/logging convention (config hash+date+key hparams); ideally a tool (MLflow/W&B/CSV-SQLite) logging config+dataset-version+metrics(loss+trading KPIs)+artifact path together.
- local-only (e.g. MLflow w/ local file backend).
  Alt:
  - W&B/cloud-hosted — rejected for now, conflicts w/ local-only; revisit if collaboration/remote-dashboard becomes a real need
  - bare CSV/SQLite log, no dedicated tool — viable fallback if MLflow local-server overhead isn't worth it
  - no formal tracking — rejected, explicitly doesn't scale, see above

### cross-architecture fairness

- architecture = categorical param in one Optuna study (not N sweeps) → fairness enforced at study level:
  - (1) same 4-way split (train / Validation A / Validation B / Final Test) every trial (see [validation & train/test splitting](02-Data, Label & Feature Engineering.md#validation--traintest-splitting));
  - (2) one shared GPU-hour budget via `estimate_total_budget()`, not per-arch;
  - (3) Hyperband pruning arch-agnostic;
  - (4) min grace-period epochs before pruning (protects slow-converging archs); post-study sanity-check trial counts per arch, top-up budget if one is starved.
- Alt:
  - separate sweeps w/ equalized budgets — rejected, old approach, wastes compute
  - fixed wall-clock per arch — rejected, same waste, time-boxed
  - compare only best trial per arch — rejected, too seed-sensitive

## Parameter Optimization

Model parameters
Training parameters
Feature parameters
Label parameters
Threshold parameters
Window parameters — per-tf sequence/context length (uniform vs. independent-per-tf vs. tapering-schedule; see [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length"), not fixed a priori

## Hyperparameter Optimization

Learning rate
Dropout
Weight decay
Optimizer
Loss-function parameters

### hyperparam search-space bounds

not fixed a priori — `profile_trial_cost()` measures real wall-clock/VRAM per arch+hparam combo on this card; `max_trials_for_budget()` derives trial cap. Search-space priors: per-tf window length (seq len) is itself an Optuna dim (placeholder default 256/tf uniform; independent-per-tf and tapering-schedule alts also candidates — see [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length"), capped per profiler-measured VRAM/wall-clock feasibility rather than a fixed value; batch size s.t. largest arch/seq combo fits VRAM at batch≥8; hidden-dim/depth kept modest vs ~1yr data (small vs NLP-scale). Concrete bounds from profiler's first pass, not hand-picked.
Alt:

- fixed ranges from DL-literature defaults w/o profiling — rejected, wrong hardware/dataset scale
- very wide ranges relying only on Hyperband — rejected as primary, wastes trials in OOM regions

## Training Engineering

Training-process decisions distinct from architecture choice ([03](03-Model & Architecture Engineering.md)) and from what/how metrics are measured (above): training-strategy selection, batch-size strategy, epoch/budget selection, loss-weight selection, training stability, sampling strategy, augmentation. Seed control / GPU determinism is a separate, **prerequisite open gap** ([99-Weakness Analysis.md § A12](99-Weakness%20Analysis.md), Imp 1) this section's discipline (≥3 seeds, paired stat test) already assumes rather than re-derives.

### training-strategy selection

- **default — train from scratch**, one Stage-1 architecture candidate per Optuna trial (per [current Stage-1 candidate set](03-Model & Architecture Engineering.md#current-stage-1-candidate-set)): cheapest, no external dependency, matches this doc's measured-evidence-only discipline since there's nothing pretrained to inherit bias from.
- **pretrain → fine-tune, already a separate scoped track**: [Time-Series Foundation Models](05-Prioritization Framework.md#time-series-foundation-models-tsfms) (Chronos/TimesFM/Moirai/Lag-Llama/PatchTST-based) — zero-shot inference first (cheap, no training) as a baseline data point, fine-tune locally only if zero-shot clears a floor worth the extra training cost. Full pretraining from scratch stays out of scope (that's what makes them "foundation" models — large corpora/compute, not a single-GPU exercise).
- **multi-stage training, already present in scattered form** — not a new mechanism: [meta-labeling](03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family) (primary DL model → secondary GBM classifier, a second stage on a narrower label) and [knowledge distillation](03-Model & Architecture Engineering.md#combination-strategies-combinedsuper-models) (N teachers → one student) are both instances of it. Both follow the same rule, which generalizes to any future case: an extra stage is adopted only if it's measured to beat the single-stage backtested KPI (per [core principle](#core-principle-error-metric--trading-objective)) by enough to be worth the added training/inference cost, never by default.
- **self-supervised pretraining** — explicitly excluded, see [99-Exclusion.md](99-Exclusion.md).
- **decision:** train-from-scratch, single-stage, is the default for every Stage-1 candidate; TSFM fine-tuning and multi-stage variants are already-scoped exceptions gated behind the same backtested-KPI bar as everything else here, not parallel default paths.
  Alt: mandate a pretrain-then-fine-tune stage for every candidate — rejected, no evidence financial-series pretraining transfers here (see [TSFM open questions](05-Prioritization Framework.md#time-series-foundation-models-tsfms)), and adds a training stage with no measured payoff yet.

### batch-size strategy

- **feasibility ceiling, already resolved:** `profile_trial_cost()` measures real VRAM per arch+seq combo on the RTX 4060 8GB card; [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints) already names batch size (not param count) as the primary VRAM lever, and [hyperparam search-space bounds](#hyperparam-search-space-bounds) already bounds it "s.t. largest arch/seq combo fits VRAM at batch≥8." Missing piece: **largest feasible ≠ best-performing.**
- **treat batch size as an Optuna-searched hyperparameter**, ranged from the profiler-measured feasibility ceiling per arch down to the existing floor (batch 8–16), not pinned at the ceiling — larger batches buy smoother gradients and GPU throughput but are a documented source of a generalization gap at this data scale; which value performs best is an empirical question the same Optuna study already answers for every other hparam, not a separate manual pass.
- **gradient accumulation** — already named as a VRAM fallback under [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints); the same mechanism doubles as a way to test an effectively-larger batch than fits in one forward/backward pass, without raising the feasibility ceiling itself.
  Alt: fix batch size at the max-feasible value for every trial, no search — rejected, conflates "largest that fits" with "best that generalizes"; a single batch size fixed independent of arch/seq — rejected, ignores the already-flagged per-arch VRAM variance.

### epoch / training-budget selection

- **pruning machinery, already resolved:** `OptunaPruningCallback` reports val_loss/epoch, Hyperband prunes unpromising/NaN-Inf trials, and [cross-architecture fairness](#cross-architecture-fairness) already sets a minimum grace-period epoch count before pruning kicks in (protects slow-converging archs). Missing: the ceiling and stop rule on the other end.
- **max-epoch ceiling**: derived from `estimate_total_budget()`/`max_trials_for_budget()` (already the mechanism for the overall search budget, per [optimization strategy](#optimization-strategy)) — a per-trial epoch cap falls out of the same wall-clock budgeting, not a separately hand-picked number.
- **early stopping**: monitor val_loss (or the weighted-sum interim proxy, per [per-head statistical metrics](#per-head-statistical-metrics-dev-diagnostics)), patience = a small multiple of the grace-period epoch count already set for Hyperband, restore best-weights on stop rather than the last epoch's.
- **minimum training budget before a trial is fairly comparable**: already the grace-period rule under [cross-architecture fairness](#cross-architecture-fairness) point 4 — same answer, not a second rule; stated here only so it isn't misread as still-open.
  Alt: fixed epoch count for every trial regardless of arch — rejected, exactly what Hyperband + grace period already avoid; no early stopping, always train to the epoch ceiling — rejected, wastes budget on trials that plateau early.

### loss-weight selection

- **the question:** per-head losses (action CE/focal, price-level quantile+MAE, confidence Brier/log-loss — see [per-head statistical metrics](#per-head-statistical-metrics-dev-diagnostics)) already combine into a "weighted-sum interim proxy" feeding Optuna's objective (matches `compute_fitness()`), but nothing states how the per-head weights themselves are set — resolved by the bullets below.
- **normalize before weighting**: per-head losses sit on incomparable native scales (cross-entropy vs. pinball loss in price units vs. Brier) — track a running EMA of each head's loss scale and normalize by it before applying weights, so a weight reflects relative importance, not an accident of units. Skipping this confounds any weight search: an apparently dominant head may just be the one with the larger native loss magnitude, not the one that matters more.
- **search the weights**: treat the small (~3-dimensional) per-head weight vector as an additional Optuna dimension within the same trial-cost budget as everything else in the study, not hand-fixed. Directly closes the risk this question's own framing names: ensuring one target doesn't dominate optimization.
- **uncertainty-weighting as a candidate alternative to static search**: homoscedastic task-uncertainty weighting (Kendall et al.) — a learned log-variance per head that dynamically re-weights losses during training instead of a fixed value tuned once — is a well-established, cheap-to-add technique (a handful of extra scalar params) worth testing head-to-head against the static-search baseline, not assumed superior.
- Weighted-sum stays an **interim search proxy only**, per [core principle](#core-principle-error-metric--trading-objective) — however the weights are chosen, final selection is still the backtested-KPI stage, unaffected by the loss-weight choice.
  Alt: fixed equal weights (1/N heads) — rejected, no reason to assume equal importance given confidence is a secondary target relative to action/price-level heads; hand-picked weights from intuition — rejected, same measured-evidence-only reasoning as everywhere else in this doc.

### training stability

- **NaN/Inf, already resolved:** `OptunaPruningCallback` already prunes NaN/Inf trials (per [optimization strategy](#optimization-strategy)) — catastrophic failure is handled. The gap is milder, non-crashing instability a NaN/Inf check doesn't catch.
- **gradient-norm clipping**: global-norm clipping as a default-on stabilizer (cheap, standard) across every candidate, not a fallback added only after a trial misbehaves.
- **head-collapse detection**: a head can silently collapse (action head predicting a single class regardless of input; a regression head's output variance collapsing near-zero) while blended val_loss still looks acceptable if that head's [loss weight](#loss-weight-selection) is small — the loss-weight choice is itself a stability risk, not only an accuracy knob. Track per-head prediction-distribution/output-variance alongside per-head loss as a monitored diagnostic (same [experiment-tracking](#experiment-tracking-current-priority) destination already planned for config/metrics logging, not a new tool).
- **recovery/exclusion**: NaN/Inf trials → already pruned by Hyperband. Non-NaN instability (oscillating per-head loss, a collapsed head) → flag and exclude from the finalist multi-seed comparison even if the trial technically completed; a degenerate trial shouldn't quietly enter the seed-averaged backtested-KPI distribution in [model-selection pipeline](#model-selection-pipeline).
  Alt: rely on NaN/Inf pruning alone — rejected, misses the collapsed-head case above, arguably easier to miss than an outright crash.

### training-data sampling strategy

- **within-epoch order — random shuffle**: standard, breaks any residual ordering artifact. Safe across the whole training pool specifically because the split is cross-symbol, not temporal (per [validation & train/test splitting](02-Data, Label & Feature Engineering.md#validation--traintest-splitting)) — there's no train/val temporal adjacency for shuffle order to leak across, so "random vs. chronological" resolves to random without the walk-forward caveats a temporal split would need.
- **regime-awareness / oversampling rare events — already partially answered, not a new lever**: rare-event underrepresentation is already addressed via class-weighted/focal loss and multi-horizon continuous features (per [class imbalance handling](02-Data, Label & Feature Engineering.md#class-imbalance-handling)) and via sample-uniqueness weighting for overlap (per [overlapping labels](02-Data, Label & Feature Engineering.md#overlapping-labels)). Regime-aware/class-balanced **batch sampling** is already named there as "deferred, complementary" — this section doesn't newly adopt it; it stays deferred until loss-reweighting alone is shown insufficient, so the training pool isn't hand-distorted away from real market frequency without evidence it's needed.
- **decision:** uniform random shuffle is the sampling strategy; regime-aware/oversampling remains the already-deferred complementary lever, not promoted here.
  Alt: chronological/curriculum ordering (easy-to-hard by volatility regime) — deferred, unproven necessity, adds complexity with no evidence gap it closes beyond what loss-reweighting already covers.

### training augmentation

- **candidates, financial-series-appropriate**: ATR-scaled jitter (noise injected proportional to local ATR, not raw price — otherwise it fights the [ATR-relative normalization](02-Data, Label & Feature Engineering.md#normalization-strategy) already resolved as primary), window-slicing/cropping, magnitude-warping. Time-warping ties directly to the already-open "pattern speed-invariance" question in [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion): if TCN multi-dilation / attention's flexible receptive field don't already handle same-pattern-different-speed cases architecturally, time-warp augmentation is a training-time complement to test — gated on that architectural check first, same "cheap architectural answer before a new lever" ordering already used there.
- **risk specific to OHLCV**: augmentation must not create unrealistic market behavior — applied post-normalization and scaled relative to the existing ATR-relative scheme, not as raw-price perturbation.
- **decision — default is no augmentation.** The training pool (full historical Binance USDT-pair universe, current + delisted, per [training symbol universe](02-Data, Label & Feature Engineering.md#training-symbol-universe-survivorship)) is large enough that this isn't the small-dataset regime augmentation is normally motivated by. Treated as a parked/Tier-2 candidate, per [prioritization framework](05-Prioritization Framework.md#decision-framework): tested only if a specific symptom appears (a rare regime/class still underrepresented after loss-reweighting + uniqueness-weighting, or measured overfitting), not adopted speculatively.
  Alt: mixup — rejected as a default candidate, unclear semantic meaning for a linear blend of two OHLCV sequences (unlike images/generic tabular rows, an interpolated price path isn't obviously a realistic market state); SMOTE-style synthetic oversampling — already rejected for the adjacent class-imbalance case in [class imbalance handling](02-Data, Label & Feature Engineering.md#class-imbalance-handling) ("awkward for sequential windows"), same reasoning applies here.

## Search & Optimization Strategies

### optimization strategy

- optimization = **one search** across (1) arch/model-combo choice + (2) each arch's hparams, not two disjoint phases.
- architecture = single categorical param inside same Optuna study as hparams (conditional sub-params per arch), not exhaustive — bad archs pruned early instead of full independent sweeps each.
- **manual/hypothesis-driven candidate selection** (which architectures/techniques enter the study at all) via the [Decision Framework](05-Prioritization Framework.md#decision-framework) tiering, **plus Optuna TPE** (sample-efficient, single-GPU budget) for the automated hyperparameter search within that selection — not full NAS/AutoML (out of scope, see [99-Exclusion.md](99-Exclusion.md)).
- Hyperband pruning (successive-halving family) — arch-agnostic, prunes unpromising/NaN-Inf trials.
- GA/NSGA-II for optional 2nd refinement stage.
- Pareto front across competing KPIs (e.g. Sortino vs max-DD).
- per-trial time measured not assumed.
  - runs real training steps per arch, measures wall-clock+peak-VRAM;
  - `estimate_total_budget()`/`max_trials_for_budget()` → projected total + trial-count cap before full study.
- `OptunaPruningCallback` reports val_loss/epoch, prunes Hyperband-unpromising or NaN/Inf trials.
- best-run selection KPI: see [error rating & model evaluation](#backtested-trading-kpis-final-selection) — primary=expectancy, guardrail=max-DD, secondary=Sortino, once the backtest module is built; until then `val_loss` remains the training-time proxy (`compute_fitness()`), explicitly interim not final.
  Alt: grid search — rejected, doesn't scale to this many hyperparameters/architectures; pure random search — rejected, TPE's sample efficiency matters under a constrained single-GPU budget.

## Model & System Alternatives

Different feature sets: candidate pool, screening methodology in [02 § candidate feature pool](02-Data, Label & Feature Engineering.md#candidate-feature-pool).
Different training strategies: [Training Engineering](#training-engineering) → training-strategy selection.
Different label strategies: the MFE/MAE/OM/TP-ladder scheme is resolved in [02 § label design](02-Data, Label & Feature Engineering.md#label-design), but benchmarking it against standard alternatives (triple-barrier, trend-scanning) is an **open gap** ([99-Weakness Analysis.md § A4](99-Weakness%20Analysis.md), Imp 2).
