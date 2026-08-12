# Error Rating & Model Evaluation

How this project measures whether a prediction is good: per-head statistical error during training/dev, confidence/calibration metrics, and the backtested trading KPIs that make the final call. This doc is the single destination for every error/metric/KPI decision; [model-architecture-planning.md](model-architecture-planning.md), [prioritization-framework.md](prioritization-framework.md), and [training-data.md](training-data.md) keep only pointers back here.

- [Error Rating \& Model Evaluation](#error-rating--model-evaluation)
  - [why a separate doc](#why-a-separate-doc)
  - [core principle: error metric ≠ trading objective](#core-principle-error-metric--trading-objective)
  - [per-head statistical metrics (dev diagnostics)](#per-head-statistical-metrics-dev-diagnostics)
  - [confidence \& calibration metrics](#confidence--calibration-metrics)
  - [backtested trading KPIs (final selection)](#backtested-trading-kpis-final-selection)
  - [statistical validity of comparisons](#statistical-validity-of-comparisons)
  - [model-selection pipeline](#model-selection-pipeline)
  - [terminology: "MAE" means two different things here](#terminology-mae-means-two-different-things-here)
  - [glossary](#glossary)
  - [related docs](#related-docs)

## why a separate doc

A reader asking "what's our error metric" needs one destination, not scattered bullets across the architecture, prioritization, and training-data docs. This doc is that destination; other docs keep only a pointer plus whatever is genuinely local to them (e.g. class-imbalance-specific loss tuning stays in [model-architecture-planning.md § class imbalance handling](model-architecture-planning.md#class-imbalance-handling), since that's about label rarity, not metric definition).

## core principle: error metric ≠ trading objective

Low statistical loss does not imply profitability. Per-head statistical metrics are a **training-time signal only** — they drive gradient descent and Optuna's interim objective — never the final model-selection criterion. Final selection is always the backtested trading KPIs below, computed from actual simulated trades derived from TP/SL predictions. Every metric choice in this doc traces back to that split: dev diagnostic, or selection criterion — never both.

## per-head statistical metrics (dev diagnostics)

No single blended loss — measured per output head, since each has a different error shape:

| head                                      | candidates to test                                  | notes                                                                                                                                            |
| ----------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| price levels (TP, SL / MAE, OM, aux MFE)  | quantile/pinball loss (primary) + MAE/MSE companion | quantile loss reused by [uncertainty-native GBM variants](model-architecture-planning.md#uncertainty-native-gbm-variants--confidence-metric-gap) |
| probabilities / confidence                | Brier vs log-loss                                   | see [confidence & calibration metrics](#confidence--calibration-metrics)                                                                         |
| action (Long/Short/None)                  | cross-entropy vs focal vs class-weighted-CE         | tuning scope (which loss, per-target gamma) lives in [class imbalance handling](model-architecture-planning.md#class-imbalance-handling)         |

Per-head metrics feed Optuna's scalar objective only as a **weighted-sum interim proxy** (matches the existing `val_loss` use in `compute_fitness()`) until the backtest module exists — real selection always stays at the backtested-KPI stage below.
Alt: single blended loss — rejected, already flagged insufficient. Per-head multi-objective Optuna — more complex, deferred (single-GPU budget). AUC-ROC — viable companion/secondary diagnostic to F1, not primary.

## confidence & calibration metrics

**Open gap:** a confidence metric is needed alongside every forecast, but no input feature carries confidence information today (see [training-data.md § model output targets](training-data.md#model-output-targets)).

Calibration is measured the same way as any probability head above — Brier score, log-loss, and a **calibration curve / Expected Calibration Error (ECE)** to check predicted vs. observed hit-rate, not just point accuracy.

The preferred fix is a model that produces calibrated uncertainty as a mechanism of itself, not via extra engineered features. Candidate techniques (quantile GBM, NGBoost, quantile regression forests) and their priority order live in [model-architecture-planning.md § uncertainty-native GBM variants](model-architecture-planning.md#uncertainty-native-gbm-variants--confidence-metric-gap), scored in [prioritization-framework.md § auxiliary tabular models](prioritization-framework.md#auxiliary-tabular-models-gbm-family) — this doc owns _what_ to measure and why it matters, that section owns _which technique_ produces it.

## backtested trading KPIs (final selection)

Computed from actual simulated trades derived from TP/SL predictions, on the BTC/USDT validation split, then the untouched final holdout (see [model-architecture-planning.md § validation & train/test splitting](model-architecture-planning.md#validation--traintest-splitting)).

- **primary — expectancy/trade** (R-multiples or %): real profitability per opportunity, less sensitive to trade frequency than profit factor.
- **guardrail — max-DD**: reject any config over acceptable drawdown/risk tolerance regardless of other numbers.
- **secondary (among guardrail-passers) — Sortino**: doesn't penalize upside vol, fits an asymmetric TP strategy, ranks finalists.
- **diagnostics only (each gameable alone)** — win rate, profit factor.
- **longer-term plan** — NSGA-II multi-objective Pareto front (e.g. Sortino vs max-DD) across competing KPIs via `run_kpi_refinement()`, once the backtest module exists; single-primary-KPI is the interim until then. Until the backtest module is built, `val_loss` remains the training-time proxy (`compute_fitness()`) — explicitly interim, not final.

Alt: Sharpe as primary — rejected, penalizes wanted upside vol. Calmar as primary — viable, not chosen; kept adjacent to the max-DD guardrail. Profit factor as primary — rejected, ignores trade frequency/opportunity cost.

## statistical validity of comparisons

Applies to every A/B decision in this project's docs (normalization scheme, activation function, architecture candidate, class-weight/focal choice) — one shared discipline so results aren't noise:

- **min 3 seeds/config**, 5 preferred if budget allows.
- **paired stat test** across matched folds (paired t-test / Wilcoxon) — require a confidence interval excluding zero, not eyeballed means.
- compare **backtested-KPI distributions**, never a single train-loss number.
- reserve multi-seed re-run budget for **top finalists post-search only** (too expensive per-trial during search) — factor into `estimate_total_budget()`/`max_trials_for_budget()`.

Alt: single seed — rejected, can't separate signal from noise. Bootstrap resampling of the val set — cheaper, complementary, combinable with the 3-seed approach. 10+ seeds per candidate during search — rejected, too expensive; finalists-only instead.

## model-selection pipeline

How the pieces above chain together, end to end:

1. **during search** — per-head losses (weighted-sum interim proxy, or plain `val_loss` pre-backtest-module) drive Optuna/Hyperband pruning. Dev diagnostics only.
2. **finalists** — re-run top configs across ≥3 seeds (see [statistical validity](#statistical-validity-of-comparisons)), compare backtested-KPI distributions: expectancy primary, max-DD guardrail, Sortino secondary.
3. **selection** — best finalist on the BTC/USDT validation split.
4. **final holdout** — run exactly once, after everything (arch/hparams/normalization/threshold) is locked in. A materially worse holdout result than validation is an overfitting-to-tuning signal → investigate, don't re-tune against it (that would require a fresh holdout).

## terminology: "MAE" means two different things here

- **Here (statistical):** Mean Absolute Error — a companion loss for price-level heads, alongside quantile/pinball loss.
- **In [training-data.md](training-data.md#glossary) and [todos/training-data-labels.md](todos/training-data-labels.md#what-drawdown-actually-measures-mae-not-peak-retracement) (trading):** Maximum Adverse Excursion — the worst adverse price move from entry before the best-case exit, used to derive SL/labels. Unrelated to the statistical metric above; always read from context.

## glossary

KPI/metric terms specific to this doc; general project terms (ATR, tf-ordered-list, Stage-1, S1/S2/S3, GBM) live in [model-architecture-planning.md § glossary](model-architecture-planning.md#glossary).

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

## related docs

- [model-architecture-planning.md](model-architecture-planning.md) — architecture candidates and where each metric plugs into search/pruning
- [prioritization-framework.md](prioritization-framework.md) — technique scoring, including GBM-family uncertainty candidates
- [training-data.md](training-data.md) — label design (TP/SL/MAE-as-trading-term) that these metrics are computed against
