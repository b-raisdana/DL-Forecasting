# TODO — evaluation & backtesting metrics

Closing the gap between [error-rating-and-evaluation.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md) (the metrics
philosophy: per-head dev diagnostics vs. backtested trading KPIs as the real target) and what actually
exists to compute either. See [master-todo.md](master-todo.md). This is the single biggest structural
gap in the whole project: every model-selection decision elsewhere is provisionally routed through
`val_loss` because the thing that's supposed to replace it doesn't exist yet.

- [TODO — evaluation \& backtesting metrics](#todo--evaluation--backtesting-metrics)
  - [todo](#todo)
  - [appendix: current implementation status](#appendix-current-implementation-status)

## todo

1. **Write a `backtest-module-design.md`** before any code: fill-simulation assumptions (matching
   [training-data.md § targeting bid price](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#targeting-bid-price)'s limit-order
   entry model once that lands, see [training-data-labels.md](training-data-labels.md)), position
   accounting across overlapping label windows (multiple anchor candles can each open a hypothetical
   position — decide how concurrent/overlapping simulated positions are accounted for, capital
   allocation per position), and walk-forward mechanics against the resolved
   [validation & train/test splitting](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#validation--traintest-splitting)
   scheme. Do this before sinking GPU-hours into Stage-1 profiling
   ([model-architecture.md](model-architecture.md)) that this module will eventually have to re-judge.
2. **Integrate `vectorbt`** (named but "not yet integrated, no imports in codebase" per
   [infrastructure.md § vectorbt](infrastructure.md#todo)) as the backtest engine, or explicitly decide
   against it and name the alternative — currently an open placeholder, not a resolved choice.
3. **Implement `expectancy/trade`** (primary KPI) from simulated trades — average R-multiple/% gained
   per trade opportunity including losers, per
   [error-rating-and-evaluation.md § backtested trading KPIs](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#backtested-trading-kpis-final-selection).
4. **Implement `max-DD`** (guardrail) — reject any config over an acceptable drawdown/risk tolerance
   regardless of other numbers. Needs to gate model selection, not just report a number.
5. **Implement `Sortino`** (secondary, ranks guardrail-passers) — downside-volatility-only return ratio.
6. **Implement the diagnostics-only pair** (win rate, profit factor) — cheap, each individually gameable,
   reported alongside the KPIs above but never used alone for selection.
7. **Wire per-head statistical metrics** (dev diagnostics, not selection criteria) per
   [error-rating-and-evaluation.md § per-head statistical metrics](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics):
   quantile/pinball loss + MAE/MSE companion for price-level heads (TP/SL/MAE/OM/aux-MFE once
   [training-data-labels.md](training-data-labels.md) lands), Brier vs. log-loss for
   probability/confidence heads, cross-entropy vs. focal vs. class-weighted-CE for the action head.
8. **Replace the `val_loss` interim proxy in `compute_fitness()`** (see appendix) with the weighted-sum
   per-head metric from step 7 once it exists, still explicitly interim until the backtest module (steps
   1-6) is wired into `run_kpi_refinement()`'s objective.
9. **Build the statistical-validity harness**: ≥3 seeds/config (5 preferred), paired stat test across
   matched folds (paired t-test/Wilcoxon) requiring a confidence interval excluding zero, comparing
   backtested-KPI distributions never a single train-loss number, per
   [error-rating-and-evaluation.md § statistical validity of comparisons](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons).
   This is the shared discipline every A/B decision across
   [model-architecture.md](model-architecture.md) and
   [input-data-channels.md](input-data-channels.md) depends on — build once, reuse everywhere, don't
   let each topic reinvent it.
10. **Wire the model-selection pipeline end to end**: search-time per-head losses → finalist re-run
    across ≥3 seeds → backtested-KPI comparison → BTC/USDT validation selection → final holdout run
    exactly once, per
    [error-rating-and-evaluation.md § model-selection pipeline](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#model-selection-pipeline).
    Currently only the first stage (search-time loss) is implemented (see appendix); stages 2-4 have no
    code.
11. **(decision) Confidence/calibration metric technique.** No input feature carries confidence
    information today. Pick a technique from
    [model-architecture-planning.md § uncertainty-native GBM variants](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#uncertainty-native-gbm-variants--confidence-metric-gap)
    (quantile GBM first — reuses the pinball-loss work from step 7 — then NGBoost) and implement Brier
    score / log-loss / calibration-curve (ECE) measurement against it, per
    [error-rating-and-evaluation.md § confidence & calibration metrics](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#confidence--calibration-metrics).
12. **Add the no-lookahead regression test to CI**, once written in
    [input-data-channels.md](input-data-channels.md) todo step 6 and
    [training-data-labels.md](training-data-labels.md) todo step 11 — wire both into whatever gate
    `xenon` already runs, per [infrastructure.md](infrastructure.md), so this stays a permanent CI check
    rather than a one-off script.

## appendix: current implementation status

Verified against `app/` directly on 2026-08-12.

- **`compute_fitness()`** ([optuna_optimizer.py](../../app/ai_modelling/parameter_optimizser/optuna_optimizer.py) —
  confirmed present by direct grep, see [model-architecture.md](model-architecture.md)'s appendix) is
  the only scoring mechanism that exists. Per
  [model-architecture-planning.md § optimization strategy](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#optimization-strategy),
  it's `val_loss`-based — the training-time proxy, explicitly interim per the docs' own framing, not the
  real expectancy/Sortino/max-DD target.
- **No backtest module exists anywhere in the codebase.** `vectorbt` is listed in
  [infrastructure.md § vectorbt](../infrastructure.md#vectorbt--not-yet-integrated) as "planned... not
  in `requirements.txt`, no imports in codebase yet." Every selection decision described in
  [error-rating-and-evaluation.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md) is therefore aspirational, not
  implemented — a fully worked-out evaluation *philosophy* with no evaluation *mechanism*.
- **No per-head metric wiring exists** — no quantile/pinball loss, no Brier/log-loss, no per-head
  cross-entropy/focal split. The model trains against whatever single loss Keras is configured with for
  `long_signal`/`short_signal` (see [training-data-labels.md](training-data-labels.md) appendix), not a
  per-head breakdown.
- **No confidence/calibration mechanism exists** — matches the "open gap" framing already in
  [error-rating-and-evaluation.md § confidence & calibration metrics](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#confidence--calibration-metrics)
  verbatim; nothing to add here beyond confirming it's still true.
- **`run_kpi_refinement()`** exists in `optuna_optimizer.py` (confirmed by grep) but its objective
  function has not been checked against whether it's already wired to anything beyond `val_loss` — flag
  for todo step 8/10 to confirm rather than assume either way.
