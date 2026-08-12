# TODO — model architecture & selection

Closing the gap between [model-architecture-planning.md](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md) +
[model-architecture-candidate-sets.md](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#stage-1-candidate-sets) +
[prioritization-framework.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#decision-framework) (the architecture-search design) and what's
actually built today. See [master-todo.md](master-todo.md) for how this fits the overall plan — this
topic is downstream of [input-data-channels.md](input-data-channels.md) (input shape) and
[training-data-labels.md](training-data-labels.md) (output shape), so sequence it after those stabilize.

- [TODO — model architecture \& selection](#todo--model-architecture--selection)
  - [todo](#todo)
  - [appendix: current implementation status](#appendix-current-implementation-status)

## todo

1. **Build the unified super-architecture skeleton** (`build_super_architecture(stage_config, profile,
   tf_list)` and its stage functions `embed`/`local_extract`/`sequential_encode`/`attend`/`fuse`/
   `global_repr`/`head`) per
   [model-architecture-candidate-sets.md § unified super-architecture skeleton](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#unified-super-architecture-skeleton).
   Currently pseudocode only — no `stage_config`-driven builder exists in `app/`. This is the
   foundation every candidate below plugs into.
2. **Express the existing CNN-LSTM(-attention) code as a `stage_config`** (`{embedding: 0,
   local_extraction: "plain_cnn", sequential: "rnn", attention: 0 | "self_attn", fusion: "concat_mlp",
   global_repr: "pool"}`) against the new skeleton, without changing its trained behavior — a
   refactor-with-regression-test step (reuse whatever test harness exists for
   `cnn_lstm_model.py`/`cnn_lstm_attention_model.py`; add one first if none exists), proving the
   skeleton reproduces the current baseline before anything new is built on it.
3. Wait for [input-data-channels.md](input-data-channels.md) todo step 9 (architecture-branch decision:
   per-tf-branch vs. flat/shared-encoder) before wiring `embedding: "linear"` + per-tf tf-id embedding —
   the two decisions are coupled; don't build the embedding stage twice.
4. Wait for [training-data-labels.md](training-data-labels.md) todo step 10 (final label/target column
   names) before wiring the `head()` stage's action/MAE-OM/MFE/confidence outputs — head shapes are
   defined by that file, not this one, per
   [training-data.md § model output targets](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#model-output-targets).
5. **Add the naive/persistence baseline** ("no change"/carry-forward last signal) — not a `stage_config`
   at all, the mandatory floor every learned candidate must beat. Cheapest possible first thing to wire
   once the eval harness in [evaluation-metrics.md](evaluation-metrics.md) can score it.
6. **Add the Tier-1 Stage-1 candidates from
   [prioritization-framework.md § current Stage-1 candidate set](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#current-stage-1-candidate-set)**
   not yet in code, each as its own `stage_config` + pseudocode block already drafted in
   [model-architecture-candidate-sets.md § architecture candidates](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#architecture-candidates):
   Transformer w/ per-tf embedding + cross-tf attention, TCN (with ModernTCN as the same block's
   large-kernel/grouped-conv parameterization), hybrid CNN→Transformer, Mamba (flag the
   non-native-Keras `MambaBlock` dependency up front), LSTM floor (GRU as the Tier-2 alt within it),
   TSMixer, DLinear. Land as separate PRs/commits, not one mega-change — each is independently
   testable against the skeleton from step 1.
7. **Wire `profile_trial_cost()`/`estimate_total_budget()`/`max_trials_for_budget()`** (already exist in
   `optuna_optimizer.py`, see appendix) against each new candidate as it lands in step 6, per
   [model-architecture-candidate-sets.md § hardware constraints](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#hardware-constraints)
   — measure real wall-clock/VRAM, don't hand-estimate, before adding it to the live Optuna study.
8. **Add architecture as a categorical Optuna dimension** across all landed Stage-1 candidates in the
   same study as their hyperparameters (conditional sub-params per arch), per
   [model-architecture-planning.md § optimization strategy](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#optimization-strategy).
   Confirm `compute_fitness()`/`OptunaPruningCallback` (both already exist, see appendix) generalize
   across the new candidates' output shapes, not just the current CNN-LSTM one.
9. **(decision) Global budget/sequencing plan.** Counting Tier-1 rows across
   [prioritization-framework.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#decision-framework): 9 full Stage-1 architectures, 4
   activation functions, 6 GBM-family techniques, 2 embedding options, 2 global-repr options, 2
   multi-tf fusion options — each gated behind its own ≥3-seed statistical-validity protocol (see
   [evaluation-metrics.md](evaluation-metrics.md)). `estimate_total_budget()` exists per-study; nothing
   rolls the other tracks (normalization → activation → fusion → GBM screen...) into one ordered plan
   with a total wall-clock estimate against the single 8GB-laptop-GPU budget. Write a short roadmap:
   ordered test phases, dependency arrows between them, estimated GPU-hours per phase, running total.
10. **Normalization-scheme test.** ATR-relative is the resolved default; the alt schemes (log-return,
    rolling z-score, hybrid ATR+log-return) are untested per
    [model-architecture-planning.md § normalization strategy](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#normalization-strategy).
    Run once step 1's skeleton exists, using the ≥3-seed protocol from
    [evaluation-metrics.md § statistical validity](evaluation-metrics.md#todo).
11. **Class-imbalance prevalence measurement** — actual prevalence (% candles peak/valley per horizon,
    % positions clearing `OM > 1`) isn't known yet; measure empirically via a data-profiling script once
    [training-data-labels.md](training-data-labels.md) lands, before finalizing the class-weight/focal
    choice in [model-architecture-planning.md § class imbalance handling](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#class-imbalance-handling).
12. **(decision) Cross-symbol validation split — potential leakage via shared calendar time.**
    [model-architecture-planning.md § validation & train/test splitting](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#validation--traintest-splitting)
    argues train-on-other-pairs / validate-on-BTC needs no walk-forward machinery since there's no
    same-symbol window overlap. But crypto pairs are heavily correlated on shared calendar time — if
    training-pair windows overlap the same weeks as the BTC validation window, the model can learn
    market-wide regime structure (a specific crash/pump) from training pairs and "predict" it on BTC for
    free, inflating validation KPIs without genuine skill. Decide and document either (a) time-disjoint
    training-pair windows from the BTC validation window, or (b) an explicit accepted-risk note plus a
    diagnostic bounding the leakage (compare validation KPIs against a rotated-symbol control, per the
    deferred "rotating leave-one-symbol-out" alt already named in that section).
13. **Resolve combination-strategy status.** Currently "unresolved, not yet measured, default =
    single-backend-wins" per
    [model-architecture-planning.md § combination strategy](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#combination-strategy).
    Only revisit once step 6's single-backend candidates have measured backtested-KPI results to compare
    a combo against — premature before then.
14. **Activation-mechanism sweep** (GELU/GLU-family/ReLU/SiLU — Tier 1 per
    [prioritization-framework.md § activation mechanisms](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#activation-mechanisms))
    — cheap post-hoc refinement within the Stage-1 categorical-search winner from step 8, not folded
    into the primary search. Sequence last, after a winner exists.

## appendix: current implementation status

Verified against `app/` directly on 2026-08-12.

- **CNN-LSTM(-attention)** — the only implemented Stage-1 candidate:
  [cnn_lstm_model.py](../../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) (no attention) and
  [cnn_lstm_attention_model.py](../../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py)
  (with attention) — plain (non-causal, non-dilated) `Conv1D` stack → LSTM stack → (attention variant
  only) self-attention → pooling → dense heads, per branch. No `stage_config`/skeleton abstraction
  wraps them; they're standalone Keras model-building functions.
- **No other Stage-1 candidate exists in code** — Transformer, TCN/ModernTCN, hybrid CNN→Transformer,
  Mamba/S4, TSMixer/DLinear, TFT, Perceiver are all planning-doc-only.
- **Optuna integration** — [optuna_optimizer.py](../../app/ai_modelling/parameter_optimizser/optuna_optimizer.py)
  implements `profile_trial_cost()`, `estimate_total_budget()`, `max_trials_for_budget()`,
  `compute_fitness()`, and `run_kpi_refinement()` (confirmed present by direct grep — all four names the
  planning docs reference actually exist as functions, not just described intentions). Whether
  `compute_fitness()` generalizes beyond the CNN-LSTM's current output shape hasn't been checked — see
  todo step 8.
- **GA optimizer** — [ga_optimizer.py](../../app/ai_modelling/parameter_optimizser/ga_optimizer.py) also
  exists alongside the Optuna path; not yet cross-referenced against
  [model-architecture-planning.md § optimization strategy](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#optimization-strategy)'s
  "GA/NSGA-II for optional 2nd refinement stage" framing — worth a quick audit alongside todo step 8 to
  confirm it's the same mechanism the docs describe, not a parallel/superseded path.
- **No unified super-architecture skeleton** (`stage_config`-driven `build_super_architecture()`) exists
  — [model-architecture-candidate-sets.md § unified super-architecture skeleton](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#unified-super-architecture-skeleton)
  is pseudocode only, not yet transcribed into `app/`.
- **Deployment/live layer is essentially absent, and the one thing that exists is disconnected.**
  `BasePatternStrategy` (see
  [training-data-labels.md § secondary mechanism](training-data-labels.md#secondary-unrelated-mechanism-livebacktest-bracket-orders))
  is the only live/backtest order-placement code that exists, and it's unrelated to the anchor-candle ML
  labels. No documented or implemented path from "trained model produces a prediction" to "an order gets
  placed." Compounding this, transaction costs/spread/slippage/latency, risk/position sizing beyond TP
  targets, and market-regime/retraining cadence are all explicitly deferred in
  [model-architecture-planning.md § deferred topics](../ML_Forecasting_System_Design/99-Exclusion.md).
  Reasonable individually during architecture search; together they mean nothing describes how a model
  result becomes a live position. Before any live/paper trading, promote these three deferred topics to
  real docs — cost-free backtest KPIs will otherwise overstate performance at a 4H horizon where fees are
  modeled but slippage isn't. Not scheduled as a numbered todo step above since it's gated behind Stage-1
  search finishing first, but flagged here so it isn't forgotten.
