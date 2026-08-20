# Master TODO

Index of every implementation gap between this project's planning docs (`docs/*.md`) and what's actually
built in `app/`, split into one file per topic so each stays readable at a glance (~1 page,
10-20 steps) instead of one growing monolith. Each topic file leads with its **todo** (the plan — read
this) and keeps a verified **current implementation status** appendix at the end (reference material,
not the point of the file).

This replaces `docs/current-code.md` and `docs/planning-weak-points-todo.md`, both retired — their
content is fully distributed across the topic files below, not lost.

- [Master TODO](#master-todo)
  - [current focus](#current-focus)
  - [topics](#topics)
  - [suggested sequencing](#suggested-sequencing)
  - [execution checklist (post-prep)](#execution-checklist-post-prep)
  - [how to use these files](#how-to-use-these-files)

## current focus

**[input data / channels preparation](01-input-data-channels.md)** — start here. The candle feature schema
in [input-features.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md) is almost entirely unimplemented (the model trains on raw
OHLCV + a fixed technical-indicator set today, none of the spec'd relative-HLC/gap/peak-valley/multi-tf
top-distance fields), and every other topic's input shape depends on this resolving first.

## topics

| topic                                                        | closes the gap in                                                                                                                                                                                                                                                                                                                                                                   | depends on                                                                                                                                       |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [input data / channels preparation](01-input-data-channels.md)  | [input-features.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md)                                                                                                                                                                                                                                                                                        | —                                                                                                                                                |
| [training data / label preparation](02-training-data-labels.md) | [training-data.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md)                                                                                                                                                                                                                                                                                         | —                                                                                                                                                |
| [model architecture & selection](04-model-architecture.md)      | [model-architecture-planning.md](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md), [model-architecture-candidate-sets.md](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#stage-1-candidate-sets), [prioritization-framework.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#decision-framework) | input shape ← input-data; output shape ← training-data-labels                                                                                    |
| [evaluation & backtesting metrics](05-evaluation-metrics.md)    | [error-rating-and-evaluation.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md)                                                                                                                                                                                                                                                                  | trained candidates ← model-architecture (partially parallelizable — the backtest-module _design_ doc, todo step 1, doesn't need a trained model) |
| [infrastructure & tooling](03-infrastructure.md)                | [infrastructure.md](../infrastructure.md)                                                                                                                                                                                                                                                                                                                                           | cross-cutting — experiment tracking (step 1-2) should land before the other topics' multi-seed test volume begins                                |

Not a topic file on its own: [timeseries-foundation-models-architecture-planning.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#time-series-foundation-models-tsfms)
covers pretrained-checkpoint TSFMs, explicitly excluded from the from-scratch architecture search — no
implementation TODO until/unless that exclusion is revisited. [prioritization-framework.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#decision-framework)
is a scoring methodology consumed by the other topics, not a pipeline stage with its own gap to close.

## suggested sequencing

Not a hard gate — parallelizable where marked, but respects the real dependency chain (input shape and
label shape must stabilize before an architecture search means anything, and the architecture search
needs somewhere to report results before it's worth running at scale):

1. [input data / channels preparation](01-input-data-channels.md) and
   [training data / label preparation](02-training-data-labels.md) in parallel — independent of each other,
   both gate everything downstream.
2. [infrastructure.md § experiment tracking](03-infrastructure.md#todo) (todo steps 1-2 only) — cheap, and
   every subsequent multi-seed test phase benefits from having it in place first.
3. [evaluation & backtesting metrics](05-evaluation-metrics.md) todo step 1 (the backtest-module _design_
   doc) — can start in parallel with 1-2, doesn't need a trained model yet.
4. [model architecture & selection](04-model-architecture.md) — once 1 has a stable input shape and
   1 (training-data-labels) has stable output columns.
5. [evaluation & backtesting metrics](05-evaluation-metrics.md) remaining steps — needs step 4's candidates
   to actually score.

## execution checklist (post-prep)

Once the prep gaps above (sequencing steps 1-3) are closed, this is the actual run loop — architecture
through KPI report. Corrects a naive "pick architecture first" ordering: architecture is chosen _for_ a
locked input/output shape, not the reverse (same dependency [suggested sequencing](#suggested-sequencing)
step 4 already states). Also splits "monitor resource usage" into a pre-flight profiling step and
continuous in-run monitoring — a single post-hoc check can't catch OOM/instability while it's happening.

1. **input/label shape locked** — output of steps 1-3 above: input channels
   ([01-input-data-channels.md](01-input-data-channels.md)) and labels
   ([02-training-data-labels.md](02-training-data-labels.md)) finalized, priority-ordered per
   [02](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md). Architecture can't be
   meaningfully chosen before this.
2. **architecture design set** — from that shape, take the priority-tiered candidates already scored in
   [03 § current Stage-1 candidate set](../ML_Forecasting_System_Design/03-Model & Architecture Engineering.md#current-stage-1-candidate-set)
   / [04 § tiered candidates by layer](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#tiered-candidates-by-layer)
   (Tier 1 first: TCN/ModernTCN, LSTM floor, CNN-LSTM, Transformer, TSMixer, DLinear, Mamba, Perceiver,
   naive baseline).
3. **data-feeding gap check** — confirm the dataset-generation pipeline actually produces the locked
   input/label shape end-to-end for that design set (remaining steps in input-data-channels.md /
   training-data-labels.md) — prerequisite engineering, not a search-time concern.
4. **pre-flight resource profiling** — before the full search, `profile_trial_cost()` measures real
   wall-clock/VRAM per arch+hparam combo on the actual GPU; `estimate_total_budget()`/
   `max_trials_for_budget()` derive the trial cap and hyperparameter search-space bounds (per
   [04 § hyperparam search-space bounds](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds))
   — not skippable, feeds step 5's budget.
5. **start training / search** — one Optuna+Hyperband study across the whole design set (architecture as
   a categorical param, not N separate sweeps), within the step-4 budget (per
   [04 § optimization strategy](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#optimization-strategy)).
6. **monitor resource usage + stability during the run, not after** — `OptunaPruningCallback` tracks
   val_loss/epoch plus NaN/Inf pruning; watch actual VRAM/wall-clock against the step-4 profiled estimate
   (catches profiler drift); track per-head loss/output-variance for silent head-collapse (per
   [04 § training stability](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#training-stability)).
7. **report KPI metrics of the design set** — per finalist (≥3 seeds, per
   [04 § statistical validity of comparisons](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons)),
   report backtested trading KPIs: expectancy/trade (primary), max-DD (guardrail), Sortino (secondary
   among guardrail-passers) — never a bare val_loss number (per
   [04 § model-selection pipeline](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#model-selection-pipeline)).

## how to use these files

- Numbered steps are ordered by dependency within their own file — treat them as a queue, not a menu.
- Steps marked **(decision)** need a one-line confirmation from the project owner before implementing
  (they change cross-cutting behavior — label shape, input shape, window size); everything else is a
  direct, self-contained fix against an already-written spec, safe to hand to an agentic coding session
  without further discussion.
- When a step is actually done (doc/code updated, not just discussed), delete it from the todo list and,
  if anything about it is worth remembering, fold a one-line note into that file's appendix instead of
  leaving a growing "closed items" archive.
- New gaps found later (code review, a failed experiment, a fresh doc pass) get added to whichever
  topic file they belong to, in the same `finding → action` shape already used throughout — not left as
  a comment or a one-off note that won't survive context compression.
