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
  - [how to use these files](#how-to-use-these-files)

## current focus

**[input data / channels preparation](input-data-channels.md)** — start here. The candle feature schema
in [input-features.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md) is almost entirely unimplemented (the model trains on raw
OHLCV + a fixed technical-indicator set today, none of the spec'd relative-HLC/gap/peak-valley/multi-tf
top-distance fields), and every other topic's input shape depends on this resolving first.

## topics

| topic                                                        | closes the gap in                                                                                                                                                                                                                                                                                                                                                                   | depends on                                                                                                                                       |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [input data / channels preparation](input-data-channels.md)  | [input-features.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md)                                                                                                                                                                                                                                                                                        | —                                                                                                                                                |
| [training data / label preparation](training-data-labels.md) | [training-data.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md)                                                                                                                                                                                                                                                                                         | —                                                                                                                                                |
| [model architecture & selection](model-architecture.md)      | [model-architecture-planning.md](../ML_Forecasting_System_Design/03-Model n Architecture Engineering.md), [model-architecture-candidate-sets.md](../ML_Forecasting_System_Design/03-Model n Architecture Engineering.md#stage-1-candidate-sets), [prioritization-framework.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#decision-framework) | input shape ← input-data; output shape ← training-data-labels                                                                                    |
| [evaluation & backtesting metrics](evaluation-metrics.md)    | [error-rating-and-evaluation.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md)                                                                                                                                                                                                                                                                  | trained candidates ← model-architecture (partially parallelizable — the backtest-module _design_ doc, todo step 1, doesn't need a trained model) |
| [infrastructure & tooling](infrastructure.md)                | [infrastructure.md](../infrastructure.md)                                                                                                                                                                                                                                                                                                                                           | cross-cutting — experiment tracking (step 1-2) should land before the other topics' multi-seed test volume begins                                |

Not a topic file on its own: [timeseries-foundation-models-architecture-planning.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#time-series-foundation-models-tsfms)
covers pretrained-checkpoint TSFMs, explicitly excluded from the from-scratch architecture search — no
implementation TODO until/unless that exclusion is revisited. [prioritization-framework.md](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#decision-framework)
is a scoring methodology consumed by the other topics, not a pipeline stage with its own gap to close.

## suggested sequencing

Not a hard gate — parallelizable where marked, but respects the real dependency chain (input shape and
label shape must stabilize before an architecture search means anything, and the architecture search
needs somewhere to report results before it's worth running at scale):

1. [input data / channels preparation](input-data-channels.md) and
   [training data / label preparation](training-data-labels.md) in parallel — independent of each other,
   both gate everything downstream.
2. [infrastructure.md § experiment tracking](infrastructure.md#todo) (todo steps 1-2 only) — cheap, and
   every subsequent multi-seed test phase benefits from having it in place first.
3. [evaluation & backtesting metrics](evaluation-metrics.md) todo step 1 (the backtest-module _design_
   doc) — can start in parallel with 1-2, doesn't need a trained model yet.
4. [model architecture & selection](model-architecture.md) — once 1 has a stable input shape and
   1 (training-data-labels) has stable output columns.
5. [evaluation & backtesting metrics](evaluation-metrics.md) remaining steps — needs step 4's candidates
   to actually score.

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
