---
name: code-layers
description: Use when adding a new module/file under app/, or deciding where code belongs, or reviewing whether a change crosses layer boundaries correctly. Applies this repo's DDD-flavored layering (Domain/Application/Infrastructure/Presentation) — not MVC, this is an offline data/ML pipeline with no request/response cycle.
---

# Code layers

Layer by dependency direction (Clean/Onion-style). Domain depends on nothing in-house. Arrows point
inward. (The directory-rename/cleanup migration this layering implies is tracked as a todo, not repeated
here — see [docs/todos/infrastructure.md](../../../docs/todos/infrastructure.md).)

## the 4 layers, outside in

1. **Presentation** — plotting, entrypoints, notebooks. Depends on Application.
2. **Application** — use-case orchestration: dataset generation, training, prediction,
   optimization, backtesting. Depends on Domain + Infrastructure.
3. **Infrastructure** — I/O/framework adapters: exchange/data fetch, model-artifact persistence,
   config, logging, TF model definitions. Depends on Domain schemas only.
4. **Domain** — market-structure TA algorithms (pure DataFrame→DataFrame, no I/O) + PanderaDFM
   schemas (value-object contracts). Depends on nothing in-house.

## placing new code

- Pure transform over an already-validated DataFrame, no I/O, no TF/plotly import → **Domain**
  (alongside `Model/TechnicalAnalysis/*`), or a new PanderaDFM schema if it defines a new
  cross-boundary shape.
- Coordinates multiple domain steps + reads/writes data for one job (build a dataset, train a
  model, run a backtest, run a hyperparameter search) → **Application**
  (`ai_modelling/dataset_generator|training|predicting|parameter_optimizser`, `Strategy/*`).
- Touches disk/exchange/GPU/config/logging directly → **Infrastructure** (`data_processing/*`,
  `Config.py`, `helper/br_py/*`). New persisted-artifact types go through a repository
  (`get`/`save` contract, see [infrastructure.md § Repository design
  pattern](../../../docs/infrastructure.md#repository-design-pattern)), not inline file/CCXT calls.
- Produces a plot, print, or CLI entrypoint → **Presentation** (`FigurePlotter/*`, `main.py`-style
  scripts).

## DDD concepts mapped

- **Value objects** = PanderaDFM schemas — no persistent identity (a candle/pivot isn't "the same
  entity" across recomputation); pandera enforces shape/dtype at every hand-off.
- **Domain services** = the TA transform chain (PeakValley → BullBearSide → BasePattern → ATR/Classic
  pivots → ftc), each a pure DataFrame→DataFrame step, no side effects.
- **Application services** = orchestrators calling domain services + repositories for one use case
  (a trainer, a predictor, an optimizer, a `Strategy`). New pipelines are new application services, not
  new domain logic.
- **Bounded contexts**, sharing the OHLCV/PanderaDFM kernel: *market structure* (TA pattern detection +
  backtesting — `Model`, `Strategy`) and *forecasting* (dataset generation, training, prediction,
  optimization — `ai_modelling/*`).

## rules

- Never import Presentation or Application from Domain. (Known existing violation:
  `Model/TechnicalAnalysis/PeakValley.py` imports a constant from `FigurePlotter/plotter.py` —
  don't copy that pattern; the fix is moving the constant to `Config.py`.)
- Don't add a second copy of a module in a new location "for the new pipeline" — this repo already
  has 3 such forks (`PreProcessing`, `predicting`, `training` each duplicated under
  `app/` top-level and `app/ai_modelling/`); extend/move the existing one instead of forking.
- Config/shared state passed into constructors/functions explicitly, not pulled from
  `app_config` mid-function — see [infrastructure.md § Dependency
  injection](../../../docs/infrastructure.md#dependency-injection).
