---
name: code-layers
description: Use when adding a module/file under app/, deciding where code belongs, or reviewing whether a change crosses layer boundaries. Applies this repo's DDD-flavored layering (Domain/Application/Infrastructure/Presentation) — not MVC; this is an offline data/ML pipeline, no request/response cycle.
---

# Code layers

Layer by dependency direction (Clean/Onion-style): Domain depends on nothing in-house, arrows point inward. (The directory-rename/cleanup migration this implies is tracked in [docs/todos/infrastructure.md](../../../docs/todos/03-infrastructure.md), not repeated here.)

## the 4 layers, outside in

1. **Presentation** — plotting, entrypoints, notebooks. Depends on Application.
2. **Application** — use-case orchestration: dataset generation, training, prediction, optimization, backtesting. Depends on Domain + Infrastructure.
3. **Infrastructure** — I/O/framework adapters: exchange/data fetch, model-artifact persistence, config, logging, TF model definitions. Depends on Domain schemas only.
4. **Domain** — market-structure TA algorithms (pure DataFrame→DataFrame, no I/O) + PanderaDFM schemas (value-object contracts). Depends on nothing in-house.

## placing new code

- Pure transform over an already-validated DataFrame, no I/O, no TF/plotly import → **Domain** (alongside `Model/TechnicalAnalysis/*`), or a new PanderaDFM schema if it defines a new cross-boundary shape.
- Coordinates multiple domain steps + reads/writes data for one job (build a dataset, train a model, run a backtest, run a hyperparameter search) → **Application** (`ai_modelling/dataset_generator|training|predicting|parameter_optimizser`, `Strategy/*`).
- Touches disk/exchange/GPU/config/logging directly → **Infrastructure** (`data_processing/*`, `Config.py`, `helper/br_py/*`). New persisted-artifact types go through a repository (`get`/`save` contract, see [infrastructure.md § Repository design pattern](../../../docs/infrastructure.md#repository-design-pattern)), not inline file/CCXT calls.
- Produces a plot, print, or CLI entrypoint → **Presentation** (`FigurePlotter/*`, `main.py`-style scripts).

## DDD concepts mapped

- **Value objects** = PanderaDFM schemas — no persistent identity (a candle/pivot isn't "the same entity" across recomputation); pandera enforces shape/dtype at every hand-off.
- **Domain services** = the TA transform chain (PeakValley → BullBearSide → BasePattern → ATR/Classic pivots → ftc), each a pure DataFrame→DataFrame step, no side effects.
- **Application services** = orchestrators calling domain services + repositories for one use case (a trainer, a predictor, an optimizer, a `Strategy`). New pipelines are new application services, not new domain logic.
- **Bounded contexts**, sharing the OHLCV/PanderaDFM kernel: *market structure* (TA pattern detection + backtesting — `Model`, `Strategy`) and *forecasting* (dataset generation, training, prediction, optimization — `ai_modelling/*`).

## rules

- Never import Presentation or Application from Domain. (Known violation: `Model/TechnicalAnalysis/PeakValley.py` imports a constant from `FigurePlotter/plotter.py` — don't copy it; fix is moving the constant to `Config.py`.)
- Don't fork a second copy of a module for a new pipeline — this repo already has 3 (`PreProcessing`, `predicting`, `training`, each duplicated under `app/` top-level and `app/ai_modelling/`); extend/move the existing one instead.
- Pass config/shared state explicitly into constructors/functions, not pulled from `app_config` mid-function — see [infrastructure.md § Dependency injection](../../../docs/infrastructure.md#dependency-injection).

## splitting an oversized file

File size policy:

| File size | Interpretation | Action |
| --- | --- | --- |
| `<300` lines | Normal | Keep new/generated files below this size. When modifying an existing file, prefer moving the touched method/function to its proper layer/responsibility module if that naturally reduces the file. |
| `300-500` lines | Potential low-priority split todo | Do not split by raw line count, but note cohesive responsibility clusters worth extracting later. |
| `>500` lines | Warning | High-priority split todo; extract one cohesive cluster at a time along the layer/responsibility boundaries above. |

The `loc` pre-commit ratchet vector
([infrastructure.md § pre-commit](../../../docs/infrastructure.md#pre-commit)) tracks total
excess-over-500 lines project-wide and blocks a commit that makes it worse, but it never requires
fixing a whole file's debt in one pass — same regression-only-blocks philosophy as the
`mypy`/`ruff`/`xenon` vectors it sits alongside
([incremental-precommit README](../../../scripts/git-hooks/incremental-precommit/README.md)).

Apply that same discipline yourself when doing the actual split, in this repo and in general:

- Extract **one cohesive cluster** per commit (functions that share a single responsibility and
  ideally have few/no external callers) — not a full rewrite of the file in one pass. Check
  callers first (`grep`/`Grep` for the function names); a cluster with zero or few external callers
  is the safest place to start.
- If the extracted names are used outside the original file, re-export them from it
  (`from new_module import name as name` — the explicit self-alias form, required for both `ruff`'s
  unused-import check and this repo's `mypy --strict` / `--no-implicit-reexport`) so existing
  importers keep working unchanged in that same commit. Repointing them to import from the new
  module directly is a separate, later cleanup — don't bundle it in.
- Log the remaining clusters as follow-up (see `docs/todos/03-infrastructure.md`'s file-length-split
  item for the current running list) instead of doing them all now — one file's full split can span
  several separate, independently-reviewed commits.
