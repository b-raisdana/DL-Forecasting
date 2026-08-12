# Code layers (DDD-flavored, not MVC)

MVC assumes a request/response UI cycle; this repo is an offline data/ML pipeline (fetch → detect
market structure → build features/datasets → train/optimize/predict → plot). The layering that fits
is Clean/Onion-style DDD: layers as **dependency direction**, not folders-for-folders'-sake. Domain
has zero dependency on anything else in-house; arrows point inward toward it.

## the 4 layers

| Layer              | Role                                                                                                                | Depends on                                          | Current code                                                                                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Domain**         | Pure market-structure logic + the value-object schemas that define what flows between stages. No I/O, no TF/plotly. | nothing in-house (pandas/numpy/pandera only)        | `Model/TechnicalAnalysis/*` (PeakValley, BullBearSide, BasePattern, pivots, ftc), `Model/Order.py`, all of `PanderaDFM/*`                                                                                                                                    |
| **Application**    | Use-case orchestration: coordinates domain services + infrastructure for one job.                                   | Domain, Infrastructure                              | `ai_modelling/dataset_generator/*` (training_datasets.py is the central orchestrator), `ai_modelling/training/*`, `ai_modelling/predicting/*`, `ai_modelling/parameter_optimizser/*`, `Strategy/*` (backtesting), `PreProcessing/*` (encoding/normalization) |
| **Infrastructure** | I/O and framework adapters: exchange fetch, disk/cache, model artifacts, config, logging, TF model definitions.     | Domain schemas only (to know what shape to persist) | `data_processing/*` (already repository-shaped: `read_multi_timeframe_ohlcv`, `read_multi_timeframe_ohlcva`), `Config.py`, `helper/br_py/*` (logging/profiling), `ai_modelling/cnn_lstm*_model.py` (TF adapters)                                             |
| **Presentation**   | Human-facing output.                                                                                                | Application                                         | `FigurePlotter/*`, `main.py`, `load_last_2_years.py`, notebooks                                                                                                                                                                                              |

## DDD concepts mapped

- **Value objects** = PanderaDFM schemas. No persistent identity (a candle/pivot isn't "the same
  entity" across recomputation) — pandera enforces shape/dtype at every hand-off, so it's the value-
  object contract layer, not incidental validation.
- **Domain services** = the TA transform chain, each a pure DataFrame→DataFrame step: PeakValley →
  BullBearSide → BasePattern → ATR/Classic pivots → ftc. No side effects, no framework imports.
- **Application services** = orchestrators that call domain services + repositories for one use case:
  `train_data_of_mt_n_profit` (build training tensors), a trainer, a predictor, an optimizer, a
  `Strategy`. New pipelines are new application services, not new domain logic.
- **Repositories** (already the stated goal in [infrastructure.md § Repository design
  pattern](infrastructure.md#repository-design-pattern)) — formalize `data_processing/ohlcv.py` /
  `atr.py` as `OhlcvRepository`, and add a `ModelArtifactRepository` for `.keras` save/load, so
  callers depend on a `get`/`save` contract, not CCXT/file layout directly.
- **Bounded contexts**, sharing the OHLCV/PanderaDFM kernel:
  - _Market structure_ — TA pattern detection + backtesting (`Model`, `Strategy`).
  - _Forecasting_ — dataset generation, training, prediction, optimization (`ai_modelling/*`).

## known violation

`Model/TechnicalAnalysis/PeakValley.py` imports `INFINITY_TIME_DELTA` from
`FigurePlotter/plotter.py` — Domain reaching into Presentation, backwards. Fix: move that constant
to `Config.py` (Domain/Infrastructure has no reason to import a plotting module).

## module inventory: duplicates/dead code found

Found while mapping modules to layers — cleanup, not a layering decision, but blocks a clean rename:

| Path                                        | Verdict                                                                                                     | Action                                                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `app/classic_indicators/obv_macd.py`        | dead, zero importers                                                                                        | delete                                                                                             |
| `app/patch/convert_H_to_h.py`               | one-shot migration, already run                                                                             | delete                                                                                             |
| `app/GPTnote.py`, `app/ai_modelling/ttt.py` | empty files                                                                                                 | delete                                                                                             |
| `app/PreProcessing/*`                       | superseded fork of `ai_modelling/training_data/PreProcessing/*` (byte-identical except one import line)     | delete top-level, keep the `ai_modelling` copy                                                     |
| `app/predicting/predictor.py`               | superseded fork of `ai_modelling/predicting/predictor.py` (older, dead imports)                             | delete top-level                                                                                   |
| `app/training/trainer.py`                   | superseded fork of `ai_modelling/training/training_batches.py`, only consumer is the dead `app/predicting/` | delete top-level                                                                                   |
| `app/br_py/*.py`                            | pure re-export shim (`from helper.br_py.br_py.X import *`); real impl lives in `helper/br_py/br_py/`        | repoint the 8 files still importing `app.br_py` to `helper.br_py.br_py` directly, then delete shim |
| `app/profit_loss/`                          | no code, just a CSV/notebook sample                                                                         | move under `data/`, out of `app/`                                                                  |

Two same-named-but-unrelated "classic indicators": `app/classic_indicators/obv_macd.py` (dead) vs
`app/ai_modelling/dataset_generator/classic_indicators.py` (active — ichimoku/bbands/rsi/mfi/cci/obv
feature engineering). Renaming the dead one away removes the collision either way.

## target module tree

```
app/
  domain/
    market_structure/      # Model/TechnicalAnalysis/* + Model/Order.py
    schemas/                # PanderaDFM/*
  application/
    dataset_generation/     # ai_modelling/dataset_generator/*
    training/                # ai_modelling/training/*
    prediction/              # ai_modelling/predicting/*
    optimization/            # ai_modelling/parameter_optimizser/*
    backtesting/              # Strategy/*
    preprocessing/             # PreProcessing/* (encoding, gap analysis)
  infrastructure/
    market_data/               # data_processing/* -> OhlcvRepository
    model_artifacts/            # new: ModelArtifactRepository (.keras save/load)
    ml_adapters/                  # ai_modelling/cnn_lstm*_model.py (TF-specific model defs)
    config/                        # Config.py
    logging/                        # helper/br_py/*
  presentation/
    plotting/                        # FigurePlotter/*
    entrypoints/                      # main.py, load_last_2_years.py
  tests/                                # mirrors the tree above
```

`helper/functions.py`, `helper/data_preparation.py`, `helper/importer.py` are cross-cutting utilities
used by every layer — leave under `helper/`, don't force them into one layer.

## migration order

1. **Cleanup** (safe, no layering risk) — delete/consolidate the table above first. Removes the
   naming collisions and forks that would otherwise get copied into two places during the rename.
2. **Fix the one reverse-dependency** (`PeakValley` → `FigurePlotter` constant).
3. **Directory rename** — mechanical `git mv` + import-path fixups per module group above. Do in
   small PRs per layer (domain first, since nothing depends on it going in; presentation last).
4. **Formalize repositories/DI** — wrap `data_processing` read/write in `OhlcvRepository`, add
   `ModelArtifactRepository`, stop mutating `app_config.GLOBAL_CACHE` as a module-level singleton
   (already the stated direction in [infrastructure.md § Dependency
   injection](infrastructure.md#dependency-injection)).

Steps 1-2 are cheap and low-risk; do them before step 3 so the rename doesn't carry dead code or
forks along with it.
