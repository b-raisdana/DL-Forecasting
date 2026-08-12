# TODO — infrastructure & tooling

Closing the gap between [infrastructure.md](../infrastructure.md) (methodology/tooling choices) and what's
actually wired up as a project-wide practice today. See [master-todo.md](master-todo.md). Most items here
are cross-cutting enablers other topics depend on rather than a pipeline stage of their own.

- [TODO — infrastructure \& tooling](#todo--infrastructure--tooling)
  - [todo](#todo)
  - [appendix: current implementation status](#appendix-current-implementation-status)

## todo

1. **(decision) Pick and wire experiment tracking.** Flagged "current priority" in
   [model-architecture-planning.md § experiment tracking](../model-architecture-planning.md#experiment-tracking-current-priority)
   but still unresolved: ad hoc `.bak`/`Copy (2)` file naming is acknowledged broken, MLflow-local vs.
   CSV/SQLite undecided. Given how many separate multi-seed test phases this project's own discipline
   requires (normalization, activation, GBM variants, ~9 Tier-1 architectures, fusion mechanism,
   per-target class-weight/focal — see [evaluation-metrics.md](evaluation-metrics.md)), lock this before
   that experiment volume begins, not reactively. Pick local-file-backed MLflow (the doc's own leaning)
   or the CSV/SQLite fallback.
2. **Start logging every training run** with config-hash + dataset-version + metrics (loss + trading
   KPIs once [evaluation-metrics.md](evaluation-metrics.md) exists) + artifact path, from the next run
   after step 1 lands — not retroactively.
3. **`vectorbt` integration** — actually owned by [evaluation-metrics.md](evaluation-metrics.md) todo
   step 2 (the backtest module needs it); listed here only as a pointer so it isn't missed while
   auditing this file, not a duplicate task.
4. **Fill or remove the empty `DDD`/`SOA` headers** in [infrastructure.md](../infrastructure.md) — dangling
   sections in the TOC of an otherwise complete doc. Either write the actual methodology notes or delete
   the headings; don't leave them empty.
5. **Extend the QA/complexity gate to label-generation correctness**, not just `radon`/`xenon` cyclomatic
   complexity (see appendix). Once the no-lookahead regression tests land (
   [input-data-channels.md](input-data-channels.md) todo step 6,
   [training-data-labels.md](training-data-labels.md) todo step 11), wire both into whatever CI step
   already runs `xenon`, so correctness and complexity are checked in the same gate rather than two
   disconnected mechanisms.
6. **Data-quality checks for the CCXT feed** — owned by
   [input-data-channels.md](input-data-channels.md) todo step 14, but the natural implementation surface
   is this file's own [Repository design pattern](../infrastructure.md#repository-design-pattern) (gap
   detection, restatement detection, and delisting/survivorship handling belong in the repository layer
   that already owns cached-artifact read/write, not scattered into feature code). Cross-referenced here
   so whoever picks up that task looks at this file's repository-pattern section first.
7. **Audit the Repository design pattern's actual adoption.** The pattern is documented
   ([infrastructure.md § Repository design pattern](../infrastructure.md#repository-design-pattern)) but
   not verified against the codebase: confirm every module reading/writing a persisted artifact (cached
   CSV/parquet/zip/npz, computed indicators) actually goes through a repository interface rather than
   calling storage/CCXT inline, per the pattern's own stated rule ("if two places implement their own
   read-or-fetch-and-cache logic for the same kind of data, that's the signal to introduce one"). Not yet
   checked line-by-line anywhere in this project's docs.
8. **Audit Dependency Injection adoption** the same way — confirm config/shared state is passed
   explicitly into constructors/functions rather than pulled from module-level globals mid-function,
   per [infrastructure.md § Dependency injection](../infrastructure.md#dependency-injection). Not yet
   checked against the actual codebase.
9. **`.gitignore` hygiene for stray editor-backup files** — a `prioritization-framework.md.bak` (~50KB)
   was previously observed in the working tree, gone by the next check, cause unconfirmed. Low priority:
   only act if it recurs — check whatever editor/tool produces it and exclude `*.bak`.
10. **Cross-reference GA vs. Optuna optimizer paths.** `ga_optimizer.py` exists alongside
    `optuna_optimizer.py` (see [model-architecture.md](model-architecture.md) appendix) but this file's
    own [Optuna](../infrastructure.md#optuna) section only documents the Optuna/TPE/Hyperband +
    NSGA-II-later plan. Confirm whether `ga_optimizer.py` is the "NSGA-II reserved for a later
    multi-objective refinement stage" piece already partially built, or a separate/superseded path, and
    update this doc's Optuna section (or add a GA section) accordingly.
11. **Evaluate, don't just document, the `lib-first` skill's candidate libraries.** That skill lists
    `numba`/`polars`/`bottleneck`/`dask.dataframe`/`ta-lib`/async CCXT fetch as candidates against real
    bottlenecks, but none are profiled or adopted yet — in particular,
    `data_processing/fetch_ohlcv.py`'s multi-symbol/multi-timeframe fetch loop is still sequential (no
    `ccxt.async_support`/thread-pool), and `bottleneck` is still commented out in `requirements.txt`.
    Profile the dataset-generation and fetch paths first; only adopt a candidate once a measured
    bottleneck justifies it.
12. **Directory-layer migration** (per the `code-layers` skill's layering; found while mapping modules to
    layers, retained here as this repo's cleanup/rename plan — not repeated in the skill, which only
    states the *current* rule):
    - **Cleanup first** (no layering risk): delete dead code — `app/classic_indicators/obv_macd.py`
      (zero importers), `app/patch/convert_H_to_h.py` (one-shot migration, already run), `app/GPTnote.py`
      and `app/ai_modelling/ttt.py` (empty files); delete superseded forks —
      `app/PreProcessing/*` (byte-identical to `ai_modelling/training_data/PreProcessing/*` except one
      import line, keep the `ai_modelling` copy), `app/predicting/predictor.py` (older, dead imports,
      superseded by `ai_modelling/predicting/predictor.py`), `app/training/trainer.py` (superseded by
      `ai_modelling/training/training_batches.py`, only consumer was the dead `app/predicting/`);
      repoint the 8 files still importing `app/br_py/*.py` (pure re-export shim) to
      `helper.br_py.br_py` directly, then delete the shim; move `app/profit_loss/` (a CSV/notebook
      sample, no code) under `data/`, out of `app/`. Renaming the dead `classic_indicators/obv_macd.py`
      away also resolves its name collision with the active
      `ai_modelling/dataset_generator/classic_indicators.py`.
    - Fix the one reverse-dependency (`Model/TechnicalAnalysis/PeakValley.py` importing
      `INFINITY_TIME_DELTA` from `FigurePlotter/plotter.py` — move the constant to `Config.py`), per the
      `code-layers` skill's "known violation" note.
    - **Directory rename** — mechanical `git mv` + import-path fixups into
      `app/{domain,application,infrastructure,presentation}/...` (domain first since nothing depends on
      it going in; presentation last):

      ```text
      app/
        domain/market_structure/    # Model/TechnicalAnalysis/* + Model/Order.py
        domain/schemas/              # PanderaDFM/*
        application/dataset_generation/  # ai_modelling/dataset_generator/*
        application/training/             # ai_modelling/training/*
        application/prediction/            # ai_modelling/predicting/*
        application/optimization/           # ai_modelling/parameter_optimizser/*
        application/backtesting/             # Strategy/*
        application/preprocessing/            # PreProcessing/* (encoding, gap analysis)
        infrastructure/market_data/            # data_processing/* -> OhlcvRepository
        infrastructure/model_artifacts/         # new: ModelArtifactRepository (.keras save/load)
        infrastructure/ml_adapters/              # ai_modelling/cnn_lstm*_model.py (TF-specific model defs)
        infrastructure/config/                    # Config.py
        infrastructure/logging/                    # helper/br_py/*
        presentation/plotting/                      # FigurePlotter/*
        presentation/entrypoints/                    # main.py, load_last_2_years.py
        tests/                                        # mirrors the tree above
      ```

      `helper/functions.py`, `helper/data_preparation.py`, `helper/importer.py` are cross-cutting
      utilities used by every layer — leave under `helper/`, don't force them into one layer.
    - **Formalize repositories/DI** last — same scope as todo steps 7-8 above (`OhlcvRepository`,
      `ModelArtifactRepository`, stop mutating `app_config.GLOBAL_CACHE` as a module-level singleton).
    Do the directory rename in small PRs per layer, after cleanup + the reverse-dependency fix so the
    rename doesn't carry dead code or forks along with it.

## appendix: current implementation status

Verified against [infrastructure.md](../infrastructure.md) directly — this file documents resolved
methodology/library choices already, so the appendix here only notes what's confirmed vs. still an open
placeholder, not a full re-derivation.

- **Resolved and in active use**: TensorFlow (`tensorflow[and-cuda]`, Docker base
  `tensorflow:25.01-tf2-py3`) as the DL framework; `pandas-ta` for technical indicators; `pandera`
  (`PanderaDFM/*`) for DataFrame schema validation; Optuna (TPE + Hyperband) for architecture/hyperparam
  search; CCXT for exchange data fetch; `plotly` for visualization; Docker for containerized runtime;
  `radon`/`xenon` as the complexity pre-commit gate.
- **Named but not yet integrated**: `vectorbt` for backtesting — "not in `requirements.txt`, no imports
  in codebase yet" per the doc's own text. This is the same gap
  [evaluation-metrics.md](evaluation-metrics.md) todo step 2 owns.
- **Empty sections**: `DDD` and `SOA` headings exist in the TOC with no content underneath — see todo
  step 4.
- **Unverified adoption**: the Repository design pattern and Dependency Injection principles are
  documented as house rules but haven't been audited against actual module code — see todo steps 7-8.
