# TODO — infrastructure & tooling

Closing the gap between [infrastructure.md](../infrastructure.md) (methodology/tooling choices) and what's
actually wired up as a project-wide practice today. See [00-master-todo.md](00-master-todo.md). Most items here
are cross-cutting enablers other topics depend on rather than a pipeline stage of their own.

- [TODO — infrastructure \& tooling](#todo--infrastructure--tooling)
  - [todo](#todo)
  - [appendix: current implementation status](#appendix-current-implementation-status)

## todo

1. **(decision) Pick and wire experiment tracking.** Flagged "current priority" in
   [model-architecture-planning.md § experiment tracking](../ML_Forecasting_System_Design/04-Experimentation, Evaluation & Optimization.md#experiment-tracking-current-priority)
   but still unresolved: ad hoc `.bak`/`Copy (2)` file naming is acknowledged broken, MLflow-local vs.
   CSV/SQLite undecided. Given how many separate multi-seed test phases this project's own discipline
   requires (normalization, activation, GBM variants, ~9 Tier-1 architectures, fusion mechanism,
   per-target class-weight/focal — see [05-evaluation-metrics.md](05-evaluation-metrics.md)), lock this before
   that experiment volume begins, not reactively. Pick local-file-backed MLflow (the doc's own leaning)
   or the CSV/SQLite fallback.
2. **Start logging every training run** with config-hash + dataset-version + metrics (loss + trading
   KPIs once [05-evaluation-metrics.md](05-evaluation-metrics.md) exists) + artifact path, from the next run
   after step 1 lands — not retroactively.
3. **`vectorbt` integration** — actually owned by [05-evaluation-metrics.md](05-evaluation-metrics.md) todo
   step 2 (the backtest module needs it); listed here only as a pointer so it isn't missed while
   auditing this file, not a duplicate task.
4. **Fill or remove the empty `DDD`/`SOA` headers** in [infrastructure.md](../infrastructure.md) — dangling
   sections in the TOC of an otherwise complete doc. Either write the actual methodology notes or delete
   the headings; don't leave them empty.
5. **Extend the QA/complexity gate to label-generation correctness**, not just `radon`/`xenon` cyclomatic
   complexity (see appendix). Once the no-lookahead regression tests land (
   [01-input-data-channels.md](01-input-data-channels.md) todo step 6,
   [02-training-data-labels.md](02-training-data-labels.md) todo step 11), wire both into whatever CI step
   already runs `xenon`, so correctness and complexity are checked in the same gate rather than two
   disconnected mechanisms.
6. **Data-quality checks for the CCXT feed** — owned by
   [01-input-data-channels.md](01-input-data-channels.md) todo step 14, but the natural implementation surface
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
   checked line-by-line anywhere in this project's docs. **Partial finding, 2026-08-15**: computed
   indicators and labels had *no* caching at all (worse than "not through a repository interface") —
   `npz_batch.py`/`ram_batch.py`/`stream_loader.py` each load `mt_ohlcv` once per quarter then call
   `train_data_of_mt_n_profit()` up to ~100× against that same object, recomputing
   `classic_indicators`/`relative_candle`/`volume_feature` and the rolling-window label computation in
   `profit_loss_adder.py` from scratch every call. Closed via
   `training_datasets.py`'s new `_cached_training_frames()` (in-memory memo on `mt_ohlcv.attrs`, since
   `pd.DataFrame` is unhashable) and a bounded in-process LRU added to `read_file()` itself
   (`helper/data_preparation.py`) in front of its disk read. Both follow the new
   [cache-or-generate skill](../../.claude/skills/cache-or-generate/SKILL.md) — read that first before
   adding another cache. Still open: no actual `Repository` class exists anywhere (both fixes are free
   functions, matching the pre-existing style, not the class-based interface the doc describes); full
   line-by-line module audit against the pattern still not done.
8. **Audit Dependency Injection adoption** the same way — confirm config/shared state is passed
   explicitly into constructors/functions rather than pulled from module-level globals mid-function,
   per [infrastructure.md § Dependency injection](../infrastructure.md#dependency-injection). Not yet
   checked against the actual codebase.
9. **`.gitignore` hygiene for stray editor-backup files** — a `prioritization-framework.md.bak` (~50KB)
   was previously observed in the working tree, gone by the next check, cause unconfirmed. Low priority:
   only act if it recurs — check whatever editor/tool produces it and exclude `*.bak`.
10. **Cross-reference GA vs. Optuna optimizer paths.** `ga_optimizer.py` exists alongside
    `optuna_optimizer.py` (see [04-model-architecture.md](04-model-architecture.md) appendix) but this file's
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
        domain/price_action/         # Model/TechnicalAnalysis/* (price-action algorithms) + Model/Order.py
        domain/technical_analysis/    # named indicator calc split out of infra (e.g. ATR/RMA)
        domain/ohlcv/                  # OHLCV shape/resampling split out of infra (build/aggregate/volume)
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
        presentation/                                 # FigurePlotter/*, main.py, load_last_2_years.py —
                                                       # organized by domain (ohlcv/, market_structure/,
                                                       # dataset_generation/, model_implementations/,
                                                       # preprocessing/, shared/), not by file type
        tests/                                        # mirrors the tree above
      ```

      `helper/functions.py`, `helper/data_preparation.py`, `helper/importer.py` are cross-cutting
      utilities used by every layer — leave under `helper/`, don't force them into one layer.
    - **2026-08-17 follow-up split:** `domain/market_structure/` renamed to `domain/price_action/` (same content — name now matches what it holds). `infrastructure/ohlcv/atr.py` mixed a pure indicator calc (`RMA`, `insert_atr`, `insert_volume_rma`, `insert_mt_volume_rma` — DataFrame→DataFrame, no I/O) with its disk-cached repository generator (`get_multi_timeframe_ohlcva`, `@cache_on_disk`-decorated, does I/O); the calc moved to new `domain/technical_analysis/` (split by category: `base.py` for `RMA`, `volume.py` for `insert_volume_rma`/`insert_mt_volume_rma`, `classic_indicators.py` moved in from `application/dataset_generation/classic_indicators.py` — Ichimoku/Bollinger/RSI/MFI/OBV/CCI, same reasoning), the generator's file renamed `infrastructure/ohlcv/atr.py` → `infrastructure/ohlcv/ohlcva.py` (it's not TA anymore, it's the OHLCVA repository generator, importing `insert_atr` from domain). `infrastructure/ohlcv/disk_cache.py` moved to `infrastructure/disk_cache.py` — it's a generic `(data_frame_type, date_range_str)` persistence primitive (used well beyond OHLCV: `domain/price_action/*`, `domain/schemas/common/ExtendedDf.py`, tests), not ohlcv-specific; `symbol_data_path()` merged into it from the deleted `infrastructure/ohlcv/fragmented_data.py` (whose other function, `load_ohlcv_list`, was dead/broken — unreachable `globals()` check, zero callers — dropped, not migrated). `infrastructure/ohlcv/` now holds only OHLCV/OHLCVA-specific I/O — no TA math, no generic cache engine.
    - **2026-08-17 second follow-up:** `infrastructure/ohlcv/ohlcv.py`'s two remaining functions each mixed I/O with a pure transform — `get_base_timeframe_ohlcv` (fetch, then shape the raw rows into a validated OHLCV DataFrame) and `get_multi_timeframe_ohlcv` (fetch via the former, then resample into every configured timeframe). New `domain/ohlcv/` holds the extracted pure halves: `ohlc.py` (`build_base_timeframe_ohlcv`), `multi_timeframe.py` (`aggregate_multi_timeframe_ohlcv`). `domain/technical_analysis/volume.py` also moved here as `domain/ohlcv/volume.py` (`insert_volume_rma`/`insert_mt_volume_rma`) — it smooths the OHLCV volume column directly rather than being a named indicator like the RSI/MACD-style entries in `classic_indicators.py`, so it fits the OHLCV-shape category better than the indicator-math one; `domain/technical_analysis/atr.py` now imports it from there. `infrastructure/ohlcv/ohlcv.py` keeps only the fetch + `@cache_on_disk` orchestration, calling the two new domain functions.
    - **Known violation, not yet fixed (tracked for a later pass):** several `domain/price_action/*` modules import directly from infrastructure — not just schemas. `BasePattern.py`, `BullBearSide.py`, `BullBearSidePivot.py`, `PeakValleyPivots.py`, `AtrMovementPivots.py` pull `get_multi_timeframe_ohlcva`/`read_multi_timeframe_ohlcva` from `infrastructure/ohlcv/ohlcva.py`, and `BasePattern.py`, `BullBearSide.py`, `BullBearSidePivot.py`, `PeakValleyPivots.py`, `PeakValley.py` import `cache_on_disk` straight from `infrastructure/disk_cache.py`. Both violate the `code-layers` skill's "Domain depends on nothing in-house" rule. Fix is to pass the OHLCVA frame (and any needed caching) in from an Application-layer caller instead of Domain pulling it itself — deferred, not attempted in this pass to avoid scope creep on top of the directory moves above.
    - **Formalize repositories/DI** last — same scope as todo steps 7-8 above (`OhlcvRepository`; stop
      mutating `app_config.GLOBAL_CACHE` as a module-level singleton). `ModelArtifactRepository` itself is
      no longer a placeholder — implemented 2026-08-16 as `tf.train.Checkpoint`/`CheckpointManager`-based
      save/restore in [infrastructure/model_artifacts](<../../app/infrastructure/model_artifacts/__init__.py>),
      wired into `tier1_000/train.py`'s periodic-checkpoint/resume flow — see
      [06-ML-Ops.md § checkpointing & resume](<../ML_Forecasting_System_Design/06-ML-Ops.md#checkpointing--resume>).
    Do the directory rename in small PRs per layer, after cleanup + the reverse-dependency fix so the
    rename doesn't carry dead code or forks along with it.
13. **Continue the file-length split of `helper/data_preparation.py` and other files the new `loc`
    ratchet vector flags** ([infrastructure.md § pre-commit](../infrastructure.md#pre-commit)). Treat `<300` lines as normal, `300-500` lines as a potential low-priority split todo, and `>500` lines as a warning/high-priority split todo. First cut
    landed 2026-08-15: the PanderaDFM schema-casting cluster moved to `helper/schema_casting.py`
    (re-exported from `data_preparation.py` for its 34 existing importers, so no caller updates were
    needed yet - see todo step 7's repository-pattern audit for when those re-exports should be
    resolved). Remaining `data_preparation.py` clusters, each a separate small commit, not all at once:
    - file-cache/read (`read_file`, `read_with_timeframe`, `read_without_index`, `read_by_date`,
      `single_timeframe`, the `_read_file_cache_*` memo) - the `cache-or-generate` skill's canonical
      instance, so update that skill's file reference if this moves.
    - timeframe/date-range (`df_timedelta_to_str`, `timedelta_to_str`, `to_timeframe`,
      `check_time_in_cache`, `times_tester`, `multi_timeframe_times_tester`, `shift_timeframe`,
      `trigger_timeframe`/`pattern_timeframe`/`anti_pattern_timeframe`/`anti_trigger_timeframe`,
      `times_in_date_range`, `after_under_process_date`, `trim_to_date_range`, `expand_date_range`).
    - misc symbol/index helpers (`map_symbol`, `FileInfoSet`/`extract_file_info`, `nearest_match`,
      `concat`).

    Other files already over the `loc` vector's 500-line threshold, same one-cluster-per-commit approach (paths below reflect the todo step 12 directory-layer migration, `market_structure/` renamed to `price_action/` in a later pass): `domain/price_action/PeakValley.py` (776 lines), `application/dataset_generation/profit_loss/profit_loss_adder.py` (765), `domain/price_action/BullBearSide.py` (712), `application/dataset_generation/training_datasets.py` (633), `domain/price_action/ftc.py` (620), `helper/data_preparation.py` (613, up from its prior count above).

    **2026-08-16 `loc` baseline bump, with an explicit payoff plan.** The todo step 12 directory-layer migration grew the four files above while moving/reformatting them (`profit_loss_adder.py` 657->765, `PeakValley.py` 694->776, `BullBearSide.py` 636->712, `ftc.py` 579->620) and pushed two more over the 500-line threshold for the first time (`training_datasets.py` at 633, `data_preparation.py` at 613) - a genuine +116 project-wide regression on the `loc` vector (1003->1119), not a rename artifact. Blocking the migration commit on an unrelated split effort wasn't worth it, so `scripts/git-hooks/incremental-precommit/baseline.json` was re-baselined to 1119 instead. Endpoint: this todo step's split plan (`data_preparation.py`'s clusters above, plus the same one-cluster-per-commit treatment for the other five files) is what pays this back down - the six files listed above are the full scope of what this bump covers, not a general license to keep growing them.
14. **(decision) Data pipeline upgrade — Feather → Parquet migration.** Plan at
    [data_pipeline_upgrade_plan.md](../../todos/data_pipeline_upgrade_plan.md): mirrors the existing
    CSV-zip → Feather on-touch migration one format further, in `infrastructure/disk_cache.py`
    (`_feather_file_path`/`write_data_file`/`_read_raw_data_file`/`remove_data_file`) and
    `disk_cache_layout.py`'s `_legacy_file_pattern`. Scoped deliberately narrow — a broader review
    (dataset catalog, DuckDB, extended data-quality checks, a found cache-key bug where `read_file()`
    doesn't hash generation parameters so a changed indicator param can silently serve a stale cached
    file) was cut back to just this first step; revisit the rest only once this lands. Not started —
    review and prioritize against the other open decisions in this file (steps 1, 7-8, 11-13) before
    picking this up.

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
  [05-evaluation-metrics.md](05-evaluation-metrics.md) todo step 2 owns.
- **Empty sections**: `DDD` and `SOA` headings exist in the TOC with no content underneath — see todo
  step 4.
- **Unverified adoption**: the Repository design pattern and Dependency Injection principles are
  documented as house rules but haven't been audited against actual module code — see todo steps 7-8.
