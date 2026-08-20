# Archive manifest

Files here are unreachable from any `app/presentation/` entrypoint (checked by static import-graph traversal, not deleted — kept for reference/recovery). Directory structure mirrors each file's original location under `app/`; test files, where they existed, moved to the same relative path under `tests/archive_not_used_trash/`. Restoring: `git mv` back to the mirrored live path and fix any import that was rewritten to point at `archive_not_used_trash.*`.

## 2026-08-19 pass

Corrected an earlier bad archival first: `domain/price_action/PeakValley.py` and `BullBearSide.py` had been moved here even though 4 live `presentation/market_structure/*_plotter.py` files still imported them directly — restored both to `app/domain/price_action/`, reverted the plotter imports.

Then archived (no live importer found via import-graph traversal from every `app/presentation/*.py`):

| Moved | Test moved |
|---|---|
| `application/backtesting/` (whole dir) | none |
| `application/dataset_generation/feeders/` (whole dir) | none |
| `application/dataset_generation/profit_loss/` (whole dir) | `tests/characterization/dataset_generator/profit_loss/test_profit_loss_adder_characterization.py` |
| `application/dataset_generation/test_normalization.py` | none |
| `application/dataset_generation/training_datasets.py` | `tests/unit/dataset_generator/test_training_datasets_cache.py` |
| `application/live_trading/` (whole dir, empty) | none |
| `application/model_implementations/cnn_lstm/` (whole dir) | none |
| `application/model_implementations/cnn_lstm_attention/` (whole dir) | none |
| `application/model_implementations/multi_branch_autoencoder_pattern/` (whole dir, empty) | none |
| `application/model_implementations/tier1_000/gbm_ensemble.py` | none |
| `application/optimization/` (whole dir) | none |
| `application/preprocessing/` (whole dir, incl. `encoder.py`) | none |
| `domain/order/SignalDf.py` | none |
| `domain/price_action/AtrMovementPivots.py`, `BasePattern.py`, `ClassicPivot.py`, `ColorTrend.py`, `PivotsHelper.py`, `RBD.py`, `SupportResistance.py` | none |
| `domain/schemas/common/BaseDFM.py` (superseded — a `BaseDFM` class now lives in `ExtendedDf.py`) | none |
| `domain/schemas/forecasting/` (whole dir) | none |
| `domain/schemas/market_structure/AtrTopPivot.py`, `BullBearSidePivot.py`, `Pivot.py` (superseded by `Pivot2.py`) | none |
| `domain/technical_analysis/base.py` | none |
| `helper/logging/do_log/log_severities.py` | none |
| `helper/logging/progressive_query.py` | none |
| `infrastructure/market_data_fetch/binance/`, `kucoin/` (whole dirs — superseded by `ccxt_client.py`) | `tests/integration/infrastructure/market_data_fetch/test_ccxt_client_live_fetch.py` (whole file); 2 of 5 tests split out of `tests/unit/infrastructure/market_data_fetch/test_fetch_ohlcv.py` (the 3 covering live `ccxt_client.py` stayed in place) |
| `infrastructure/order_execution/` (whole dir, empty stubs) | none |

Internal imports between files that moved together were rewritten to keep resolving (`archive_not_used_trash.*` prefix added where the target also moved here).

Fixed `pytest.ini`: `testpaths` pointed at the nonexistent `app/tests` (pre-existing, unrelated — real suite is top-level `tests/`), so a no-path `pytest` invocation silently fell back to a whole-repo recursive search and swept up `test_normalization.py` above (matches `test_*.py`, isn't a real test) — corrected to `testpaths = tests` and added `norecursedirs = archive_not_used_trash` so archived tests stay excluded from default runs (still runnable by pointing pytest at them directly).

**Not touched, flagged instead:** `application/preprocessing/encoder.py`/`application/backtesting/BasePatternStrategy.py` (and, before this file's move below, `presentation/market_structure/{Pivot,PeakValley}_plotter.py`) import `infrastructure.ohlcv.{ohlcv,ohlcva}`, which doesn't exist anywhere in the tree (pre-existing, unrelated to this pass — likely meant `domain.ohlcv.*`).

## 2026-08-20 pass

Outside this session, `presentation/market_structure/*_plotter.py` moved to `market_structure/` here and `presentation/model_implementations/` was renamed to `presentation/ai_models/` — re-ran the import-graph sweep against that new state, confirmed with you before treating the plotter move as intentional, and found `domain/price_action/{PeakValley,BullBearSide}.py`, `domain/order/`, and `domain/schemas/market_structure/` newly orphaned by it — archived (mirrored paths, no live importers, no tests reference any of them). `app/presentation/ai_models/tier1_000_training.py` and `app/infrastructure/options_settings.py` were mid-edit (syntax errors in the working tree) — left untouched per your instruction; reachability through them was evaluated from their last-committed content.

Then did a **method-level** pass (same request, extended to functions/classes, not just files): for every live file, found top-level functions/classes with zero references anywhere in live code, as a transitive fixed point (a helper only used by a now-dead function cascades to dead too) — excluding dunders, `@computed_field`/CLI-`@command`-decorated, and known framework-dispatched method names (`on_epoch_end`, `emit`, `dispatch`, etc; a class's own reachability is still checked even when its methods carry such names — a `Handler`/`Callback` subclass that's never instantiated is still dead). Per your instruction: in the original file the dead function/class is commented out (not removed), live code stays; in a duplicate under this archive tree the dead code stays active, and any live code it depended on is copied alongside it too so the archived copy still parses/imports (the *original* keeps the sole live copy of that dependency). Two files had every remaining function dead — moved whole instead of split: `helper/logging/base.py`, `helper/logging/profiling/serialization.py`.

Files split (dead symbol(s) — note):

| File | Dead symbol(s) | Note |
|---|---|---|
| `application/dataset_generation/relative_candle.py` | `relative_candle_columns` | also a test *utility* in `test_scales_inversely_with_atr` — that usage repointed at the archive import, not duplicated |
| `application/dataset_generation/volume_feature.py` | `volume_feature_columns`, `log_sma_volume_feature_columns` | |
| `application/model_implementations/shared/base.py` | `setup_tensorboard`, `dataset_folder`, `pre_train_model`, `overlapped_quarters`, `build_model`, `check_dataset_shape_change`, `CustomEpochLogger` | class never instantiated |
| `config/Config.py` | `TREND`, `TopTYPE`, `CandleSize` | only consumers were today's newly-archived price_action/market_structure files; re-exports removed from `config/__init__.py` |
| `domain/price_action/CausalExtremum.py` | `observed_extremum_tf_minutes` | test split to `tests/archive_not_used_trash/unit/price_action/test_causal_extremum.py` |
| `domain/technical_analysis/classic_indicators.py` | `add_ichimoku`, `add_bbands`, `add_classic_indicators`, `zz_bollinger_width` | |
| `helper/data_preparation.py` | `date_range_of_data`, `df_timedelta_to_str`, `timedelta_to_str`, `dict_of_list`, `shift_timeframe`, `trigger_timeframe`, `pattern_timeframe`, `anti_pattern_timeframe`, `anti_trigger_timeframe`, `FileInfoSet`, `extract_file_info`, `expand_date_range`, `nearest_match` | |
| `helper/logging/do_log/log_it.py` | `init_logger` | unreachable now that `br_lib_init` moved; its `_intercept_stdlib_logging`/`InterceptHandler`/`_nearest_level_name` deps duplicated in, still live in the original for `_init_default_logger`'s own use |
| `helper/logging/do_log/ray_id.py` | `ContextVarMiddleware` | never wired into any app |
| `helper/logging/profiling/base.py` | `profile_to_db`, `init_global_profile_to_db`, `profile_func` | |
| `helper/schema_casting.py` | `cast_and_validate2`, `apply_as_type2`, `column_dtypes`, `index_names` | |
| `infrastructure/datastore_engine/disk_cache.py` | `read_without_index` | legacy-feather-migration path; 2 tests moved to `tests/archive_not_used_trash/unit/infrastructure/test_disk_cache.py` |
| `presentation/shared/plotter.py` | `plot_multiple_figures`, `save_figure`, `file_id`, `update_figure_layout`, `timeframe_color` | lost its remaining live callers when the market_structure plotters moved above |
