# Archive manifest

Files here are unreachable from any `app/presentation/` entrypoint (checked by static import-graph
traversal, not deleted — kept for reference/recovery). Directory structure mirrors each file's original
location under `app/`; test files, where they existed, moved to the same relative path under
`tests/archive_not_used_trash/`. Restoring: `git mv` back to the mirrored live path and fix any import
that was rewritten to point at `archive_not_used_trash.*`.

## 2026-08-19 pass

Corrected an earlier bad archival first: `domain/price_action/PeakValley.py` and `BullBearSide.py` had
been moved here even though 4 live `presentation/market_structure/*_plotter.py` files still imported them
directly — restored both to `app/domain/price_action/`, reverted the plotter imports.

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

Internal imports between files that moved together were rewritten to keep resolving
(`archive_not_used_trash.*` prefix added where the target also moved here).

Fixed `pytest.ini`: `testpaths` pointed at the nonexistent `app/tests` (pre-existing, unrelated — real
suite is top-level `tests/`), so a no-path `pytest` invocation silently fell back to a whole-repo recursive
search and swept up `test_normalization.py` above (matches `test_*.py`, isn't a real test) — corrected to
`testpaths = tests` and added `norecursedirs = archive_not_used_trash` so archived tests stay excluded from
default runs (still runnable by pointing pytest at them directly).

**Not touched, flagged instead:** `presentation/market_structure/{Pivot,PeakValley}_plotter.py` and
`application/preprocessing/encoder.py`/`application/backtesting/BasePatternStrategy.py` import
`infrastructure.ohlcv.{ohlcv,ohlcva}`, which doesn't exist anywhere in the tree (pre-existing, unrelated to
this pass — likely meant `domain.ohlcv.*`).
