# Data gap-filling plan

Open work on how missing or incomplete candle data is detected, handled, and surfaced to the trainer. Complements the caching strategy in [02-Appendix-Feature-Label-Preparation.md](../ML_Forecasting_System_Design/02-Appendix-Feature-Label-Preparation.md#missing-or-incomplete-data).

## Modifications to existing methods

Ordered from model feed down to raw data preparation.

- `app/application/model_implementations/tier1_000/train.py:run_training` — `epochs=1_000_000` hardcoded at `model.fit()`; make it a parameter derived from time budget.
- `app/application/model_implementations/tier1_000/model.py:Tier1000Model.call` — `BRANCH_TIMEFRAMES`, `BRANCH_WINDOW_LENGTHS`, `CANDLE_FEATURE_COLUMNS` hardcoded at module level; extract into a shared config so the datafeeder can reference the same source.
- `app/application/model_implementations/tier1_000/datafeeder_input3_outcome1.py:make_tf_dataset` — `inputs` dict keys hardcoded to `BRANCH_TIMEFRAMES`; derive from shared config instead.
- `app/application/model_implementations/tier1_000/datafeeder_input3_outcome1.py:DatasetBundle` — add fields for any new labels or feature modalities introduced by gap-handling features.
- `app/application/model_implementations/tier1_000/datafeeder_input3_outcome1.py:build_dataset` — `BRANCH_TIMEFRAMES` loop and `BRANCH_WINDOW_LENGTHS` lookup hardcoded; parameterize. `_ATR_LENGTH_OVERRIDE` hardcoded dict; make config-driven. Add new computation paths for gap-related features.
- `app/application/model_implementations/tier1_000/datafeeder_input3_outcome1.py:_branch_features` — add new feature computation calls for gap indicators.
- `app/application/dataset_generation/extremum_features.py:compute_higher_extremum_distance` — extend from plus2/plus3 only to nearest-top/valley across all eligible timeframes.
- `app/application/dataset_generation/extremum_features.py:_one_target` — outer anchor loop is a CPU bottleneck; vectorize across anchors.
- `app/infrastructure/datastore_engine/disk_cache.py:cache_on_disk` — remove `windowed: bool` parameter and the `if windowed:`/`else:` dispatch; `read_file_windowed()` is the only active entry point.

## New methods to create

- **Shared config module** — dataclass or module holding `BRANCH_TIMEFRAMES`, `BRANCH_WINDOW_LENGTHS`, `CANDLE_FEATURE_COLUMNS`, `AUX_FEATURE_DIM`, and gap-handling parameters; referenced by both `model.py` and `datafeeder_input3_outcome1.py`.
- **Gap detection in fetch path** — new method (or layer) in `app/infrastructure/market_data_fetch/ccxt_client.py` that distinguishes exchange gaps from network failures; retries with backoff before marking as gap.
- **Gap tagging in cache writer** — new method in `disk_cache.py` or the DuckDB writer that records gap origin (missing candle vs. incomplete OHLCV field) and source timestamp alongside cached data.
- **Per-timeframe gap alignment propagator** — new method that takes base-TF (5min) gap flags and propagates them to derived higher timeframes (15min, 1H, etc.) according to the specified inclusion rule.
- **Gap-aware cache backfill handler** — new method that, given a corrected range, recomputes and appends only that range instead of rebuilding the full timeframe table.
- **Feed-time NaN filtering contract** — new validation method in the datafeeder that enforces which NaN patterns drop a sample vs. are tolerated, replacing the current implicit behavior.

## current state

- Missing candles in an input series are skipped at fetch time.
- `build_dataset` fills gaps with NaN rather than dropping or erroring, to simplify adoption.
- NaN-containing selected samples are filtered or dropped later at feed time, not at cache time.

## out of scope

- Full gap audit across the existing Binance history (deferred until the fetch-time gap detection above is stable).
- Partitioning strategy for gap-tagged cache entries.
