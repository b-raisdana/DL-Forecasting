---
name: pytest
description: Use whenever writing, running, or debugging a pytest test in this repo (app/tests/). Covers how to actually execute tests here (Windows has no usable Python env for this project — the real one is a WSL conda env), plus repo conventions for markers, fixtures, and structure. Load test-strategy first to pick the right test type before using this skill to write it.
---

# pytest (this repo)

Companion to [test-strategy](../test-strategy/SKILL.md) (which type to write) and
[docs/testing.md](../../../docs/testing.md) (full layout reference). This skill covers the mechanics.

## Running tests — read this first

This machine's Windows Python installs (`py -3.9`...`3.14`) do **not** have this project's dependencies
(`pandas`, `pandas_ta`, `tensorflow`, `pandera`, ...) — don't `pip install` them into a throwaway venv,
that fights meson/build-from-source errors on Windows for no reason. The real environment is the WSL
conda env `tf`:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf &&
  cd /mnt/c/Code/DL-Forecasting &&
  pytest -m "unit or characterization or regression or smoke"
'
```

Windows path `C:\Code\DL-Forecasting` = WSL path `/mnt/c/Code/DL-Forecasting` (same clone, not a
separate checkout). TensorFlow prints CUDA/cuDNN registration warnings on import — harmless noise, not a
failure signal; check the actual pytest summary line. WSL has no outbound network (pip installs will
fail with a DNS error) — `pytest` and the full dep set are already installed in `tf`, so this is rarely
needed; if a genuinely new dependency is required, ask the user rather than assuming network access.

**Data cache lives off `/mnt/c`.** The code clone above stays on the `drvfs` mount (needed for Windows
git/VS Code access), but the real cached OHLCV under `data/` (`Config.path_of_data`) has been moved to
a native ext4 path, `/home/brais/dlf-data` — `drvfs` is the worst case for its ~19.5k small per-day zip
files (each open crosses the WSL2 9p protocol). The `tf` conda env's `activate.d`/`deactivate.d` hooks
set `DLF_DATA_ROOT=/home/brais/dlf-data` automatically, so `conda activate tf` (as in the command above)
already points there — nothing extra to do. Only `e2e`/`perf` tests touch real `data/` at all (see
[test-strategy](../test-strategy/SKILL.md) fixture/data policy); unit/characterization/regression/smoke
never read it. Full rationale: [docs/infrastructure.md § environments](../../../docs/infrastructure.md#environments).

Fast gate vs full run — see [docs/testing.md § markers and running](../../../docs/testing.md#markers-and-running).

## Repo config

`pytest.ini` (repo root): `testpaths = app/tests`, `pythonpath = app` (so `from Config import app_config`
style absolute imports resolve without a `src.`/`app.` prefix, matching how the app itself imports),
`--import-mode=importlib`, `--strict-markers`. Markers are pre-registered there — using an unregistered
one is an error by design; add it to `pytest.ini` if a genuinely new type is needed (check
[test-strategy](../test-strategy/SKILL.md) first, this should be rare).

## Where a new test goes

`app/tests/<type>/<mirrors the app/ package path of what's under test>/test_<module>.py` — e.g. a
characterization test for `app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py` goes in
`app/tests/characterization/dataset_generator/profit_loss/test_profit_loss_adder_characterization.py`.
Only create the folders you're actually populating.

## Structure: Arrange-Act-Assert

```python
def test_max_profit_n_loss_flat_candles_zero_distance(flat_ohlc):
    # Arrange
    ohlc = flat_ohlc(n=5)
    # Act
    result = max_profit_n_loss(ohlc, position_max_bars=2, action_delay=1, rolling_window=3)
    # Assert
    assert result["max_high_distance"].dropna().eq(0).all()
```

One assertion focus per test — if two behaviors need separate reasoning to debug a failure, they're two
tests, not one with multiple unrelated asserts.

## Fixtures

Shared synthetic-data builders go in `app/tests/conftest.py` as factory fixtures (a fixture that returns
a function, so each test controls size/shape), not fixed DataFrames — e.g. `flat_ohlc(n)`,
`trending_ohlc(n, direction)`. Keep fixtures small (5-30 rows) and deterministic. Don't read from
`data/` (real cached OHLCV) in unit/characterization/regression/smoke tests — see
[docs/testing.md § fixture/data policy](../../../docs/testing.md#fixturedata-policy).

## Naming

`test_<unit>_<state_under_test>_<expected_result>` where the state/expected fit on one line; for
characterization tests, `test_<function>_<fixture_name>` is fine since the point is pinning a value, not
describing a behavior claim.

## Parametrize over copy-pasted near-duplicates

```python
@pytest.mark.parametrize("action_delay,expected_open", [(1, 105), (2, 106)])
def test_worst_long_open_by_action_delay(flat_ohlc, action_delay, expected_open):
    ...
```

Reach for this the moment two test functions differ only in literals.
