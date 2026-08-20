---
name: pytest
description: Use whenever writing, running, or debugging a pytest test in this repo (app/tests/). Covers how to actually execute tests here (Windows has no usable Python env for this project — the real one is a WSL conda env), plus repo conventions for markers, fixtures, and structure. Load test-strategy first to pick the right test type before using this skill to write it.
---

# pytest (this repo)

Companion to [test-strategy](../test-strategy/SKILL.md) (which type to write) — this skill covers mechanics: layout, running, fixtures, naming.

## Running tests — read this first

Windows Python installs (`py -3.9`...`3.14`) lack this project's deps (`pandas`, `pandas_ta`, `tensorflow`, `pandera`, ...) — don't `pip install` into a throwaway venv, that hits meson/build-from-source errors on Windows for nothing. The real environment is the WSL conda env `tf`:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf &&
  cd /home/brais/code/DL-Forecasting &&
  pytest -m unit
'
```

The repo is a native WSL ext4 clone at `/home/brais/code/DL-Forecasting` (no separate Windows-side checkout). TensorFlow's CUDA/cuDNN registration warnings on import are harmless noise, not a failure signal — check the actual pytest summary line. WSL has no outbound network (pip installs fail with a DNS error); `pytest` and the full dep set are already installed in `tf`, so this is rarely needed — ask the user before assuming network access for a genuinely new dependency.

**Data cache lives in-repo.** Cached OHLCV under `data/` (`Config.path_of_data`) needs no override — it defaults to `<repo_root>/data`, and since the whole clone is native ext4, that's already the fast path (no `drvfs` tax to route around). `DLF_DATA_ROOT` still exists as a generic override but is unused by default now. Only `e2e`/`perf` tests touch real `data/` at all (see [test-strategy](../test-strategy/SKILL.md) fixture/data policy); unit/characterization/regression/smoke never read it. Full rationale: [docs/infrastructure.md § environments](../../../docs/infrastructure.md#environments).

Fast gate (every commit): `pytest -m unit`. Full run (nightly/manual, includes integration, `e2e`, and `perf`): `pytest`. Live broker OHLCV fetches are integration tests, never pre-commit tests. The fast gate is wired into `.pre-commit-config.yaml` alongside the `mypy`/`ruff`/`xenon` incremental ratchet ([infrastructure.md § incremental ratchet](../../../docs/infrastructure.md#incremental-ratchet-mypyruffxenon-scope)) — local/per-commit only, bypassable with `--no-verify`. No CI workflow exists yet; promote the same fast-gate command to a required check when one lands.

## Repo config

`pytest.ini` (repo root): `testpaths = app/tests`, `pythonpath = app` (so `from Config import app_config` style absolute imports resolve without a `src.`/`app.` prefix, matching how the app itself imports), `--import-mode=importlib`, `--strict-markers`. Registered markers: `unit`, `characterization`, `integration`, `regression`, `smoke`, `e2e`, `perf` — `--strict-markers` makes an unregistered marker an authoring error, not a silently-ignored typo; add a new one to `pytest.ini` only if a genuinely new type is needed (check [test-strategy](../test-strategy/SKILL.md) first, this should be rare).

## Directory layout

```
pytest.ini                      # repo root: markers, testpaths, pythonpath
app/
  tests/
    conftest.py                 # shared fixtures (synthetic OHLC builders, etc.)
    unit/<mirrors app/ package path>/test_<module>.py
    characterization/<mirrors app/ package path>/test_<module>_characterization.py
    integration/<mirrors app/ package path>/test_<flow>.py
    regression/test_<invariant>.py
    smoke/test_<surface>.py
    e2e/test_<pipeline>.py
    perf/test_<budget>.py
```

One tree under `app/tests/`, split by type first then mirroring the source package path — not colocated `test_*.py` next to source, and not one flat folder. Type is the axis test selection runs on (`-m unit`, skip `e2e` by default), so it has to be the top split; mirroring source under that keeps a test's home discoverable from its target's path. Only create a subfolder when it has a test in it — don't pre-scaffold empty type folders.

A new test goes at `app/tests/<type>/<mirrors the app/ package path of what's under test>/test_<module>.py` — e.g. a characterization test for `app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py` goes in `app/tests/characterization/dataset_generator/profit_loss/test_profit_loss_adder_characterization.py`.

`app/ai_modelling/dataset_generator/test_normalization.py` (pre-existing) is not a pytest test — no `assert`s, prints/shows plots for manual inspection. Leave it in place as an analysis script; don't migrate it into `app/tests/` as-is. If it grows real assertions later, split the assertions into `app/tests/integration/dataset_generator/test_normalization.py` and keep the plotting script separate.

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

One assertion focus per test — if two behaviors need separate reasoning to debug a failure, they're two tests, not one with multiple unrelated asserts.

## Fixtures

Shared synthetic-data builders go in `app/tests/conftest.py` as factory fixtures (a fixture that returns a function, so each test controls size/shape), not fixed DataFrames — e.g. `flat_ohlc(n)`, `trending_ohlc(n, direction)`. Keep fixtures small (5-30 rows) and deterministic.

Fixture/data policy by type:

- Unit/characterization/regression/smoke: synthetic in-memory DataFrames only — small (5-30 candles), deterministic, no disk/network reads. Real cached OHLCV under `data/` is large and non-deterministic across cache refreshes, which would make failures un-diagnosable.
- Integration: still synthetic by default; real cached data only when the test's actual point is the I/O/repository layer itself (rare — most of `app/`'s logic is transform, not storage).
- E2E: the one place real (or a small pinned real-data snapshot) OHLCV is acceptable, because the point is proving the full chain, not isolating a bug.

## Naming

`test_<unit>_<state_under_test>_<expected_result>` where the state/expected fit on one line; for characterization tests, `test_<function>_<fixture_name>` is fine since the point is pinning a value, not describing a behavior claim.

## Parametrize over copy-pasted near-duplicates

```python
@pytest.mark.parametrize("action_delay,expected_open", [(1, 105), (2, 106)])
def test_worst_long_open_by_action_delay(flat_ohlc, action_delay, expected_open):
    ...
```

Reach for this the moment two test functions differ only in literals.
