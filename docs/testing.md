# testing strategy

Test-type taxonomy, directory layout, and tagging convention for `app/`. Complements the QA/complexity
gate (`radon`/`xenon`) in [infrastructure.md](infrastructure.md#qa) — that gate catches complexity, this
strategy catches correctness.

- [testing strategy](#testing-strategy)
  - [nature of this project](#nature-of-this-project)
  - [test types](#test-types)
  - [directory layout](#directory-layout)
  - [naming](#naming)
  - [fixture/data policy](#fixturedata-policy)
  - [markers and running](#markers-and-running)
  - [CI gate](#ci-gate)
  - [open items](#open-items)

## nature of this project

Offline data/ML pipeline (pandas dataframes → TensorFlow model → `backtrader` strategy), not a deployed
service: no HTTP API, no UI, no other team's service calling in. That rules out several categories from
the Microsoft engineering-playbook taxonomy (automated-testing) outright — see table below.

## test types

| type | applicable | meaning here | marker |
| --- | --- | --- | --- |
| Unit | yes | pure function, no I/O — e.g. one `profit_loss_adder.py` function on an in-memory synthetic OHLC frame | `unit` |
| Characterization | yes | pins today's *actual* output of legacy/unspecced code (no independent spec to assert against) before refactor — Michael Feathers' term, not in the MS list but explicitly needed here (see [training-data-labels.md](todos/training-data-labels.md) step 1) | `characterization` |
| Integration | yes | multiple modules/stages wired together — e.g. `add_long_n_short_profit` → `train_data_of_mt_n_profit` dataset assembly, or a `PanderaDFM` schema gating a repository read | `integration` |
| Regression | yes | guards one specific previously-true invariant so it can't silently break — e.g. the no-lookahead check in [training-data-labels.md](todos/training-data-labels.md) step 11 | `regression` |
| Smoke | yes | minimal "still imports / still builds / one batch runs" check — cheap enough for every commit | `smoke` |
| E2E | yes, expensive | fetch → dataset → 1-epoch train → predict → strategy signal, real (or cached real) data | `e2e` |
| Load/Performance | yes, narrow | throughput budget for vectorized dataset generation (ties to the "prefer vectorized pandas ops" rule in [infrastructure.md](infrastructure.md#general-guide)) | `perf` |
| CDC | no | no separately-deployed consumer/provider pair we own — CCXT is a third-party API, not a contract we version | — |
| Synthetic monitoring | no | nothing deployed/serving traffic yet | — |
| Shadow | no | no production system to compare old/new against | — |
| Fault injection | not yet | relevant once live trading exists (broker/network failures); revisit then | — |
| UI | no | plotly usage here is diagnostic plotting, not a product UI | — |

## directory layout

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

One tree under `app/tests/`, split by type first then mirroring the source package path — not
colocated `test_*.py` next to source, and not one flat folder. Rationale: type is the axis CI selects
on (`-m unit`, skip `e2e` by default), so it has to be the top split; mirroring source under that keeps
a test's home discoverable from its target's path. Only create a subfolder when it has a test in it
(YAGNI, per [infrastructure.md](infrastructure.md#principles)) — don't pre-scaffold empty type folders.

`app/ai_modelling/dataset_generator/test_normalization.py` (pre-existing) is not a pytest test — no
`assert`s, prints/shows plots for manual inspection. Leave it in place as an analysis script; don't
migrate it into `app/tests/` as-is. If it grows real assertions later, split the assertions into
`app/tests/integration/dataset_generator/test_normalization.py` and keep the plotting script separate.

## naming

`test_<unit>_<state_under_test>_<expected_result>` (MS playbook convention) where practical, e.g.
`test_max_profit_n_loss_flat_candles_zero_distance`. For characterization tests, the "state" is the
input fixture and "expected" is literally today's captured output, so a shorter
`test_<function>_<fixture_name>` is fine — the point is pinning a value, not describing intent.

## fixture/data policy

- Unit/characterization/regression/smoke: synthetic in-memory `DataFrame`s built in `conftest.py` or the
  test file — small (5-30 candles), deterministic, no disk/network reads. Matches the "avoid reading from
  disk" rule from the MS unit-testing guidance: real cached OHLCV under `data/` is large, non-deterministic
  across cache refreshes, and would make failures un-diagnosable.
- Integration: still synthetic data by default; real cached data only when the test's actual point is the
  I/O/repository layer itself (rare — most of `app/`'s logic is transform, not storage).
- E2E: the one place real (or a small pinned real-data snapshot) OHLCV is acceptable, because the point is
  proving the full chain, not isolating a bug.

## markers and running

Registered in `pytest.ini`: `unit`, `characterization`, `integration`, `regression`, `smoke`, `e2e`,
`perf`. `--strict-markers` — an unregistered marker is a test-authoring bug, not a typo to silently
ignore.

- Fast gate (every commit): `pytest -m "unit or characterization or regression or smoke"`
- Full (nightly/manual): `pytest` (everything, including `e2e`/`perf`)

Execution environment: this repo's runtime deps (`pandas`, `pandas_ta`, `tensorflow`, `pandera`, ...)
live in the WSL conda env `tf` (`/home/brais/miniconda3/envs/tf`), not in any Windows Python — see the
`pytest` skill for the exact invocation.

## CI gate

No CI workflow file exists yet (see [infrastructure.md](infrastructure.md)). The fast test gate
(`pytest -m "unit or characterization or regression or smoke"`) is wired into `.pre-commit-config.yaml`
alongside the `xenon` complexity gate and `mypy` strict-typing gate (see
[infrastructure.md § pre-commit](infrastructure.md#pre-commit)) — local/per-commit only, new/modified
files only, bypassable with `--no-verify`. When CI lands, promote the same command to a required check,
per [training-data-labels.md](todos/training-data-labels.md) step 11 ("wire into whatever CI gate `xenon`
runs").

## open items

- No CI runner defined yet — commands above are the contract CI should eventually run, not something
  currently enforced.
- `fault_injection` marker: add once live trading (real broker/network calls) exists.
