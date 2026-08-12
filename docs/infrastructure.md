# methodology, tools and infrastructure selection

- [methodology, tools and infrastructure selection](#methodology-tools-and-infrastructure-selection)
  - [general guide](#general-guide)
  - [methodologies for follow](#methodologies-for-follow)
    - [DDD](#ddd)
    - [SOA](#soa)
    - [principles](#principles)
    - [Repository design pattern](#repository-design-pattern)
    - [Dependency injection](#dependency-injection)
  - [environments](#environments)
  - [QA](#qa)
    - [pre-commit](#pre-commit)
    - [monitor and maintain code complexity](#monitor-and-maintain-code-complexity)
    - [tests](#tests)
  - [libraries](#libraries)
    - [TensorFlow](#tensorflow)
    - [pandas-ta](#pandas-ta)
    - [pandera](#pandera)
    - [Optuna](#optuna)
    - [CCXT](#ccxt)
    - [plotly](#plotly)
    - [Docker](#docker)
    - [vectorbt — not yet integrated](#vectorbt--not-yet-integrated)

## general guide

prefer vectorized pandas ops; tag any non-vectorized fallback with `NOT_VECTORIZED_OPERATION`.
prefered to do via well-known libs instead of self-implementation (research first / then implement).

## environments

- **WSL `tf` conda env** (`/home/brais/miniconda3/envs/tf`) — the one place this repo's real deps
  (`pandas`, `tensorflow`, `pandera`, `pandas_ta`, ...) and dev tooling (`pytest`, `mypy`, `pre-commit`)
  live. `tensorflow[and-cuda]` doesn't install on Windows (CUDA extras are Linux-only), so there's no
  Windows equivalent - don't create one.
- **Windows** only runs `git.exe` itself (commits happen from PowerShell/VS Code). `.git/hooks/pre-commit`
  is a thin shim (tracked at `scripts/git-hooks/pre-commit`, installed via `scripts/git-hooks/install.sh`
  since `.git/hooks/` isn't version-controlled) that hops into WSL once via `wsl.exe -d Ubuntu-24.04` and
  runs `pre-commit` natively from there - so mypy/pytest see real types and imports instead of falling
  back to `Any` on missing stubs. No persistent Windows Python venv is needed or maintained for this.
- Rejected: a dedicated Windows venv running mypy/ruff/xenon directly. Doubles the environments to
  maintain, can't install the real deps (see above), so mypy would ignore-missing-import everything into
  implicit `Any` - defeats the point of the strict/no-`Any` gate below.
- **Filesystem location**: the clone lives on the Windows-mounted drive (`C:\Code\DL-Forecasting` =
  WSL `/mnt/c/Code/DL-Forecasting`, via `drvfs`), not a native WSL ext4 volume (e.g. under `~/`).
  Deliberate: it's the one location both Windows (`git.exe`, VS Code) and WSL (`tf` conda env) can
  read/write as the same clone, with no separate checkout or sync step. Cost: `drvfs` I/O is
  substantially slower than a native Linux filesystem, so large data-pipeline runs (dataset generation,
  bulk OHLCV read/write under `data/`) pay a throughput penalty. Rejected fix: moving the clone to a
  native WSL path — Windows-side git/VS Code would then need `\\wsl$\...` (or a second checkout kept in
  sync), reintroducing the dual-environment problem this layout avoids. If pipeline I/O becomes a
  measured bottleneck, revisit as a scoped decision rather than a blanket relocation.

## methodologies for follow

### DDD

MVC doesn't fit (no request/response UI cycle) — layer by dependency direction instead
(Clean/Onion-style): Domain (market-structure TA logic + PanderaDFM schemas as value objects) →
Application (dataset generation, training, prediction, optimization, backtesting orchestration) →
Infrastructure (exchange/data-fetch, model-artifact persistence, config, logging) → Presentation
(plotting, entrypoints). Full layer-to-module mapping, violations found, and migration order:
[architecture-layers.md](architecture-layers.md).

### SOA

Not adopted — single-process offline pipeline, no independently-deployed services. The
"application services" in the DDD layering above (orchestrators like `train_data_of_mt_n_profit`,
a trainer, a predictor) are service-*shaped* (one entry point, coordinate lower layers) but stay
in-process; revisit only if a stage needs independent scaling/deployment.

### principles

- SOLID
- DRY
- YAGNI

### Repository design pattern

- any module that reads or writes a persisted artifact (exchange data, cached CSV/parquet/zip/npz,
  computed indicators) should go through a repository interface instead of calling the storage/exchange
  API inline — callers depend on a `get`/`save` contract, not on CCXT, file layout, or cache format.
- one repository per artifact type (not per caller); if two places implement their own
  read-or-fetch-and-cache logic for the same kind of data, that's the signal to introduce one.
- swapping the underlying store (e.g. CSV → parquet, local disk → object storage) or mocking data
  access in tests should require touching only the repository, not every caller.

### Dependency injection

- config and other shared state should be passed into constructors/functions explicitly, not pulled
  from a module-level singleton mid-function — a function's dependencies should be visible in its
  signature.
- never mutate shared/global config to pass state between calls; it makes call order matter and rules
  out running things concurrently.
- favor DI wherever a component currently reaches out to global state instead of receiving what it
  needs as arguments — this also makes the component trivially testable/mockable in isolation.

## QA

### pre-commit

Config: `.pre-commit-config.yaml`. Runs inside WSL `tf` (see [environments](#environments)); one-time
setup per clone is `bash scripts/git-hooks/install.sh`.

- Hygiene: `pre-commit-hooks` (trailing-whitespace, end-of-file-fixer, check-yaml/toml,
  check-added-large-files at 1MB - this repo has a history of committed multi-MB `.keras`/`.zip` files,
  check-merge-conflict, debug-statements).
- `ruff` — lint (`E,F,I,UP,B,C4,SIM`) + format, 120-char line length (`pyproject.toml`
  `[tool.ruff]`). Not `C90`/complexity - that's xenon's job, not duplicated here.
- `xenon`/`radon` — complexity gate, see below.
- `mypy --strict --disallow-any-explicit` (`pyproject.toml` `[tool.mypy]`) — every function
  signature must be typed, `Any` may never appear explicitly in this repo's own code. Third-party
  imports without stubs still resolve as implicit `Any` (`ignore_missing_imports = true`) rather than
  erroring - full `--disallow-any-unimported` was rejected as impractical given `ccxt`/`pandas_ta`/
  `prophet` ship no stubs; would need stub packages authored for each first.
- `pytest -m "unit or characterization or regression or smoke"` — fast gate, see [testing.md](testing.md).
- **Scope: new/modified files only**, not the whole repo. A strict-mypy baseline run found 911
  pre-existing errors across 80/132 files; gating the whole repo at once would block unrelated commits
  until all of that legacy debt was fixed first. `mypy`/`ruff`/`xenon` only see the files pre-commit
  passes them (git diff), and `follow-imports=silent` on mypy stops that check from cascading into
  untouched files' errors. Files tighten to the strict standard as they're touched, not all at once.
- Bypassable with `git commit --no-verify` — pre-commit is a local convenience gate, not a substitute for
  CI (none exists yet, see [testing.md § CI gate](testing.md#ci-gate)).

### monitor and maintain code complexity

`radon` for cyclomatic complexity / maintainability index metrics; `xenon` (wraps `radon`,
non-zero exit above threshold) as the pre-commit gate: `--max-absolute B --max-modules A --max-average A`.

### tests

See [testing.md](testing.md) for the test-type taxonomy, `app/tests/` layout, and pytest marker
convention — complements this complexity gate with a correctness gate.

## libraries

### TensorFlow

DL framework (CNN-LSTM-attention models). GPU via `tensorflow[and-cuda]`; Docker base image `tensorflow:25.01-tf2-py3`.

### pandas-ta

technical analysis indicators.

### pandera

DataFrame schema/dtype validation (`PanderaDFM/*`).

### Optuna

architecture + hyperparameter search (`optuna_optimizer.py`): TPE sampler + Hyperband
pruning for the main search (architecture is one categorical param in the same study as
its hyperparameters, so bad architectures get pruned early instead of each getting an
exhaustive run); NSGA-II reserved for a later multi-objective refinement stage once
trading-KPI backtesting exists.

### CCXT

crypto exchange/broker communication (data fetch).

### plotly

visualization.

### Docker

containerized runtime (`Dockerfile`, `docker-compose.yml`).

### vectorbt — not yet integrated

planned for backtesting; not in `requirements.txt`, no imports in codebase yet.
