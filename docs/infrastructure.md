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
    - [incremental ratchet (mypy/ruff/xenon scope)](#incremental-ratchet-mypyruffxenon-scope)
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
Full performance/concurrency/resource-usage rules for all new code, plus a candidate list of
pandas/numpy-adjacent libraries evaluated for this repo: [performance-and-concurrency.md](performance-and-concurrency.md)
(enforced day to day by the skills indexed in [performance-skills.md](performance-skills.md)).

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
- **Filesystem location — code**: the clone lives on the Windows-mounted drive (`C:\Code\DL-Forecasting`
  = WSL `/mnt/c/Code/DL-Forecasting`, via `drvfs`), not a native WSL ext4 volume (e.g. under `~/`).
  Deliberate: it's the one location both Windows (`git.exe`, VS Code) and WSL (`tf` conda env) can
  read/write as the same clone, with no separate checkout or sync step. Rejected fix: moving the whole
  clone to a native WSL path — Windows-side git/VS Code would then need `\\wsl$\...` (or a second
  checkout kept in sync), reintroducing the dual-environment problem this layout avoids.
- **Filesystem location — data cache**: unlike the clone, the training-data cache under `data/` is
  gitignored, so Windows-side tooling never touches it — only the WSL `tf` env reads/writes it (dataset
  generation, training; `pytest`'s fast markers don't touch real `data/`, see [testing.md § fixture/data
  policy](testing.md#fixturedata-policy)). `drvfs` I/O is substantially slower than a native Linux
  filesystem, and this cache is the worst case for it: ~19.5k small per-day zip files under
  `data/Kucoin/Spot/*` (~8GB), each open crossing the WSL2 9p protocol. Since nothing requires dual-OS
  access here, `Config.path_of_data` is overridable via the `DLF_DATA_ROOT` env var (defaults to
  `<repo_root>/data` if unset); the `tf` conda env's `activate.d`/`deactivate.d` hooks set it to
  `/home/brais/dlf-data`, a native ext4 path, so WSL-side runs skip the `drvfs` tax entirely while
  Windows keeps the unchanged default.

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
  `[tool.ruff]`). Not `C90`/complexity - that's xenon's job, not duplicated here. `ruff check` runs
  with `--exit-zero` (auto-fixes what's safely fixable, never hard-blocks on the rest) because `ruff`
  is also an incremental-ratchet vector below - it shouldn't independently force-fixing a whole touched
  legacy file's unrelated violations on top of the ratchet's own gate.
- `mypy --strict --disallow-any-explicit` (`pyproject.toml` `[tool.mypy]`) — every function
  signature must be typed, `Any` may never appear explicitly in this repo's own code. Third-party
  imports without stubs still resolve as implicit `Any` (`ignore_missing_imports = true`) rather than
  erroring - full `--disallow-any-unimported` was rejected as impractical given `ccxt`/`pandas_ta`/
  `prophet` ship no stubs; would need stub packages authored for each first.
- `pytest -m "unit or characterization or regression or smoke"` — fast gate, see [testing.md](testing.md).
- Bypassable with `git commit --no-verify` — pre-commit is a local convenience gate, not a substitute for
  CI (none exists yet, see [testing.md § CI gate](testing.md#ci-gate)).

### incremental ratchet (mypy/ruff/xenon scope)

`mypy` and `radon`/`xenon` analyze whole files, not diff hunks, so an early "must be 100% clean on any
touched file" design (per-file, strict) turned a one-line edit to a legacy file into a forced cleanup of
every unrelated pre-existing violation in that file - a chain reaction into a dramatically bigger diff
than the change called for. A first baseline run found 911 pre-existing strict-mypy errors across
80/132 files, so that design would've blocked nearly any commit to a legacy file.

Replaced with `scripts/git-hooks/incremental-precommit/` (full design in that folder's README): track
each tool's ("vector": `mypy`, `ruff`, `xenon`) total problem count project-wide in a committed
`baseline.json`. A commit is only blocked if a vector's count goes **up** past its baseline (a real
regression you introduced) - never for pre-existing debt in a file you happen to touch. As debt gets
fixed (by anyone), once a vector's improvement reaches `chunk_size` (`config.json`, default 50), the
baseline ratchets down to the new lower count and is committed - locking the improvement in.

Mutation-safety note: fixing a real (non-mechanical) violation means hand-editing legacy code, which
risks silently changing behavior while "just satisfying the linter." No mutation-testing tool for this
(considered, rejected as too slow to run every commit - see the incremental-precommit README) - the
safeguard is test discipline instead: `ratchet_check.py` prints a reminder (not a block) when a commit
ratchets a baseline down without touching `app/tests/{characterization,unit,regression}/`, pointing at
[testing.md](testing.md)'s characterization-test discipline.

xenon's own thresholds (used by the `xenon` vector's count): `--max-absolute B --max-modules A
--max-average A` (blocks ranked worse than `B`).

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
