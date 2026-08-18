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
    - [ClickHouse](#clickhouse)
    - [vectorbt — not yet integrated](#vectorbt--not-yet-integrated)

## general guide

prefer vectorized pandas ops; tag any non-vectorized fallback with `NOT_VECTORIZED_OPERATION`.
prefered to do via well-known libs instead of self-implementation (research first / then implement).
Performance/concurrency/resource-usage rules for all new code, plus the candidate list of
pandas/numpy-adjacent libraries evaluated for this repo, are enforced day to day by the
`vectorized-pandas-numpy`, `lib-first`, and `concurrency-and-blocking` skills.

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
- **Filesystem location — code and data**: as of 2026-08-18, the whole clone (code and the `data/`
  cache alike) lives on a native WSL ext4 volume, `/home/brais/code/DL-Forecasting` — not the
  Windows-mounted `drvfs` drive. `Config.path_of_data` needs no override (`DLF_DATA_ROOT` still exists
  as a generic escape hatch, just unused here); it resolves to `<repo_root>/data` by default, which is
  now correct since `<repo_root>` itself is already native ext4. This replaced an earlier drvfs-based
  layout (`C:\Code\DL-Forecasting` = WSL `/mnt/c/Code/DL-Forecasting`) that put code on `drvfs` for
  dual Windows/WSL access, with the `data/` cache carved out separately to a native path
  (`/home/brais/dlf-data`) because `drvfs` was the worst case for its ~19.5k small per-day cache files
  (each open crossing the WSL2 9p protocol). That split no longer exists — `/mnt/c/Code/DL-Forecasting`
  isn't a live clone (checked: only a stale `.zip` backup remains on `C:\Code`), and `dlf-data` was
  merged back into `data/` and removed. Windows-side access (VS Code, Explorer) now goes through the
  `\\wsl.localhost\Ubuntu-24.04\home\brais\code\DL-Forecasting` UNC path instead of a drive-letter
  mount — the "second checkout to keep in sync" concern that previously ruled this out doesn't apply,
  since it's the same native filesystem, not a separate copy.
- **Clickable repo-path references in comments/docstrings**: VS Code's built-in path-link detection stops at the first whitespace, so a referenced filename containing a literal space (e.g. some `(handmade)` designset files) never lights up as clickable no matter how the surrounding text is formatted. The `DanLevett.pattern-links` ("Link Patterns") extension fixes this via two custom regex rules, but its `linkTarget` is passed straight to `vscode.Uri.parse()` with no `${workspaceFolder}` substitution or relative-to-document resolution — the target must be a hardcoded absolute `file://` path. That's machine-specific, so it's configured per-machine in the WSL remote's `~/.vscode-server/data/Machine/settings.json` (untracked, not `.vscode/settings.json`), not committed to the repo:
  - repo-root-relative paths (no leading `/`) starting with `app/`, `docs/`, or `scripts/` get the repo root prepended;
  - already-absolute paths (leading `/`, not preceded by a word character — avoids double-matching a relative path's inner segments) are used as-is;
  - `./sibling.py`-style paths resolve against a hardcoded folder (currently `app/application/model_implementations/tier1_000/`, since the rule has no per-document `${fileDirname}` — the extension only substitutes regex capture groups, not editor context) — update that hardcoded folder, or add another rule, if `./`-relative links are needed elsewhere.
  - All three stay on a single line (the regexes exclude newlines) — a path reference wrapped across lines in source needs joining onto one line to become clickable.

## methodologies for follow

### DDD

MVC doesn't fit (no request/response UI cycle) — layer by dependency direction instead
(Clean/Onion-style): Domain (market-structure TA logic + PanderaDFM schemas as value objects) →
Application (dataset generation, training, prediction, optimization, backtesting orchestration) →
Infrastructure (exchange/data-fetch, model-artifact persistence, config, logging) → Presentation
(plotting, entrypoints). Full layer-to-module mapping and placement rules: the `code-layers` skill;
migration order/cleanup plan: [todos/infrastructure.md](todos/03-infrastructure.md#todo) item 12.

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
- `pytest -m "unit or characterization or regression or smoke"` — fast gate, see the `pytest` skill.
- Bypassable with `git commit --no-verify` — pre-commit is a local convenience gate, not a substitute for
  CI (none exists yet).

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
the `test-strategy` skill's characterization-test discipline.

xenon's own thresholds (used by the `xenon` vector's count): `--max-absolute B --max-modules A
--max-average A` (blocks ranked worse than `B`).

`loc` vector: sum of `max(0, line_count - 500)` over every `app/**/*.py` file - a file-length countermeasure `xenon` doesn't cover (cyclomatic complexity per function, not raw file size). Size policy: `<300` lines is normal, and new/generated files should stay below this size; when modifying a file, prefer moving the touched method/function to its proper location if that naturally reduces the file. `300-500` lines is a potential low-priority split todo. `>500` lines is a warning and high-priority split todo. The ratchet's 500-line threshold is stricter than SonarQube's `S104`/pylint's `C0302` default of 1000, deliberately, so it starts counting excess on files already in the 500-1000 range instead of only reacting once they cross 1000.

### tests

See the `test-strategy` skill for the test-type taxonomy and the `pytest` skill for `app/tests/` layout
and marker convention — complements this complexity gate with a correctness gate.

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

### ClickHouse

Client-server OLAP database — evaluated as a target store for migrating `infrastructure.disk_cache`'s on-disk (feather/ZSTD) artifact cache off flat files, then parked: a single-process offline pipeline with no other concurrent consumer doesn't need a network-served store's daemon/port management (`infrastructure.md` § SOA). Kept wired up (not removed) for the scenario that would actually justify one: live/paper trading with multiple concurrent readers hitting one instance at once. The disk-cache migration itself starts elsewhere — see [data_pipeline_upgrade_plan.md](data_pipeline_upgrade_plan.md) (currently scoped to its first step, Feather → Parquet).

- **Runtime**: `docker-compose.yml` `clickhouse` service (`clickhouse/clickhouse-server:24.8`), HTTP `:8123` + native TCP `:9000`, data/logs bind-mounted to `docker_volume/clickhouse/` (native ext4, already gitignored — same rationale as [environments](#environments)'s filesystem-location note: avoids the WSL2 9p/drvfs penalty).
- **WSL access**: `localhost:8123` works from both WSL and Windows — WSL2's `localhostForwarding` (on by default) makes this transparent regardless of which Docker runtime backs it, no IP/hostname juggling needed.
- **Python client**: `clickhouse-connect` (`requirements.txt`) — official HTTP client; `query_df`/`insert_df` read/write pandas DataFrames directly, a better fit for this repo's pandas-centric pipeline than the older TCP-only `clickhouse-driver`.
- **Config**: `Config.clickhouse_host/_port/_user/_password/_database` (all `DLF_CLICKHOUSE_*`-overridable), defaults matching `docker-compose.yml`'s own `CLICKHOUSE_USER=dlf`/`CLICKHOUSE_PASSWORD=dlf`/`CLICKHOUSE_DB=dl_forecasting`. `infrastructure/clickhouse_client.py` (`get_clickhouse_client()`, `clickhouse_is_reachable()`) is the connection factory — first building block only, see below.
- **Docker runtime blocker (as of 2026-08-18)**: this WSL distro has no `docker`/`dockerd` — Docker Desktop is installed on Windows but WSL integration isn't enabled for `Ubuntu-24.04`, and a native install needs `sudo` (password-protected, unavailable to the agent that set this up). Either unblocks it:
  - Docker Desktop → Settings → Resources → WSL Integration → enable `Ubuntu-24.04` → Apply & Restart. Fastest — no packages to install, backend already present.
  - Native engine inside WSL: `curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER` (relogin or `newgrp docker` after) — skips Docker Desktop's translation layer entirely; needs the `sudo` password once.

  Either way: `docker compose up -d clickhouse` from the repo root once a runtime is live.
- **Not the disk-cache migration target**: `infrastructure.disk_cache`'s generic `(data_frame_type, date_range_str)` engine backs ~20+ callers (`domain/price_action/*`, `domain/schemas/common/ExtendedDf.py`, `infrastructure/ohlcv/*`, ...) in a single-process offline pipeline with no other concurrent consumer — a client-server store adds daemon/port management this repo's architecture (`infrastructure.md` § SOA) doesn't call for. DuckDB gets the same range-query win over Parquet with zero server; see the linked plan doc for the full per-layer breakdown.

### vectorbt — not yet integrated

planned for backtesting; not in `requirements.txt`, no imports in codebase yet.
