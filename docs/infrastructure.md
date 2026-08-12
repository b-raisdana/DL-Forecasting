# methodology, tools and infrastructure selection

- [methodology, tools and infrastructure selection](#methodology-tools-and-infrastructure-selection)
  - [general guide](#general-guide)
  - [methodologies for follow](#methodologies-for-follow)
    - [DDD](#ddd)
    - [SOA](#soa)
    - [principles](#principles)
    - [Repository design pattern](#repository-design-pattern)
    - [Dependency injection](#dependency-injection)
  - [QA](#qa)
    - [monitor and maintain code complexity](#monitor-and-maintain-code-complexity)
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

## methodologies for follow

### DDD

### SOA

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

### monitor and maintain code complexity

`radon` for cyclomatic complexity / maintainability index metrics; `xenon` (wraps `radon`,
non-zero exit above threshold) as the pre-commit gate.

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
