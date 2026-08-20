# OpenTelemetry migration & observability stack plan

Two halves of one effort: instrument the codebase with OpenTelemetry (replacing today's ad hoc
`profile_it` timing decorator), then stand up the backend that receives, stores, and visualizes what
gets emitted (Prometheus for metrics, Jaeger for traces, Grafana on top). A different concern from
experiment tracking (config/metrics/artifact-per-run comparison), which stays
[MLflow](../ML_Forecasting_System_Design/todo/03-infrastructure.md#todo) (todo item 1, still unresolved) — that answers "what config
produced this result," this stack answers "how is the system behaving and where did time go in one
specific run."

- [current state](#current-state)
- [decision: spans, not a blanket 1:1 port](#decision-spans-not-a-blanket-11-port)
- [why a Prometheus + Jaeger + Grafana backend](#why-a-prometheus--jaeger--grafana-backend)
- [phase 0 — dependencies & config](#phase-0--dependencies--config)
- [phase 1 — core tracing module](#phase-1--core-tracing-module)
- [phase 2 — migrate by layer, lowest-risk first](#phase-2--migrate-by-layer-lowest-risk-first)
- [phase 3 — metrics](#phase-3--metrics)
- [phase 4 — backend stack rollout](#phase-4--backend-stack-rollout)
- [grafana dashboards](#grafana-dashboards)
- [testing & rollout](#testing--rollout)
- [non-goals / deferred](#non-goals--deferred)

## current state

`profile_it` (`app/helper/logging/profiling/base.py:15-49`) is a wall-clock decorator: `time.time()`
before/after, logs `"func(...) started"` / `"func(...) executed in X.XXXs"` via Loguru (`log_d`),
colorama-colored by duration bucket. It stringifies args/kwargs for the log line (`parameters_to_str`,
line 116) but produces no structured data — no parent/child nesting, no exception capture, nothing
queryable. A sibling `profile_to_db` and a `sys.setprofile`-based global profiler exist in the same file
but have no call sites — dead code, candidate for deletion during this work.

39 active call sites across 21 files:

- `domain/price_action` (19 sites, hottest loops) — `BullBearSide.py` (6), `AtrMovementPivots.py` (6),
  `PeakValley.py` (2), plus `BasePattern.py`, `ColorTrend.py`, `BullBearSidePivot.py`,
  `PeakValleyPivots.py`; `domain/ohlcv/ohlcv.py` (2)
- `presentation` (13 sites, low frequency) — `OHLVC_plotter.py` (5), the market-structure plotters,
  `shared/plotter.py`
- `application` (4 sites) — `BasePatternStrategy.py`, `ExtendedStrategy.py`, `rolling_mean_std.py`
- `infrastructure` (1 site) — `ccxt_client.py:99`
- No use in the model/training layer today — an instrumentation gap, not a migration target.

No structlog, no OpenTelemetry, no metrics/tracing framework anywhere in the repo (`requirements.txt`,
all `.py`). Logging runs on Loguru + colorama (`app/helper/logging/do_log/log_it.py`). A FastAPI service
exists (`ray_id.py:37-55`, `ContextVarMiddleware`) using a contextvar-based correlation ID (`ray_id_var`)
set per request — the same shape OTel context propagation uses, so it's a bridge target, not a parallel
system to keep.

## decision: spans, not a blanket 1:1 port

Replace `profile_it` with a `traced` decorator that opens an OTel span — but only at real call
boundaries, not automatically at all 39 existing sites. Two different needs hide inside "profile_it
usage":

- **Boundary calls** (infra I/O, backtest orchestration, plot rendering, top-level pattern-detection
  entry points) — genuinely want a span: nesting, duration, exceptions, attributes. Low call frequency,
  low overhead risk.
- **Hot inner loops** (`BullBearSide`, `AtrMovementPivots` called per-candle/per-pivot over OHLCV data)
  — a span per call adds creation/export overhead and floods trace volume for no analytical benefit.
  These should emit **metrics** (a duration histogram + call counter per function) instead of a span per
  invocation.

Mixing these up is the main way this migration goes wrong.

## why a Prometheus + Jaeger + Grafana backend

OTel is vendor-neutral instrumentation only — it doesn't store or visualize anything itself. Once spans
and metrics are being emitted (phases 1-3), something needs to receive them:

- **Prometheus** — pull-based metrics store, cheap to query/alert on. Fits a long-running process (a
  training loop) scraping an in-process `/metrics` endpoint naturally; a short-lived batch job (one CCXT
  fetch, one indicator-compute pass) needs a Pushgateway instead, since there's nothing alive to scrape
  once it exits. Cardinality caution: label by pipeline stage/symbol/timeframe (bounded sets), never by
  run ID or timestamp.
- **Jaeger** — trace store, answers "for this one run, what happened and where did time go," not
  trends. Payoff here is per-stage/per-Optuna-trial waterfalls, not cross-service debugging — this repo
  is a single-process offline pipeline with [SOA not adopted](infrastructure.md#soa), so there's no
  service boundary yet for trace context to cross. Revisit if/when a live-trading service splits out.
- **Grafana** — shared visualization over both, Prometheus (PromQL) and Jaeger side by side, with
  exemplars linking a metric spike directly to the Jaeger trace for that sample.
- **OTel Collector** sits between the app and both backends: the app only ever speaks OTLP, the
  Collector fans it out to Prometheus's scrape format and Jaeger's ingest — swapping either backend
  later doesn't touch instrumented code.

## phase 0 — dependencies & config

- Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` to `requirements.txt`;
  `opentelemetry-instrumentation-fastapi` if the FastAPI service is in scope.
- Exporter is env-gated: default to a console/dev exporter (zero infra, keeps today's local workflow);
  optional OTLP exporter → Collector → Jaeger/Prometheus for deep-dive sessions or the full dashboard
  stack. Don't require Docker infra for the default dev loop.
- Sampler: `AlwaysOn` for local dev; leave root untouched for later if this ever runs as a deployed
  service.
- `Config` additions (`app/infrastructure/config`, pattern matching `Config.clickhouse_host` etc.):
  `Config.otel_collector_endpoint`, `DLF_OTEL_*`-overridable.
- **Prerequisite for the backend stack (not for phases 1-3, which need no Docker)**: this WSL distro has
  no `docker`/`dockerd` — the same blocker already documented for ClickHouse
  ([infrastructure.md § ClickHouse](infrastructure.md#clickhouse), "Docker runtime blocker as of
  2026-08-18"). Fix once, shared by both: enable Docker Desktop's WSL integration for `Ubuntu-24.04`, or
  install the native engine (`curl -fsSL https://get.docker.com | sudo sh`).

## phase 1 — core tracing module

New module, e.g. `app/helper/logging/tracing/` (sibling to existing `profiling/`, per this repo's
layering conventions):

- `init_tracing()` — `TracerProvider`, `Resource(service.name=...)`, `BatchSpanProcessor` (or
  `SimpleSpanProcessor` for tests), exporter selected by env var.
- Custom `SpanProcessor`/exporter that reproduces today's colored console line (`func(...) executed in
  X.XXXs`) from span name + duration + status, so local dev UX doesn't regress during migration.
- `traced` decorator (the `profile_it` replacement) — wraps the function body in
  `tracer.start_as_current_span(name)`; OTel records duration and nesting automatically. Reuse
  `parameters_to_str` (base.py:116) logic to populate span attributes instead of baking args into the
  log message. On exception: `span.record_exception` + `span.set_status(ERROR)`, then re-raise — this is
  strictly more than `profile_it` does today (its exception-handling branches are already commented out,
  lines 22-30).
- A separate lightweight `@counted`/`@timed_metric` decorator for hot-loop functions — increments a
  counter and records a histogram via the OTel Metrics API, no span, no per-call export.
- Bridge `ray_id` into OTel context: either drop `ray_id_var` in favor of the span's
  `trace_id`/`span_id` surfaced into the Loguru format, or keep `ray_id` as a span attribute during the
  transition so nothing downstream breaks immediately.
- Note on globals: OTel's SDK convention is a global provider (`trace.get_tracer(__name__)`) — that's
  the library's own API shape, not app state being smuggled through globals, so it doesn't violate this
  repo's [DI rule against mutating shared config](infrastructure.md#dependency-injection). Keep app-level
  data (stage name, symbol, timeframe) passed explicitly as span/metric attributes, not read from
  globals.

## phase 2 — migrate by layer, lowest-risk first

1. **Presentation** (13 sites) — plotting is low-frequency; safe first migration, validates the
   console-parity exporter end to end.
2. **Application** (4 sites) — backtesting/preprocessing; moderate frequency, still boundary-shaped
   calls.
3. **Infrastructure** (1 site, `ccxt_client.py:99`) — good candidate for real span attributes (symbol,
   timeframe, HTTP status) since it's genuine external I/O.
4. **Domain/price_action** (19 sites) — last, and *not* a straight swap: audit call frequency per
   function first. Entry points into a detection module (called once per run) become spans; per-candle/
   per-pivot inner calls become the metrics decorator instead.
5. Delete `profile_it`, `profile_to_db`, and the `sys.setprofile` global hook once nothing references
   them.

## phase 3 — metrics

Instrument things `profile_it` never touched — mapped to actual pipeline stages, not generic
placeholders:

| stage | metric |
|---|---|
| disk cache (`infrastructure/disk_cache.py`, `disk_cache_gaps.py`) | hit/miss rate, read/write duration, format-migration events (feather→parquet, relevant given the recent Parquet migration) |
| CCXT fetch (`infrastructure/market_data_fetch`) | fetch duration, rows fetched, gap-fill events, exchange API errors/rate-limit hits |
| indicator/feature computation (`domain/price_action`) | computation duration per indicator set (via the `@counted`/`@timed_metric` decorator from phase 1) |
| dataset generation (`train_data_of_mt_n_profit`, ram dataset producer) | duration, rows produced |
| training (CNN-LSTM trainer, Optuna) — currently zero instrumentation in `ai_models` | epoch duration, GPU utilization/memory (via GPU exporter, phase 4), Optuna trial duration, trial count |
| backtesting/evaluation | run counts, `forecast_vs_actual_error`, `model_last_trained_timestamp`, `feature_drift_score` |
| ClickHouse service | native metrics endpoint, scraped directly by Prometheus |

This is where OTel earns its keep beyond what timing logs already gave you — treat as a second iteration
after phase 2's span migration lands.

## phase 4 — backend stack rollout

Needs the [docker runtime prerequisite](#phase-0--dependencies--config) resolved first. Each step below
needs the previous one stable — don't parallelize:

1. OTel Collector (routing hub) up via docker-compose, no exporters wired yet.
2. Point the app's OTLP exporter (phase 0) at the Collector.
3. Prometheus scraping the Collector's Prometheus exporter.
4. Jaeger receiving traces via OTLP from the Collector.
5. Grafana on top of both, exemplar linking enabled last once Prometheus + Jaeger are both stable.

Docker-compose services, bind-mounted to `docker_volume/{otel,prometheus,jaeger,grafana}/` — same
native-ext4/gitignored convention as the existing `clickhouse` service
([infrastructure.md § ClickHouse](infrastructure.md#clickhouse)):

- `otel-collector` (`otel/opentelemetry-collector-contrib`) — receivers: OTLP; exporters: Prometheus,
  OTLP→Jaeger.
- `prometheus` — scrape config: the Collector's Prometheus exporter, ClickHouse's own metrics endpoint,
  and the GPU exporter.
- `jaeger` (all-in-one image, sufficient for single-node local use — no separate storage backend needed
  at this scale).
- `grafana` — datasources: Prometheus + Jaeger, provisioned dashboards (below).
- `pushgateway` — target for short-lived batch jobs (CCXT fetch, indicator compute) that exit before
  Prometheus could scrape them.
- GPU exporter (`nvidia-gpu-exporter` or DCGM) — this machine has one GPU (`nvidia-smi -L`: RTX 4060
  Laptop). Needs the NVIDIA container toolkit wired into whichever Docker runtime lands in phase 0 —
  check availability once that's resolved, don't assume it's present.

## grafana dashboards

- System/GPU utilization (resource usage during training).
- Pipeline-stage duration (RED: rate/errors/duration) across fetch → cache → features → dataset →
  train → backtest.
- ClickHouse service health.
- Optuna trial explorer via the Jaeger datasource (trace waterfall) with exemplar links from the
  training-duration histogram straight into the matching trace.
- Non-goal: a Grafana↔MLflow link (e.g. querying MLflow's backend store as a Grafana SQL datasource) —
  MLflow already has its own UI for run comparison; don't duplicate it here (YAGNI).

## testing & rollout

- Add a pytest fixture installing an in-memory `SpanExporter` so tests can assert spans/attributes exist
  for migrated boundary calls (see repo pytest conventions); same fixture pattern covers the phase-3
  metrics decorator (assert counter/histogram recorded, no real exporter).
- Roll out layer by layer per phase 2; each layer's PR should be independently revertable — don't
  migrate all 21 files in one change.
- No CI exists yet ([infrastructure.md § pre-commit](infrastructure.md#pre-commit)) — bringing the
  compose stack up and confirming Prometheus targets are `UP` / Grafana dashboards render is a manual
  check per phase 4, not an automated gate.

## non-goals / deferred

- **Logs signal**: OTel supports it (Collector → Loki), but out of scope for this pass — bridge the
  existing `app/helper/logging` module to OTel logs later if trace-to-log correlation becomes needed.
- **Alerting rules**: don't build these before dashboards/metrics have a stabilized baseline to alert
  against.
- **Cross-process/service tracing**: no payoff until a live-trading service actually splits out of this
  single-process pipeline ([infrastructure.md § SOA](infrastructure.md#soa)) — revisit then.
- **Revisiting the MLflow decision**: this plan doesn't replace [../ML_Forecasting_System_Design/todo/03-infrastructure.md item
  1](../ML_Forecasting_System_Design/todo/03-infrastructure.md#todo) — lock that decision independently; it covers experiment/artifact
  tracking, this plan covers operational observability.
