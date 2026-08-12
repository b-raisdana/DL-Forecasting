---
name: test-strategy
description: Use whenever deciding what kind of test a change needs, or reviewing whether a PR/change is adequately tested. Maps the Microsoft engineering-playbook test-type taxonomy onto this repo's actual nature (offline data/ML pipeline, no deployed service) so you pick the right category — unit, characterization, integration, regression, smoke, e2e, perf — instead of defaulting to "add a unit test". See docs/testing.md for the full, versioned reference this skill summarizes.
source: https://microsoft.github.io/code-with-engineering-playbook/automated-testing/
---

# Test Strategy

Full reference: [docs/testing.md](../../../docs/testing.md). This skill is the quick-decision version —
read the doc for directory layout, naming, fixture policy, marker config.

## This repo's nature

Offline data/ML pipeline: pandas dataframes → TensorFlow model → `backtrader` strategy. No HTTP API, no
UI, no other team's service calling in yet. That rules out several categories the MS playbook otherwise
lists — don't reach for them here (see "not applicable" below).

## Picking a type

| if the change is... | write a... | marker |
| --- | --- | --- |
| a pure function, no I/O (indicator math, label math, scaling) | unit test on a synthetic in-memory fixture | `unit` |
| touching legacy code with no independent spec to assert against (e.g. anything in `profit_loss_adder.py` before its spec-alignment lands) | characterization test — pin *today's actual* output first, don't hand-derive expected values | `characterization` |
| wiring multiple modules/stages together (dataset assembly, a repository read gated by a `PanderaDFM` schema) | integration test | `integration` |
| fixing a bug or protecting an invariant that broke before (e.g. no-lookahead) | regression test, named after the invariant, not the bug ticket | `regression` |
| a broad "does it still work at all" check safe to run every commit | smoke test | `smoke` |
| the full fetch→dataset→train→predict→strategy chain | e2e test, real/pinned data, not run on every commit | `e2e` |
| a vectorization/throughput claim (e.g. "no Python-level per-row loop") | perf test with an explicit budget | `perf` |

## Characterization tests: the key discipline

Never hand-compute expected values for legacy/unspecced code — run the real function against the fixture
and capture what it *actually* outputs today, then assert that. The goal is a safety net for refactoring,
not a spec-conformance check (that's what a regression test against the written spec is for, once one
exists). Getting this backwards (writing what you *think* it should output) produces a test that's wrong
from day one and blocks refactors with false failures.

## Not applicable here (and why)

- **CDC (consumer-driven contract)** — no separately-deployed consumer/provider pair this repo owns;
  CCXT is a third-party API, not a versioned contract to test against.
- **Synthetic monitoring / shadow testing** — nothing deployed/serving traffic.
- **UI testing** — plotly usage is diagnostic plotting, not a product UI.
- **Fault injection** — not yet; revisit once live trading makes real broker/network calls.

## When reviewing a PR/change for test coverage

Ask: what type does this change actually need, per the table above — not "does it have *a* test." A
correctness fix to legacy label-generation code needs a characterization test capturing the old and new
values (proving the change is intentional), not just a unit test asserting the new behavior in isolation.
