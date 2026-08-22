# ratchet_check.py — analysis review and upgrade todo plan

## current role and scope

`scripts/git-hooks/incremental-precommit/ratchet_check.py` is the live gate for every commit.
It re-measures four project-wide static-analysis vectors (`mypy`, `ruff`, `xenon`, `loc`) and
blocks only when a vector's count goes **up** past its committed baseline in `baseline.json`.
It never forces fixing pre-existing debt in touched files — that is the whole point of the
incremental ratchet design documented in `scripts/git-hooks/incremental-precommit/README.md`
and `docs/infrastructure.md` § incremental ratchet.

Because this file runs on every commit, correctness, error handling, and testability matter more
than raw performance. The upgrades below are ordered by risk and impact.

---

## architecture overview

```
main()
  ├── load baseline.json
  ├── for each vector in VECTORS:
  │     ├── project-wide count_fn()              ← mypy_count, ruff_count, xenon_count, loc_count
  │     ├── compare with baseline
  │     ├── candidate_regressions  (current > base)
  │     ├── improved               (current < base)
  │     └── bootstrapped           (no baseline yet)
  ├── staged_app_python_files()                    ← git diff --cached
  ├── for each candidate_regression:
  │     └── detail_counter(staged_paths)           ← counts problems only in staged files
  │           ├── regressed       (staged count > 0)
  │           └── ignored_regression (staged count == 0)
  ├── block if any regressed
  ├── ratchet down baselines where improvement >= 3%
  ├── rewrite baseline.json + git add it
  └── print summary / characterization-test reminder
```

---

## detailed findings

### F1 — silent tool-failure masking (P0)

**Location**: `run()` (`ratchet_check.py:39-41`), `run_output()` (`ratchet_check.py:44-46`),
all count functions (`mypy_count`, `ruff_count`, `xenon_count`, `loc_count`).

**Weakness**: `subprocess.run(..., check=False)` ignores non-zero exit codes. If `mypy`,
`ruff`, or `radon` is not installed, crashes, or times out, the function receives an empty
stdout and returns 0 — silently passing the commit. `run()` also discards `result.stderr`
entirely; `run_output()` concatenates it but the callers only look for success patterns.

**Upgrade**:
- Add `result.check_returncode()` (or at least inspect `result.returncode`) and raise a
  descriptive `RuntimeError` when the tool exits non-zero.
- Add a per-tool timeout (e.g. `timeout=300`) so a hung `mypy` on a pathological file does
  not freeze the commit indefinitely.
- When stdout cannot be parsed, treat it as an error, not as "0 problems".

**Factor**: correctness / error handling. Hot path: yes — runs every commit. Mutation-safety:
N/A (behavioral fix — wrong answers become loud failures instead of silent passes).

---

### F2 — duplicate project-wide vs. detail-count logic (P1)

**Partially resolved**: the per-rule-code split (see the README's "keys" section) introduced `_group_ruff`/`_group_mypy`, shared parsing helpers now called by both the project-wide and detail counters for those two tools - the JSON-decode/regex-match duplication F2 flagged for `mypy`/`ruff` is gone. `xenon`/`loc` still have separate project/detail functions, and the `run`/`run_output` split is untouched; the rest of this item still applies to those.

**Location**: `mypy_count` / `mypy_detail_count` (`ratchet_check.py:49-59` vs. `ratchet_check.py:131-137`),
`ruff_count` / `ruff_detail_count` (`ratchet_check.py:62-68` vs. `ratchet_check.py:148-156`),
`xenon_count` / `xenon_detail_count` (`ratchet_check.py:75-89` vs. `ratchet_check.py:183-195`),
`loc_count` / `loc_detail_count` (`ratchet_check.py:92-99` vs. `ratchet_check.py:211-216`).

**Weakness**: Each vector has two near-identical functions that differ only in whether they
operate on all project files or a filtered `paths` list. The `run` / `run_output` split and
the JSON-parsing / regex-parsing blocks are also duplicated. Four vectors × two variants = eight
functions where four would suffice with a shared `count_violations(tool, paths=None)` helper.

**Upgrade**:
- Introduce a `_count_violations(vector_name, paths=None)` helper that accepts an optional
  file list and delegates to the tool with either the project target or the explicit paths.
- Collapse `run` and `run_output` into a single `_run(args, cwd, capture_stderr=False)` with a
  flag, or always capture both and let callers pick.
- Share the JSON-decode / regex-match / rank-filter logic in one place per vector.

**Factor**: duplication/simplification. Hot path: yes — called every commit. Mutation-safety:
pending — extract carefully; add regression tests for the helper before collapsing (see F3).

---

### F3 — zero automated tests for the gate itself (P0)

**Partially covered**: `tests/unit/git_hooks/test_ratchet_check.py` now exists and covers rule-code grouping, bootstrap, pass/block/ignored-regression, baseline resync + key self-pruning, and the characterization-test reminder (cases 1, 2, 5, 6, 10 below, adapted for the per-key/resync-on-success design - cases 3/4 no longer apply since the chunk_size/percentage threshold was removed, see F4). Cases 7-9 (loud failure on tool crash / malformed baseline / unparseable output) are still open - they depend on F1, which hasn't been implemented.

**Location**: entire file.

**Weakness**: The incremental ratchet is the single highest-trust component in the pre-commit
pipeline — if it breaks, every commit either silently passes bad code or blocks good code — yet
there are no tests for it anywhere under `app/tests/`. The four vectors, the baseline-bootstrap
path, the ratchet-down threshold, the ignored-regression branch, and the characterization-test
reminder are all untested.

**Upgrade**:
- Add `tests/unit/git_hooks/test_ratchet_check.py` (or `tests/regression/...` given the
  file's role as infrastructure).
- Test cases:
  1. Baseline missing → bootstraps at current count, does not block.
  2. Current count == baseline → passes.
  3. Current count < baseline but < 3% improvement → records improvement, does not ratchet.
  4. Current count < baseline and >= 3% improvement → ratchets baseline, stages file.
  5. Project-wide regression + staged regression → blocks with exit code 1.
  6. Project-wide regression + zero staged regression → passes (ignored regression branch).
  7. Tool not installed / exits non-zero → loud failure, not silent pass.
  8. `baseline.json` malformed → loud failure, not silent pass.
  9. `mypy` output format change (no "Found N error" match) → treated as parse error, not 0.
  10. Characterization-test reminder prints when ratcheting without touching test dirs.
- Use `unittest.mock` to mock `subprocess.run`, `Path.exists`, `Path.read_text`,
  `Path.write_text`, and `staged_app_python_files` so tests run without real tools or git state.

**Factor**: correctness / coverage. Hot path: yes — runs every commit. Mutation-safety:
required — write these tests before any other refactor (F2, F4, F5).

---

### F4 — hardcoded constants that belong in config (P1)

**Resolved by removal, not by wiring config.json in.** `RATCHET_IMPROVEMENT_RATIO`/`chunk_size` are gone: baseline.json now fully resyncs to the fresh counts after every successful commit instead of waiting for a threshold, so there's no ratio/chunk_size left to configure. `config.json` was deleted. See `scripts/git-hooks/incremental-precommit/README.md` § "keeping baseline current". The rest of this item (below) is left for history; `TARGET`/`COMPLEXITY_RANKS`/`max_lines`/`max_absolute` are still hardcoded and the config-extraction idea still applies to those if wanted later.

**Location**: `RATCHET_IMPROVEMENT_RATIO = 0.03` (`ratchet_check.py:30`),
`TARGET = "app"` (`ratchet_check.py:28`),
`COMPLEXITY_RANKS = "ABCDEF"` (`ratchet_check.py:29`),
`max_lines: int = 500` in `loc_count` / `loc_detail_count` (`ratchet_check.py:92,211`),
`max_absolute: str = "B"` in `xenon_count` / `xenon_detail_count` (`ratchet_check.py:75,183`).

**Weakness**: `config.json` held only `chunk_size: 3` (this doc and the README each cited a different, wrong default before the resolution above). The ratchet improvement threshold, the xenon rank ceiling, the loc line threshold, and the target directory are all hardcoded in the script despite being policy decisions that already have prose documentation in `infrastructure.md` and the README. Moving them to `config.json` makes them editable without touching code and keeps policy and implementation in one place.

**Upgrade** (for the still-hardcoded constants only - `chunk_size`/`ratchet_improvement_ratio` no longer apply, see the resolution note above):
- Introduce a `config.json` schema, if wanted:
  ```json
  {
    "xenon_max_absolute": "B",
    "loc_max_lines": 500,
    "target": "app"
  }
  ```
- Load these values in `main()` or at module level, falling back to the current defaults when
  keys are absent.
- `COMPLEXITY_RANKS` can stay as a constant (it is an enum, not a policy knob), but document
  why it is `"ABCDEF"` and not a longer/shorter string.

**Factor**: configuration / policy coupling. Hot path: N/A (read once per commit). Mutation-safety:
N/A (defaults preserve current behavior; missing keys fall back to current hardcoded values).

---

### F5 — fragile `baseline.json` write and git-staging race (P1)

**Location**: `BASELINE_PATH.write_text(...)` (`ratchet_check.py:311-312`).

**Weakness**:
1. `write_text` is not atomic — if the process receives SIGTERM mid-write, `baseline.json` is
   left as a truncated/partial file, breaking every subsequent commit until someone manually
   fixes it.
2. `subprocess.run(["git", "add", ...])` stages the file unconditionally whenever `bootstrapped
   or ratcheted`. If the hook itself fails later (e.g. `main()` raises after the write), the
   user ends up with a partially staged `baseline.json` in their index — a confusing state.
3. If two commits run the ratchet concurrently (e.g. two terminals), both read the same
   baseline, both compute new values, and the last writer wins — potentially losing a ratchet
   the other commit just earned.

**Upgrade**:
- Write to a temp file in the same directory, then `os.replace()` (atomic on POSIX) onto
  `baseline.json`.
- Stage `baseline.json` only after the full `main()` logic succeeds and the return value is
  confirmed as 0 (pass). Better: return the path to stage and let the caller (the hook wrapper)
  stage it, separating the check from the mutation.
- For the concurrency case: add a simple advisory lock (e.g. `fasteners.InterProcessLock`
  or a `.lock` file with `fcntl.flock`) around the read-modify-write of `baseline.json`.

**Factor**: correctness / concurrency. Hot path: yes — runs every commit. Mutation-safety:
pending — the atomic-replace and lock changes are mechanical, but the "stage only on success"
split changes the hook wrapper's contract and needs the tests from F3 first.

---

### F6 — regex and JSON parsing fragility (P2)

**Location**: `re.search(r"Found (\d+) error", stdout)` in `mypy_count` (`ratchet_check.py:58`)
and `mypy_detail_count` (`ratchet_check.py:136`).

**Weakness**: The regex assumes mypy's exact English output phrasing `"Found N error"`. A
mypy version bump that switches to `"Found N errors"` (plural), `"N error(s) found"`, or a
localized output would cause the regex to return `None` and the function to silently return 0.
`ruff`'s JSON parsing already handles the empty/invalid case, but `xenon`'s rank lookup uses
`block.get("rank", "A")` which silently accepts a missing key.

**Upgrade**:
- Prefer machine-readable flags over text parsing: `mypy --show-error-codes --no-error-summary
  --json` (mypy 1.0+ exposes `--show-traceback` and JSON output modes; if not available, the
  regex should at least be case-insensitive and accept both singular/plural).
- If JSON output is unavailable, make the regex more defensive: `re.search(r"Found\s+(\d+)\s+error", stdout, re.IGNORECASE)` and log a warning when the match is `None` instead of returning 0.
- For `xenon`, change `block.get("rank", "A")` to an explicit default with a logged warning
  when the key is missing, so a radon API change is visible.

**Factor**: robustness. Hot path: yes — runs every commit. Mutation-safety: N/A (parsing
tightening; current default of 0 on miss is the bug being fixed).

---

### F7 — `_exclude_tests` inconsistency (P2)

**Location**: `xenon_detail_count` calls `_exclude_tests(paths)` (`ratchet_check.py:184`), but
`xenon_count` does not (`ratchet_check.py:75-89`).

**Weakness**: The project-wide xenon count includes test files and `archive_not_used_trash`,
while the staged-only xenon detail count excludes them. This means a regression detected by
`xenon_count` may be "ignored" by `xenon_detail_count` not because staged files are clean, but
because the project-wide count includes files that the detail count filters out. The two counts
are measuring different scopes but are compared against the same baseline.

**Upgrade**: Either exclude tests/archive from both, or include them in both. The README and
`infrastructure.md` describe xenon's scope as `app/` (excluding tests), so the project-wide
`xenon_count` should call `_exclude_tests` too. Alternatively, make the exclusion list
configurable and apply it uniformly.

**Factor**: correctness. Hot path: yes. Mutation-safety: pending — changing the baseline's
meaning requires a baseline reset (document the reset step).

---

### F8 — `loc_count` scope drift (P2)

**Location**: `loc_count` (`ratchet_check.py:92-99`) vs. `loc_detail_count` (`ratchet_check.py:211-216`).

**Weakness**: `loc_count` walks `(ROOT / TARGET).rglob("*.py")` and excludes `__pycache__` and
`archive_not_used_trash`, but does **not** exclude `tests/`. `loc_detail_count` counts only
staged paths (which are already filtered by `staged_app_python_files` to exclude
`archive_not_used_trash` but not `tests/`). The project-wide loc count can grow because of
test-file growth, but the detail count on a given commit may not reflect that — same scope
mismatch as F7.

**Upgrade**: Apply the same exclusion policy to both functions. The `infrastructure.md` loc
policy does not explicitly exclude tests, but if tests are excluded from xenon they should
probably be excluded from loc too for consistency. If the decision is to keep tests in the
loc count, document that explicitly and ensure `loc_detail_count` includes them too.

**Factor**: correctness / policy clarity. Hot path: yes. Mutation-safety: pending (baseline
meaning change).

---

### F9 — `run` / `run_output` duplication and stderr swallowing (P2)

**Location**: `run()` (`ratchet_check.py:39-41`), `run_output()` (`ratchet_check.py:44-46`).

**Weakness**: Two functions that differ only in whether they append `result.stderr`. Every
caller that needs stderr uses `run_output`; every caller that doesn't uses `run`. This split
means:
- `run` silently swallows stderr, hiding tool warnings that may indicate a problem.
- Adding a new caller requires choosing between the two, with no clear rule.

**Upgrade**: Collapse into one `_run(args, cwd, include_stderr=False)` that always captures
stderr and returns either stdout or stdout+stderr based on the flag. Always-include-stderr is
also safer because tool warnings (e.g. ruff's `"warning: ..."`) are not lost.

**Factor**: duplication/simplification. Hot path: yes. Mutation-safety: N/A (mechanical).

---

### F10 — hardcoded `ROOT` path resolution fragility (P3)

**Location**: `ROOT = Path(__file__).resolve().parents[3]` (`ratchet_check.py:24`).

**Weakness**: The script assumes it lives exactly three directory levels below the repo root
(`scripts/git-hooks/incremental-precommit/` → parents[3] = repo root). If the file is moved
or the folder structure changes, `ROOT` silently points to the wrong directory. `HERE` is
robust (relative to the file itself), but `ROOT` is not.

**Upgrade**: Compute `ROOT` from a repo marker (e.g. walk up from `HERE` looking for
`.git/`, `pyproject.toml`, or a known directory like `app/`). This is a common pattern:
```python
ROOT = HERE
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
```
Alternatively, accept `ROOT` as an environment variable override for non-standard layouts.

**Factor**: robustness. Hot path: N/A (computed once). Mutation-safety: N/A.

---

### F11 — no CLI / dry-run / verbosity controls (P3)

**Location**: `main()` (`ratchet_check.py:252-333`).

**Weakness**: The script has no CLI interface — it always runs with defaults, always writes
`baseline.json`, always stages it, and always prints at the same verbosity. There is no way to:
- Run it in dry-run mode to see what would happen without mutating `baseline.json`.
- Override the config path or baseline path for testing.
- Increase verbosity to debug why a particular vector regressed.

**Upgrade**: Add a minimal `argparse` (or `click`) interface:
```
ratchet_check.py [--config PATH] [--baseline PATH] [--dry-run] [--verbose]
```
Keep the zero-argument default behavior unchanged so the existing pre-commit hook wiring
(`scripts/git-hooks/pre-commit`) does not need to change.

**Factor**: usability / testability. Hot path: N/A. Mutation-safety: N/A.

---

### F12 — `print_regression_details` only prints for actually regressed vectors (P3)

**Location**: `print_regression_details` (`ratchet_check.py:234-243`).

**Weakness**: The function takes `regressed` (only the vectors that blocked) and prints
details only for those. For `ignored_regressions` (project-wide up, staged count == 0), the
user sees a one-line summary but no detail on *which* staged files would have triggered the
regression if they had been touched — useful context for understanding why the vector is
drifting.

**Upgrade**: Optionally print details for `ignored_regressions` too, or add a `--verbose`
flag (see F11) that surfaces them. Not a bug, but a missed observability opportunity.

**Factor**: usability. Hot path: N/A. Mutation-safety: N/A.

---

## prioritized upgrade todo

| id | priority | title | key files |
|----|----------|-------|-----------|
| T1 | P0 | Add unit/regression tests for all ratchet branches | new `tests/.../test_ratchet_check.py` |
| T2 | P0 | Make tool failures loud instead of silent (return-code check, timeouts) | `run`, `run_output`, all count functions |
| T3 | P1 | Move configurable constants into `config.json` (ratio, loc threshold, xenon rank, target) | `config.json`, `ratchet_check.py` |
| T4 | P1 | Collapse project-wide / detail-count duplication into a shared helper per vector | `ratchet_check.py` |
| T5 | P1 | Make `baseline.json` write atomic + stage only on full success + add inter-process lock | `ratchet_check.py`, hook wrapper |
| T6 | P2 | Unify `run` / `run_output`; always capture stderr | `ratchet_check.py` |
| T7 | P2 | Harden `mypy` regex and xenon rank parsing; log on parse failure instead of returning 0 | `mypy_count`, `mypy_detail_count`, `xenon_count`, `xenon_detail_count` |
| T8 | P2 | Fix `_exclude_tests` scope mismatch between `xenon_count` and `xenon_detail_count` | `xenon_count`, `_exclude_tests` |
| T9 | P2 | Clarify and align `loc_count` / `loc_detail_count` test exclusion policy | `loc_count`, `loc_detail_count` |
| T10 | P3 | Replace `parents[3]` `ROOT` resolution with walk-up-from-file or env-var override | `ratchet_check.py:24` |
| T11 | P3 | Add `argparse` CLI with `--dry-run`, `--config`, `--baseline`, `--verbose` | `main()` |
| T12 | P3 | Print ignored-regression details for observability (behind `--verbose`) | `print_regression_details`, `main` |

---

## recommended execution order

1. **T1 (P0, tests)** — write the test suite first. Every later refactor (T2–T12) is gated by
   these tests passing.
2. **T2 (P0, loud failures)** — add return-code checks and timeouts. Run the new tests; they
   should now catch the silent-pass bug.
3. **T3 (P1, config)** — move constants to `config.json` with backward-compatible defaults.
4. **T4 (P1, DRY)** — collapse the eight count functions into four shared helpers. Run tests.
5. **T5 (P1, atomic write + lock)** — fix the baseline write race. Run tests.
6. **T6–T9 (P2, hardening)** — parse robustness, scope fixes.
7. **T10–T12 (P3, polish)** — path resolution, CLI, observability.

---

## mutation-safety policy for this file

This file runs on every commit. The repo's own policy (see `infrastructure.md` and the
ratchet README) is "test discipline, not mutation testing." Apply that here:

- **T1 must land before any other item**. Without tests, a refactor to T2–T5 is unguarded.
- **T5's baseline-meaning change** (T7/T9: fixing scope mismatches) requires a one-time
  `baseline.json` key deletion per affected vector so the new count is bootstrapped cleanly —
  document that in the commit message.
- **T2's loud-failure change** is behavior-preserving for the success path (same counts, same
  block decision) but changes the failure path from silent-pass to loud-fail. That is the
  intended fix, not a regression, but it should be called out in the commit message.

---

## related references

- `scripts/git-hooks/incremental-precommit/README.md` — design rationale and vector table.
- `docs/infrastructure.md` § incremental ratchet — policy and threshold rationale.
- `scripts/git-hooks/incremental-precommit/config.json` — current (minimal) config.
- `docs/todos/code_optimization.md` — existing optimization backlog format (P0/P1/P2/P3 tiers).
- `docs/todos/data_pipeline_upgrade_plan.md` — existing upgrade-plan format.
