# incremental pre-commit ratchet

Problem: `mypy`/`radon` analyze whole files, not diff hunks. A strict "must be 100% clean on any touched file" gate turns a one-line edit to a legacy file into a forced cleanup of every unrelated pre-existing violation in that file - a chain reaction into a dramatically bigger diff than the change actually called for.

Fix: track a project-wide problem count per **key** in [baseline.json](baseline.json), committed to the repo. `ratchet_check.py` re-measures every key on every commit and blocks only when a key's count goes **up** past its baseline **and** the files you staged contain a problem of that exact key - a real regression you introduced, not pre-existing debt in a file you happened to touch. The staged-file check matters because the counters scan the whole `app/` working tree, not the git index, so it stops an unrelated dirty file from getting blamed for your commit.

## keys

| key | tool | counts |
| --- | --- | --- |
| `mypy:code` | `mypy --config-file pyproject.toml app` | one key per bracketed `[code]` on each `error:` line; lines with no code fall into `mypy:uncoded` |
| `ruff:code` | `ruff check app --output-format=json` | one key per violation's `code` field |
| `xenon` | `radon cc app -j` | single count: blocks ranked worse than `B` (matches `.pre-commit-config.yaml`'s `--max-absolute B`) |
| `loc` | walks `app/**/*.py` | single count: sum of `max(0, line_count - 500)` per file - `<300` normal, `300-500` low-priority split todo, `>500` warning/high-priority split todo |

`mypy` and `ruff` are split per rule/error code instead of one aggregate number each: fixing ten `mypy:attr-defined` errors while introducing one `mypy:arg-type` error still gets caught, because `mypy:arg-type`'s own baseline catches it - no per-instance diffing needed, just a finer count. A brand-new code (a rule just enabled, or one never seen before) bootstraps at whatever it currently finds instead of blocking, so turning a rule on or off is never retroactively blamed on anyone.

## keeping baseline current

After every successful (non-blocked) commit, `baseline.json` is fully rewritten to the just-measured counts and staged in the same commit - no threshold, no lazy accumulation. A key with zero violations left simply drops out of the file (self-pruning), whether that's because the last violation was fixed or the rule was retired.

## mutation-safety note (no new tool - policy only)

Fixing a real violation (not just a formatting/import-order nit) means touching legacy code by hand, which risks silently changing behavior while "just satisfying the linter." This repo's safeguard for that is test discipline, not a mutation-testing tool (see the `test-strategy` skill's characterization-test discipline): pin the function's _current actual_ output with a characterization test before changing it. Whenever a commit lowers baseline for any key (something got fixed), `ratchet_check.py` checks whether the same commit touches `app/tests/{characterization,unit,regression}/` and prints a reminder (not a block) if it doesn't - a nudge, not a gate, per the "policy only" decision for this repo.

## bootstrapping / resetting a key

Delete (or edit) its entry in `baseline.json` and run `python ratchet_check.py` once - it re-measures and records the current count as the new baseline for that key, same as first-time setup.

## why not per-line `# noqa`/`# type: ignore` baselines instead

Considered (e.g. `mypy-baseline`, ignore-comment sprinkling) and rejected: those mark _specific lines_ as pre-existing debt, which is more precise but requires generating and maintaining a per-line baseline file that drifts every time a legacy file is edited (even unrelated changes shift line numbers). A project-wide count per key is coarser but self-maintaining - no regeneration step, no merge conflicts in a baseline file every time two branches touch the same legacy file.

## inline commentary

### design

Incremental pre-commit ratchet - see README.md in this folder for the full design.

Blocks a commit only when a key's project-wide problem count goes UP past its recorded baseline AND the staged files contribute a problem of that exact key. Never requires fixing a touched file's pre-existing debt just to commit - that's the "chain reaction" this replaces: mypy/radon analyze whole files, so a strict per-file gate turns a one-line edit into a forced cleanup of every unrelated legacy violation in that file.

`mypy`/`ruff` are split per rule/error code (not one aggregate count each) so that fixing a pile of one rule's violations can never mask introducing a new violation of a different rule. Every successful commit resyncs baseline.json to the fresh counts immediately - no chunk/percentage threshold, no accumulation window.

### mypy_count cwd choice

`cwd=app` (not ROOT, target "app") deliberately: this repo's modules import each other bare (`from ai_modelling.base import ...`, matching pytest.ini's pythonpath=app - see the pytest skill's "Repo config" section), so mypy needs app/ itself as its implicit search-path root. Running from ROOT with target="app" made every file resolve under two conflicting module names (via explicit_package_bases finding repo-root vs app/ as the package base, since app/**init**.py exists) and mypy aborted after 1 file ("found twice").

### ruff key scope

`ruff_project_counts`/`ruff_detail_counts` run plain `ruff check` (project config, no `--select`/`--ignore` override), so these keys only ever see whatever `pyproject.toml`'s `[tool.ruff.lint] select` includes. `Q`/`RUF`/`T10`/`T20`/`ERA` (quote conventions, Ruff-specific likely-bug checks, debugger statements, print() calls, commented-out code) are deliberately left out of that list - they're advisory, surfaced instead via precommit_wrapper.py's stdout warning + logs/pre-commit/pre-commit.log, not editor squiggles or a commit gate.

### xenon excludes tests and archive

Cyclomatic complexity isn't a meaningful signal for test code (parametrized/assert-heavy loops are idiomatic there, not a design smell), so app/tests/ is excluded entirely - see docs/infrastructure.md#incremental-ratchet-mypyruffxenon-scope. archive_not_used_trash/ is unreachable-from-presentation code kept for reference, not linted - see app/archive_not_used_trash/README.md.
