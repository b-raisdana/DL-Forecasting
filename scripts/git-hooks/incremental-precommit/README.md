# incremental pre-commit ratchet

Problem: `mypy`/`radon` analyze whole files, not diff hunks. A strict "must be 100% clean on any touched file" gate turns a one-line edit to a legacy file into a forced cleanup of every unrelated pre-existing violation in that file - a chain reaction into a dramatically bigger diff than the change actually called for.

Fix: two independent layers, computed fresh on every commit - no persisted per-file baseline to drift or merge-conflict on.

1. **Blocking gate - per-touched-file diff.** For every file touched in the commit, compare its own violation count (`mypy`/`ruff`/`xenon`) or line count (`loc`) before this commit vs. after, and block only the files that got worse. "Before" comes from a throwaway `git worktree` checked out at `HEAD` (removed again right after); "after" reuses the same tool runs already computed for the trend layer below - no extra invocation needed there. A regression in one file can't be masked by an unrelated file improving elsewhere in the same commit, and there's no "first time this rule/file has ever been seen" loophole - see the rules table.
2. **Trend metric - project-wide aggregate.** `baseline.json` still tracks one count per key project-wide and resyncs to the fresh counts after every successful commit (self-pruning: a key with nothing left just drops out). This layer **never blocks** - it exists purely as a long-term "is total debt trending down" signal and to decide when to remind about characterization tests (see below).

## per-file blocking rules

| tool | rule |
| --- | --- |
| `mypy` / `ruff` / `xenon` | zero tolerance: a touched file's own violation count may not increase at all. A new file has an implicit before of 0, so any violation in it already blocks - no special case needed. |
| `loc` | a file already over 500 lines may grow by at most a 5-line slack (`LOC_SLACK`) - not a hard freeze, so a small legitimate fix isn't blocked outright. A brand-new file must be ≤500 lines at introduction (the slack-diff rule alone can't cover a file with no "before"). A file crossing 500 lines for the first time in this commit isn't blocked by this rule - only the non-blocking project-wide sum notices it. |

Splitting an over-limit file into two or more files is a legitimate way to get back under the loc cap - there's no equivalent escape for `mypy`/`ruff`/`xenon` since those count actual defects, not size. Whether a split was a meaningful seam vs. arbitrary chopping to dodge the check isn't something line counts alone can judge - left as a review norm, not a mechanical check.

Renamed files (git similarity detection, `-M`) look up their "before" state under the old path; a file reported as newly added has no "before" lookup at all (forced to 0 / no prior line count).

## performance: this doubles mypy/ruff/xenon's runtime on commits touching `app/*.py`

Getting an accurate per-file "before" count for `mypy` needs the whole `app/` package as it looked at `HEAD` (it resolves types across files, so a single file in isolation gives wrong answers) - that means a full second `mypy`/`ruff`/`radon` pass against a temporary worktree, on top of the pass already run for "after". This is the accepted cost of exact, driftless per-file diffing instead of a persisted per-line baseline file (see "why not per-line baselines" below). `loc`'s before/after comes from `git show HEAD:path` line counts instead - no worktree, no extra tool run.

## keys (trend / ratchet-down input only - not the blocking gate)

| key | tool | counts |
| --- | --- | --- |
| `mypy:code` | `mypy --config-file pyproject.toml app` | one key per bracketed `[code]` on each `error:` line; lines with no code fall into `mypy:uncoded` |
| `ruff:code` | `ruff check app --output-format=json` | one key per violation's `code` field |
| `xenon` | `radon cc app -j` | single count: blocks ranked worse than `B` (matches `.pre-commit-config.yaml`'s `--max-absolute B`) |
| `loc` | walks `app/**/*.py` | single count: sum of `max(0, line_count - 500)` per file |

`mypy`/`ruff` are split per rule/error code so the trend stays legible as rules are added or removed over time (each code bootstraps/retires independently) - this has no effect on blocking, which is entirely the per-file gate above.

## keeping the trend baseline current

After every successful (non-blocked) commit, `baseline.json` is fully rewritten to the just-measured counts and staged in the same commit - no threshold, no lazy accumulation. A key with zero violations left simply drops out of the file. This never affects whether the commit itself was allowed through - that's decided entirely by the per-file gate above.

## mutation-safety note (no new tool - policy only)

Fixing a real violation (not just a formatting/import-order nit) means touching legacy code by hand, which risks silently changing behavior while "just satisfying the linter." This repo's safeguard for that is test discipline, not a mutation-testing tool (see the `test-strategy` skill's characterization-test discipline): pin the function's _current actual_ output with a characterization test before changing it. Whenever a commit lowers the trend baseline for any key (something got fixed), `ratchet_check.py` checks whether the same commit touches `app/tests/{characterization,unit,regression}/` and prints a reminder (not a block) if it doesn't - a nudge, not a gate, per the "policy only" decision for this repo.

## bootstrapping / resetting a key

Delete (or edit) its entry in `baseline.json` and run `python ratchet_check.py` once - it re-measures and records the current count as the new trend baseline for that key, same as first-time setup. This only affects the trend layer; there's nothing to bootstrap for the per-file gate since it's recomputed fresh every run.

## why not per-line `# noqa`/`# type: ignore` baselines instead

Considered (e.g. `mypy-baseline`, ignore-comment sprinkling) and rejected: those mark _specific lines_ as pre-existing debt, which is more precise but requires generating and maintaining a per-line baseline file that drifts every time a legacy file is edited (even unrelated changes shift line numbers). The per-file gate gets the same precision (which file actually regressed) without that drift, by recomputing both sides fresh from git every run instead of persisting anything; the project-wide trend count is coarser but self-maintaining for the same reason.

## inline commentary

### design

Incremental pre-commit ratchet - see README.md in this folder for the full design.

Two layers: a per-touched-file before/after diff is the blocking gate (zero tolerance for mypy/ruff/xenon, a 5-line slack for loc on files already over the 500-line cap, a hard 500-line cap for brand-new files). A project-wide count per key is kept only as a trend metric feeding the mutation-safety reminder - it never blocks a commit by itself, so an unrelated file improving elsewhere can't mask a regression in the file you touched, and there's no "first time this rule's been seen" loophole the way a project-wide-only baseline would have.

### mypy_count cwd choice

`cwd=app` (not ROOT, target "app") deliberately: this repo's modules import each other bare (`from ai_modelling.base import ...`, matching pytest.ini's pythonpath=app - see the pytest skill's "Repo config" section), so mypy needs app/ itself as its implicit search-path root. Running from ROOT with target="app" made every file resolve under two conflicting module names (via explicit_package_bases finding repo-root vs app/ as the package base, since app/**init**.py exists) and mypy aborted after 1 file ("found twice"). The same reasoning applies to the "before" run against the `HEAD` worktree - it's invoked the same way, rooted at the worktree instead of `ROOT`.

### ruff filename is absolute, unlike mypy/radon

`ruff check --output-format=json` reports each violation's `filename` as an absolute path, while `mypy` and `radon cc -j` both report paths relative to their invocation `cwd`. `_group_ruff_by_file` relativizes against whichever root that particular run used (repo root for "after", the temporary worktree for "before") so both sides key on the same `app/...` strings before comparing.

### xenon excludes tests and archive

Cyclomatic complexity isn't a meaningful signal for test code (parametrized/assert-heavy loops are idiomatic there, not a design smell), so app/tests/ is excluded entirely - see docs/infrastructure.md#incremental-ratchet-mypyruffxenon-scope. archive_not_used_trash/ is unreachable-from-presentation code kept for reference, not linted - see app/archive_not_used_trash/README.md.
