# incremental pre-commit ratchet

Problem: `mypy`/`radon` analyze whole files, not diff hunks. A strict "must be 100% clean on any
touched file" gate turns a one-line edit to a legacy file into a forced cleanup of every unrelated
pre-existing violation in that file - a chain reaction into a dramatically bigger diff than the
change actually called for.

Fix: track total problem count per **vector** (one static-analysis tool) project-wide in
[baseline.json](baseline.json), committed to the repo. `ratchet_check.py` re-measures each vector on
every commit and only blocks when the count goes **up** past baseline (a real regression - you added
a new violation somewhere). It never requires fixing pre-existing debt just because you touched the
file it's in.

## vectors

| vector | tool | counts |
| --- | --- | --- |
| `mypy` | `mypy --config-file pyproject.toml app` | `Found N errors` |
| `ruff` | `ruff check app --output-format=json` | violations in the JSON array |
| `xenon` | `radon cc app -j` | blocks ranked worse than `B` (matches `.pre-commit-config.yaml`'s `--max-absolute B`) |

## ratcheting down

Debt only ever gets fixed gradually (whoever happens to touch a file, or a dedicated cleanup pass).
Once a vector's improvement since the last recorded baseline reaches `chunk_size`
([config.json](config.json), default 50), `ratchet_check.py` rewrites `baseline.json` to the new,
lower count and stages it in the same commit - locking the improvement in so the regression check
above stops anyone drifting back up past it. Smaller improvements (< chunk_size) are still measured
but not yet written back, so partial progress isn't lost - it just accumulates toward the next
chunk.

## mutation-safety note (no new tool - policy only)

Fixing a real violation (not just a formatting/import-order nit) means touching legacy code by hand,
which risks silently changing behavior while "just satisfying the linter." This repo's safeguard for
that is test discipline, not a mutation-testing tool (see the `test-strategy` skill's
characterization-test discipline): pin the
function's *current actual* output with a characterization test before changing it. When a commit
ratchets a baseline down, `ratchet_check.py` checks whether the same commit touches
`app/tests/{characterization,unit,regression}/` and prints a reminder (not a block) if it doesn't -
a nudge, not a gate, per the "policy only" decision for this repo.

## bootstrapping / resetting a vector

Delete (or edit) its key in `baseline.json` and run `python ratchet_check.py` once - it re-measures
and records the current count as the new baseline for that vector, same as first-time setup.

## why not per-vector `# noqa`/`# type: ignore` baselines instead

Considered (e.g. `mypy-baseline`, ignore-comment sprinkling) and rejected: those mark *specific
lines* as pre-existing debt, which is more precise but requires generating and maintaining a
per-line baseline file that drifts every time a legacy file is edited (even unrelated changes shift
line numbers). A single project-wide count per vector is coarser but self-maintaining - no
regeneration step, no merge conflicts in a baseline file every time two branches touch the same
legacy file.
