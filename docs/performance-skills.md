# performance & speed-optimization skills

Dedicated index of the `.claude/skills/*` skills that enforce
[performance-and-concurrency.md](performance-and-concurrency.md) day to day. Each skill's `SKILL.md` is
the trigger source of truth; this file is only the catalog so the full set is visible at a glance.

| Skill                                                                           | Triggers on                               | Enforces                                                                                                                                | [Principles](performance-and-concurrency.md#principles) covered |
| ------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [vectorized-pandas-numpy](../.claude/skills/vectorized-pandas-numpy/SKILL.md)   | writing/reviewing pandas/numpy code       | fully vectorized DataFrame/ndarray ops instead of Python-level loops — pushes heavy compute into pandas/numpy's C backend               | 1, 4, 5 (pandas/numpy-specific case of library-first)           |
| [lib-first](../.claude/skills/lib-first/SKILL.md)                               | before implementing new non-trivial logic | research an existing well-maintained library before hand-writing an algorithm                                                           | 1, 4, 5                                                         |
| [concurrency-and-blocking](../.claude/skills/concurrency-and-blocking/SKILL.md) | adding I/O calls or CPU-heavy fan-out     | asyncio/thread-pool for I/O-bound, vectorize-then-process-pool for CPU-bound; avoids blocking the caller and minimizes memory footprint | 1, 2, 3, 4                                                      |

## adding a new performance skill

Add the row here in the same edit that creates the `SKILL.md` (skills-mirror still separately mirrors
`.claude/skills/` into `docs/skills-mirror/` — this file is an index on top of that, not a replacement).
