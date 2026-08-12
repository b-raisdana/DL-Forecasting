---
name: skills-mirror
description: Use automatically whenever a file under .claude/skills/ is created, edited, renamed, or deleted in this repo. Mirrors the current state of .claude/skills into docs/skills-mirror so skill definitions are version-controlled and recoverable.
---

# Skills Mirror

Whenever anything under `.claude/skills/` changes, mirror it into `docs/skills-mirror/` in the same turn, before ending the response.

## How

1. Copy the changed file(s) to the matching path under `docs/skills-mirror/` (e.g. `.claude/skills/foo/SKILL.md` → `docs/skills-mirror/foo/SKILL.md`), preserving the full skill directory (SKILL.md plus any `references/`, `scripts/`, etc.).
2. On delete or rename in `.claude/skills/`, delete/rename the corresponding mirror so both trees stay identical — no orphaned mirrors.
3. `docs/skills-mirror/` is a plain mirror of current state, not a history — one copy per skill, overwritten on each change.
