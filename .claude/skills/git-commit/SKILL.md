---
name: git-commit
description: 'Execute git commit with conventional commit message analysis, intelligent staging, and message generation. Use when user asks to commit changes, create a git commit, or mentions "/commit". Supports: (1) Auto-detecting type and scope from changes, (2) Generating conventional commit messages from diff, (3) Interactive commit with optional type/scope/description overrides, (4) Intelligent file staging for logical grouping'
license: MIT
allowed-tools: Bash
source: https://github.com/github/awesome-copilot/blob/main/skills/git-commit/SKILL.md
---

# Git Commit with Conventional Commits

Message format: `<type>[optional scope]: <description>` — present tense, imperative mood
("add"/"fix", not "added"/"fixes"), <72 chars, derived from the actual diff (not the ticket title).
Staging, safety rules (never `--force`/`--no-verify`, never commit unless asked, new commit over
amend, etc.), and the commit workflow itself follow this session's standing git instructions — this
skill only adds the Conventional Commits taxonomy below.

## Types

| Type | Purpose |
| --- | --- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting/style (no logic) |
| `refactor` | Code refactor (no feature/fix) |
| `perf` | Performance improvement |
| `test` | Add/update tests |
| `build` | Build system/dependencies |
| `ci` | CI/config changes |
| `chore` | Maintenance/misc |
| `revert` | Revert commit |

## Breaking changes

`feat!: remove deprecated endpoint`, or a footer:

```
feat: allow config to extend other configs

BREAKING CHANGE: `extends` key behavior changed
```

## Optional footers

`Closes #123`, `Refs #456`.
