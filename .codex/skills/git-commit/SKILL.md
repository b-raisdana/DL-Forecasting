---
name: git-commit
description: 'Execute git commit with conventional commit message analysis, intelligent staging, and message generation. Use when the user asks to commit changes, create a git commit, or types "/commit". Covers: auto-detecting type/scope from the diff, generating conventional commit messages, interactive overrides, and grouping files into logical commits.'
license: MIT
allowed-tools: Bash
source: https://github.com/github/awesome-copilot/blob/main/skills/git-commit/SKILL.md
---

# Git Commit with Conventional Commits

Message format: `<type>[optional scope]: <description>` — present tense, imperative mood ("add"/"fix", not "added"/"fixes"), under 72 chars, derived from the actual diff, not the ticket title. Staging, safety rules (never `--force`/`--no-verify`, never commit unless asked, new commit over amend, etc.) and the commit workflow follow this session's standing git instructions — this skill only adds the Conventional Commits taxonomy below.

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
