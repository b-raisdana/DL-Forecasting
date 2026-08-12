---
name: git-commit
description: 'Execute git commit with conventional commit message analysis, intelligent staging, and message generation. Use when user asks to commit changes, create a git commit, or mentions "/commit". Supports: (1) Auto-detecting type and scope from changes, (2) Generating conventional commit messages from diff, (3) Interactive commit with optional type/scope/description overrides, (4) Intelligent file staging for logical grouping'
license: MIT
allowed-tools: Bash
source: https://github.com/github/awesome-copilot/blob/main/skills/git-commit/SKILL.md
---

# Git Commit with Conventional Commits

## Overview

Create standardized, semantic git commits using the Conventional Commits specification. Analyze the actual diff to determine appropriate type, scope, and message.

## Conventional Commit Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Commit Types

| Type       | Purpose                        |
| ---------- | ------------------------------ |
| `feat`     | New feature                    |
| `fix`      | Bug fix                        |
| `docs`     | Documentation only             |
| `style`    | Formatting/style (no logic)    |
| `refactor` | Code refactor (no feature/fix) |
| `perf`     | Performance improvement        |
| `test`     | Add/update tests               |
| `build`    | Build system/dependencies      |
| `ci`       | CI/config changes              |
| `chore`    | Maintenance/misc               |
| `revert`   | Revert commit                  |

## Breaking Changes

```
# Exclamation mark after type/scope
feat!: remove deprecated endpoint

# BREAKING CHANGE footer
feat: allow config to extend other configs

BREAKING CHANGE: `extends` key behavior changed
```

## Workflow

### 1. Analyze Diff

```bash
git diff --staged   # if files are staged
git diff             # otherwise, working tree diff
git status --porcelain
```

### 2. Stage Files (if needed)

```bash
git add path/to/file1 path/to/file2   # specific files
git add *.test.*                       # by pattern
git add -p                             # interactive
```

**Never commit secrets** (.env, credentials.json, private keys).

### 3. Generate Commit Message

Determine from the diff:

- **Type**: what kind of change
- **Scope**: what area/module is affected
- **Description**: one-line summary, present tense, imperative mood, <72 chars

### 4. Execute Commit

```bash
git commit -m "<type>[scope]: <description>"

# Multi-line with body/footer
git commit -m "$(cat <<'EOF'
<type>[scope]: <description>

<optional body>

<optional footer>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Best Practices

- One logical change per commit
- Present tense, imperative mood: "add" / "fix", not "added" / "fixes"
- Reference issues: `Closes #123`, `Refs #456`
- Description under 72 characters
- Append the `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer, per this repo's commit convention

## Git Safety Protocol

- NEVER update git config
- NEVER run destructive commands (`--force`, hard reset) without explicit request
- NEVER skip hooks (`--no-verify`) unless the user asks
- NEVER force push to main/master
- If commit fails due to a hook, fix the issue and create a NEW commit (don't amend)
- Only commit when the user explicitly asks
