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

## Grouping multiple reasons into one commit message

A diff can contain changes with genuinely different *reasons* — e.g. a refactor plus an unrelated perf change plus a new feature, all sitting staged at once. A single flat `<type>: <description>` line cannot honestly describe all of that, but that doesn't mean splitting into several commits — the changes still land as **one commit**, with the message body divided into one topic block per reason.

1. **Cluster changes by underlying reason, not by file or by which `type` they'd get.** Look at *why* each hunk changed, not just what changed. Read the actual diff content (not just filenames) to tell reasons apart — a rename plus a behavior tweak inside the same file is still one reason if the tweak only exists to support the rename.
2. **Open a new topic block only when reasons are truly independent** — i.e. you could describe one change without mentioning the other, and reverting one would not require touching the other. Do not split just because a single reason happens to touch several files, several directories, or would otherwise map to different conventional-commit types. Example: a refactor that necessarily breaks and then fixes a test is one `refactor` topic, not `refactor` + `fix`.
3. **When unsure whether two changes share a reason, keep them together.** Over-splitting (topic sprawl in one message) is worse than a slightly broader topic; under-splitting is the failure mode this section exists to prevent, but it only applies once independence is clear.
4. **Write one `<type>[scope]: <description>` line per topic**, each followed by its own supporting bullets, in one commit message:

   ```
   <primary-type>[scope]: <description of the primary/dominant topic>

   refactor(domain): rename market_structure to price_action package
   - split pure indicator math out of infrastructure/ohlcv/atr.py into domain/technical_analysis/atr.py

   perf(ohlcv): window disk cache to eliminate duplicate cache files
   - add cache_window_freq_overrides / generation-rate monitor config

   feat(tier1_000): add six-timeframe hybrid temporal model and configs
   - add input3/outcome1 datafeeder, ModernTCN+LSTM branches, new designsets

   chore: update tooling permissions and git-commit skill grouping
   ```

   The header line reuses the topic block judged most significant as the commit's subject (pick `feat` over `fix` over `refactor`/`perf` over `docs`/`test` over `chore`/`build`/`ci`/`style`, in that order, among the topics actually present) — every topic still gets its own labeled block in the body, the header doesn't replace it.
5. Present the proposed topic breakdown (files + type + summary per topic) before committing if the split is non-obvious, so the user can confirm or merge/split further.
6. A single, clearly-scoped diff (one reason) still gets exactly one plain `<type>[scope]: <description>` message — this section changes nothing about the common case.

## Breaking changes

`feat!: remove deprecated endpoint`, or a footer:

```
feat: allow config to extend other configs

BREAKING CHANGE: `extends` key behavior changed
```

## Optional footers

`Closes #123`, `Refs #456`.
