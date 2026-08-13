#!/bin/sh
# Keep mirrored skill files identical to their canonical source.
#
# .github/git-commit/SKILL.md is a copy of .claude/skills/git-commit/SKILL.md, kept
# around for tools (e.g. GitHub Copilot) that look under .github/ instead of .claude/.
# Windows here doesn't have symlinks enabled (core.symlinks=false, no elevated
# session), so this hook is the single-source-of-truth mechanism instead: copy
# canonical -> mirror and fail the commit if that changed anything, same pattern as
# the trailing-whitespace/ruff --fix hooks above it in .pre-commit-config.yaml.
set -eu

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
changed=0

sync_pair() {
    canonical="$repo_root/$1"
    mirror="$repo_root/$2"
    if [ ! -f "$mirror" ] || ! cmp -s "$canonical" "$mirror"; then
        mkdir -p "$(dirname "$mirror")"
        cp "$canonical" "$mirror"
        echo "synced $2 <- $1"
        changed=1
    fi
}

sync_pair ".claude/skills/git-commit/SKILL.md" ".github/git-commit/SKILL.md"

exit "$changed"
