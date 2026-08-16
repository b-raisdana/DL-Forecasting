#!/bin/sh
# Keep mirrored skill files identical to their canonical source.
#
# .claude/skills and .codex/skills are bidirectional mirrors: edit either side,
# and this hook copies the newest version to the other side. .github/git-commit
# is the Copilot / VS Code commit-message mirror (see .vscode/settings.json).
#
# Windows here doesn't have symlinks enabled (core.symlinks=false, no elevated
# session), so this hook uses regular file copies and fails the commit if it had
# to update a mirror, same pattern as the trailing-whitespace/ruff --fix hooks
# above it in .pre-commit-config.yaml.
set -eu

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
changed=0

copy_if_needed() {
    source="$repo_root/$1"
    target="$repo_root/$2"
    if [ ! -f "$target" ] || ! cmp -s "$source" "$target"; then
        mkdir -p "$(dirname "$target")"
        cp "$source" "$target"
        echo "synced $2 <- $1"
        changed=1
    fi
}

newer_file() {
    left="$1"
    right="$2"

    if [ ! -f "$repo_root/$left" ]; then
        printf '%s\n' "$right"
        return
    fi

    if [ ! -f "$repo_root/$right" ]; then
        printf '%s\n' "$left"
        return
    fi

    if [ "$repo_root/$left" -nt "$repo_root/$right" ]; then
        printf '%s\n' "$left"
    else
        printf '%s\n' "$right"
    fi
}

sync_bidirectional_pair() {
    left="$1"
    right="$2"

    if [ ! -f "$repo_root/$left" ] && [ ! -f "$repo_root/$right" ]; then
        return
    fi

    source=$(newer_file "$left" "$right")
    if [ "$source" = "$left" ]; then
        copy_if_needed "$left" "$right"
    else
        copy_if_needed "$right" "$left"
    fi
}

for skill_parent in "$repo_root"/.claude/skills "$repo_root"/.codex/skills; do
    [ -d "$skill_parent" ] || continue
    for skill_file in "$skill_parent"/*/SKILL.md; do
        [ -f "$skill_file" ] || continue
        skill_dir=$(basename "$(dirname "$skill_file")")
        sync_bidirectional_pair ".claude/skills/$skill_dir/SKILL.md" ".codex/skills/$skill_dir/SKILL.md"
    done
done

sync_bidirectional_pair ".claude/skills/git-commit/SKILL.md" ".github/git-commit/SKILL.md"
sync_bidirectional_pair ".codex/skills/git-commit/SKILL.md" ".github/git-commit/SKILL.md"

exit "$changed"
