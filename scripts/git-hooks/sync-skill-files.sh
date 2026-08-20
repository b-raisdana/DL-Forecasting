#!/bin/sh
# Keep mirrored skill files identical to their canonical source.
#
# .claude/skills, .codex/skills, .devin/skills, .qoder/skills, .copilot/skills, .kiro/skills, and .kilo/skills are bidirectional mirrors:
# edit either side, and this hook copies the newest version to the other side.
# .github/git-commit is the Copilot / VS Code commit-message mirror (see .vscode/settings.json).
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

sync_multiple_agents() {
    skill_name="$1"
    shift
    agent_paths="$@"

    # Find the newest file among all agents
    newest=""
    for path in $agent_paths; do
        if [ -f "$repo_root/$path" ]; then
            if [ -z "$newest" ] || [ "$repo_root/$path" -nt "$repo_root/$newest" ]; then
                newest="$path"
            fi
        fi
    done

    # If no file exists, return
    if [ -z "$newest" ]; then
        return
    fi

    # Copy newest to all other agents
    for path in $agent_paths; do
        if [ "$path" != "$newest" ]; then
            copy_if_needed "$newest" "$path"
        fi
    done
}

# Collect all unique skill names across all agents
seen_skills=""

for skill_parent in "$repo_root"/.claude/skills "$repo_root"/.codex/skills "$repo_root"/.devin/skills "$repo_root"/.qoder/skills "$repo_root"/.copilot/skills "$repo_root"/.kiro/skills "$repo_root"/.kilo/skills; do
    [ -d "$skill_parent" ] || continue
    for skill_file in "$skill_parent"/*/SKILL.md; do
        [ -f "$skill_file" ] || continue
        skill_dir=$(basename "$(dirname "$skill_file")")
        # Add to seen_skills if not already present
        case " $seen_skills " in
            *" $skill_dir "*) ;;
            *) seen_skills="$seen_skills $skill_dir" ;;
        esac
    done
done

# Sync each skill across all agents
for skill_dir in $seen_skills; do
    [ "$skill_dir" = "use-aget-skills" ] && continue
    sync_multiple_agents "$skill_dir" \
        ".claude/skills/$skill_dir/SKILL.md" \
        ".codex/skills/$skill_dir/SKILL.md" \
        ".devin/skills/$skill_dir/SKILL.md" \
        ".qoder/skills/$skill_dir/SKILL.md" \
        ".copilot/skills/$skill_dir/SKILL.md" \
        ".kiro/skills/$skill_dir/SKILL.md" \
        ".kilo/skills/$skill_dir/SKILL.md"
done

# Special case: git-commit skill also syncs to .github/git-commit/SKILL.md
sync_multiple_agents "git-commit" \
    ".claude/skills/git-commit/SKILL.md" \
    ".codex/skills/git-commit/SKILL.md" \
    ".devin/skills/git-commit/SKILL.md" \
    ".qoder/skills/git-commit/SKILL.md" \
    ".copilot/skills/git-commit/SKILL.md" \
    ".kiro/skills/git-commit/SKILL.md" \
    ".kilo/skills/git-commit/SKILL.md" \
    ".github/git-commit/SKILL.md"

exit "$changed"
