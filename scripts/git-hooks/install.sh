#!/bin/sh
# One-time setup per clone: installs the tracked pre-commit hook shim into .git/hooks/
# (not version-controlled by git itself). Run from repo root: bash scripts/git-hooks/install.sh
set -eu
repo_root="$(git rev-parse --show-toplevel)"
cp "$repo_root/scripts/git-hooks/pre-commit" "$repo_root/.git/hooks/pre-commit"
chmod +x "$repo_root/.git/hooks/pre-commit"
echo "Installed .git/hooks/pre-commit"
