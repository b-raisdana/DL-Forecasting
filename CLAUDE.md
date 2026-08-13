# DL-Forecasting

## Python environment

This repo's Python lives only in the WSL `tf` conda env — Windows has no valid interpreter for it (full rationale: [docs/infrastructure.md § environments](docs/infrastructure.md#environments)).

- Distro `Ubuntu-24.04`, conda env `tf`, interpreter `/home/brais/miniconda3/envs/tf/bin/python` (Python 3.12.9).
- Run any command against it from Windows: `wsl.exe -d Ubuntu-24.04 -- bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf && cd /mnt/c/Code/DL-Forecasting && <command>'`.
- Test-running mechanics (markers, fast gate, fixtures): [pytest skill](.claude/skills/pytest/SKILL.md).
- VS Code: `.vscode/settings.json` pins `python.defaultInterpreterPath` to that env, project-wide (not a global/User setting) — it only resolves once the folder is reopened via the Remote-WSL extension (`.vscode/extensions.json` recommends it); a plain local Windows window correctly shows no matching interpreter.
