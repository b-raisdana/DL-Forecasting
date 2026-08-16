---
name: pre-commit-check
description: Use after finishing any code modification in this repo, before the final response, to run the repository pre-commit checks and confirm the edits did not break formatting, linting, incremental ratchet checks, fast pytest gates, or mirrored skill-file synchronization.
---

# Pre-Commit Check

## End-of-Edit Gate

After modifying code, run the repo's pre-commit gate before final handoff unless the user explicitly asks not to run checks. Prefer checking only changed files first because it matches what a real commit will run and avoids unrelated legacy failures:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf &&
  cd /mnt/c/Code/DL-Forecasting &&
  pre-commit run
'
```

Use `pre-commit run --all-files` only when the change is broad, touches shared configuration, or the user asks for a full check.

## Failure Handling

If hooks auto-modify files, inspect the diff, keep intended fixes, and rerun `pre-commit run` until it passes or only unrelated pre-existing failures remain.

If a hook fails:

- Fix failures caused by the current edits.
- Do not clean up unrelated legacy violations unless they block proving the current change and the smallest practical fix is clear.
- For pytest failures, use the `pytest` skill before debugging or rerunning tests.
- For Python files under `app/`, remember the `optimization-review` skill still applies after functional edits; do that review before the final pre-commit gate when both skills trigger.

## Reporting

In the final response, state whether `pre-commit run` passed. If it did not pass, include the failing hook names and whether the failures appear related to the current edits.
