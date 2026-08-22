"""Wraps `pre-commit run --hook-stage pre-commit` with failure-snapshotting, logging, and
non-blocking advisory lint warnings - see README.md in this directory for the full design.

Invoked by scripts/git-hooks/pre-commit (tracked source of .git/hooks/pre-commit, see
scripts/git-hooks/install.sh) from inside the WSL "tf" conda env - see
docs/infrastructure.md#environments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = REPO_ROOT / "logs" / "pre-commit" / "pre-commit.log"
SEQ_FILE = REPO_ROOT / ".git" / "pc-seq"
CHECK_PIPELINE_CMD = ["pre-commit", "run", "--hook-stage", "pre-commit"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _staged_files() -> list[str]:
    output = _git("diff", "--cached", "--name-only")
    return output.splitlines() if output else []


def _read_seq() -> int:
    if not SEQ_FILE.exists():
        return 0
    try:
        return int(SEQ_FILE.read_text().strip())
    except ValueError:
        return 0


def _advance_seq(n: int) -> None:
    SEQ_FILE.write_text(f"{n + 1}\n")


def _snapshot_branch(n: int, human_ts: str) -> str:
    tree = _git("write-tree")
    commit = _git("commit-tree", tree, "-p", "HEAD", "-m", f"snapshot: pre-commit failed (fix-{n}) {human_ts}")
    # -f force-moves the branch if it already exists from a prior failed attempt in this same
    # session (same N - see _read_seq/_advance_seq) - so repeated failures overwrite, not
    # accumulate. The human-readable timestamp in the commit message still identifies exactly
    # which attempt this was, even though the branch ref itself only ever points at the latest.
    branch = f"snapshot/fix-{n}"
    _git("branch", "-f", branch, commit)
    return branch


# Deliberately excluded from pyproject.toml's [tool.ruff.lint] select (so the VS Code ruff
# extension never highlights them) but still worth surfacing somewhere non-blocking: quote
# conventions, Ruff-specific likely-bug checks, debugger statements, print() calls, and
# commented-out code.
ADVISORY_RUFF_RULES = "Q,RUF,T10,T20,ERA"


def _advisory_lint_warnings(staged: list[str]) -> list[dict[str, object]]:
    py_files = [f for f in staged if f.startswith("app/") and f.endswith(".py") and (REPO_ROOT / f).exists()]
    if not py_files:
        return []
    result = subprocess.run(
        ["ruff", "check", "--select", ADVISORY_RUFF_RULES, "--output-format=json", *py_files],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        violations = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [
        {
            "file": Path(v["filename"]).resolve().relative_to(REPO_ROOT).as_posix(),
            "line": v["location"]["row"],
            "code": v["code"],
            "message": v["message"],
        }
        for v in violations
    ]


def _log(entry: dict[str, object]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _human_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _run_pipeline() -> tuple[bool, str]:
    # Popen + line-by-line tee (rather than subprocess.run(capture_output=True)) so the pipeline's
    # output still streams live to the terminal - mypy/pytest can take a while - while the exact
    # same text is also captured for the per-failed-attempt log file below.
    proc = subprocess.Popen(
        CHECK_PIPELINE_CMD, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    proc.wait()
    return proc.returncode == 0, "".join(lines)


def _write_failure_detail(n: int, human_ts: str, pipeline_output: str, entry: dict[str, object]) -> Path:
    # One folder per fix-N session (matches the snapshot branch name) so every attempt within a
    # session accumulates here, even though the branch itself only ever keeps the latest (-f).
    # Files inside are named by human-readable timestamp - one pair per attempt.
    detail_dir = LOG_FILE.parent / f"fix-{n}"
    detail_dir.mkdir(parents=True, exist_ok=True)
    (detail_dir / f"{human_ts}.log").write_text(pipeline_output)
    (detail_dir / f"{human_ts}.json").write_text(json.dumps(entry, indent=2) + "\n")
    return detail_dir


def main() -> int:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    staged = _staged_files()

    passed, pipeline_output = _run_pipeline()
    advisory_lint_warnings = _advisory_lint_warnings(staged)

    entry: dict[str, object] = {
        "timestamp": timestamp,
        "branch": branch,
        "staged_files": staged,
        "checks_run": CHECK_PIPELINE_CMD,
        "result": "pass" if passed else "fail",
        "advisory_lint_warnings": advisory_lint_warnings,
    }

    if advisory_lint_warnings:
        sys.stdout.write(
            f"warning: {len(advisory_lint_warnings)} advisory lint finding(s) in staged files "
            "(non-blocking - quote/print/debugger/commented-out-code/Ruff-specific checks, see "
            "logs/pre-commit/pre-commit.log):\n"
        )
        for hit in advisory_lint_warnings:
            sys.stdout.write(f"  {hit['file']}:{hit['line']} {hit['code']} {hit['message']}\n")

    n = _read_seq()

    if passed:
        _advance_seq(n)
        _log(entry)
        return 0

    human_ts = _human_timestamp()
    snapshot_branch = _snapshot_branch(n, human_ts)
    entry["snapshot_branch"] = snapshot_branch
    entry["snapshot_seq"] = n
    detail_dir = _write_failure_detail(n, human_ts, pipeline_output, entry)
    entry["detail_dir"] = detail_dir.relative_to(REPO_ROOT).as_posix()
    _log(entry)
    sys.stderr.write(
        f"pre-commit checks failed - staged changes preserved as '{snapshot_branch}'"
        f" (working tree, index, and HEAD untouched); full output logged to"
        f" '{detail_dir.relative_to(REPO_ROOT).as_posix()}/{human_ts}.log'. Fix the reported"
        " issues, then re-stage and commit again.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
