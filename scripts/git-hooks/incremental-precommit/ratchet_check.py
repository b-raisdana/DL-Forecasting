"""Incremental pre-commit ratchet - see README.md in this folder for the full design.

Blocks a commit only when a vector's project-wide problem count goes UP past its
recorded baseline (a real regression). Never requires fixing a touched file's
pre-existing debt just to commit - that's the "chain reaction" this replaces: mypy/
radon analyze whole files, so a strict per-file gate turns a one-line edit into a
forced cleanup of every unrelated legacy violation in that file.

As debt gets fixed (by anyone, in any commit), once a vector's improvement reaches
chunk_size (config.json, default 50), the baseline ratchets down to the new, lower
count and baseline.json is rewritten + staged - locking the improvement in so it
can't silently regress later without tripping the check above.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "baseline.json"
CONFIG_PATH = HERE / "config.json"
TARGET = "app"
COMPLEXITY_RANKS = "ABCDEF"


def load_json(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout


def mypy_count() -> int:
    # cwd=app (not ROOT, target "app") deliberately: this repo's modules import each other
    # bare (`from ai_modelling.base import ...`, matching pytest.ini's pythonpath=app - see
    # docs/testing.md), so mypy needs app/ itself as its implicit search-path root. Running
    # from ROOT with target="app" made every file resolve under two conflicting module names
    # (via explicit_package_bases finding repo-root vs app/ as the package base, since
    # app/__init__.py exists) and mypy aborted after 1 file ("found twice").
    stdout = run("mypy", "--config-file", str(ROOT / "pyproject.toml"), ".", cwd=ROOT / TARGET)
    match = re.search(r"Found (\d+) error", stdout)
    return int(match.group(1)) if match else 0


def ruff_count() -> int:
    stdout = run("ruff", "check", TARGET, "--output-format=json")
    try:
        violations = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        violations = []
    return len(violations)


def xenon_count(max_absolute: str = "B") -> int:
    stdout = run("radon", "cc", TARGET, "-j")
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    threshold = COMPLEXITY_RANKS.index(max_absolute)
    return sum(
        1 for blocks in data.values() for block in blocks if COMPLEXITY_RANKS.index(block.get("rank", "A")) > threshold
    )


VECTORS: dict[str, Callable[[], int]] = {
    "mypy": mypy_count,
    "ruff": ruff_count,
    "xenon": xenon_count,
}


def characterization_test_touched() -> bool:
    stdout = run("git", "diff", "--cached", "--name-only")
    test_dirs = ("app/tests/characterization", "app/tests/unit", "app/tests/regression")
    return any(line.startswith(test_dirs) for line in stdout.splitlines())


def main() -> int:
    config = load_json(CONFIG_PATH)
    chunk_size = config.get("chunk_size", 50)
    baseline = load_json(BASELINE_PATH)

    regressed: list[tuple[str, int, int]] = []
    improved: dict[str, tuple[int, int, int]] = {}
    current_counts: dict[str, int] = {}
    bootstrapped = False

    for name, count_fn in VECTORS.items():
        current = count_fn()
        current_counts[name] = current
        base = baseline.get(name)
        if base is None:
            baseline[name] = current
            bootstrapped = True
            print(f"[{name}] no baseline yet - bootstrapping at {current}")
            continue
        if current > base:
            regressed.append((name, base, current))
        elif current < base:
            improved[name] = (base, current, base - current)

    if regressed:
        print("Incremental pre-commit ratchet: BLOCKED - new problems introduced\n")
        for name, base, current in regressed:
            print(f"  {name}: baseline {base} -> now {current} (+{current - base} new)")
        print(
            "\nThis only blocks NEW violations you introduced - it never requires fixing "
            "pre-existing debt in files you happen to touch. Run the tool directly on your "
            "changed files to see what's new (e.g. `ruff check <file>`, "
            "`mypy --config-file pyproject.toml <file>`)."
        )
        return 1

    ratcheted: list[tuple[str, int, int]] = []
    for name, (base, current, progress) in improved.items():
        if progress >= chunk_size:
            baseline[name] = current
            ratcheted.append((name, base, current))

    if bootstrapped or ratcheted:
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        subprocess.run(["git", "add", str(BASELINE_PATH)], cwd=ROOT, check=False)
    if ratcheted:
        print("Incremental pre-commit ratchet: progress locked in\n")
        for name, base, current in ratcheted:
            print(f"  {name}: baseline {base} -> {current} (-{base - current} fixed, chunk size {chunk_size})")
        if not characterization_test_touched():
            print(
                "\nNote: this commit fixed enough pre-existing problems to ratchet a baseline down, "
                "but doesn't touch app/tests/{characterization,unit,regression}. If any of these were "
                "behavior-affecting fixes (not just type annotations/formatting), pin the before/after "
                "behavior with a characterization test first - see docs/testing.md. This is a reminder, "
                "not a block: this repo's mutation-safety net is test discipline, not a mutation-testing "
                "tool (see docs/infrastructure.md#pre-commit)."
            )

    print("\nIncremental pre-commit ratchet: OK (no regressions)")
    for name, current in current_counts.items():
        print(f"  {name}: {current} / baseline {baseline.get(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
