"""Incremental pre-commit ratchet - see README.md in this folder for the full design.

Blocks a commit only when a vector's project-wide problem count goes UP past its
recorded baseline (a real regression). Never requires fixing a touched file's
pre-existing debt just to commit - that's the "chain reaction" this replaces: mypy/
radon analyze whole files, so a strict per-file gate turns a one-line edit into a
forced cleanup of every unrelated legacy violation in that file.

As debt gets fixed (by anyone, in any commit), once a vector's improvement reaches
3% of its recorded baseline, the baseline ratchets down to the new, lower count
and baseline.json is rewritten + staged - locking the improvement in so it can't
silently regress later without tripping the check above.
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
RATCHET_IMPROVEMENT_RATIO = 0.03


def load_json(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout


def run_output(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout + result.stderr


def mypy_count() -> int:
    # cwd=app (not ROOT, target "app") deliberately: this repo's modules import each other
    # bare (`from ai_modelling.base import ...`, matching pytest.ini's pythonpath=app - see
    # the pytest skill's "Repo config" section), so mypy needs app/ itself as its implicit
    # search-path root. Running
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


def _exclude_tests(paths: list[Path]) -> list[Path]:
    return [path for path in paths if "tests" not in path.parts and "archive_not_used_trash" not in path.parts]


def xenon_count(max_absolute: str = "B") -> int:
    # Cyclomatic complexity isn't a meaningful signal for test code (parametrized/assert-heavy
    # loops are idiomatic there, not a design smell), so app/tests/ is excluded entirely - see
    # docs/infrastructure.md#incremental-ratchet-mypyruffxenon-scope. archive_not_used_trash/ is
    # unreachable-from-presentation code kept for reference, not linted - see
    # app/archive_not_used_trash/README.md.
    stdout = run("radon", "cc", TARGET, "-j", "-i", "tests,archive_not_used_trash")
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    threshold = COMPLEXITY_RANKS.index(max_absolute)
    return sum(
        1 for blocks in data.values() for block in blocks if COMPLEXITY_RANKS.index(block.get("rank", "A")) > threshold
    )


def loc_count(max_lines: int = 500) -> int:
    total = 0
    for path in (ROOT / TARGET).rglob("*.py"):
        if "__pycache__" in path.parts or "archive_not_used_trash" in path.parts:
            continue
        line_count = sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))
        total += max(0, line_count - max_lines)
    return total


VECTORS: dict[str, Callable[[], int]] = {
    "mypy": mypy_count,
    "ruff": ruff_count,
    "xenon": xenon_count,
    "loc": loc_count,
}


def staged_app_python_files() -> list[Path]:
    stdout = run("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    paths: list[Path] = []
    for line in stdout.splitlines():
        path = Path(line)
        if path.parts[:1] == (TARGET,) and path.suffix == ".py" and (ROOT / path).exists():
            if "archive_not_used_trash" in path.parts:
                continue
            paths.append(path)
    return sorted(set(paths))


def print_mypy_details(paths: list[Path]) -> None:
    app_paths = [path.relative_to(TARGET).as_posix() for path in paths]
    if not app_paths:
        print("    no staged app Python files to inspect")
        return
    output = run_output("mypy", "--config-file", str(ROOT / "pyproject.toml"), *app_paths, cwd=ROOT / TARGET)
    print(output.rstrip() or "    mypy reported no errors on staged app Python files")


def mypy_detail_count(paths: list[Path]) -> int:
    app_paths = [path.relative_to(TARGET).as_posix() for path in paths]
    if not app_paths:
        return 0
    output = run_output("mypy", "--config-file", str(ROOT / "pyproject.toml"), *app_paths, cwd=ROOT / TARGET)
    match = re.search(r"Found (\d+) error", output)
    return int(match.group(1)) if match else 0


def print_ruff_details(paths: list[Path]) -> None:
    if not paths:
        print("    no staged app Python files to inspect")
        return
    output = run_output("ruff", "check", *(path.as_posix() for path in paths))
    print(output.rstrip() or "    ruff reported no errors on staged app Python files")


def ruff_detail_count(paths: list[Path]) -> int:
    if not paths:
        return 0
    stdout = run("ruff", "check", *(path.as_posix() for path in paths), "--output-format=json")
    try:
        violations = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return 0
    return len(violations)


def print_xenon_details(paths: list[Path], max_absolute: str = "B") -> None:
    paths = _exclude_tests(paths)
    if not paths:
        print("    no staged app Python files to inspect")
        return
    stdout = run("radon", "cc", "-j", *(path.as_posix() for path in paths))
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        print(stdout.rstrip() or "    radon output could not be parsed")
        return
    threshold = COMPLEXITY_RANKS.index(max_absolute)
    printed = False
    for file_path, blocks in sorted(data.items()):
        for block in blocks:
            rank = block.get("rank", "A")
            if COMPLEXITY_RANKS.index(rank) <= threshold:
                continue
            print(f"    {file_path}:{block.get('lineno')} {block.get('type')} {block.get('name')} rank {rank}")
            printed = True
    if not printed:
        print("    xenon/radon reported no rank > B blocks on staged app Python files")


def xenon_detail_count(paths: list[Path], max_absolute: str = "B") -> int:
    paths = _exclude_tests(paths)
    if not paths:
        return 0
    stdout = run("radon", "cc", "-j", *(path.as_posix() for path in paths))
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return 0
    threshold = COMPLEXITY_RANKS.index(max_absolute)
    return sum(
        1 for blocks in data.values() for block in blocks if COMPLEXITY_RANKS.index(block.get("rank", "A")) > threshold
    )


def print_loc_details(paths: list[Path], max_lines: int = 500) -> None:
    printed = False
    for path in paths:
        line_count = sum(1 for _ in (ROOT / path).open(encoding="utf-8", errors="ignore"))
        overage = line_count - max_lines
        if overage <= 0:
            continue
        print(f"    {path.as_posix()}: {line_count} lines (+{overage} over {max_lines})")
        printed = True
    if not printed:
        print("    no staged app Python files exceed the loc threshold")


def loc_detail_count(paths: list[Path], max_lines: int = 500) -> int:
    total = 0
    for path in paths:
        line_count = sum(1 for _ in (ROOT / path).open(encoding="utf-8", errors="ignore"))
        total += max(0, line_count - max_lines)
    return total


DETAIL_PRINTERS: dict[str, Callable[[list[Path]], None]] = {
    "mypy": print_mypy_details,
    "ruff": print_ruff_details,
    "xenon": print_xenon_details,
    "loc": print_loc_details,
}

DETAIL_COUNTERS: dict[str, Callable[[list[Path]], int]] = {
    "mypy": mypy_detail_count,
    "ruff": ruff_detail_count,
    "xenon": xenon_detail_count,
    "loc": loc_detail_count,
}


def print_regression_details(regressed: list[tuple[str, int, int]]) -> None:
    paths = staged_app_python_files()
    print("\nDetails from staged app Python files:")
    for name, _, _ in regressed:
        print(f"\n  {name}:")
        detail_printer = DETAIL_PRINTERS.get(name)
        if detail_printer is None:
            print("    no detail printer configured")
            continue
        detail_printer(paths)


def characterization_test_touched() -> bool:
    stdout = run("git", "diff", "--cached", "--name-only")
    test_dirs = ("app/tests/characterization", "app/tests/unit", "app/tests/regression")
    return any(line.startswith(test_dirs) for line in stdout.splitlines())


def main() -> int:
    baseline = load_json(BASELINE_PATH)

    candidate_regressions: list[tuple[str, int, int]] = []
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
            candidate_regressions.append((name, base, current))
        elif current < base:
            improved[name] = (base, current, base - current)

    staged_paths = staged_app_python_files()
    regressed: list[tuple[str, int, int]] = []
    ignored_regressions: list[tuple[str, int, int]] = []
    for name, base, current in candidate_regressions:
        detail_counter = DETAIL_COUNTERS.get(name)
        staged_problem_count = detail_counter(staged_paths) if detail_counter is not None else 0
        if staged_problem_count > 0:
            regressed.append((name, base, current))
        else:
            ignored_regressions.append((name, base, current))

    if regressed:
        print("Incremental pre-commit ratchet: BLOCKED - new problems introduced\n")
        for name, base, current in regressed:
            print(f"  {name}: baseline {base} -> now {current} (+{current - base} new)")
        print_regression_details(regressed)
        print(
            "\nThis only blocks NEW violations you introduced - it never requires fixing "
            "pre-existing debt in files you happen to touch. The details above are scoped "
            "to staged app Python files; the blocking comparison is still the project-wide total."
        )
        return 1
    if ignored_regressions:
        print("Incremental pre-commit ratchet: project-wide regression ignored for untouched vectors\n")
        for name, base, current in ignored_regressions:
            print(
                f"  {name}: baseline {base} -> now {current} (+{current - base} project-wide), "
                "but staged app Python files report 0 problems for this vector"
            )

    ratcheted: list[tuple[str, int, int]] = []
    for name, (base, current, _progress) in improved.items():
        if current < base - (base * RATCHET_IMPROVEMENT_RATIO):
            baseline[name] = current
            ratcheted.append((name, base, current))

    if bootstrapped or ratcheted:
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        subprocess.run(["git", "add", str(BASELINE_PATH)], cwd=ROOT, check=False)
    if ratcheted:
        print("Incremental pre-commit ratchet: progress locked in\n")
        for name, base, current in ratcheted:
            print(
                f"  {name}: baseline {base} -> {current} "
                f"(-{base - current} fixed, threshold {RATCHET_IMPROVEMENT_RATIO:.0%})"
            )
        if not characterization_test_touched():
            print(
                "\nNote: this commit fixed enough pre-existing problems to ratchet a baseline down, "
                "but doesn't touch app/tests/{characterization,unit,regression}. If any of these were "
                "behavior-affecting fixes (not just type annotations/formatting), pin the before/after "
                "behavior with a characterization test first - see the test-strategy skill. This is a "
                "reminder, not a block: this repo's mutation-safety net is test discipline, not a mutation-testing "
                "tool (see docs/infrastructure.md#pre-commit)."
            )

    print("\nIncremental pre-commit ratchet: OK (no regressions)")
    for name, current in current_counts.items():
        print(f"  {name}: {current} / baseline {baseline.get(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
