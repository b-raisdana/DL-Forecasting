# print is the intended user-facing output for this pre-commit hook script
# See README.md: "design" for the module-level overview.
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "baseline.json"
TARGET = "app"
COMPLEXITY_RANKS = "ABCDEF"

MYPY_CODED_ERROR_RE = re.compile(r": error: .*\[([\w-]+)\]\s*$")
MYPY_UNCODED_ERROR_RE = re.compile(r": error: ")


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


def _group_ruff(stdout: str) -> dict[str, int]:
    try:
        violations = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        violations = []
    return dict(Counter(f"ruff:{v['code']}" for v in violations if v.get("code")))


def _group_mypy(output: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in output.splitlines():
        coded = MYPY_CODED_ERROR_RE.search(line)
        if coded:
            counts[f"mypy:{coded.group(1)}"] += 1
        elif MYPY_UNCODED_ERROR_RE.search(line):
            counts["mypy:uncoded"] += 1
    return dict(counts)


def _tool_of(key: str) -> str:
    return key.split(":", 1)[0]


def _exclude_tests(paths: list[Path]) -> list[Path]:
    return [path for path in paths if "tests" not in path.parts and "archive_not_used_trash" not in path.parts]


def ruff_project_counts() -> dict[str, int]:
    stdout = run("ruff", "check", TARGET, "--output-format=json")
    return _group_ruff(stdout)


def mypy_project_counts() -> dict[str, int]:
    # See README.md: "mypy_count cwd choice"
    output = run_output("mypy", "--config-file", str(ROOT / "pyproject.toml"), ".", cwd=ROOT / TARGET)
    return _group_mypy(output)


def xenon_project_counts(max_absolute: str = "B") -> dict[str, int]:
    # See README.md: "xenon excludes tests and archive"
    stdout = run("radon", "cc", TARGET, "-j", "-i", "tests,archive_not_used_trash")
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    threshold = COMPLEXITY_RANKS.index(max_absolute)
    count = sum(
        1 for blocks in data.values() for block in blocks if COMPLEXITY_RANKS.index(block.get("rank", "A")) > threshold
    )
    return {"xenon": count}


def loc_project_counts(max_lines: int = 500) -> dict[str, int]:
    total = 0
    for path in (ROOT / TARGET).rglob("*.py"):
        if "__pycache__" in path.parts or "archive_not_used_trash" in path.parts:
            continue
        line_count = sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))
        total += max(0, line_count - max_lines)
    return {"loc": total}


PROJECT_COUNTERS: list[Callable[[], dict[str, int]]] = [
    ruff_project_counts,
    mypy_project_counts,
    xenon_project_counts,
    loc_project_counts,
]


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


def ruff_detail_counts(paths: list[Path]) -> dict[str, int]:
    if not paths:
        return {}
    stdout = run("ruff", "check", *(path.as_posix() for path in paths), "--output-format=json")
    return _group_ruff(stdout)


def mypy_detail_counts(paths: list[Path]) -> dict[str, int]:
    app_paths = [path.relative_to(TARGET).as_posix() for path in paths]
    if not app_paths:
        return {}
    output = run_output("mypy", "--config-file", str(ROOT / "pyproject.toml"), *app_paths, cwd=ROOT / TARGET)
    return _group_mypy(output)


def xenon_detail_counts(paths: list[Path], max_absolute: str = "B") -> dict[str, int]:
    paths = _exclude_tests(paths)
    if not paths:
        return {}
    stdout = run("radon", "cc", "-j", *(path.as_posix() for path in paths))
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return {}
    threshold = COMPLEXITY_RANKS.index(max_absolute)
    count = sum(
        1 for blocks in data.values() for block in blocks if COMPLEXITY_RANKS.index(block.get("rank", "A")) > threshold
    )
    return {"xenon": count}


def loc_detail_counts(paths: list[Path], max_lines: int = 500) -> dict[str, int]:
    total = 0
    for path in paths:
        line_count = sum(1 for _ in (ROOT / path).open(encoding="utf-8", errors="ignore"))
        total += max(0, line_count - max_lines)
    return {"loc": total}


DETAIL_COUNTERS: list[Callable[[list[Path]], dict[str, int]]] = [
    ruff_detail_counts,
    mypy_detail_counts,
    xenon_detail_counts,
    loc_detail_counts,
]


def print_ruff_details(paths: list[Path]) -> None:
    if not paths:
        print("    no staged app Python files to inspect")
        return
    output = run_output("ruff", "check", *(path.as_posix() for path in paths))
    print(output.rstrip() or "    ruff reported no errors on staged app Python files")


def print_mypy_details(paths: list[Path]) -> None:
    app_paths = [path.relative_to(TARGET).as_posix() for path in paths]
    if not app_paths:
        print("    no staged app Python files to inspect")
        return
    output = run_output("mypy", "--config-file", str(ROOT / "pyproject.toml"), *app_paths, cwd=ROOT / TARGET)
    print(output.rstrip() or "    mypy reported no errors on staged app Python files")


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


DETAIL_PRINTERS: dict[str, Callable[[list[Path]], None]] = {
    "ruff": print_ruff_details,
    "mypy": print_mypy_details,
    "xenon": print_xenon_details,
    "loc": print_loc_details,
}


def print_regression_details(regressed: list[tuple[str, int, int]]) -> None:
    paths = staged_app_python_files()
    print("\nDetails from staged app Python files:")
    printed_tools: set[str] = set()
    for name, _, _ in regressed:
        tool = _tool_of(name)
        if tool in printed_tools:
            continue
        printed_tools.add(tool)
        print(f"\n  {tool}:")
        detail_printer = DETAIL_PRINTERS.get(tool)
        if detail_printer is None:
            print("    no detail printer configured")
            continue
        detail_printer(paths)


def characterization_test_touched() -> bool:
    stdout = run("git", "diff", "--cached", "--name-only")
    test_dirs = ("app/tests/characterization", "app/tests/unit", "app/tests/regression")
    return any(line.startswith(test_dirs) for line in stdout.splitlines())


def main() -> int:
    old_baseline = load_json(BASELINE_PATH)

    current_counts: dict[str, int] = {}
    for counter in PROJECT_COUNTERS:
        current_counts.update(counter())

    candidate_regressions: list[tuple[str, int, int]] = []
    improved: list[tuple[str, int, int]] = []
    for key in sorted(set(old_baseline) | set(current_counts)):
        current = current_counts.get(key, 0)
        base = old_baseline.get(key)
        if base is None:
            print(f"[{key}] no baseline yet - bootstrapping at {current}")
        elif current > base:
            candidate_regressions.append((key, base, current))
        elif current < base:
            improved.append((key, base, current))

    staged_paths = staged_app_python_files()
    staged_detail: dict[str, int] = {}
    for detail_counter in DETAIL_COUNTERS:
        staged_detail.update(detail_counter(staged_paths))

    regressed: list[tuple[str, int, int]] = []
    ignored_regressions: list[tuple[str, int, int]] = []
    for key, base, current in candidate_regressions:
        if staged_detail.get(key, 0) > 0:
            regressed.append((key, base, current))
        else:
            ignored_regressions.append((key, base, current))

    if regressed:
        print("Incremental pre-commit ratchet: BLOCKED - new problems introduced\n")
        for key, base, current in regressed:
            print(f"  {key}: baseline {base} -> now {current} (+{current - base} new)")
        print_regression_details(regressed)
        print(
            "\nThis only blocks NEW violations you introduced - it never requires fixing "
            "pre-existing debt in files you happen to touch. The details above are scoped "
            "to staged app Python files; the blocking comparison is still the project-wide total."
        )
        return 1
    if ignored_regressions:
        print("Incremental pre-commit ratchet: project-wide regression ignored for untouched keys\n")
        for key, base, current in ignored_regressions:
            print(
                f"  {key}: baseline {base} -> now {current} (+{current - base} project-wide), "
                "but staged app Python files report 0 problems for this key"
            )

    new_baseline = dict(current_counts)
    if new_baseline != old_baseline:
        BASELINE_PATH.write_text(json.dumps(new_baseline, indent=2, sort_keys=True) + "\n")
        subprocess.run(["git", "add", str(BASELINE_PATH)], cwd=ROOT, check=False)

    if improved:
        print("Incremental pre-commit ratchet: progress locked in\n")
        for key, base, current in improved:
            print(f"  {key}: baseline {base} -> {current} (-{base - current} fixed)")
        if not characterization_test_touched():
            print(
                "\nNote: this commit fixed pre-existing problems, but doesn't touch "
                "app/tests/{characterization,unit,regression}. If any of these were "
                "behavior-affecting fixes (not just type annotations/formatting), pin the before/after "
                "behavior with a characterization test first - see the test-strategy skill. This is a "
                "reminder, not a block: this repo's mutation-safety net is test discipline, not a mutation-testing "
                "tool (see docs/infrastructure.md#pre-commit)."
            )

    tool_totals: dict[str, int] = {"ruff": 0, "mypy": 0, "xenon": 0, "loc": 0}
    tool_baseline_totals: dict[str, int] = {"ruff": 0, "mypy": 0, "xenon": 0, "loc": 0}
    for key, count in current_counts.items():
        tool_totals[_tool_of(key)] = tool_totals.get(_tool_of(key), 0) + count
    for key, count in new_baseline.items():
        tool_baseline_totals[_tool_of(key)] = tool_baseline_totals.get(_tool_of(key), 0) + count

    print("\nIncremental pre-commit ratchet: OK (no regressions)")
    for tool in sorted(set(tool_totals) | set(tool_baseline_totals)):
        print(f"  {tool}: {tool_totals.get(tool, 0)} / baseline {tool_baseline_totals.get(tool, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
