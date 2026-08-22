import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/git-hooks/incremental-precommit"))
import ratchet_check  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(ratchet_check, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(ratchet_check, "staged_app_python_files", lambda: [])
    monkeypatch.setattr(ratchet_check, "characterization_test_touched", lambda: True)
    monkeypatch.setattr(ratchet_check.subprocess, "run", lambda *a, **k: None)
    return baseline_path


def _seed(baseline_path: Path, data: dict[str, int]) -> None:
    baseline_path.write_text(json.dumps(data))


def _wire(monkeypatch, current: dict[str, int], staged_detail: dict[str, int] | None = None) -> None:
    monkeypatch.setattr(ratchet_check, "PROJECT_COUNTERS", [lambda: dict(current)])
    monkeypatch.setattr(ratchet_check, "DETAIL_COUNTERS", [lambda paths: dict(staged_detail or {})])


def test_ruff_json_groups_by_rule_code():
    stdout = json.dumps([{"code": "E501"}, {"code": "E501"}, {"code": "F401"}])
    assert ratchet_check._group_ruff(stdout) == {"ruff:E501": 2, "ruff:F401": 1}


def test_mypy_groups_by_error_code_and_ignores_notes_and_summary():
    output = "\n".join(
        [
            "a.py:1: error: bad type  [type-arg]",
            "a.py:1: error: bad type  [type-arg]",
            "a.py:2: error: no overload  [call-overload]",
            "a.py:2: note: Possible overload variants:",
            "a.py:2: note:     def f() -> int",
            "Found 3 errors in 1 file (checked 1 source file)",
        ]
    )
    assert ratchet_check._group_mypy(output) == {"mypy:type-arg": 2, "mypy:call-overload": 1}


def test_mypy_error_without_code_falls_back_to_uncoded_bucket():
    output = "a.py:1: error: something old-style with no bracket code"
    assert ratchet_check._group_mypy(output) == {"mypy:uncoded": 1}


def test_bootstrap_new_key_does_not_block(hermetic, monkeypatch, capsys):
    _wire(monkeypatch, current={"ruff:E501": 3})

    exit_code = ratchet_check.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[ruff:E501] no baseline yet - bootstrapping at 3" in out
    assert json.loads(hermetic.read_text()) == {"ruff:E501": 3}


def test_current_equal_to_baseline_passes(hermetic, monkeypatch, capsys):
    _seed(hermetic, {"ruff:E501": 3})
    _wire(monkeypatch, current={"ruff:E501": 3})

    exit_code = ratchet_check.main()

    assert exit_code == 0
    assert "BLOCKED" not in capsys.readouterr().out


def test_regression_with_staged_problem_blocks(hermetic, monkeypatch, capsys):
    _seed(hermetic, {"ruff:E501": 3})
    _wire(monkeypatch, current={"ruff:E501": 5}, staged_detail={"ruff:E501": 2})

    exit_code = ratchet_check.main()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "ruff:E501: baseline 3 -> now 5 (+2 new)" in out


def test_regression_without_staged_problem_is_ignored_but_other_key_still_blocks(hermetic, monkeypatch, capsys):
    _seed(hermetic, {"ruff:E501": 3, "ruff:F401": 1})
    _wire(
        monkeypatch,
        current={"ruff:E501": 5, "ruff:F401": 2},
        staged_detail={"ruff:F401": 1},
    )

    exit_code = ratchet_check.main()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "ruff:F401: baseline 1 -> now 2 (+1 new)" in out
    assert "ruff:E501: baseline 3 -> now 5" not in out


def test_fixing_many_of_one_rule_does_not_mask_a_new_violation_of_another(hermetic, monkeypatch, capsys):
    _seed(hermetic, {"ruff:A": 10, "ruff:B": 1})
    _wire(monkeypatch, current={"ruff:A": 3, "ruff:B": 2}, staged_detail={"ruff:B": 1})

    exit_code = ratchet_check.main()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "ruff:B: baseline 1 -> now 2 (+1 new)" in out
    assert "ruff:A" not in out.split("Details from staged app Python files")[0]


def test_successful_commit_resyncs_baseline_and_drops_retired_key(hermetic, monkeypatch):
    _seed(hermetic, {"ruff:E501": 3, "ruff:OLDRULE": 2})
    _wire(monkeypatch, current={"ruff:E501": 3})

    exit_code = ratchet_check.main()

    assert exit_code == 0
    assert json.loads(hermetic.read_text()) == {"ruff:E501": 3}


def test_characterization_reminder_prints_only_when_tests_untouched(hermetic, monkeypatch, capsys):
    _seed(hermetic, {"ruff:E501": 5})
    _wire(monkeypatch, current={"ruff:E501": 3})
    monkeypatch.setattr(ratchet_check, "characterization_test_touched", lambda: False)

    exit_code = ratchet_check.main()

    assert exit_code == 0
    assert "characterization test first" in capsys.readouterr().out


def test_characterization_reminder_suppressed_when_tests_touched(hermetic, monkeypatch, capsys):
    _seed(hermetic, {"ruff:E501": 5})
    _wire(monkeypatch, current={"ruff:E501": 3})
    monkeypatch.setattr(ratchet_check, "characterization_test_touched", lambda: True)

    exit_code = ratchet_check.main()

    assert exit_code == 0
    assert "characterization test first" not in capsys.readouterr().out
