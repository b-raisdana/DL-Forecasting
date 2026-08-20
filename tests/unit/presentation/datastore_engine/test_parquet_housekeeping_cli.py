import pytest
from application.datastore_engine.parquet_housekeeping import CompactionSummary, MigrationSummary, RepairSummary
from presentation.datastore_engine import parquet_housekeeping_cli
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.unit
def test_migrate_reports_success_and_exits_zero(monkeypatch):
    summary = MigrationSummary(legacy_files_converted=["a.parquet"])
    monkeypatch.setattr(parquet_housekeeping_cli, "run_legacy_migration", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, ["migrate"])

    assert result.exit_code == 0
    assert "converted 1 legacy file(s) to Parquet" in result.stdout


@pytest.mark.unit
def test_migrate_dry_run_forwards_flag_and_uses_conditional_verb(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return MigrationSummary()

    monkeypatch.setattr(parquet_housekeeping_cli, "run_legacy_migration", fake_run)

    result = runner.invoke(parquet_housekeeping_cli.app, ["migrate", "--dry-run"])

    assert result.exit_code == 0
    assert captured["dry_run"] is True
    assert "would convert" in result.stdout


@pytest.mark.unit
def test_migrate_exits_nonzero_and_prints_failures_when_summary_has_failures(monkeypatch):
    summary = MigrationSummary(legacy_files_failed=[("bad.feather", "boom")])
    monkeypatch.setattr(parquet_housekeeping_cli, "run_legacy_migration", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, ["migrate"])

    assert result.exit_code == 1
    assert "legacy conversion failure(s)" in result.stdout
    assert "bad.feather: boom" in result.stdout


@pytest.mark.unit
def test_fix_index_reports_success_and_exits_zero(monkeypatch):
    summary = RepairSummary(ran=True, parquet_files_scanned=5, parquet_files_repaired=["a.parquet"])
    monkeypatch.setattr(parquet_housekeeping_cli, "run_index_repair", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, ["fix-index"])

    assert result.exit_code == 0
    assert "scanned 5 Parquet file(s), repaired 1" in result.stdout


@pytest.mark.unit
def test_fix_index_reports_skip_when_not_flagged(monkeypatch):
    summary = RepairSummary(ran=False)
    monkeypatch.setattr(parquet_housekeeping_cli, "run_index_repair", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, ["fix-index"])

    assert result.exit_code == 0
    assert "no repair flagged" in result.stdout


@pytest.mark.unit
def test_fix_index_exits_nonzero_and_prints_failures_when_summary_has_failures(monkeypatch):
    summary = RepairSummary(ran=True, parquet_files_scanned=1, parquet_files_failed=[("bad.parquet", "boom")])
    monkeypatch.setattr(parquet_housekeeping_cli, "run_index_repair", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, ["fix-index"])

    assert result.exit_code == 1
    assert "Parquet repair failure(s)" in result.stdout
    assert "bad.parquet: boom" in result.stdout


@pytest.mark.unit
def test_compact_reports_success_and_exits_zero(monkeypatch):
    summary = CompactionSummary(batches_merged=["merged.parquet"])
    monkeypatch.setattr(parquet_housekeeping_cli, "run_compaction", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, ["compact"])

    assert result.exit_code == 0
    assert "merged 1 batch(es)" in result.stdout


@pytest.mark.unit
def test_compact_dry_run_forwards_flag_and_uses_conditional_verb(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return CompactionSummary()

    monkeypatch.setattr(parquet_housekeeping_cli, "run_compaction", fake_run)

    result = runner.invoke(parquet_housekeeping_cli.app, ["compact", "--dry-run"])

    assert result.exit_code == 0
    assert captured["dry_run"] is True
    assert "would merge" in result.stdout
