import pytest
from application.datastore_engine.parquet_housekeeping import ConversionSummary
from presentation.datastore_engine import parquet_housekeeping_cli
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.unit
def test_main_reports_success_and_exits_zero(monkeypatch):
    summary = ConversionSummary(legacy_files_converted=["a.parquet"], parquet_files_scanned=5)
    monkeypatch.setattr(parquet_housekeeping_cli, "run_datastore_conversion", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, [])

    assert result.exit_code == 0
    assert "converted 1 legacy file(s)" in result.stdout
    assert "scanned 5 Parquet file(s)" in result.stdout


@pytest.mark.unit
def test_main_dry_run_forwards_flag_and_uses_conditional_verbs(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return ConversionSummary()

    monkeypatch.setattr(parquet_housekeeping_cli, "run_datastore_conversion", fake_run)

    result = runner.invoke(parquet_housekeeping_cli.app, ["--dry-run"])

    assert result.exit_code == 0
    assert captured["dry_run"] is True
    assert "would convert" in result.stdout


@pytest.mark.unit
def test_main_exits_nonzero_and_prints_failures_when_summary_has_failures(monkeypatch):
    summary = ConversionSummary(parquet_files_failed=[("bad.parquet", "boom")])
    monkeypatch.setattr(parquet_housekeeping_cli, "run_datastore_conversion", lambda **kwargs: summary)

    result = runner.invoke(parquet_housekeeping_cli.app, [])

    assert result.exit_code == 1
    assert "Parquet repair failure(s)" in result.stdout
    assert "bad.parquet: boom" in result.stdout
