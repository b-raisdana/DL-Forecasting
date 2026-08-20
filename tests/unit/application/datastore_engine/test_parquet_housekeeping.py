import pandas as pd
import pytest
from application.datastore_engine.parquet_housekeeping import run_compaction, run_index_repair, run_legacy_migration
from infrastructure.datastore_engine.parquet_housekeeping import parquet_file_has_index_bug


def _flat_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1.0, 2.0, 3.0]})


@pytest.mark.unit
def test_run_legacy_migration_converts_feather_to_parquet(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    feather_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    _flat_ohlcv_df().to_feather(feather_path)

    summary = run_legacy_migration(root=tmp_path)

    assert summary.ok
    assert feather_path.with_suffix(".parquet") in summary.legacy_files_converted
    assert not feather_path.exists()


@pytest.mark.unit
def test_run_legacy_migration_dry_run_changes_nothing_on_disk(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    feather_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    _flat_ohlcv_df().to_feather(feather_path)

    summary = run_legacy_migration(root=tmp_path, dry_run=True)

    assert summary.ok
    assert feather_path.exists()  # untouched
    assert len(summary.legacy_files_converted) == 1


@pytest.mark.unit
def test_run_index_repair_repairs_bad_parquet_and_leaves_good_parquet_alone(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    good_parquet_path = type_dir / "ohlcv.24-01-03.00-00T24-01-03.23-59.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(bad_parquet_path)
    _flat_ohlcv_df().to_parquet(good_parquet_path)

    summary = run_index_repair(root=tmp_path, force=True)

    assert summary.ran
    assert summary.ok
    assert summary.parquet_files_scanned == 2
    assert bad_parquet_path in summary.parquet_files_repaired
    assert good_parquet_path not in summary.parquet_files_repaired
    assert parquet_file_has_index_bug(bad_parquet_path) is False


@pytest.mark.unit
def test_run_index_repair_dry_run_changes_nothing_on_disk(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(bad_parquet_path)

    summary = run_index_repair(root=tmp_path, dry_run=True, force=True)

    assert summary.ok
    assert parquet_file_has_index_bug(bad_parquet_path) is True  # untouched
    assert bad_parquet_path in summary.parquet_files_repaired


@pytest.mark.unit
def test_run_index_repair_isolates_one_bad_file_from_the_rest(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    good_parquet_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.parquet"
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(good_parquet_path)
    bad_parquet_path.write_bytes(b"not-a-real-parquet-file")

    summary = run_index_repair(root=tmp_path, force=True)

    assert len(summary.parquet_files_failed) == 1
    assert summary.parquet_files_failed[0][0] == bad_parquet_path
    assert not summary.ok
    assert good_parquet_path in summary.parquet_files_repaired  # the other file still got fixed


@pytest.mark.unit
def test_run_index_repair_skips_scan_when_not_flagged_and_not_forced(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(bad_parquet_path)

    summary = run_index_repair(root=tmp_path)

    assert summary.ran is False
    assert parquet_file_has_index_bug(bad_parquet_path) is True  # untouched, scan never ran


@pytest.mark.unit
def test_run_compaction_merges_a_contiguous_run(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    paths = []
    for day in ["24-01-01", "24-01-02", "24-01-03"]:
        path = type_dir / f"ohlcv.{day}.00-00T{day}.23-59.parquet"
        _flat_ohlcv_df().to_parquet(path)
        paths.append(path)

    summary = run_compaction(root=tmp_path)

    assert summary.ok
    assert len(summary.batches_merged) == 1
    assert summary.batches_merged[0].name == "ohlcv.24-01-01.00-00T24-01-03.23-59.parquet"
    for path in paths:
        assert not path.exists()


@pytest.mark.unit
def test_run_compaction_dry_run_changes_nothing_on_disk(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    paths = []
    for day in ["24-01-01", "24-01-02", "24-01-03"]:
        path = type_dir / f"ohlcv.{day}.00-00T{day}.23-59.parquet"
        _flat_ohlcv_df().to_parquet(path)
        paths.append(path)

    summary = run_compaction(root=tmp_path, dry_run=True)

    assert summary.ok
    assert len(summary.batches_merged) == 1
    for path in paths:
        assert path.exists()  # untouched


@pytest.mark.unit
def test_run_compaction_finds_nothing_for_an_isolated_file(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    _flat_ohlcv_df().to_parquet(type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.parquet")

    summary = run_compaction(root=tmp_path)

    assert summary.ok
    assert summary.batches_merged == []
