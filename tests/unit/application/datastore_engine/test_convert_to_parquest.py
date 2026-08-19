import pandas as pd
import pytest
from application.datastore_engine.convert_to_parquest import run_datastore_conversion
from infrastructure.datastore_engine.convert_to_parquest import parquet_file_has_index_bug


def _flat_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1.0, 2.0, 3.0]})


@pytest.mark.unit
def test_run_datastore_conversion_converts_legacy_and_repairs_parquet(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    feather_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    good_parquet_path = type_dir / "ohlcv.24-01-03.00-00T24-01-03.23-59.parquet"
    _flat_ohlcv_df().to_feather(feather_path)
    _flat_ohlcv_df().set_index("date").to_parquet(bad_parquet_path)
    _flat_ohlcv_df().to_parquet(good_parquet_path)

    summary = run_datastore_conversion(root=tmp_path)

    assert summary.ok
    assert feather_path.with_suffix(".parquet") in summary.legacy_files_converted
    assert not feather_path.exists()
    assert bad_parquet_path in summary.parquet_files_repaired
    assert good_parquet_path not in summary.parquet_files_repaired
    assert summary.parquet_files_scanned == 3  # the converted feather file + the two originally-parquet files
    assert parquet_file_has_index_bug(bad_parquet_path) is False


@pytest.mark.unit
def test_run_datastore_conversion_dry_run_changes_nothing_on_disk(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    feather_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    _flat_ohlcv_df().to_feather(feather_path)
    _flat_ohlcv_df().set_index("date").to_parquet(bad_parquet_path)

    summary = run_datastore_conversion(root=tmp_path, dry_run=True)

    assert summary.ok
    assert feather_path.exists()  # untouched
    assert parquet_file_has_index_bug(bad_parquet_path) is True  # untouched
    assert len(summary.legacy_files_converted) == 1
    assert bad_parquet_path in summary.parquet_files_repaired


@pytest.mark.unit
def test_run_datastore_conversion_isolates_one_bad_file_from_the_rest(tmp_path, monkeypatch):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    good_parquet_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.parquet"
    bad_parquet_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(good_parquet_path)
    bad_parquet_path.write_bytes(b"not-a-real-parquet-file")

    summary = run_datastore_conversion(root=tmp_path)

    assert len(summary.parquet_files_failed) == 1
    assert summary.parquet_files_failed[0][0] == bad_parquet_path
    assert not summary.ok
    assert good_parquet_path in summary.parquet_files_repaired  # the other file still got fixed
