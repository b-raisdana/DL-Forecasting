import pandas as pd
import pytest
from infrastructure.datastore_engine.convert_to_parquest import (
    convert_legacy_file,
    find_legacy_files,
    find_parquet_files,
    parquet_file_has_index_bug,
    repair_parquet_file,
)


def _flat_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1.0, 2.0, 3.0]})


@pytest.mark.unit
def test_find_legacy_files_finds_feather_and_zip_recursively(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    feather_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    zip_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.zip"
    parquet_path = type_dir / "ohlcv.24-01-03.00-00T24-01-03.23-59.parquet"
    _flat_ohlcv_df().to_feather(feather_path)
    _flat_ohlcv_df().to_csv(zip_path, compression="zip", index=False)
    _flat_ohlcv_df().to_parquet(parquet_path)

    found = find_legacy_files(tmp_path)

    assert set(found) == {feather_path, zip_path}


@pytest.mark.unit
def test_find_parquet_files_finds_only_parquet_recursively(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    parquet_path = type_dir / "ohlcv.24-01-01.00-00T24-01-01.23-59.parquet"
    _flat_ohlcv_df().to_parquet(parquet_path)
    (type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.feather").write_bytes(b"not-checked-here")

    assert find_parquet_files(tmp_path) == [parquet_path]


@pytest.mark.unit
def test_convert_legacy_file_migrates_feather_to_parquet_and_deletes_source(tmp_path):
    feather_path = tmp_path / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    _flat_ohlcv_df().to_feather(feather_path)

    result_path = convert_legacy_file(feather_path)

    assert result_path == feather_path.with_suffix(".parquet")
    assert result_path.exists()
    assert not feather_path.exists()
    df = pd.read_parquet(result_path)
    assert df["date"].tolist() == _flat_ohlcv_df()["date"].tolist()


@pytest.mark.unit
def test_convert_legacy_file_flattens_a_date_indexed_feather_source(tmp_path):
    """Regression: a legacy Feather file that was itself written with `date` as its own index (the
    root cause of the pre-existing Parquet index bug — see infrastructure.datastore_engine.
    convert_to_parquest module docstring) must not carry that shape into the converted Parquet file."""
    feather_path = tmp_path / "ohlcv.24-01-01.00-00T24-01-01.23-59.feather"
    _flat_ohlcv_df().set_index("date").to_feather(feather_path)

    result_path = convert_legacy_file(feather_path)

    assert result_path is not None
    assert parquet_file_has_index_bug(result_path) is False
    assert "date" in pd.read_parquet(result_path).columns


@pytest.mark.unit
def test_convert_legacy_file_migrates_csv_zip_to_parquet_and_deletes_source(tmp_path):
    zip_path = tmp_path / "ohlcv.24-01-01.00-00T24-01-01.23-59.zip"
    _flat_ohlcv_df().to_csv(zip_path, compression="zip", index=False)

    result_path = convert_legacy_file(zip_path)

    assert result_path == zip_path.with_suffix(".parquet")
    assert result_path.exists()
    assert not zip_path.exists()


@pytest.mark.unit
def test_convert_legacy_file_returns_none_for_unrecognized_extension(tmp_path):
    other_path = tmp_path / "notes.txt"
    other_path.write_text("hello")

    assert convert_legacy_file(other_path) is None


@pytest.mark.unit
def test_parquet_file_has_index_bug_true_when_date_was_written_as_index(tmp_path):
    path = tmp_path / "bad.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(path)

    assert parquet_file_has_index_bug(path) is True


@pytest.mark.unit
def test_parquet_file_has_index_bug_false_for_flat_file(tmp_path):
    path = tmp_path / "good.parquet"
    _flat_ohlcv_df().to_parquet(path)

    assert parquet_file_has_index_bug(path) is False


@pytest.mark.unit
def test_repair_parquet_file_fixes_a_date_indexed_file_atomically(tmp_path):
    path = tmp_path / "bad.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(path)

    repaired = repair_parquet_file(path)

    assert repaired is True
    assert parquet_file_has_index_bug(path) is False
    df = pd.read_parquet(path)
    assert "date" in df.columns
    assert not (tmp_path / "bad.parquet.repair.tmp").exists()


@pytest.mark.unit
def test_repair_parquet_file_is_noop_for_already_flat_file(tmp_path):
    path = tmp_path / "good.parquet"
    original = _flat_ohlcv_df()
    original.to_parquet(path)
    mtime_before = path.stat().st_mtime_ns

    repaired = repair_parquet_file(path)

    assert repaired is False
    assert path.stat().st_mtime_ns == mtime_before


@pytest.mark.unit
def test_repair_parquet_file_dry_run_reports_without_writing(tmp_path):
    path = tmp_path / "bad.parquet"
    _flat_ohlcv_df().set_index("date").to_parquet(path)

    repaired = repair_parquet_file(path, dry_run=True)

    assert repaired is True
    assert parquet_file_has_index_bug(path) is True  # untouched
