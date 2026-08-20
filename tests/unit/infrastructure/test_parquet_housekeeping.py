import pandas as pd
import pytest
from infrastructure.datastore_engine.parquet_housekeeping import (
    _contiguous_merge_batches,
    _split_run_by_size,
    clear_repair_flag,
    convert_legacy_file,
    find_legacy_files,
    find_merge_batches,
    find_parquet_files,
    flag_repair_required,
    merge_batch,
    parquet_file_has_index_bug,
    repair_parquet_file,
    repair_required,
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
    parquet_housekeeping module docstring) must not carry that shape into the converted Parquet file."""
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


@pytest.mark.unit
def test_repair_flag_round_trips(tmp_path):
    assert repair_required(tmp_path) is False

    flag_repair_required(tmp_path)
    assert repair_required(tmp_path) is True

    clear_repair_flag(tmp_path)
    assert repair_required(tmp_path) is False


@pytest.mark.unit
def test_clear_repair_flag_is_noop_when_never_flagged(tmp_path):
    clear_repair_flag(tmp_path)  # must not raise
    assert repair_required(tmp_path) is False


def _daily_ohlcv_file(type_dir, day: str, size_bytes: int | None = None):
    path = type_dir / f"ohlcv.{day}.00-00T{day}.23-59.parquet"
    _flat_ohlcv_df().to_parquet(path)
    if size_bytes is not None:
        path.write_bytes(b"x" * size_bytes)  # overwrite with junk of an exact known size, for size-threshold tests
    return path


class _FakePath:
    def __init__(self, name, size):
        self.name = name
        self._size = size

    def stat(self):
        return type("S", (), {"st_size": self._size})()

    def __repr__(self):
        return self.name


@pytest.mark.unit
def test_split_run_by_size_flushes_once_target_crossed():
    range_to_path = {
        "a": _FakePath("a", 40),
        "b": _FakePath("b", 40),
        "c": _FakePath("c", 40),
    }

    batches = _split_run_by_size("ohlcv", ["a", "b", "c"], range_to_path, target_bytes=50)

    assert len(batches) == 1
    data_frame_type, files = batches[0]
    assert data_frame_type == "ohlcv"
    assert files == [range_to_path["a"], range_to_path["b"]]  # "c" left as a dangling single-file batch, dropped


@pytest.mark.unit
def test_split_run_by_size_drops_trailing_single_file_batch():
    range_to_path = {"a": _FakePath("a", 10)}
    assert _split_run_by_size("ohlcv", ["a"], range_to_path, target_bytes=50) == []


# A target so large no run crosses it mid-accumulation — isolates "does contiguity grouping/splitting
# work" from _split_run_by_size's own crossing-the-target behavior (covered separately above). A whole
# contiguous run of >=2 files still flushes as one trailing batch once the run ends (see
# _split_run_by_size's docstring: only a trailing batch of exactly 1 file is dropped).
_HUGE_TARGET_BYTES = 10**9


@pytest.mark.unit
def test_contiguous_merge_batches_merges_contiguous_daily_files(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    for day in ["24-01-01", "24-01-02", "24-01-03"]:
        _daily_ohlcv_file(type_dir, day)

    batches = _contiguous_merge_batches(type_dir, "ohlcv", target_bytes=_HUGE_TARGET_BYTES)

    assert len(batches) == 1
    data_frame_type, files = batches[0]
    assert data_frame_type == "ohlcv"
    assert len(files) == 3


@pytest.mark.unit
def test_contiguous_merge_batches_splits_on_a_gap(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    for day in ["24-01-01", "24-01-02", "24-01-04", "24-01-05"]:  # 24-01-03 missing
        _daily_ohlcv_file(type_dir, day)

    batches = _contiguous_merge_batches(type_dir, "ohlcv", target_bytes=_HUGE_TARGET_BYTES)

    assert len(batches) == 2
    assert {len(files) for _dft, files in batches} == {2}


@pytest.mark.unit
def test_contiguous_merge_batches_ignores_already_multi_window_files(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    # a single file already spanning 2 days (a legacy pre-windowing range, or a previous compaction's
    # own output) must not be treated as a merge candidate alongside single-window files.
    _flat_ohlcv_df().to_parquet(type_dir / "ohlcv.24-01-01.00-00T24-01-02.23-59.parquet")
    _daily_ohlcv_file(type_dir, "24-01-03")

    assert _contiguous_merge_batches(type_dir, "ohlcv", target_bytes=_HUGE_TARGET_BYTES) == []


@pytest.mark.unit
def test_merge_batch_merges_and_deletes_sources(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    files = [_daily_ohlcv_file(type_dir, day) for day in ["24-01-01", "24-01-02", "24-01-03"]]

    merged_path = merge_batch("ohlcv", files)

    assert merged_path is not None
    assert merged_path.name == "ohlcv.24-01-01.00-00T24-01-03.23-59.parquet"
    assert merged_path.exists()
    for path in files:
        assert not path.exists()
    merged_df = pd.read_parquet(merged_path)
    assert len(merged_df) == 3 * len(_flat_ohlcv_df())
    assert parquet_file_has_index_bug(merged_path) is False


@pytest.mark.unit
def test_merge_batch_leaves_sources_on_read_failure(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    good_path = _daily_ohlcv_file(type_dir, "24-01-01")
    bad_path = type_dir / "ohlcv.24-01-02.00-00T24-01-02.23-59.parquet"
    bad_path.write_bytes(b"not a parquet file")

    merged_path = merge_batch("ohlcv", [good_path, bad_path])

    assert merged_path is None
    assert good_path.exists()
    assert bad_path.exists()


@pytest.mark.unit
def test_find_merge_batches_merges_a_trailing_run_even_under_the_default_target_size(tmp_path):
    """A contiguous run that never crosses app_config.parquet_target_chunk_size_mb (100MB by default —
    these 3 tiny files are nowhere close) still merges as one trailing batch once the run ends, rather
    than being left fragmented forever — see _split_run_by_size's docstring."""
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    for day in ["24-01-01", "24-01-02", "24-01-03"]:
        _daily_ohlcv_file(type_dir, day)

    batches = find_merge_batches(tmp_path)

    assert len(batches) == 1
    data_frame_type, files = batches[0]
    assert data_frame_type == "ohlcv"
    assert len(files) == 3


@pytest.mark.unit
def test_find_merge_batches_finds_nothing_for_a_single_isolated_file(tmp_path):
    type_dir = tmp_path / "ohlcv" / "Spot" / "BTCUSDT" / "Kucoin"
    type_dir.mkdir(parents=True)
    _daily_ohlcv_file(type_dir, "24-01-01")

    assert find_merge_batches(tmp_path) == []
