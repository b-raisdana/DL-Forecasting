"""Unit tests for the archived disk_cache.read_without_index() — see
app/archive_not_used_trash/README.md.
"""

import os

import pandas as pd
import pytest
from archive_not_used_trash.infrastructure.datastore_engine import disk_cache as archived_disk_cache
from infrastructure.datastore_engine import disk_cache

pytestmark = pytest.mark.unit


def _df(rows: int = 3, **columns: list) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=rows, tz="UTC"), **columns})
    return frame.set_index("date")


def test_read_migrates_legacy_feather_file_to_parquet_on_whole_file_read(tmp_path):
    file_path = str(tmp_path)
    df = _df(value=[1.0, 2.0, 3.0])
    disk_cache._data_frame_type_dir("legacy_feather_type", file_path)  # ensure the type dir exists
    feather_path = disk_cache._feather_file_path("legacy_feather_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    df.reset_index().to_feather(feather_path, compression="zstd")

    result = archived_disk_cache.read_without_index(
        "legacy_feather_type", "20-01-01.00-00T20-01-03.00-00", file_path, n_rows=None, skip_rows=None
    )

    parquet_path = disk_cache._parquet_file_path("legacy_feather_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    assert not os.path.exists(feather_path)
    assert os.path.exists(parquet_path)
    assert result["value"].tolist() == [1.0, 2.0, 3.0]


def test_read_prefers_parquet_over_a_coexisting_legacy_feather_file(tmp_path):
    file_path = str(tmp_path)
    disk_cache._data_frame_type_dir("both_formats_type", file_path)
    feather_path = disk_cache._feather_file_path("both_formats_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    parquet_path = disk_cache._parquet_file_path("both_formats_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    _df(value=[1.0, 2.0, 3.0]).reset_index().to_feather(feather_path, compression="zstd")
    _df(value=[9.0, 9.0, 9.0]).reset_index().to_parquet(parquet_path, compression="zstd")

    result = archived_disk_cache.read_without_index(
        "both_formats_type", "20-01-01.00-00T20-01-03.00-00", file_path, n_rows=None, skip_rows=None
    )

    assert result["value"].tolist() == [9.0, 9.0, 9.0]
    assert os.path.exists(feather_path)  # untouched — parquet already covers this file, no migration needed
