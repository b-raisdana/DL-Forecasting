import os
from typing import Annotated

import pandas as pd
import pandera
import pytest
from infrastructure.datastore_engine import disk_cache
from pandera import typing as pt


class _NullableColSchema(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]
    nullable_col: pt.Series[float] = pandera.Field(nullable=True)


class _ExtraColSchema(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]

    class Config:
        strict = False


def _df(rows: int = 3, **columns: list) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=rows, tz="UTC"), **columns})
    return frame.set_index("date")


@pytest.mark.unit
def test_write_data_file_places_file_under_per_type_subfolder(tmp_path):
    file_path = str(tmp_path)
    df = _df(value=[1.0, 2.0, 3.0])

    disk_cache.write_data_file(df, "myindicator", "20-01-01.00-00T20-01-03.00-00", file_path)

    type_dir = os.path.join(file_path, "myindicator")
    assert os.listdir(file_path) == ["myindicator"]
    assert os.listdir(type_dir) == ["myindicator.20-01-01.00-00T20-01-03.00-00.parquet"]


@pytest.mark.unit
def test_data_frame_type_dir_migrates_pre_existing_flat_file(tmp_path):
    file_path = str(tmp_path)
    flat_path = os.path.join(file_path, "legacytype.20-01-01.00-00T20-01-02.00-00.feather")
    _df(rows=2, value=[1.0, 2.0]).reset_index().to_feather(flat_path, compression="zstd")

    type_dir = disk_cache._data_frame_type_dir("legacytype", file_path)

    assert not os.path.exists(flat_path)
    assert os.path.exists(os.path.join(type_dir, "legacytype.20-01-01.00-00T20-01-02.00-00.feather"))


@pytest.mark.unit
def test_read_migrates_legacy_feather_file_to_parquet_on_whole_file_read(tmp_path):
    file_path = str(tmp_path)
    df = _df(value=[1.0, 2.0, 3.0])
    disk_cache._data_frame_type_dir("legacy_feather_type", file_path)  # ensure the type dir exists
    feather_path = disk_cache._feather_file_path("legacy_feather_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    df.reset_index().to_feather(feather_path, compression="zstd")

    result = disk_cache.read_without_index(
        "legacy_feather_type", "20-01-01.00-00T20-01-03.00-00", file_path, n_rows=None, skip_rows=None
    )

    parquet_path = disk_cache._parquet_file_path("legacy_feather_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    assert not os.path.exists(feather_path)
    assert os.path.exists(parquet_path)
    assert result["value"].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_read_prefers_parquet_over_a_coexisting_legacy_feather_file(tmp_path):
    file_path = str(tmp_path)
    disk_cache._data_frame_type_dir("both_formats_type", file_path)
    feather_path = disk_cache._feather_file_path("both_formats_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    parquet_path = disk_cache._parquet_file_path("both_formats_type", "20-01-01.00-00T20-01-03.00-00", file_path)
    _df(value=[1.0, 2.0, 3.0]).reset_index().to_feather(feather_path, compression="zstd")
    _df(value=[9.0, 9.0, 9.0]).reset_index().to_parquet(parquet_path, compression="zstd")

    result = disk_cache.read_without_index(
        "both_formats_type", "20-01-01.00-00T20-01-03.00-00", file_path, n_rows=None, skip_rows=None
    )

    assert result["value"].tolist() == [9.0, 9.0, 9.0]
    assert os.path.exists(feather_path)  # untouched — parquet already covers this file, no migration needed


@pytest.mark.unit
def test_write_data_file_raises_on_unexpected_nan(tmp_path):
    df = _df(value=[1.0, float("nan"), 3.0])

    with pytest.raises(Exception, match="value"):
        disk_cache.write_data_file(
            df, "strict_type", "20-01-01.00-00T20-01-03.00-00", str(tmp_path), nan_allowed_columns=frozenset()
        )


@pytest.mark.unit
def test_write_data_file_allows_nan_in_explicitly_allowed_column(tmp_path):
    file_path = str(tmp_path)
    df = _df(value=[1.0, float("nan"), 3.0])

    disk_cache.write_data_file(
        df, "allowed_type", "20-01-01.00-00T20-01-03.00-00", file_path, nan_allowed_columns=frozenset({"value"})
    )

    type_dir = os.path.join(file_path, "allowed_type")
    assert os.listdir(type_dir) == ["allowed_type.20-01-01.00-00T20-01-03.00-00.parquet"]


@pytest.mark.unit
def test_write_data_file_skips_nan_guard_when_nan_allowed_columns_is_none(tmp_path):
    df = _df(value=[1.0, float("nan"), 3.0])

    disk_cache.write_data_file(df, "unguarded_type", "20-01-01.00-00T20-01-03.00-00", str(tmp_path))


@pytest.mark.unit
def test_read_file_generator_write_allows_schema_nullable_nan_without_explicit_param(tmp_path):
    def generator(date_range_str, **kwargs):
        return _df(value=[1.0, 2.0, 3.0], nullable_col=[float("nan"), 2.0, 3.0])

    result = disk_cache.read_file(
        "20-01-01.00-00T20-01-03.00-00", "nullable_type", generator, _NullableColSchema, file_path=str(tmp_path)
    )

    assert result["nullable_col"].isna().sum() == 1


@pytest.mark.unit
def test_read_file_generator_write_raises_on_nan_in_non_nullable_column(tmp_path):
    def generator(date_range_str, **kwargs):
        return _df(value=[1.0, float("nan"), 3.0], nullable_col=[1.0, 2.0, 3.0])

    with pytest.raises(Exception, match="value"):
        disk_cache.read_file(
            "20-01-01.00-00T20-01-03.00-00", "strict_read_type", generator, _NullableColSchema, file_path=str(tmp_path)
        )


@pytest.mark.unit
def test_cache_on_disk_nan_allowed_columns_permits_extra_nonschema_column_nan(tmp_path):
    def generator(date_range_str, **kwargs) -> pt.DataFrame[_ExtraColSchema]:
        return _df(value=[1.0, 2.0, 3.0], scratch_col=[float("nan"), 2.0, 3.0])

    dataset = disk_cache.CachableDataset(
        dataset_folder_name="extra_type", nan_allowed_columns=frozenset({"scratch_col"})
    )
    get_extra = disk_cache.cache_on_disk(dataset)(generator)

    result = get_extra("20-01-01.00-00T20-01-03.00-00", file_path=str(tmp_path))

    assert result["scratch_col"].isna().sum() == 1
