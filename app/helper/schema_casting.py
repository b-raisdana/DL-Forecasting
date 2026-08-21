import logging
import string
from collections.abc import Hashable, Sequence
from typing import cast

import pandas as pd
import pandera
from helper.functions import Pandera_DFM_Type
from helper.logging.do_log import log_d, log_e
from helper.pandera import pandera_validate
from pandera import DataType


def all_annotations(cls: type, include_indexes: bool = False) -> dict[str, object]:
    """Returns a dictionary-like ChainMap that includes annotations for all
    attributes defined in cls or inherited from superclasses."""
    all_classes_list = [c.__annotations__ for c in cls.__mro__ if hasattr(c, "__annotations__")]
    annotations: dict[str, object] = {}
    drop_list = ["Config"] if include_indexes else ["date", "timeframe", "Config"]
    for single_class_annotations in all_classes_list:
        for attr_name, attr_type in single_class_annotations.items():
            if attr_name not in drop_list and "__" not in attr_name:
                annotations[attr_name] = attr_type
    return annotations  # ChainMap(*(c.__annotations__ for c in cls.__mro__ if '__annotations__' in c.__dict__))


@pandera_validate(allow_pandas_dataframe=True)
def cast_and_validate(
    data: pd.DataFrame,
    model_class: type[Pandera_DFM_Type],
    return_bool: bool = False,
    zero_size_allowed: bool = False,
    unique_index: bool = False,
) -> pd.DataFrame | bool:
    if len(data) == 0:
        if not zero_size_allowed:
            raise Exception("Zero size data!")
        else:
            if return_bool:
                return True
            else:
                return cast(pd.DataFrame, empty_df(model_class))
    if unique_index and not data.index.is_unique:
        log_e("Not tested")
        raise Exception(f"Expected to be unique but found duplicates:{data.index[data.index.duplicated()]}")
    try:
        data = apply_as_type(data, model_class)
        data = _coerce_index_to_ns_utc(data)
    except KeyError as e:
        if return_bool:
            log_d(str(e))
            return False
        else:
            raise e
    if return_bool:
        try:
            model_class.validate(data, lazy=True)
        except pandera.errors.SchemaErrors as exc:
            log_d(str(exc.schema_errors), logging.WARNING)
            return False
    else:
        model_class.validate(
            data,
            lazy=True,
        )
    if return_bool:
        return True
    if model_class.to_schema().strict:
        columns_to_keep: list[str] = [
            column for column in model_class.__fields__ if column not in ["timeframe", "date"]
        ]
        data = data[columns_to_keep]
    return data


def _coerce_datetime_to_ns_utc(series: pd.Series) -> pd.Series:  # type: ignore[explicit-any]
    """Coerce any datetime-like Series to ``datetime64[ns, UTC]``: localize naive timestamps, convert
    aware ones to UTC, and force nanosecond precision (Pandera validates against ``datetime64[ns, UTC]``
    and pandas 3.x may hand us ``datetime64[us, <tz>]`` straight from DuckDB/Parquet)."""
    if getattr(series.dt, "tz", None) is None or series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    return series.astype("datetime64[ns, UTC]")


@pandera_validate(allow_pandas_dataframe=True)
def _coerce_index_to_ns_utc(data: pd.DataFrame) -> pd.DataFrame:
    """Coerce every DatetimeIndex level (e.g. the ``date`` index) to ``datetime64[ns, UTC]`` so the
    schema's ``DatetimeTZDtype("ns", "UTC")`` index passes validation regardless of the incoming
    resolution or timezone."""
    if isinstance(data.index, pd.MultiIndex):
        new_levels: list[pd.Index] = []  # type: ignore[explicit-any]
        for _level_name, level_values in zip(data.index.names, data.index.levels, strict=True):
            if isinstance(level_values, pd.DatetimeIndex):
                # Wrap in DatetimeIndex (not `.values`): a tz-aware Series' `.values` drops the
                # timezone in pandas 3.x, which would lose the UTC localization we just applied.
                new_levels.append(pd.DatetimeIndex(_coerce_datetime_to_ns_utc(pd.Series(level_values))))
            else:
                new_levels.append(level_values)
        data.index = data.index.set_levels(cast(Sequence[Sequence[Hashable]], new_levels))  # type: ignore[arg-type]
    elif isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.DatetimeIndex(_coerce_datetime_to_ns_utc(pd.Series(data.index)))
    return data


@pandera_validate(allow_pandas_dataframe=True)
def apply_as_type(data: pd.DataFrame, model_class: type[Pandera_DFM_Type]) -> pd.DataFrame:
    as_types: dict[str, str] = {}
    _all_annotations = all_annotations(model_class)
    for attr_name, attr_type in _all_annotations.items():
        if attr_name not in data.dtypes and (hasattr(data.index, "names") and attr_name not in data.index.names):
            raise KeyError(f"'{attr_name}' in {model_class.__name__} but not in data:{data.dtypes}")
        col_dtype = data.dtypes.get(attr_name) if attr_name in data.dtypes else None
        attr_type_str = str(attr_type).lower()
        is_datetime_annotation = (
            "timestamp" in attr_type_str or "datetimetzdtype" in attr_type_str or "datetime64" in attr_type_str
        )
        if col_dtype is not None and (
            isinstance(col_dtype, pd.DatetimeTZDtype) or str(col_dtype).startswith("datetime64")
        ):
            if is_datetime_annotation and not str(col_dtype).lower().startswith("datetime64[ns, utc"):
                as_types[attr_name] = "datetime64[ns, UTC]"
        elif col_dtype is None and attr_name in data.index.names and is_datetime_annotation:
            idx = data.index.get_level_values(attr_name)
            if isinstance(idx, pd.DatetimeIndex) and not str(idx.dtype).startswith("datetime64[ns, utc"):
                data.index = data.index.set_levels(
                    cast(
                        pd.DatetimeIndex,
                        pd.DatetimeIndex(data.index.get_level_values(attr_name).astype("datetime64[ns, UTC]")),
                    ),
                    level=attr_name,
                )
        elif "timedelta" in attr_type_str and col_dtype is not None and "timedelta" not in str(col_dtype).lower():
            as_types[attr_name] = "timedelta64[s]"
        elif "pandera.typing.pandas.Series" in str(attr_type):
            astype = str(attr_type).replace("pandera.typing.pandas.Series[", "").replace("]", "")
            trans_table = str.maketrans("", "", string.digits)
            astype = astype.translate(trans_table)
            if astype != "str" and attr_name in data.columns and astype not in str(data.dtypes.loc[attr_name]).lower():
                as_types[attr_name] = astype
    if len(as_types) > 0:
        data = data.astype(as_types)
    return data


# def cast_and_validate2(
# data,
# model_class: type[Pandera_DFM_Type],
# return_bool: bool = False,
# zero_size_allowed: bool = False,
# unique_index: bool = False,
# ) -> Any:
# if len(data) == 0:
# if not zero_size_allowed:
# raise Exception("Zero size data!")
# else:
# if return_bool:
# return True
# else:
# return empty_df(model_class)
# if unique_index:
# if not data.index.is_unique:
# raise NotImplementedError
# raise Exception(f"Expected to be unique but found duplicates:{data.index[data.index.duplicated()]}")
# try:
# column_annotations = column_dtypes(data, model_class)
# data = apply_as_type2(data, model_class, column_annotations)
# except KeyError as e:
# if return_bool:
# log_d(e)
# return False
# else:
# raise e
# if return_bool:
# try:
# model_class.validate(data, lazy=True)
# except pandera.errors.SchemaErrors as exc:
# log_w(str(exc.schema_errors))
# return False
# else:
# model_class.validate(
# data,
# lazy=True,
# )
# if return_bool:
# return True
# data = data[column_annotations.keys()]
# return data


# def apply_as_type2(data, model_class, _column_dtypes) -> ptd:
# as_types = {}
# for attr_name, attr_type in _column_dtypes.items():
# if attr_name not in data.dtypes.keys() and (hasattr(data.index, "names") and attr_name not in data.index.names):
# raise KeyError(f"'{attr_name}' in {model_class.__name__} but not in data:{data.dtypes}")
# if "timestamp" in str(attr_type).lower() and "timestamp" not in str(data.dtypes.loc[attr_name]).lower():
# as_types[attr_name] = "datetime64[ns, UTC]"
# if "datetimetzdtype" in str(attr_type).lower():
# if "datetimetzdtype" not in str(data.dtypes.loc[attr_name]).lower():
# as_types[attr_name] = "datetime64[ns, UTC]"
# elif "timedelta" in str(attr_type).lower() and "timedelta" not in str(data.dtypes.loc[attr_name]).lower():
# as_types[attr_name] = "timedelta64[s]"
# # as_types[attr_name] = pandera.typing.Timedelta
# elif "pandera.typing.pandas.Series" in str(attr_type):
# astype = str(attr_type).replace("pandera.typing.pandas.Series[", "").replace("]", "")
# trans_table = str.maketrans("", "", string.digits)
# astype = astype.translate(trans_table)
# if astype != "str" and attr_name in data.columns and astype not in str(data.dtypes.loc[attr_name]).lower():
# as_types[attr_name] = astype
# if len(as_types) > 0:
# data = data.astype(as_types)
# return data


# def column_dtypes(data, model_class) -> dict[str, DataType]:
# _all_annotations = all_annotations(model_class)
# data_index_names = index_names(data)
# column_annotations = {k: a for k, a in _all_annotations.items() if k not in data_index_names}
# d_type: str
# return column_annotations


# def index_names(data):
# _index_names = []
# if hasattr(data.index, "names"):
# _index_names += data.index.names
# elif hasattr(data.index, "name"):
# if data.index.name is None or data.index.name == "":
# raise Exception("Set name of index as title!")
# _index_names = [data.index.name]
# return _index_names


def index_fields(model_class: type[Pandera_DFM_Type]) -> dict[str, DataType | str]:
    if "PeakValleys" in model_class.__name__:
        pass
    names: dict[str, DataType | str]
    if hasattr(model_class.to_schema().index, "columns"):
        # model_class has a MultiIndex
        # names = list(model_class.to_schema().index.columns.keys())
        names = model_class.to_schema().index.dtypes
    else:
        # model_class has a single Index
        all_fields = all_annotations(model_class, include_indexes=True)
        names = {}
        for k, v in all_fields.items():
            if hasattr(v, "__origin__") and "pandera.typing.pandas.Index" in str(v.__origin__):
                names[k] = model_class.to_schema().index.dtype
    return names


def column_fields(model_class: type[Pandera_DFM_Type]) -> dict[str, DataType]:
    return model_class.to_schema().dtypes
    # return list(model_class.to_schema().columns.keys())


def empty_df[T: Pandera_DFM_Type](model_class: type[T]) -> T:  # type: ignore[valid-type]
    as_types: dict[str, str] = {}
    for name, dtype in column_fields(model_class).items():
        as_types[name] = str(dtype.type.name)
    for name, idx_dtype in index_fields(model_class).items():
        if isinstance(idx_dtype, DataType):
            as_types[name] = str(idx_dtype.type.name)
        else:
            as_types[name] = str(idx_dtype)
    # Create an empty DataFrame with Pandas-compatible data types
    empty_data: dict[str, list[object]] = {column: [] for column in as_types}
    _empty_df = pd.DataFrame(empty_data)
    _empty_df = _empty_df.astype(as_types)
    index_names = list(index_fields(model_class).keys())

    if index_names:
        _empty_df = _empty_df.set_index(index_names)
    validated: T = cast(T, model_class(_empty_df))
    return validated
