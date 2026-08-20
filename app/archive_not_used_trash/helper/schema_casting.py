import logging
import string
from typing import Any

import pandas as pd
import pandera
from helper.functions import Pandera_DFM_Type
from helper.logging.do_log import log_d, log_e, log_w
from pandera import DataType


def cast_and_validate2(
    data,
    model_class: type[Pandera_DFM_Type],
    return_bool: bool = False,
    zero_size_allowed: bool = False,
    unique_index: bool = False,
) -> Any:
    if len(data) == 0:
        if not zero_size_allowed:
            raise Exception("Zero size data!")
        else:
            if return_bool:
                return True
            else:
                return empty_df(model_class)
    if unique_index:
        if not data.index.is_unique:
            raise NotImplementedError
            raise Exception(f"Expected to be unique but found duplicates:{data.index[data.index.duplicated()]}")
    try:
        column_annotations = column_dtypes(data, model_class)
        data = apply_as_type2(data, model_class, column_annotations)
    except KeyError as e:
        if return_bool:
            log_d(e)
            return False
        else:
            raise e
    if return_bool:
        try:
            model_class.validate(data, lazy=True)
        except pandera.errors.SchemaErrors as exc:
            log_w(str(exc.schema_errors))
            return False
    else:
        model_class.validate(
            data,
            lazy=True,
        )
    if return_bool:
        return True
    data = data[column_annotations.keys()]
    return data


def apply_as_type2(data, model_class, _column_dtypes) -> pd.DataFrame:
    as_types = {}
    for attr_name, attr_type in _column_dtypes.items():
        if attr_name not in data.dtypes.keys() and (hasattr(data.index, "names") and attr_name not in data.index.names):
            raise KeyError(f"'{attr_name}' in {model_class.__name__} but not in data:{data.dtypes}")
        if "timestamp" in str(attr_type).lower() and "timestamp" not in str(data.dtypes.loc[attr_name]).lower():
            as_types[attr_name] = "datetime64[ns, UTC]"
        if "datetimetzdtype" in str(attr_type).lower():
            if "datetimetzdtype" not in str(data.dtypes.loc[attr_name]).lower():
                as_types[attr_name] = "datetime64[ns, UTC]"
            elif "timedelta" in str(attr_type).lower() and "timedelta" not in str(data.dtypes.loc[attr_name]).lower():
                as_types[attr_name] = "timedelta64[s]"
                # as_types[attr_name] = pandera.typing.Timedelta
        elif "pandera.typing.pandas.Series" in str(attr_type):
            astype = str(attr_type).replace("pandera.typing.pandas.Series[", "").replace("]", "")
            trans_table = str.maketrans("", "", string.digits)
            astype = astype.translate(trans_table)
            if astype != "str" and attr_name in data.columns and astype not in str(data.dtypes.loc[attr_name]).lower():
                as_types[attr_name] = astype
    if len(as_types) > 0:
        data = data.astype(as_types)
    return data


def column_dtypes(data, model_class) -> dict[str, DataType]:
    _all_annotations = all_annotations(model_class)
    data_index_names = index_names(data)
    column_annotations = {k: a for k, a in _all_annotations.items() if k not in data_index_names}
    d_type: str
    return column_annotations


def index_names(data):
    _index_names = []
    if hasattr(data.index, "names"):
        _index_names += data.index.names
    elif hasattr(data.index, "name"):
        if data.index.name is None or data.index.name == "":
            raise Exception("Set name of index as title!")
        _index_names = [data.index.name]
    return _index_names


    # return list(model_class.to_schema().columns.keys())


# duplicated from app/helper/schema_casting.py (still live there; a dead function here depends on it)
def empty_df(model_class: type[Pandera_DFM_Type]) -> pd.DataFrame:
    as_types = dict(column_fields(model_class), **index_fields(model_class))
    # Create an empty DataFrame with Pandas-compatible data types
    empty_data = {column: [] for column in as_types}
    _empty_df = pd.DataFrame(empty_data)
    for _name, _type in as_types.items():
        as_types[_name] = _type.type.name

    _empty_df = _empty_df.astype(as_types)
    # if len(index_fields(model_class).keys()) == 0:
    #     pass
    _empty_df = _empty_df.set_index(list(index_fields(model_class).keys()))
    _empty_df = model_class(_empty_df)
    return _empty_df


# duplicated from app/helper/schema_casting.py (still live there; a dead function here depends on it)
def all_annotations(cls, include_indexes=False) -> dict:
    """Returns a dictionary-like ChainMap that includes annotations for all
    attributes defined in cls or inherited from superclasses."""
    all_classes_list = [c.__annotations__ for c in cls.__mro__ if hasattr(c, "__annotations__")]
    annotations = {}
    if include_indexes:
        drop_list = ["Config"]
    else:
        drop_list = ["date", "timeframe", "Config"]
    for single_class_annotations in all_classes_list:
        for attr_name, attr_type in single_class_annotations.items():
            if attr_name not in drop_list and "__" not in attr_name:
                annotations[attr_name] = attr_type
    return annotations  # ChainMap(*(c.__annotations__ for c in cls.__mro__ if '__annotations__' in c.__dict__))


# duplicated from app/helper/schema_casting.py (still live there; a dead function here depends on it)
def column_fields(model_class: type[Pandera_DFM_Type]) -> dict[str, DataType]:
    return model_class.to_schema().dtypes


# duplicated from app/helper/schema_casting.py (still live there; a dead function here depends on it)
def index_fields(model_class: type[Pandera_DFM_Type]) -> dict[str, str]:
    if "PeakValleys" in model_class.__name__:
        pass
    if hasattr(model_class.to_schema().index, "columns"):
        # model_class has a MultiIndex
        # names = list(model_class.to_schema().index.columns.keys())
        names = model_class.to_schema().index.dtypes
    else:
        # model_class has a single Index
        all_fields = all_annotations(model_class, include_indexes=True)
        names = {
            k: model_class.to_schema().index.dtype
            for k, v in all_fields.items()
            if "pandera.typing.pandas.Index" in str(v.__origin__)
        }
    return names
