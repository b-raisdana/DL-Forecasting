from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pandera
from config import app_config
from helper.data_preparation import after_under_process_date, concat
from helper.logging import log_d, log_w
from helper.schema_casting import all_annotations
from infrastructure.ohlcv.disk_cache import (
    datarange_is_not_cachable,
    read_without_index,
    remove_data_file,
    write_data_file,
)
from infrastructure.ohlcv.fragmented_data import symbol_data_path
from pandera import DataType
from pandera import typing as pt


class BasePanderaDFM(pandera.DataFrameModel):
    pass

    # @classmethod
    # def empty(cls):
    #     if cls._sample_obj is None:
    #         raise Exception("_sample_obj should be set manually before.")
    #     return cls._sample_obj.drop(cls._sample_obj.index)


class BaseDFM(pandera.DataFrameModel):
    class Config:
        # to resolve pandera.errors.SchemaError: column ['XXXX'] not in dataframe
        add_missing_columns = True
        # to resolve pandera.errors.SchemaError: expected series ['XXXX'/None] to have type datetime64[ns, UTC]
        # , got object
        coerce = True


class ExtendedDf:
    _column_dtypes = None
    _index_names = None
    column_d_type_assertion_checked = False
    schema_data_frame_model: BasePanderaDFM = None
    _sample_df: pd.DataFrame = None
    _empty_df: pd.DataFrame = None

    @classmethod
    def new(cls, dictionary_of_data: dict = None, strict: bool = True) -> pt.DataFrame[BasePanderaDFM]:
        if app_config.check_assertions and cls._sample_df is None:
            raise AssertionError(f"{cls}._sample_obj should be defined before!")
        if cls._empty_df is None:
            _empty = cls._sample_df.drop(cls._sample_df.index)
            cls._empty_df = cls.schema_data_frame_model.to_schema().validate(_empty)

        if dictionary_of_data is not None:
            if not isinstance(dictionary_of_data, dict):
                raise ValueError(f"dictionary_of_data should be dict but {type(dictionary_of_data)} given.")
            if any([isinstance(v, list) for k, v in dictionary_of_data.items()]):
                raise NotImplementedError("List values as dictionary_of_data values!")
            _new = cls._empty_df.copy()
            _index_names = cls.index_names()
            if len(_index_names) > 1:
                try:
                    the_index = tuple([dictionary_of_data[k] for k in _index_names])
                except KeyError:
                    raise Exception(
                        f"Indexes {_index_names} should have value in the dictionary_of_data: {dictionary_of_data}"
                    )
            else:
                try:
                    the_index = dictionary_of_data[_index_names[0]]
                except KeyError:
                    raise Exception(
                        f"Indexes {_index_names} should have value in the dictionary_of_data: {dictionary_of_data}"
                    )
            unused_keys = []
            for key in dictionary_of_data:
                if not strict or key in cls.schema_data_frame_model.to_schema().columns.keys():
                    if key not in _index_names:
                        # _new[key] = pd.Series()
                        # log_d(f"{key}({type(dictionary_of_data[key])})={dictionary_of_data[key]}")
                        _new.loc[the_index, key] = dictionary_of_data[key]
                elif key not in _index_names:
                    unused_keys += [key]
            if len(unused_keys) > 0:
                raise Exception(f"Unused keys in the dictionary: {','.join(unused_keys)}")
            try:
                _new = cls.schema_data_frame_model.to_schema().validate(_new, lazy=True)
            except pandera.errors.SchemaErrors as e:
                if "coerce_dtype('int64')         [nan]".replace(" ", "") in str(e).replace(" ", ""):
                    raise TypeError(
                        "Use pt.Series[pd.Int8Dtype] instead of pt.Series[int] to allow nullable int series: " + str(e)
                    )
                else:
                    raise e
            return _new
        _new = cls._empty_df.copy()
        return _new

    @classmethod
    def cast_and_validate(
        cls: type[ExtendedDf],
        df: ExtendedDf | pd.DataFrame,
        return_bool: bool = False,
        zero_size_allowed: bool = False,
    ) -> pt.DataFrame[BasePanderaDFM] | bool:
        if not zero_size_allowed:
            if len(df) == 0:
                raise Exception("Zero size data not allowed in parameters!")
        if app_config.check_assertions and cls.schema_data_frame_model is None:
            raise AssertionError(f"Define cls.schema_data_frame_model in child class:{cls.__name__}")
        try:
            result = cls.schema_data_frame_model.validate(df)
        except pandera.errors.SchemaError as e:
            if return_bool:
                log_d(f"data is not valid according to {cls.schema_data_frame_model.__name__}")
                return False
            else:
                raise e
        if return_bool:
            return True
        return result

    @classmethod
    def read_file(
        cls,
        date_range_str: str,
        data_frame_type: str,
        generator: Callable,
        skip_rows=None,
        n_rows=None,
        file_path: str = None,
        zero_size_allowed: None | bool = None,
    ) -> pt.DataFrame[BasePanderaDFM]:
        """
        Read data from a file and return a DataFrame. If the file does not exist or the DataFrame does not
        match the expected columns, the generator function is used to build and persist the DataFrame.

        This function reads data from a file with the specified `data_frame_type` and `date_range_str` parameters.
        If the file exists and the DataFrame columns match the expected columns defined in the configuration, the
        function returns the DataFrame. Otherwise, `generator` is called to build the DataFrame; this single
        wrapper (not the generator) persists it via write_data_file() and hands back the in-memory result
        directly, so a cache miss never pays for a redundant re-read of the file it just wrote (see
        cache-or-generate skill; mirrors helper.data_preparation.read_file()).

        Parameters:
            date_range_str (str): The date range string used to construct the filename.
            data_frame_type (str): The type of DataFrame to read/write, e.g., 'ohlcva', 'multi_timeframe_ohlcva', etc.
            generator (Callable): Builds and returns the DataFrame for date_range_str; must not write it itself.
            skip_rows (Optional[int]): The number of rows to skip while reading the file.
            n_rows (Optional[int]): The maximum number of rows to read from the file.
            file_path (str, optional): The path to the directory containing the data files.

        Returns:
            pd.DataFrame: The DataFrame read from the file or generated by the generator function.

        Raises:
            Exception: If the expected columns are not defined in the configuration or if the generated DataFrame
                       does not match the expected columns.

        Example:
            # Assuming you have a generator function 'generate_ohlcva' returning a DataFrame
            df = read_file(date_range_str='17-10-06.00-00T17-10-06', data_frame_type='ohlcva',
                           generator=generate_ohlcva)

        Note:
            This function first attempts to read the file based on the provided parameters. If the file is not found
            or the DataFrame does not match the expected columns, the generator function is called to build the
            DataFrame, which is then persisted and returned as-is (no re-read).
            :param zero_size_allowed:
            :param file_path:
            :param n_rows:
            :param skip_rows:
            :param date_range_str:
            :param data_frame_type:
            :param generator:
        """
        if file_path is None:
            file_path = symbol_data_path()
        if date_range_str is None:
            date_range_str = app_config.processing_date_range
        df = None
        try:
            df = cls.read_and_index(data_frame_type, date_range_str, file_path, n_rows, skip_rows)
        except FileNotFoundError:
            pass
        if zero_size_allowed is None:
            zero_size_allowed = after_under_process_date(date_range_str)
        if df is None or not cls.cast_and_validate(df, return_bool=True, zero_size_allowed=zero_size_allowed):
            if skip_rows or n_rows is not None:
                # generator() returns the full, already-indexed range; slicing is only meaningful
                # against a flat, not-yet-indexed disk read, so combining a cache miss with a partial
                # read isn't supported (no caller does this today).
                raise Exception(
                    "read_file() cannot generate on a cache miss when skip_rows/n_rows is set; "
                    "pre-populate the cache with a full read first."
                )
            df = generator(date_range_str)
            write_data_file(df, data_frame_type, date_range_str, file_path)
            df = cls.cast_and_validate(df, zero_size_allowed=zero_size_allowed)
        else:
            df = cls.cast_and_validate(df, zero_size_allowed=zero_size_allowed)
        if datarange_is_not_cachable(date_range_str):
            remove_data_file(data_frame_type, date_range_str, file_path)
        return df

    @classmethod
    def read_and_index(cls, data_frame_type, date_range_str, file_path, n_rows, skip_rows):
        df = read_without_index(data_frame_type, date_range_str, file_path, n_rows, skip_rows)
        index_names = cls.index_names()
        df.set_index(index_names, inplace=True)
        return df

    @classmethod
    def concat(
        cls, left: pt.DataFrame[BasePanderaDFM], right: pt.DataFrame[BasePanderaDFM]
    ) -> pt.DataFrame[BasePanderaDFM]:
        result: pt.DataFrame[BasePanderaDFM] = concat(left, right)
        return result

    @classmethod
    def index_names(cls, sample_df=None):
        if sample_df is not None:
            # implement extracting index names from a sample df if needed.
            raise NotImplementedError
        if cls._index_names is not None:
            return cls._index_names
        _index_names = []
        schema = cls.schema_data_frame_model.to_schema()
        if hasattr(schema.index, "names"):
            _index_names += schema.index.names
        elif hasattr(schema.index, "name"):
            log_w("Never been tested")
            if schema.index.name is None or schema.index.name == "":
                raise AttributeError("Set name of index as title!")
            _index_names = [schema.index.name]
        if len(_index_names) == 0 or None in _index_names:
            raise ValueError(
                "Use = pandera.Field(check_name=True) for single index.len(_index_names) == 0 or None in _index_names"
            )
        cls._index_names = _index_names
        return cls._index_names

    @classmethod
    def index_id(cls, index_name: str):
        return cls.index_names().index(index_name)

    @classmethod
    def column_dtypes(cls) -> dict[str, DataType]:
        if cls._column_dtypes is not None:
            return cls._column_dtypes
        _all_annotations = all_annotations(cls.schema_data_frame_model)
        data_index_names = cls.index_names()
        column_annotations = {k: a for k, a in _all_annotations.items() if k not in data_index_names}
        cls.column_d_type_assertion(column_annotations)
        # given:
        # _all_annotations['end'] =
        # pandera.typing.pandas.Series[typing.Annotated[pandas.core.dtypes.dtypes.DatetimeTZDtype, 'ns', 'UTC']]
        # then:
        # _all_annotations['end'].__args__[0] =
        # typing.Annotated[pandas.core.dtypes.dtypes.DatetimeTZDtype, 'ns', 'UTC']
        column_annotations = {k: a.__args__[0] for k, a in _all_annotations.items() if k not in data_index_names}
        cls._column_dtypes = column_annotations
        return cls._column_dtypes

    @classmethod
    def column_d_type_assertion(cls, column_annotations):
        if cls.column_d_type_assertion_checked:
            return
        d_type: str
        for key, d_type in column_annotations.items():
            if not str(d_type).startswith("pandera.typing.pandas.Series["):
                raise NotImplementedError
        cls.column_d_type_assertion_checked = True
