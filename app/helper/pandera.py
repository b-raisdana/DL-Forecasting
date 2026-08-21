"""
Drop-in replacement for ``@pa.check_types(lazy=True)`` that adds two
behaviours pandera does not provide:

1. Warns when legacy bare ``pd.DataFrame`` annotations are used.
2. (Reserved) Recursive discovery of ``pt.DataFrame[Schema]`` inside nested
   container annotations.

Use ``allow_pandas_dataframe=True`` to skip the warning and delegate
directly to ``pa.check_types(lazy=True)`` — zero extra overhead.
"""

from __future__ import annotations

from functools import wraps
from typing import (
    Annotated,
    Any,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

import pandas as pd
import pandera.pandas as pa
from config import app_config
from helper.logging.do_log.log_it import log_w
from pandera import typing as pt


def _dataframe_schema(annotation: object) -> type[pa.DataFrameModel] | None:
    """Return the Pandera schema from a DataFrame[Schema] annotation."""
    origin = get_origin(annotation)

    if origin is pt.DataFrame:
        args = get_args(annotation)
        if len(args) == 1 and isinstance(args[0], type):
            return args[0]

    return None


def _walk_dataframe_annotations(
    annotation: object,
) -> list[type[pa.DataFrameModel]]:
    """
    Recursively find all Pandera DataFrame schemas inside an annotation.

    Handles, for example:
        DataFrame[A]
        tuple[DataFrame[A], DataFrame[B]]
        list[DataFrame[A]]
        dict[str, DataFrame[A]]
        DataFrame[A] | None
        Optional[DataFrame[A]]
        tuple[DataFrame[A], list[DataFrame[B] | None]]
        Annotated[DataFrame[A], ...]
    """
    if annotation is Any:
        return []

    schema = _dataframe_schema(annotation)
    if schema is not None:
        return [schema]

    origin = get_origin(annotation)

    if origin is None:
        return []

    # Annotated[T, ...] → inspect T only.
    if origin is Annotated:
        args = get_args(annotation)
        return _walk_dataframe_annotations(args[0]) if args else []

    # Literal[...] contains values, not types.
    if origin is Literal:
        return []

    schemas: list[type[pa.DataFrameModel]] = []

    for arg in get_args(annotation):
        # Union[T, None], T | None, etc.
        if arg is type(None):
            continue

        schemas.extend(_walk_dataframe_annotations(arg))

    return schemas


def _contains_legacy_pandas_dataframe(annotation: object) -> bool:
    """Return True when annotation contains bare/legacy pandas DataFrame."""
    if annotation is pd.DataFrame:
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    return any(_contains_legacy_pandas_dataframe(arg) for arg in get_args(annotation) if arg is not type(None))


def pandera_validate(func: Any = None, *, allow_pandas_dataframe: bool = False) -> Any:
    """
    Runtime Pandera validation decorator.

    - ``allow_pandas_dataframe=False`` (default): warns on bare ``pd.DataFrame``
      annotations, then applies ``pa.check_types(lazy=True)``.
    - ``allow_pandas_dataframe=True``: skips the warning and do
      ``pa.check_types(lazy=True)(func)`` directly.
    """

    def decorator(func: Any) -> Any:
        if app_config.environment == "production":
            return func

        if allow_pandas_dataframe:
            return pa.check_types(lazy=True)(func)

        hints = get_type_hints(func, include_extras=True)

        input_annotations = {name: hints[name] for name in func.__annotations__ if name in hints}

        output_annotation = hints.get("return")

        for name, annotation in input_annotations.items():
            if _contains_legacy_pandas_dataframe(annotation):
                log_w(
                    f"{func.__qualname__}: parameter {name} uses pandas.DataFrame instead of pandera.typing.DataFrame",
                    stack_offset=2,
                )

        if output_annotation is not None and _contains_legacy_pandas_dataframe(output_annotation):
            log_w(
                f"{func.__qualname__}: return annotation uses pandas.DataFrame instead of pandera.typing.DataFrame",
                stack_offset=2,
            )

        @pa.check_types(lazy=True)
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
