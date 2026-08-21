from __future__ import annotations

import logging
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
import pandera as pa
from pandera import typing as pt

logger = logging.getLogger(__name__)


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


def pandera_transform(func: Any = None, *, allow_pandas_dataframe: bool = False) -> Any:
    """
    Runtime Pandera validation decorator.

    - Wraps the function with ``pandera.check_types(lazy=True)`` so every
      ``pt.DataFrame[Schema]``-annotated input/output is validated at runtime.
    - Recursively inspects annotations and emits a warning whenever a bare
      ``pandas.DataFrame`` / ``pd.DataFrame`` is found.
    - Pass ``allow_pandas_dataframe=True`` to silence that legacy-DataFrame
      warning for functions where no Pandera schema applies. This does **not**
      disable validation of other ``pt.DataFrame[Schema]`` annotations in the
      same signature.
    """

    def decorator(func: Any) -> Any:
        hints = get_type_hints(func, include_extras=True)

        input_annotations = {name: hints[name] for name in func.__annotations__ if name in hints}

        output_annotation = hints.get("return")

        for name, annotation in input_annotations.items():
            if not allow_pandas_dataframe and _contains_legacy_pandas_dataframe(annotation):
                logger.warning(
                    "%s: parameter %r uses pandas.DataFrame instead of pandera.typing.DataFrame",
                    func.__qualname__,
                    name,
                )

        if (
            output_annotation is not None
            and not allow_pandas_dataframe
            and _contains_legacy_pandas_dataframe(output_annotation)
        ):
            logger.warning(
                "%s: return annotation uses pandas.DataFrame instead of pandera.typing.DataFrame",
                func.__qualname__,
            )

        @pa.check_types(lazy=True)
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
