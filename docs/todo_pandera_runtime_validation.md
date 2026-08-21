# Pandera DataFrame Validation

## Where schemas live

- Value-object schemas (shape only, no behavior) live in `domain/schemas/<subpackage>/`, mirroring the domain module that owns the data.
  Example: `domain/schemas/price_action/CausalExtremum.py` → `CausalExtremumOHLC`, `CausalExtremumResult`.
- Transforms (functions that compute/derive) live in `domain/` or `application/` and **import** schemas from `domain/schemas/` — never define them inline.
- One schema file per concern. Don't pile schemas into `domain/schemas/common/`; split into subpackages (`price_action/`, `ohlcv/`, `market_structure/`, ...) as the domain grows.

## Defining a schema

```python
import pandera
from pandera import typing as pt
from typing import Annotated
import pandas as pd


class MyInput(pandera.DataFrameModel):
    class Config:
        coerce = True          # int→float, datetime coercion — never str→unparseable-float

    high: pt.Series[float]
    low: pt.Series[float]


class MyResult(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]   # always ns UTC
    true_peak_reach_minutes: pt.Series[float]
    extremum_sign: pt.Series[int]
```

Rules:
- `coerce = True` on **input** schemas only. Never on output schemas — a wrong output dtype is a bug to catch, not convert.
- Index must be `Annotated[pd.DatetimeTZDtype, "ns", "UTC"]`. pandas 3.x defaults to microsecond precision, so call `.astype("datetime64[ns, UTC]")` explicitly before returning.
- Column names match the DataFrame's actual columns exactly (lowercase snake_case).
- Use `pandera.DataFrameModel` / `pandera.typing as pt` — not the deprecated top-level `pandera.DataFrameModel` re-export shortcuts flagged as FutureWarning on 0.32+ (they still work today, but new code should be ready for the `pandera.pandas` move).

## Function signatures and automatic validation

Every DataFrame-in/DataFrame-out function is decorated with `pandera_validate` (defined in `app/helper/pandera.py` — import it, never redefine it inline):

```python
from helper.pandera import pandera_validate
from pandera import typing as pt


@pandera_validate
def transform(ohlc: pt.DataFrame[MyInput]) -> pt.DataFrame[MyResult]:
    # ... compute ...
    return result
```

- `pt.DataFrame[Schema]` is the *only* acceptable way to annotate DataFrame params/returns in application/domain code — the `pt.` prefix keeps pandera-validated DataFrames visually distinct from plain `pd.DataFrame`. Never annotate with bare `pd.DataFrame`.
- `pandera_validate` validates every `pt.DataFrame[Schema]`-annotated input on call and the return value on return (`check_types(lazy=True)` under the hood) — no manual `Schema.validate(...)` calls needed in the body. `lazy=True` collects all schema errors into a single `SchemaErrors` exception instead of failing on the first.
- The index-precision cast is **not** automatic: you must still call `.astype("datetime64[ns, UTC]")` explicitly before returning if your output schema requires ns-UTC precision.
- Every function with a `pt.DataFrame[Schema]`-annotated parameter or return **must** carry `@pandera_transform`. An annotation without the decorator is silently unenforced — that's the one manual-placement step that can't be skipped.
- Exception: generic helper/infra functions with no stable expected shape may use bare `pd.DataFrame` and skip the decorator — this is the exception, not the default. If a function's shape is stable, give it a schema and decorate it.
- Already covered by `cast_and_validate()` (from `helper/schema_casting.py`)? Skip `@pandera_transform` for that same shape — but still decorate if the function produces a *new* intermediate shape no existing schema covers.
- Using a bare `pd.DataFrame` annotation anywhere in a decorated function's signature logs a warning at import time (not a hard failure) — treat that warning as a signal to switch to `pt.DataFrame[Schema]`, not as noise to ignore.

### `pandera_validate` source

```python
from __future__ import annotations

import logging
from functools import wraps
from typing import Any, get_args, get_origin, get_type_hints

import pandas as pd
import pandera as pa
from pandera import typing as pt

logger = logging.getLogger(__name__)


def _dataframe_schema(annotation: Any) -> type[pa.DataFrameModel] | None:
    """Return the Pandera schema from a DataFrame[Schema] annotation."""
    origin = get_origin(annotation)

    if origin is pt.DataFrame:
        args = get_args(annotation)
        if len(args) == 1 and isinstance(args[0], type):
            return args[0]

    return None


def _walk_dataframe_annotations(
    annotation: Any,
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


def _contains_legacy_pandas_dataframe(annotation: Any) -> bool:
    """Return True when annotation contains bare/legacy pandas DataFrame."""
    if annotation is pd.DataFrame:
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    return any(
        _contains_legacy_pandas_dataframe(arg)
        for arg in get_args(annotation)
        if arg is not type(None)
    )


def pandera_transform(func):
    hints = get_type_hints(func, include_extras=True)

    input_annotations = {
        name: hints[name]
        for name in func.__annotations__
        if name in hints
    }

    output_annotation = hints.get("return")

    for name, annotation in input_annotations.items():
        if _contains_legacy_pandas_dataframe(annotation):
            logger.warning(
                "%s: parameter %r uses pandas.DataFrame instead of "
                "pandera.typing.DataFrame",
                func.__qualname__,
                name,
            )

    if output_annotation is not None and _contains_legacy_pandas_dataframe(
        output_annotation
    ):
        logger.warning(
            "%s: return annotation uses pandas.DataFrame instead of "
            "pandera.typing.DataFrame",
            func.__qualname__,
        )

    @pa.check_types(lazy=True)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
```

### Known limitations (accepted, not fixed)

Note these in the docstring of the final code:
- **No automatic datetime-index normalization.** `pandera_validate` does not cast returned `DatetimeIndex` values to `datetime64[ns, UTC]`. You must call `.astype("datetime64[ns, UTC]")` explicitly before returning if your output schema requires ns-UTC precision.
- **No production bypass.** `pandera_validate` always runs `pa.check_types(lazy=True)`. There is no `app_config`-driven bypass — if you need to skip validation in production, remove the decorator rather than relying on a config flag.
- **Nested-container schema validation isn't independently confirmed.** `pa.check_types(lazy=True)` validates top-level `pt.DataFrame[Schema]` return values, but whether it enforces schemas nested inside `tuple`/`list`/`dict` containers hasn't been verified against this repo's pandera version. Prefer a single top-level `pt.DataFrame[Schema]` return where possible.
- **A prod bug that only shows up under real data shapes fails loudly.** Unlike a bypassed validator, `pandera_validate` always raises `SchemaErrors` on violations — there is no silent prod bypass. This is the intended trade-off: correctness over raw perf in the hot path.

## What each direction catches

| Direction | Catches | Mechanism |
|---|---|---|
| Input | missing required columns | Field absence |
| Input | wrong dtype (e.g. str where float expected) | `coerce=True` fixes int→float; str→unparseable-float still raises |
| Input | NaN in non-nullable fields | pandera's default `nullable=False` |
| Output | columns silently dropped | Field absence |
| Output | wrong index precision (us vs ns) | explicit `datetime64[ns, UTC]` index type before return, then schema dtype check |
| Output | dtype drift (float→int, etc.) | schema dtype check |

## Required regression tests

Every function decorated with `@pandera_transform` needs tests proving broken data actually fails — otherwise a future refactor can quietly drop the decorator or weaken a schema with nothing to catch it. Add to `tests/unit/.../test_<module>.py`, minimum four:

```python
import pandera
import pytest


def test_missing_required_input_column_raises_schema_errors() -> None:
    bad = df({"low": [1.0, 2.0]}, index=...)
    with pytest.raises(pandera.errors.SchemaErrors):
        transform(bad)


def test_wrong_dtype_input_raises_schema_errors() -> None:
    bad = df({"high": ["a", "b"], "low": [1.0, 2.0]}, index=...)   # uncoercible values
    with pytest.raises(pandera.errors.SchemaErrors):
        transform(bad)


def test_result_missing_column_raises_schema_errors() -> None:
    result = transform(_make_ohlc(...))
    broken = result.drop(columns=["some_required_col"])
    with pytest.raises(pandera.errors.SchemaErrors):
        MyResult.validate(broken, lazy=True)


def test_result_wrong_index_dtype_raises_schema_errors() -> None:
    # Bypasses transform()'s output to prove the schema itself
    # would catch wrong precision if the cast were ever skipped/broken.
    result = transform(_make_ohlc(...))
    broken = result.copy()
    broken.index = broken.index.astype("datetime64[us, UTC]")   # simulate pandas 3.x default
    with pytest.raises(pandera.errors.SchemaErrors):
        MyResult.validate(broken, lazy=True)
```

Rules:
- `pytest.raises(pandera.errors.SchemaErrors)` only — never assert on error message text, it changes between pandera versions.
- Dtype tests need genuinely uncoercible values (`["a", "b", "c"]`, not `["1", "2", "3"]` — `coerce=True` will happily parse numeric strings).

## When *not* to add pandera

- Function returns a plain scalar, ndarray, or dict — pandera is for DataFrame shapes only.
- Throwaway analysis script or notebook cell — schemas are for production pipeline code.
- Function already covered by `cast_and_validate()` and produces no new shape (see exception above) — don't stack `@pandera_transform` on top of it.

## Import style

```python
import pandera
from pandera import typing as pt
from helper.pandera import pandera_validate
from helper.importer import df
```

- `pt` → schema definitions (`pt.Series[...]`, `pt.Index[...]`) **and** function signatures (`pt.DataFrame[MySchema]`).
- DataFrame construction and `isinstance` checks use plain `pd.DataFrame` or the `df` alias from `helper.importer` (e.g. `df({"col": [1.0, 2.0]})`, `isinstance(value, pd.DataFrame)`).

## Policy

Pre-commit: If any function input/output annotation contains `pd.DataFrame` or `pt.DataFrame[...]`, require `@pandera_transform`. Do not enforce schema details or special cases.

@pandera_transform: `pa.check_types(lazy=True)` is the runtime authority and warns when legacy `pd.DataFrame` is used.

`allow_pandas_dataframe=True`: parameter to `@pandera_transform`. Suppresses the legacy-`pd.DataFrame` warning for functions where no Pandera schema applies; schema validation for any `pt.DataFrame[Schema]` in the same signature still runs normally. Pre-commit still requires `@pandera_transform` even when this flag is set.

No separate exemption decorator: the switch on `@pandera_transform` is the sole opt-out.

Pre-commit does not interpret `allow_pandas_dataframe`: it only enforces that `@pandera_transform` is present. Runtime `@pandera_transform` decides whether the raw pandas annotation is allowed.

### Pre-commit checker

The checker lives in `scripts/check_pandera_decorator.py`. It is added as a local
pre-commit hook (`.pre-commit-config.yaml` → `check-pandera-decorator`) and is
dependency-light: it parses the AST without importing application modules.

CI usage:
  pre-commit run --all-files
