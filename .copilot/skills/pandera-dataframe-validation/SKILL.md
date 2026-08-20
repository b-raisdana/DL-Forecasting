---
name: pandera-dataframe-validation
description: Use when adding or modifying a Pandera DataFrameModel schema, or when annotating a DataFrame→DataFrame function's input/output types. Covers this repo's conventions for schema placement, coerce policy, input/output validation placement, index precision (ns UTC), and the required negative/regression tests that prove broken data is rejected.
---

# Pandera DataFrame validation

Trigger: adding a new `pandera.DataFrameModel`, annotating a DataFrame-returning function, or adding `lazy=True` validation guards to an existing transform.

## Where schemas live

- **Value-object schemas** (the shape of a DataFrame, no behavior) go in `domain/schemas/<subpackage>/`, mirroring the domain module that owns the data. Example: `domain/schemas/price_action/CausalExtremum.py` holds `CausalExtremumOHLC` and `CausalExtremumResult`.
- **Domain transforms** (functions that compute/derive) live in `domain/` or `application/` and import their schemas from `domain/schemas/`. They never define schemas inline.
- One schema file per concern — don't pile every schema into `domain/schemas/common/`. Split into subpackages (`price_action/`, `ohlcv/`, `market_structure/`, ...) as the domain grows.

## Schema definition conventions

```python
import pandera
from pandera import typing as pt
from typing import Annotated

import pandas as pd


class MyInput(pandera.DataFrameModel):
    class Config:
        coerce = True          # int→float, datetime coercion — but NOT str→unparseable-float

    high: pt.Series[float]
    low: pt.Series[float]


class MyResult(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]  # always ns UTC
    true_peak_reach_minutes: pt.Series[float]
    extremum_sign: pt.Series[int]
```

Rules:
- `coerce = True` only on **input** schemas (the function should tolerate int inputs that are semantically float). Never coerce on output schemas — if a result is the wrong dtype, that's a bug to catch, not a type to silently convert.
- Index must be `datetime64[ns, UTC]` (`Annotated[pd.DatetimeTZDtype, "ns", "UTC"]`). pandas 3.x defaults to microsecond precision; explicit `astype("datetime64[ns, UTC]")` before returning is required.
- Column names must match the DataFrame's actual column names exactly (lowercase snake_case, matching the rest of the codebase).

## Function signature pattern

```python
def compute_something(ohlc: ptd[MyInput]) -> ptd[MyResult]:
```

Rules:
- **Always** use `ptd[<Schema>]` for DataFrame parameters and return types in application/domain code.
- `ptd` is the project alias for `pd.DataFrame`, imported from `helper.importer`. It is the **only** acceptable way to write DataFrame type annotations here.
- `pt` is `pandera.typing` — use it **only inside schema definitions** for `pt.Series[...]`, `pt.Index[...]`, etc. Never use `pt.DataFrame[...]` in function signatures.
- Never use bare `pd.DataFrame` in a type annotation.
- Generic helper/infrastructure functions that intentionally accept/return any DataFrame shape (no specific schema exists) may use bare `ptd`. This is an exception, not the default — create a schema if the function has a stable expected shape.

```python
from helper.importer import ptd

def transform(df: ptd[OHLCV]) -> ptd[OHLCV]:
    ...
```

The function body validates **both** directions with `lazy=True`:

```python
def transform(df: ptd[OHLCV]) -> ptd[OHLCV]:
    OHLCV.validate(df, lazy=True)          # guard: bad input never enters the algorithm

    # ... compute ...

    result.index = result.index.astype("datetime64[ns, UTC]")  # enforce ns precision
    OHLCV.validate(result, lazy=True)      # guard: broken post-processing is caught
    return result
```

Placement: input validation at the very top (before any computation), output validation after the DataFrame is fully assembled but before the `return`. Both use `lazy=True` so all schema errors are collected into a single `SchemaErrors` exception rather than failing on the first one.

## What to validate

| Direction | What to catch | How |
|-----------|--------------|-----|
| Input | missing required columns | schema Field absence |
| Input | wrong dtype (str where float expected) | `coerce=True` handles int→float; str→unparseable-float still raises |
| Input | NaN in non-nullable fields | pandera's default `nullable=False` |
| Output | missing columns someone dropped | schema Field absence |
| Output | wrong index precision (us vs ns) | explicit `datetime64[ns, UTC]` index type |
| Output | dtype drift (float→int, etc.) | schema dtype check |

## Required tests — schema regression

Every function with pandera guards needs **at least one test that proves broken code fails**. Without these, a future refactor can silently remove or weaken validation and nothing catches it.

Add these to the function's `tests/unit/.../test_<module>.py`:

```python
import pandera
import pytest


def test_missing_required_input_column_raises_schema_errors() -> None:
    bad = ptd({"low": [1.0, 2.0]}, index=...)
    with pytest.raises(pandera.errors.SchemaErrors):
        transform(bad)


def test_wrong_dtype_input_raises_schema_errors() -> None:
    bad = ptd({"high": ["a", "b"], "low": [1.0, 2.0]}, index=...)
    with pytest.raises(pandera.errors.SchemaErrors):
        transform(bad)


def test_result_missing_column_raises_schema_errors() -> None:
    ohlc = _make_ohlc(...)
    result = transform(ohlc)
    broken = result.drop(columns=["some_required_col"])
    with pytest.raises(pandera.errors.SchemaErrors):
        MyResult.validate(broken, lazy=True)


def test_result_wrong_index_dtype_raises_schema_errors() -> None:
    ohlc = _make_ohlc(...)
    result = transform(ohlc)
    broken = result.copy()
    broken.index = broken.index.astype("datetime64[us, UTC]")   # simulate pandas 3.x default
    with pytest.raises(pandera.errors.SchemaErrors):
        MyResult.validate(broken, lazy=True)
```

Rules:
- Use `pytest.raises(pandera.errors.SchemaErrors)` — never `assert` on error messages, they change between pandera versions.
- At minimum: one missing-column input test, one wrong-dtype input test, one missing-column output test, one wrong-index-precision output test.
- Tests use genuinely uncoercible values for dtype tests (e.g. `["a", "b", "c"]` not `["1", "2", "3"]`, because `coerce=True` will happily parse numeric strings).

## When NOT to add pandera

- A function that returns a plain scalar, ndarray, or dict — pandera is for tabular (DataFrame) shapes only.
- A throwaway analysis script or notebook cell — schemas are for production pipeline code.
- A function already wrapped by `cast_and_validate()` from `helper/schema_casting.py` — that helper already validates against a schema after type-coercion; adding a second inline `validate()` call is redundant. (Exception: the function's own *output* should still be validated if it's a new intermediate shape that no existing schema covers.)

## Import style

```python
import pandera
from pandera import typing as pt
from helper.importer import ptd
```

- `pt` = `pandera.typing`. Use it **only inside schema definitions** for `pt.Series[...]`, `pt.Index[...]`, etc.
- `ptd` = `pd.DataFrame` alias. Use it for:
  - DataFrame type annotations: `def foo(df: ptd[MySchema]) -> ptd[MySchema]:`
  - DataFrame construction: `df = ptd({"col": [1.0, 2.0]})`
  - `isinstance` checks: `isinstance(value, ptd)`

Do **not** use the deprecated top-level re-exports (`pandera.DataFrameModel`, `pandera.typing.Series`) in new code — they emit a `FutureWarning` on pandera 0.32+ and will be removed. Use `pandera.DataFrameModel` and `pandera.typing as pt` (from `pandera.pandas` in a future version; for now the top-level `pandera` module still re-exports them but the warning says to switch to `pandera.pandas`).
