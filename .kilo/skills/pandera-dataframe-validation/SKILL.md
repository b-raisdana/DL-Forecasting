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
from helper.pandera import pandera_transform
from pandera import typing as pt

@pandera_transform
def compute_something(ohlc: pt.DataFrame[MyInput]) -> pt.DataFrame[MyResult]:
    ...
```

Rules:
- **Always** use `pt.DataFrame[<Schema>]` for DataFrame parameters and return types in application/domain code. `pt` (`pandera.typing`) is used both for schema fields (`pt.Series[...]`, `pt.Index[...]`) and for these signatures — it is the *only* acceptable DataFrame annotation. The old `ptd` alias (`helper.importer.ptd`) has been removed from the codebase; don't reintroduce it.
- Never use bare `pd.DataFrame` in a type annotation, **except**: generic helper/infrastructure functions with no stable expected shape (e.g. `cast_and_validate`, `index_by_date`) may use bare `pd.DataFrame` and skip the decorator below — this is the exception, not the default. Create a schema if the function's shape is actually stable.
- Every function with a `pt.DataFrame[Schema]`-annotated parameter or return **must** carry `@pandera_transform` (from `app/helper/pandera.py`) — it runs `pandera.check_types(lazy=True)` under the hood, validating every annotated input on call and the return value on return. An annotation without the decorator is silently unenforced.
- `@pandera_transform` also warns (at import time, not a hard failure) whenever a bare `pd.DataFrame` shows up in a decorated function's signature — treat that as a signal to switch to `pt.DataFrame[Schema]`. Pass `allow_pandas_dataframe=True` to silence the warning for a function where no schema legitimately applies.
- The index-precision cast is **not** automatic — still call `.astype("datetime64[ns, UTC]")` explicitly before returning if the output schema requires ns-UTC precision.
- Rolling `@pandera_transform` onto an existing function that doesn't have it yet is a separate, incremental migration (tracked in `docs/todo_pandera_runtime_validation.md`) — don't add it as a side effect of unrelated edits unless asked.

```python
from helper.pandera import pandera_transform
from pandera import typing as pt

@pandera_transform
def transform(df: pt.DataFrame[OHLCV]) -> pt.DataFrame[OHLCV]:
    # ... compute ...
    result.index = result.index.astype("datetime64[ns, UTC]")  # enforce ns precision
    return result
```

`@pandera_transform`'s `check_types(lazy=True)` replaces manual `Schema.validate(df, lazy=True)` calls in the body — don't hand-roll validation for a function that already carries the decorator.

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

Every function decorated with `@pandera_transform` needs **at least one test that proves broken code fails**. Without these, a future refactor can silently drop the decorator or weaken a schema and nothing catches it.

Add these to the function's `tests/unit/.../test_<module>.py`:

```python
import pandas as pd
import pandera
import pytest


def test_missing_required_input_column_raises_schema_errors() -> None:
    bad = pd.DataFrame({"low": [1.0, 2.0]}, index=...)
    with pytest.raises(pandera.errors.SchemaErrors):
        transform(bad)


def test_wrong_dtype_input_raises_schema_errors() -> None:
    bad = pd.DataFrame({"high": ["a", "b"], "low": [1.0, 2.0]}, index=...)
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
- Anything under `app/archive_not_used_trash/` — dead code kept for reference only, still on the old pre-`pandera_transform` style (manual `pt.DataFrame[Model]` annotations, no decorator). Don't "fix" its annotations to match current convention; it's out of scope for pandera work unless a file is explicitly being revived out of the archive.

## Import style

```python
import pandera
from pandera import typing as pt
from helper.pandera import pandera_transform
```

- `pt` = `pandera.typing`. Use it for schema fields (`pt.Series[...]`, `pt.Index[...]`) **and** for function signatures (`pt.DataFrame[MySchema]`).
- `pandera_transform` = the runtime-validation decorator (`app/helper/pandera.py`); apply it to every function with a `pt.DataFrame[Schema]`-annotated parameter or return.
- DataFrame construction / `isinstance` checks use plain `pd.DataFrame` — e.g. `pd.DataFrame({"col": [1.0, 2.0]})`, `isinstance(value, pd.DataFrame)`. There is no project alias for this anymore; `helper.importer.ptd` was removed.

Do **not** use the deprecated top-level re-exports (`pandera.DataFrameModel`, `pandera.typing.Series`) in new code — they emit a `FutureWarning` on pandera 0.32+ and will be removed. Use `pandera.DataFrameModel` and `pandera.typing as pt` (from `pandera.pandas` in a future version; for now the top-level `pandera` module still re-exports them but the warning says to switch to `pandera.pandas`).
