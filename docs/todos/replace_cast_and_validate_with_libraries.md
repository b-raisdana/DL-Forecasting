# TODO — Replace `cast_and_validate()` with library-based features

Current state: `app/helper/schema_casting.py` carries a hand-rolled `cast_and_validate()` (lines 26-68) plus its type-coercion helper `apply_as_type()` (lines 101-131) that together perform work already expressible in pandera, pydantic, and pandas primitives. This document tracks the incremental replacements.

## Scope

- `cast_and_validate()` has 37 call-sites across the repo (per repo-wide grep).
- The function does three jobs: (1) empty-df/zero-size guard + unique-index guard, (2) dtype coercion (datetime → `datetime64[ns, UTC]`, timedelta → `timedelta64[s]`, Series-type unwrapping), (3) pandera `validate()` + column pruning.
- `apply_as_type()` is the custom coercion logic.

## Prerequisites / blockers

- Confirm pandera version in `requirements*.txt` (the `coerce=True` flag and `DataType` API differ across pandera 1.x / 2.x lines).
- Confirm whether `Pandera_DFM_Type` subclasses already set `coerce=True` or `strict=True` at the schema level today.
- Audit the 37 call-sites for any caller that relies on the current failure mode (KeyError vs SchemaError vs return-bool) before changing behavior.

## todo

1. **[P1] Replace custom datetime coercion with pandas/pandera primitives**
   - `_coerce_datetime_to_ns_utc()` (lines 71-79) and `_coerce_index_to_ns_utc()` (lines 82-98) exist because pandera's `DatetimeTZDtype("ns", "UTC")` is strict about precision/timezone.
   - Investigate: pandera 2.x `Check` with `coerce=True` + pandas `pd.to_datetime(..., utc=True)` can likely replace both helpers entirely.
   - Factor: library delegation / dead-code elimination.

2. **[P1] Replace `apply_as_type()` with schema-level coercion**
   - `apply_as_type()` (lines 101-131) manually builds an `astype` dict for datetime, timedelta, and `pandera.typing.pandas.Series[...]` columns.
   - Investigate: set `coerce=True` on the `DataFrameModel`'s `dtypes` declarations, or use pandera's `DataType` coercion hooks, so the schema itself handles conversion before validation.
   - Factor: library delegation / duplication elimination.

3. **[P1] Replace empty-df / unique-index guards with pandera checks**
   - Lines 33-43 handle zero-size and duplicate-index cases before pandera ever runs.
   - Investigate: pandera `Check` lambdas (`Check(lambda df: len(df) > 0 or zero_size_allowed, ...)`) or a pydantic `RootModel` validator on the input could enforce these at the schema level.
   - Factor: library delegation.

4. **[P2] Evaluate pydantic `BaseModel` / `RootModel` for the public API boundary**
   - Where `cast_and_validate()` is called at I/O boundaries (disk-cache read, dataset-generator output), a pydantic model could own coercion + validation in one place.
   - Compare against current pandera `DataFrameModel` approach: pydantic gives per-field validators and better error messages but is slower on wide DataFrames.
   - Factor: library delegation / performance.

5. **[P2] Evaluate pandas `convert_dtypes()` / `infer_objects()` for the simple cases**
   - For artifacts that only need basic nullable-dtype inference (no strict schema), `df.convert_dtypes()` or `df.infer_objects()` can replace the custom Series-type unwrapping logic in `apply_as_type()`.
   - Factor: library delegation.

6. **[P2] Column pruning — schema-driven column selection**
   - Lines 66-68 filter the dataframe to schema columns minus `["timeframe", "date"]`.
   - Investigate: pandera `DataFrameModel.to_schema().columns.keys()` already exposes the canonical column list; replace the hand-rolled list comprehension with a schema call or a pandera `Check` that rejects extra columns.
   - Factor: duplication elimination.

## Non-goals (keep as-is)

- `return_bool=True` short-circuit path (lines 37-40, 47-52, 53-58, 64-65) — this is a caller-visible API contract; any replacement must preserve the boolean fast-fail behavior.
- `empty_df()` (lines 245-258) — this is a factory, not validation; the column-selection refactor in item 6 may touch it but the factory itself stays.

## Success criteria

- `app/helper/schema_casting.py` shrinks by ≥50 lines (removing `cast_and_validate2`, `apply_as_type2`, `column_dtypes`, `index_names`, and the custom coercion helpers once replaced).
- All 37 call-sites still pass their existing tests after the swap (no behavior change at the boundary).
- New pandera/pydantic/pandas features are configured at the schema/model layer, not re-implemented in free functions.
