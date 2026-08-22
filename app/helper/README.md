# `helper/` modules

## `pandera.py`

Runtime Pandera validation decorator with n_return/NaN enforcement and static NaN-fill detection. Drop-in replacement for `@pa.check_types(lazy=True)`.

Import: `from helper.pandera import pandera_validate`

### What it adds over `@pa.check_types(lazy=True)`

1. Legacy-annotation warning: logs when a bare `pd.DataFrame` annotation is used instead of `pt.DataFrame[Schema]`.
2. Call-time `n_return` enforcement: caller declares how many valid (non-NaN) rows it expects; wrapper drops NaN rows, checks length against `n_return`, raises `InsufficientDataError` on short/NaN output instead of silently returning it.
3. Static NaN-fill detection: at decoration time, AST-scans the wrapped function's source for NaN-fill/impute calls (`fillna`, `bfill`, `ffill`, `interpolate`, `SimpleImputer`, ...). Per project policy NaNs should be *prevented* (sufficient warmup/lookback) rather than filled — any hit is logged, or raised as `NanFillDetectedError` with `forbid_nan_fill=True`.

### Decorator options

- `allow_pandas_dataframe=False`: skip the legacy bare-`pd.DataFrame` warning.
- `trim_to_n_return=True`: once the sufficiency check passes, trim output to exactly the last `n_return` valid rows (deterministic shape, not "at least n_return").
- `warn_on_nan_fill=True`: log any NaN-fill hit at decoration time (runs in every env, including prod — costs nothing per-call).
- `forbid_nan_fill=False`: raise `NanFillDetectedError` at decoration time instead of warning — fails the import outright.
- `deep_nan_fill_scan=False`: also follow plain-name calls (not attribute/method calls, which can't be statically resolved) to helpers in the same module, up to `nan_fill_scan_depth` levels, best-effort.
- `nan_fill_scan_depth=2`: max follow-depth for deep scan.
- `extra_nan_fill_names=frozenset()`: extra call names to treat as NaN-fills, on top of `DEFAULT_NAN_FILL_NAMES`.

Production bypass: when `app_config.environment == "production"`, the *runtime* schema check and n_return/NaN enforcement are skipped entirely (same all-or-nothing bypass as plain `check_types`). The static NaN-fill scan and legacy-annotation warning are decoration-time-only and always run regardless of environment.

### Call-time reserved kwargs (popped before reaching the wrapped function)

- `n_return: int` (required unless `allow_return_nan=True`): number of valid rows the caller expects back.
- `allow_return_nan: bool = False`: skip NaN-drop + length enforcement for this call; raw output (including NaNs) returned as-is. For exploratory calls or intentionally-raw output.
- `discard_n_return: bool = False`: if True, `n_return` is NOT forwarded to the wrapped function's own call (for functions that don't accept it). If False, it is forwarded so the function can use it internally (e.g. to slice `df.tail(n_return + warmup)` before returning).

Constraint: `func` must be an ordinary Python function (not a bound method, lambda-behind-partial, or arbitrary callable) — the decorator relies on `__qualname__`, `__globals__`, `__annotations__` for schema checking and NaN-fill scanning, which only genuine `types.FunctionType` objects expose.

### Exceptions

- `InsufficientDataError`: fewer valid rows than `n_return` after NaN-row drop.
- `NanFillDetectedError`: NaN-fill call found during decoration with `forbid_nan_fill=True` (also raised for hits in a locally-resolvable helper when `deep_nan_fill_scan=True`).
