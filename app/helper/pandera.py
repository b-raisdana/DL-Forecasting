"""Runtime Pandera validation decorator with n_return/NaN enforcement and
static NaN-fill detection. See app/helper/README.md for full docs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import FunctionType
from typing import TypeVar, cast, get_args, get_origin, get_type_hints, overload

import optree
import pandas as pd
import pandera
import pandera.pandas as pa
from config import app_config
from helper.logging.do_log.log_it import log_w

Pandera_DFM_Type = TypeVar("Pandera_DFM_Type", bound=pandera.DataFrameModel)


def _contains_legacy_pandas_dataframe(annotation: object) -> bool:
    """Return True when annotation contains bare/legacy pandas DataFrame."""
    if annotation is pd.DataFrame:
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    return any(_contains_legacy_pandas_dataframe(arg) for arg in get_args(annotation) if arg is not type(None))


# ---------------------------------------------------------------------------
# Static NaN-fill detection
# ---------------------------------------------------------------------------

DEFAULT_NAN_FILL_NAMES = frozenset(
    {
        "fillna",
        "bfill",
        "backfill",
        "ffill",
        "pad",
        "interpolate",
        "nan_to_num",
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
    }
)


class NanFillDetectedError(ValueError):
    """NaN-fill call found at decoration time with ``forbid_nan_fill=True``."""


@dataclass(frozen=True)
class NanFillHit:
    call_name: str
    filename: str
    lineno: int
    chain: tuple[str, ...]  # outer-most caller first, empty for a direct hit

    def __str__(self) -> str:
        location = f".{self.call_name}(...) at {self.filename}:{self.lineno}"
        return f"{location} (call chain: {' -> '.join((*self.chain, self.call_name))})" if self.chain else location


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _scan_nan_fills(
    fn: FunctionType,
    *,
    nan_fill_names: frozenset[str],
    deep: bool,
    max_depth: int,
    _chain: tuple[str, ...] = (),
    _visited: set[int] | None = None,
) -> list[NanFillHit]:
    visited = _visited if _visited is not None else set()
    if id(fn) in visited or max_depth < 0:
        return []
    visited.add(id(fn))

    try:
        source = textwrap.dedent(inspect.getsource(fn))
        _, start_line = inspect.getsourcelines(fn)
        filename = inspect.getsourcefile(fn) or "<unknown>"
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []

    hits: list[NanFillHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        hits.extend(
            _process_nan_fill_call(
                node, fn, source, start_line, filename, nan_fill_names, deep, max_depth, _chain, visited
            )
        )
    return hits


def _process_nan_fill_call(
    node: ast.Call,
    fn: FunctionType,
    source: str,
    start_line: int,
    filename: str,
    nan_fill_names: frozenset[str],
    deep: bool,
    max_depth: int,
    _chain: tuple[str, ...],
    visited: set[int],
) -> list[NanFillHit]:
    name = _call_name(node)
    if name is None:
        return []
    if name in nan_fill_names:
        return [NanFillHit(name, filename, start_line + node.lineno - 1, _chain)]
    if deep and isinstance(node.func, ast.Name):
        return _deep_scan_call(node.func, fn, nan_fill_names, deep, max_depth - 1, _chain, visited)
    return []


def _deep_scan_call(
    func_node: ast.Name,
    fn: FunctionType,
    nan_fill_names: frozenset[str],
    deep: bool,
    max_depth: int,
    _chain: tuple[str, ...],
    visited: set[int],
) -> list[NanFillHit]:
    target: object = fn.__globals__.get(func_node.id)
    if not isinstance(target, FunctionType):
        return []
    return _scan_nan_fills(
        target,
        nan_fill_names=nan_fill_names,
        deep=deep,
        max_depth=max_depth,
        _chain=(*_chain, fn.__qualname__),
        _visited=visited,
    )


def _report_nan_fills(func: FunctionType, hits: list[NanFillHit], *, forbid: bool) -> None:
    for hit in hits:
        message = (
            f"{func.__qualname__}: potential NaN-fill call {hit}. "
            f"Policy: prevent NaNs via sufficient warmup/lookback instead of filling them."
        )
        if forbid:
            raise NanFillDetectedError(message)
        log_w(message, stack_offset=3)


# ---------------------------------------------------------------------------
# n_return / NaN enforcement
# ---------------------------------------------------------------------------


class InsufficientDataError(ValueError):
    """Fewer valid rows than ``n_return`` after NaN-row drop."""


def _enforce_dataframe(df: pd.DataFrame, *, n_return: int, trim_to_n_return: bool, qualname: str) -> pd.DataFrame:
    cleaned = df.dropna()  # drop any row containing a NaN, in any column

    if len(cleaned) < n_return:
        offending = df.columns[df.isna().any()].tolist()
        raise InsufficientDataError(
            f"{qualname}: insufficient valid data after NaN-row drop. "
            f"Required n_return={n_return}, got {len(cleaned)} valid rows "
            f"(input had {len(df)} rows). Columns with NaNs: {offending or 'none'}."
        )

    return cleaned.tail(n_return) if trim_to_n_return else cleaned


def _enforce_output[T](result: T, *, n_return: int, trim_to_n_return: bool, qualname: str) -> T:
    """Walk nested DataFrame / tuple / list / dict via optree, enforcing
    NaN-drop + n_return on every DataFrame leaf."""

    def _leaf(x: object) -> object:
        if isinstance(x, pd.DataFrame):
            return _enforce_dataframe(x, n_return=n_return, trim_to_n_return=trim_to_n_return, qualname=qualname)
        return x

    return cast(T, optree.tree_map(_leaf, result, is_leaf=lambda x: isinstance(x, pd.DataFrame)))  # type: ignore[arg-type]  # optree's stub wants PyTree[Never]; _T is caller-supplied and structurally fine here


@overload
def pandera_validate[**P, R](
    func: Callable[P, R],
    *,
    allow_pandas_dataframe: bool = ...,
    trim_to_n_return: bool = ...,
    warn_on_nan_fill: bool = ...,
    forbid_nan_fill: bool = ...,
    deep_nan_fill_scan: bool = ...,
    nan_fill_scan_depth: int = ...,
    extra_nan_fill_names: frozenset[str] = ...,
) -> Callable[P, R]: ...


@overload
def pandera_validate[**P, R](
    func: None = None,
    *,
    allow_pandas_dataframe: bool = ...,
    trim_to_n_return: bool = ...,
    warn_on_nan_fill: bool = ...,
    forbid_nan_fill: bool = ...,
    deep_nan_fill_scan: bool = ...,
    nan_fill_scan_depth: int = ...,
    extra_nan_fill_names: frozenset[str] = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def pandera_validate[**P, R](
    func: Callable[P, R] | None = None,
    *,
    allow_pandas_dataframe: bool = False,
    trim_to_n_return: bool = True,
    warn_on_nan_fill: bool = True,
    forbid_nan_fill: bool = False,
    deep_nan_fill_scan: bool = False,
    nan_fill_scan_depth: int = 2,
    extra_nan_fill_names: frozenset[str] = frozenset(),
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Runtime Pandera validation decorator. See app/helper/README.md for full docs
    (decorator options, call-time kwargs, production bypass, exceptions).
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # `func` is typed as Callable[_P, _R] for correct call-site checking,
        # but every decorated target is a plain `def`-defined function, so
        # it is also a types.FunctionType exposing the dunder attributes the
        # checks below need.
        func_obj = cast(FunctionType, func)

        if not allow_pandas_dataframe:
            hints = get_type_hints(func_obj, include_extras=True)
            for name in func_obj.__annotations__:
                annotation: object | None = hints.get(name)
                if annotation is not None and _contains_legacy_pandas_dataframe(annotation):
                    where = "return annotation" if name == "return" else f"parameter {name}"
                    log_w(
                        f"{func_obj.__qualname__}: {where} uses pandas.DataFrame instead of pandera.typing.DataFrame",
                        stack_offset=2,
                    )

        if warn_on_nan_fill or forbid_nan_fill:
            hits = _scan_nan_fills(
                func_obj,
                nan_fill_names=DEFAULT_NAN_FILL_NAMES | extra_nan_fill_names,
                deep=deep_nan_fill_scan,
                max_depth=nan_fill_scan_depth,
            )
            _report_nan_fills(func_obj, hits, forbid=forbid_nan_fill)

        if app_config.environment == "production":
            return func

        inner = pa.check_types(lazy=True)(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            n_return_raw: object = kwargs.pop("n_return", None)
            allow_return_nan = bool(kwargs.pop("allow_return_nan", False))
            discard_n_return = bool(kwargs.pop("discard_n_return", False))

            n_return_in_sig = "n_return" in inspect.signature(func_obj).parameters

            if n_return_in_sig:
                enforcement_active = not allow_return_nan
                n_return_valid = isinstance(n_return_raw, int) and not isinstance(n_return_raw, bool)

                if enforcement_active:
                    if n_return_raw is None:
                        raise TypeError(
                            f"{func_obj.__qualname__}: 'n_return' is required "
                            f"(pass allow_return_nan=True to skip enforcement)."
                        )
                    if not n_return_valid:
                        raise TypeError(
                            f"{func_obj.__qualname__}: 'n_return' must be an int, got {type(n_return_raw).__name__}."
                        )
            else:
                enforcement_active = False
                n_return_valid = False

            call_kwargs: dict[str, object] = dict(kwargs)
            if n_return_valid and not discard_n_return:
                call_kwargs["n_return"] = n_return_raw

            # call_kwargs is rewritten dynamically (n_return/allow_return_nan/
            # discard_n_return popped and conditionally re-added), so it no
            # longer matches _P.kwargs exactly from the type checker's view —
            # this is the one unavoidable seam between ParamSpec preservation
            # and runtime kwarg rewriting.
            result: R = inner(*args, **call_kwargs)  # type: ignore[arg-type]

            if not enforcement_active:
                return result

            assert n_return_valid  # guaranteed by the raise above when enforcement_active
            return _enforce_output(
                result,
                n_return=cast(int, n_return_raw),
                trim_to_n_return=trim_to_n_return,
                qualname=func_obj.__qualname__,
            )

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator


# ---------------------------------------------------------------------------
# Self-test / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    class _FakeConfig:
        environment = "development"

    app_config = _FakeConfig()  # type: ignore[assignment]

    def log_w(message: str, stack_limit: int = 10, stack_offset: int = 2) -> None:
        pass

    @pandera_validate(allow_pandas_dataframe=True)
    def make_series(length: int, n_return: int | None = None) -> pd.DataFrame:
        # simulate a feature with 10-bar warmup: first 10 rows are NaN
        vals = np.arange(length, dtype=float)
        vals[:10] = np.nan
        return pd.DataFrame({"atr": vals})

    # 1) sufficient data -> trimmed to exactly n_return valid rows
    out = make_series(50, n_return=30)
    assert len(out) == 30

    # 2) insufficient data -> raises
    try:
        make_series(15, n_return=30)
        raise AssertionError("expected InsufficientDataError")
    except InsufficientDataError:
        pass

    # 3) allow_return_nan bypass -> raw output, NaNs included
    # `allow_return_nan`/`discard_n_return` are wrapper-injected kwargs, not
    # part of the wrapped function's own signature, so a type checker
    # correctly can't see them in `_P` — see the decorator's docstring note
    # on the ParamSpec/kwarg-rewriting seam. Silenced here, not upstream.
    out3 = make_series(15, n_return=30, allow_return_nan=True)  # type: ignore[call-arg]
    assert len(out3) == 15

    # 4) missing n_return without allow_return_nan -> raises
    try:
        make_series(50)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass

    # 5) nested dict/tuple of DataFrames -> each leaf enforced independently
    @pandera_validate(allow_pandas_dataframe=True)
    def make_nested(length: int, n_return: int | None = None) -> dict[str, object]:
        return {
            "main": make_series(length, allow_return_nan=True),  # type: ignore[call-arg]
            "aux": (make_series(length, allow_return_nan=True),),  # type: ignore[call-arg]
        }

    nested: dict[str, object] = make_nested(50, n_return=30)
    main: pd.DataFrame = nested["main"]  # type: ignore[assignment]
    aux: tuple[pd.DataFrame, ...] = nested["aux"]  # type: ignore[assignment]
    assert len(main) == 30 and len(aux[0]) == 30

    # 6) NaN-fill detected -> warns by default (doesn't block the call)
    @pandera_validate(allow_pandas_dataframe=True)
    def bad_fill(df: pd.DataFrame, n_return: int | None = None) -> pd.DataFrame:
        return df.fillna(0)

    bad_fill(pd.DataFrame({"x": [1.0, None]}), n_return=1, allow_return_nan=True)  # type: ignore[call-arg]

    # 7) forbid_nan_fill=True -> raises at decoration time, before any call
    try:

        @pandera_validate(allow_pandas_dataframe=True, forbid_nan_fill=True)
        def bad_fill_forbidden(df: pd.DataFrame, n_return: int | None = None) -> pd.DataFrame:
            return df.bfill()

        raise AssertionError("expected NanFillDetectedError")
    except NanFillDetectedError:
        pass

    # 8) deep scan follows a plain-name helper call in the same module
    def _impute_helper(df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate()

    try:

        @pandera_validate(allow_pandas_dataframe=True, forbid_nan_fill=True, deep_nan_fill_scan=True)
        def uses_helper(df: pd.DataFrame, n_return: int | None = None) -> pd.DataFrame:
            return _impute_helper(df)

        raise AssertionError("expected NanFillDetectedError via deep scan")
    except NanFillDetectedError:
        pass
