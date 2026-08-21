"""Tests for helper.pandera - the runtime pandera validation decorator.

Critical regression guard: @pandera_validate must remain a drop-in for
``@pa.check_types(lazy=True)`` - every decorated function has to keep working
when called WITHOUT the new ``n_return`` kwarg (the entire existing codebase
calls it that way). The ``n_return``/NaN enforcement is opt-in.
"""

import pandas as pd
import pytest
from helper.pandera import (
    DEFAULT_NAN_FILL_NAMES,
    InsufficientDataError,
    NanFillDetectedError,
    _scan_nan_fills,
    pandera_validate,
)

pytestmark = pytest.mark.unit


@pandera_validate(allow_pandas_dataframe=True)
def _plain_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"a": range(n)})


@pandera_validate(allow_pandas_dataframe=True)
def _frame_with_warmup(length: int, n_return: int | None = None) -> pd.DataFrame:
    # first 10 rows NaN (indicator warmup), rest valid
    vals = [float("nan")] * 10 + [float(i) for i in range(10, length)]
    return pd.DataFrame({"atr": vals})


def _interpolate_helper(df: pd.DataFrame) -> pd.DataFrame:
    """Module-level helper so the deep static scan can resolve it via __globals__."""  # noqa: E501
    return df.interpolate()


def _deep_caller(df: pd.DataFrame) -> pd.DataFrame:
    return _interpolate_helper(df)


def test_decorated_callable_without_n_return() -> None:
    """The decorator must not require n_return - existing callers omit it."""
    out = _plain_frame(5)
    assert len(out) == 5


def test_n_return_enforcement_is_opt_in() -> None:
    """Passing n_return drops NaN rows and trims to exactly n_return valid rows."""
    out = _frame_with_warmup(50, n_return=30)
    assert len(out) == 30
    assert not out["atr"].isna().any()


def test_n_return_insufficient_raises() -> None:
    with pytest.raises(InsufficientDataError):
        _frame_with_warmup(15, n_return=30)


def test_allow_return_nan_bypasses_enforcement() -> None:
    out = _frame_with_warmup(15, n_return=30, allow_return_nan=True)
    assert len(out) == 15  # raw output, NaNs included


def test_n_return_must_be_int() -> None:
    with pytest.raises(TypeError):
        _frame_with_warmup(50, n_return="30")  # type: ignore[arg-type]


def test_static_nan_fill_detection_direct() -> None:
    def f(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0)

    hits = _scan_nan_fills(f, nan_fill_names=DEFAULT_NAN_FILL_NAMES, deep=False, max_depth=2)
    assert [h.call_name for h in hits] == ["fillna"]


def test_forbid_nan_fill_raises_at_decoration() -> None:
    with pytest.raises(NanFillDetectedError):

        @pandera_validate(allow_pandas_dataframe=True, forbid_nan_fill=True)
        def f(df: pd.DataFrame, n_return: int | None = None) -> pd.DataFrame:
            return df.bfill()


def test_deep_scan_follows_helper() -> None:
    hits = _scan_nan_fills(_deep_caller, nan_fill_names=DEFAULT_NAN_FILL_NAMES, deep=True, max_depth=2)
    assert any(h.call_name == "interpolate" for h in hits)


def test_deep_scan_respects_depth_budget() -> None:
    # depth=0 scans caller's body but following the helper needs depth-1 (blocked)
    hits = _scan_nan_fills(_deep_caller, nan_fill_names=DEFAULT_NAN_FILL_NAMES, deep=True, max_depth=0)
    assert hits == []
