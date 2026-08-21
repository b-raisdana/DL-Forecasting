import logging
from typing import Any

import pandas as pd
import pandera
import pandera.typing as pt
import pytest
from helper.pandera import pandera_transform

pytestmark = pytest.mark.unit


class _SimpleSchema(pandera.DataFrameModel):
    close: pandera.typing.pandas.Series[float]


class _OtherSchema(pandera.DataFrameModel):
    open: pandera.typing.pandas.Series[float]


@pytest.fixture(autouse=True)
def _warn_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging.getLogger("helper.pandera"), "warning", lambda *a, **k: None)


def test_default_warns_on_legacy_pd_dataframe_param() -> None:
    @pandera_transform
    def f(df: pd.DataFrame) -> Any:
        return df

    assert f is not None


def test_default_warns_on_legacy_pd_dataframe_return() -> None:
    @pandera_transform
    def f() -> pd.DataFrame:
        return pd.DataFrame({"close": [1.0]})

    assert f is not None


def test_allow_pandas_dataframe_suppresses_param_warning() -> None:
    @pandera_transform(allow_pandas_dataframe=True)
    def f(df: pd.DataFrame) -> Any:
        return df

    assert f is not None


def test_allow_pandas_dataframe_suppresses_return_warning() -> None:
    @pandera_transform(allow_pandas_dataframe=True)
    def f() -> pd.DataFrame:
        return pd.DataFrame({"close": [1.0]})

    assert f is not None


def test_validation_still_catches_bad_typed_dataframe() -> None:
    @pandera_transform
    def f(df: pt.DataFrame[_SimpleSchema]) -> pt.DataFrame[_SimpleSchema]:
        return df

    bad = pd.DataFrame({"close": ["not_a_float"]})

    with pytest.raises(pandera.errors.SchemaErrors):
        f(bad)


def test_allow_pandas_dataframe_does_not_disable_pt_validation() -> None:
    @pandera_transform(allow_pandas_dataframe=True)
    def f(df: pd.DataFrame, typed: pt.DataFrame[_SimpleSchema]) -> pt.DataFrame[_SimpleSchema]:
        return typed

    bad = pd.DataFrame({"close": ["not_a_float"]})

    with pytest.raises(pandera.errors.SchemaErrors):
        f(pd.DataFrame(), bad)


def test_bare_decorator_usage() -> None:
    @pandera_transform
    def f(df: pt.DataFrame[_SimpleSchema]) -> pt.DataFrame[_SimpleSchema]:
        return df

    good = pd.DataFrame({"close": [1.0]})
    result = f(good)
    pd.testing.assert_frame_equal(result, good)
