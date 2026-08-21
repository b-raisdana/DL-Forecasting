import ast
from pathlib import Path

import pytest

from scripts.check_pandera_decorator import (
    _has_pandera_transform,
    _is_dataframe_annotation,
    _walk_annotation,
    check_file,
)

pytestmark = pytest.mark.unit


def _parse(code: str) -> ast.Module:
    return ast.parse(code)


def _func_with_annotations(code: str) -> ast.FunctionDef:
    module = _parse(code)
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise AssertionError("no function found")


class TestIsDataframeAnnotation:
    def test_bare_pd_dataframe(self) -> None:
        node = _func_with_annotations("def f(x: pd.DataFrame): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is True

    def test_bare_pt_dataframe(self) -> None:
        node = _func_with_annotations("def f(x: pt.DataFrame): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is True

    def test_parameterized_pt_dataframe(self) -> None:
        node = _func_with_annotations("def f(x: pt.DataFrame[Schema]): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is True

    def test_pandera_typing_dataframe(self) -> None:
        node = _func_with_annotations("def f(x: pandera.typing.DataFrame): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is True

    def test_pandas_dataframe(self) -> None:
        node = _func_with_annotations("def f(x: pandas.DataFrame): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is True

    def test_from_pandas_import_dataframe(self) -> None:
        node = _func_with_annotations("def f(x: DataFrame): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is True

    def test_non_dataframe_passes(self) -> None:
        node = _func_with_annotations("def f(x: int): pass").args.args[0].annotation
        assert _is_dataframe_annotation(node) is False

    def test_nested_inside_list(self) -> None:
        node = _func_with_annotations("def f(x: list[pt.DataFrame[Schema]]): pass").args.args[0].annotation
        found = any(_is_dataframe_annotation(n) for n in _walk_annotation(node))
        assert found is True

    def test_nested_inside_union(self) -> None:
        node = _func_with_annotations("def f(x: pt.DataFrame[Schema] | None): pass").args.args[0].annotation
        found = any(_is_dataframe_annotation(n) for n in _walk_annotation(node))
        assert found is True


class TestHasPanderaTransform:
    def test_bare_decorator(self) -> None:
        func = _func_with_annotations("@pandera_transform\ndef f(): pass")
        assert _has_pandera_transform(func.decorator_list) is True

    def test_module_decorator(self) -> None:
        func = _func_with_annotations("@helper.pandera_transform\ndef f(): pass")
        assert _has_pandera_transform(func.decorator_list) is True

    def test_decorator_with_args(self) -> None:
        func = _func_with_annotations("@pandera_transform(allow_pandas_dataframe=True)\ndef f(): pass")
        assert _has_pandera_transform(func.decorator_list) is True

    def test_other_decorator(self) -> None:
        func = _func_with_annotations("@profile_it\ndef f(): pass")
        assert _has_pandera_transform(func.decorator_list) is False

    def test_no_decorator(self) -> None:
        func = _func_with_annotations("def f(): pass")
        assert _has_pandera_transform(func.decorator_list) is False


class TestCheckFile:
    def test_missing_decorator_with_pd_dataframe_param(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(x: pd.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1
        assert "f" in violations[0]
        assert "pd.DataFrame" in violations[0] or "DataFrame annotations" in violations[0]

    def test_missing_decorator_with_pt_dataframe_return(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f() -> pt.DataFrame[Schema]: pass")
        violations = check_file(path)
        assert len(violations) == 1
        assert "f" in violations[0]
        assert "pt.DataFrame" in violations[0] or "DataFrame annotations" in violations[0]

    def test_missing_decorator_with_bare_pt_dataframe(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(x: pt.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1

    def test_decorator_present_cleans(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("@pandera_transform\ndef f(x: pd.DataFrame): pass")
        violations = check_file(path)
        assert violations == []

    def test_decorator_with_args_cleans(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("@pandera_transform(allow_pandas_dataframe=True)\ndef f(x: pd.DataFrame): pass")
        violations = check_file(path)
        assert violations == []

    def test_non_dataframe_annotation_skips(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(x: int): pass")
        violations = check_file(path)
        assert violations == []

    def test_nested_tuple_annotation_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(x: tuple[pt.DataFrame[A], pt.DataFrame[B]]): pass")
        violations = check_file(path)
        assert len(violations) == 1

    def test_mixed_aliases_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text(
            "import pandas as pd\nimport pandera.typing as pt\ndef f(x: pd.DataFrame, y: pt.DataFrame[Schema]): pass"
        )
        violations = check_file(path)
        assert len(violations) == 1

    def test_async_function_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("async def f(x: pd.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1

    def test_class_method_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("class C:\n    def method(self, x: pd.DataFrame): pass\n")
        violations = check_file(path)
        assert len(violations) == 1

    def test_bare_function_without_decorator_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(x: pd.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1
        assert "f" in violations[0]

    def test_allow_pandas_dataframe_does_not_bypass_precommit(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("@pandera_transform(allow_pandas_dataframe=True)\ndef f(x: pd.DataFrame): pass")
        violations = check_file(path)
        assert violations == []

    def test_return_type_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f() -> pd.DataFrame: pass")
        violations = check_file(path)
        assert len(violations) == 1

    def test_syntax_error_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(: pass")
        violations = check_file(path)
        assert violations == []

    def test_kwonly_arg_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(*, x: pd.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1

    def test_vararg_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(*args: pd.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1

    def test_kwarg_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text("def f(**kwargs: pd.DataFrame): pass")
        violations = check_file(path)
        assert len(violations) == 1
