"""AST-based pre-commit checker for the Pandera DataFrame validation policy.

Policy:
  - If a function's input/output annotations contain ``pd.DataFrame`` or
    ``pt.DataFrame[...]``, it must carry ``@pandera_transform``.
  - Schema details and ``allow_pandas_dataframe`` are **not** inspected here
    — runtime ``@pandera_transform`` is the authority for those.

The checker is dependency-light and deterministic: it only parses the AST
and never imports application modules.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _is_dataframe_annotation(node: ast.AST) -> bool:
    """Return True when *node* represents a DataFrame type annotation.

    Matches the following shapes (with any import alias):

    * ``pd.DataFrame``
    * ``pt.DataFrame``
    * ``pandera.typing.DataFrame``
    * ``pandas.DataFrame``
    * ``DataFrame`` (when imported directly)
    """
    # Subscript: pt.DataFrame[Schema] → inspect the value (pt.DataFrame)
    if isinstance(node, ast.Subscript):
        return _is_dataframe_annotation(node.value)

    # BinOp: pt.DataFrame[Schema] | None → inspect both sides
    if isinstance(node, ast.BinOp):
        return _is_dataframe_annotation(node.left) or _is_dataframe_annotation(node.right)

    # Bare ``DataFrame`` (from pandas import DataFrame / from pandera.typing import DataFrame)
    if isinstance(node, ast.Name) and node.id == "DataFrame":
        return True

    if not isinstance(node, ast.Attribute):
        return False

    # pd.DataFrame / pt.DataFrame / pandas.DataFrame / pandera.typing.DataFrame
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.insert(0, current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.insert(0, current.id)

    if not parts or parts[-1] != "DataFrame":
        return False

    prefix = ".".join(parts[:-1])
    return prefix in {"pd", "pt", "pandera.typing", "pandas"}


def _walk_annotation(node: ast.AST) -> ast.AST:
    """Recursively yield every sub-node inside a type annotation."""
    yield node
    if isinstance(node, ast.Subscript):
        yield from _walk_annotation(node.value)
        if isinstance(node.slice, ast.Slice):
            for dim in (node.slice.lower, node.slice.upper, node.slice.step):
                if dim is not None:
                    yield from _walk_annotation(dim)
        else:
            yield from _walk_annotation(node.slice)
    elif isinstance(node, ast.BinOp):
        yield from _walk_annotation(node.left)
        yield from _walk_annotation(node.right)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            yield from _walk_annotation(elt)
    elif isinstance(node, (ast.Attribute, ast.Index)):
        yield from _walk_annotation(node.value)


def _has_pandera_transform(decorator_list: list[ast.expr]) -> bool:
    """Return True if any decorator is ``pandera_transform`` (with any alias or arguments)."""
    for decorator in decorator_list:
        # @pandera_transform or @pandera_transform(...)
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Name) and decorator.id == "pandera_transform":
            return True
        # @module.pandera_transform or @module.pandera_transform(...)
        if isinstance(decorator, ast.Attribute) and decorator.attr == "pandera_transform":
            return True
    return False


def _function_uses_dataframe_annotation(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function's annotations mention a DataFrame type."""
    # Parameters
    for arg in func.args.args:
        if arg.annotation is not None:
            for node in _walk_annotation(arg.annotation):
                if _is_dataframe_annotation(node):
                    return True
    for arg in func.args.kwonlyargs:
        if arg.annotation is not None:
            for node in _walk_annotation(arg.annotation):
                if _is_dataframe_annotation(node):
                    return True
    if func.args.vararg is not None and func.args.vararg.annotation is not None:
        for node in _walk_annotation(func.args.vararg.annotation):
            if _is_dataframe_annotation(node):
                return True
    if func.args.kwarg is not None and func.args.kwarg.annotation is not None:
        for node in _walk_annotation(func.args.kwarg.annotation):
            if _is_dataframe_annotation(node):
                return True

    # Return annotation
    if func.returns is not None:
        for node in _walk_annotation(func.returns):
            if _is_dataframe_annotation(node):
                return True

    return False


def check_file(path: Path) -> list[str]:
    """Return a list of violation strings for *path* (empty if clean)."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        if not _function_uses_dataframe_annotation(node):
            continue

        if _has_pandera_transform(node.decorator_list):
            continue

        violations.append(
            f"{path}:{node.lineno}:{node.col_offset}: "
            f"function '{node.name}' uses DataFrame annotations but lacks @pandera_transform"
        )

    return violations


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]

    if not argv:
        print("usage: check_pandera_decorator <file>...", file=sys.stderr)
        return 2

    all_violations: list[str] = []
    for filename in argv:
        path = Path(filename)
        if not path.is_file():
            continue
        all_violations.extend(check_file(path))

    if all_violations:
        for v in all_violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
