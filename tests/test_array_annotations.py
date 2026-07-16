"""Repository-wide array annotation contracts."""

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "body_models"
JAXTYPING_WRAPPERS = {"Bool", "Complex", "Float", "Inexact", "Int", "Integer", "Num", "Shaped", "UInt"}
ARRAY_TYPES = {"Array", "Tensor", "jax.Array", "np.ndarray", "torch.Tensor"}


def test_array_annotations_use_jaxtyping() -> None:
    failures = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for annotation in _annotations(tree):
            bare = _bare_array_types(annotation)
            if bare:
                location = f"{path.relative_to(SOURCE_ROOT)}:{annotation.lineno}"
                failures.append(f"{location}: {', '.join(sorted(bare))}")
        for field in _typed_dict_fields(tree):
            if any(_qualified_name(node) == "Any" for node in ast.walk(field.annotation)):
                location = f"{path.relative_to(SOURCE_ROOT)}:{field.annotation.lineno}"
                failures.append(f"{location}: TypedDict field {field.target.id!r} uses Any")

    assert not failures, "Bare array annotations must be wrapped with jaxtyping:\n" + "\n".join(failures)


def _annotations(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            yield node.annotation
        elif isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            yield node.returns


def _typed_dict_fields(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_qualified_name(base) == "TypedDict" for base in node.bases):
            continue
        for field in node.body:
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                yield field


def _bare_array_types(annotation: ast.AST, *, wrapped: bool = False) -> set[str]:
    name = _qualified_name(annotation)
    if not wrapped and name in ARRAY_TYPES:
        return {name}

    if isinstance(annotation, ast.Subscript) and _qualified_name(annotation.value) in JAXTYPING_WRAPPERS:
        return _bare_array_types(annotation.slice, wrapped=True)

    bare = set()
    for child in ast.iter_child_nodes(annotation):
        bare.update(_bare_array_types(child, wrapped=wrapped))
    return bare


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None
