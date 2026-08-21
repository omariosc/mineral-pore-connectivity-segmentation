"""Regression checks for the content-addressed neural execution source map."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from src.training.screen_selection import EXECUTION_SOURCE_FILES


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("config", "scripts", "src")
FROZEN_PYTHON_ENTRYPOINTS = {
    "scripts/train_patches.py",
    "scripts/evaluate_confirmatory_checkpoint.py",
    "scripts/report_validation_screen_tiles.py",
    "scripts/build_smoke_preflight_manifest.py",
    "scripts/build_selected_method_lock.py",
    "scripts/build_neural_freeze_manifest.py",
    "scripts/record_protocol_cell_failure.py",
}
ACTIVE_LAZY_IMPORT_SITES = {
    # These imports deliberately occur after CLI/artifact validation or inside
    # the authenticated scientific factory. Inspect their function bodies so a
    # future repository-local import cannot silently escape the source map.
    "scripts/train_patches.py": "main",
    "src/training/neural_freeze.py": "_load_selected_retraining_cell",
    "scripts/evaluate_confirmatory_checkpoint.py": "create_model_from_state",
    "src/models/focal_loss.py": "create_advanced_loss_function",
}
ACTIVE_CONDITIONAL_IMPORT_BRANCHES = {
    # Inspect only the frozen R3 branch. The containing method also retains
    # unsupported historical loss factories that are not execution sources.
    (
        "src/training/patch_trainer.py",
        "PatchTrainer",
        "create_model_and_optimizer",
        "loss_type",
        "focal_dice",
    ),
}
EXPECTED_NON_PYTHON_SOURCES = {
    "config/pipeline_config.yaml",
    "config/confirmatory_splits.json",
    "scripts/aire_confirmatory.slurm",
    "scripts/aire_validation_screen.slurm",
    "scripts/aire_validation_smoke.slurm",
    "scripts/aire_selected_retrain.slurm",
    "scripts/aire_locked_evaluation.slurm",
    "scripts/aire_validation_report.slurm",
}


def _module_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for source_root in PYTHON_ROOTS:
        for path in (ROOT / source_root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).with_suffix("")
            parts = list(relative.parts)
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            index[module] = path
    return index


def _runtime_imports(nodes: Iterable[ast.AST]) -> Iterable[ast.AST]:
    """Yield imports executed in this scope while pruning deferred functions."""

    def visit(node: ast.AST) -> Iterable[ast.AST]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        # Class bodies, including try handlers and match cases nested within
        # them, execute when the surrounding scope executes.
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    for node in nodes:
        yield from visit(node)


def _package_initializers(module: str, index: dict[str, Path]) -> set[Path]:
    parts = module.split(".")
    initializers = set()
    for end in range(1, len(parts)):
        path = index.get(".".join(parts[:end]))
        if path is not None and path.name == "__init__.py":
            initializers.add(path)
    return initializers


def _local_import_paths(
    path: Path,
    import_nodes: Iterable[ast.AST],
    index: dict[str, Path],
) -> set[Path]:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    package = parts[:-1] if parts[-1] != "__init__" else parts[:-1]
    imported_modules: set[str] = set()
    for node in import_nodes:
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
            continue
        if node.level:
            base = package[: len(package) - node.level + 1]
            if node.module:
                base += node.module.split(".")
            base_module = ".".join(base)
        else:
            base_module = node.module or ""
        if base_module:
            imported_modules.add(base_module)
        for alias in node.names:
            candidate = f"{base_module}.{alias.name}" if base_module else alias.name
            if candidate in index:
                imported_modules.add(candidate)

    paths: set[Path] = set()
    for module in imported_modules:
        imported = index.get(module)
        if imported is None:
            continue
        paths.add(imported)
        paths.update(_package_initializers(module, index))
    return paths


def _module_scope_local_imports(path: Path, index: dict[str, Path]) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _local_import_paths(path, _runtime_imports(tree.body), index)


def _named_callable_local_imports(
    path: Path,
    callable_name: str,
    index: dict[str, Path],
) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == callable_name
    ]
    assert len(matches) == 1, f"Expected one top-level callable {callable_name} in {path}"
    return _local_import_paths(path, _runtime_imports(matches[0].body), index)


def _selected_method_branch_local_imports(
    path: Path,
    class_name: str,
    method_name: str,
    attribute_name: str,
    selector_literal: str,
    index: dict[str, Path],
) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, f"Expected one class {class_name} in {path}"
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    assert len(methods) == 1, f"Expected one method {class_name}.{method_name}"

    def is_selected_equality(node: ast.If) -> bool:
        comparison = node.test
        if (
            not isinstance(comparison, ast.Compare)
            or len(comparison.ops) != 1
            or not isinstance(comparison.ops[0], ast.Eq)
            or len(comparison.comparators) != 1
        ):
            return False
        pairs = (
            (comparison.left, comparison.comparators[0]),
            (comparison.comparators[0], comparison.left),
        )
        return any(
            isinstance(attribute, ast.Attribute)
            and isinstance(attribute.value, ast.Name)
            and attribute.value.id == "self"
            and attribute.attr == attribute_name
            and isinstance(literal, ast.Constant)
            and literal.value == selector_literal
            for attribute, literal in pairs
        )

    branches = [
        node
        for node in ast.walk(methods[0])
        if isinstance(node, ast.If) and is_selected_equality(node)
    ]
    assert len(branches) == 1, (
        f"Expected one {class_name}.{method_name} branch for "
        f"self.{attribute_name} == {selector_literal!r}"
    )
    return _local_import_paths(path, _runtime_imports(branches[0].body), index)


def _reachable_python_sources() -> set[str]:
    index = _module_index()
    pending = {ROOT / identifier for identifier in FROZEN_PYTHON_ENTRYPOINTS}
    for identifier, callable_name in ACTIVE_LAZY_IMPORT_SITES.items():
        pending.update(
            _named_callable_local_imports(ROOT / identifier, callable_name, index)
        )
    for branch in ACTIVE_CONDITIONAL_IMPORT_BRANCHES:
        identifier, class_name, method_name, attribute_name, selector_literal = branch
        pending.update(
            _selected_method_branch_local_imports(
                ROOT / identifier,
                class_name,
                method_name,
                attribute_name,
                selector_literal,
                index,
            )
        )
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        pending.update(_module_scope_local_imports(path, index) - visited)
    return {path.relative_to(ROOT).as_posix() for path in visited}


def test_execution_source_map_is_exact_reachable_python_closure():
    attested = {
        identifier for identifier in EXECUTION_SOURCE_FILES if identifier.endswith(".py")
    }
    reachable = _reachable_python_sources()
    assert attested == reachable, {
        "unattested_reachable_imports": sorted(reachable - attested),
        "unreachable_attested_python": sorted(attested - reachable),
    }


def test_execution_source_map_has_exact_non_python_contract_files():
    attested = {
        identifier
        for identifier in EXECUTION_SOURCE_FILES
        if not identifier.endswith(".py")
    }
    assert attested == EXPECTED_NON_PYTHON_SOURCES


def test_every_attested_source_is_a_real_repository_file():
    for identifier in EXECUTION_SOURCE_FILES:
        path = ROOT / identifier
        assert path.is_file(), identifier
        assert not path.is_symlink(), identifier
