"""The import graph must stay acyclic and point one way.

Each module sits at a numbered layer, and a module may only import from a
strictly lower one. This is what keeps a layer reachable on its own: the point
of the check is not tidiness, it is that the contract table can be imported
without pulling in the graph rewriter, and the backend can be imported without pulling
in what the rewrite pipeline happens to look like today.

The layers here are the compiler's own, not the eventual package layout. When
a module moves into a subpackage (``passes.py`` into ``passes/``,
``apple_metal.py`` into ``codegen/``) the entry moves with it and the rule does
not change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "aotolith"

# Lower number = lower layer. A module may import only from a strictly smaller
# number, so nothing at the bottom knows what is above it. A package's
# ``__init__`` may also re-export from its own submodules, which is the one
# allowed same-layer edge.
LAYERS: dict[str, int] = {
    # No intra-package imports at all: pure data and shared primitives.
    "op_sets": 0,
    "shapes": 0,
    "ir": 0,
    "ir.types": 0,
    "ir.op": 1,  # Operation, built from types
    "ir.graph": 2,  # Graph, built from both
    "ir.ops": 3,  # the contract table, over the whole IR
    "ir.ops.rules": 3,
    "ir.ops.shared": 3,
    "ir.ops.view": 3,
    "ir.ops.constants": 3,
    "ir.ops.reduction": 3,
    "ir.ops.elementwise": 3,
    "ir.ops.indexing": 3,
    "ir.ops.kernels": 3,
    "ir.ops.table": 4,  # assembles every family
    "selection": 0,
    "_version": 0,
    # Facts about the operator set, expressed in terms of the IR types.
    "planning": 1,
    # Graph rewriting: knows operators and the IR, not the backend.
    "passes": 1,
    # The decomposition table: a filter over Torch's rules. It reaches back for
    # nothing, so capture can depend on it rather than the reverse.
    "decompose": 1,
    # Tracing only: it knows the IR and takes the decomposition keep set as an
    # argument, so it stays below the legalizer.
    "capture.tracer": 2,
    # Legalization: FX -> core ops. Sits above capture because it consumes what
    # tracing produced, and never the other way round.
    "legalize": 3,
    "legalize.targets": 2,
    "legalize.fx": 3,
    "legalize.refs": 3,
    "legalize.state": 3,
    "legalize.views": 4,
    "legalize.elementwise": 4,
    "legalize.indexing": 4,
    "legalize.reductions": 4,
    "legalize.padding": 4,
    "legalize.linalg": 4,
    "legalize.attention": 4,
    # The composition point: trace, then legalize. It sits above both halves and
    # is imported by neither, which is what keeps the two one-way.
    "capture": 5,
    # One target's implementation: codegen, runtime dispatch, plan execution.
    "backends": 3,
    "backends.apple_metal": 3,
    # Serialization of a finished plan.
    "artifact": 4,
    # Sequencing the whole pipeline.
    "compiler": 6,
    # Entry points.
    "cli": 7,
    "models.qwen2": 7,
}

MODULE_PATHS = {
    "op_sets": "op_sets.py",
    "shapes": "shapes.py",
    "ir": "ir/__init__.py",
    "ir.types": "ir/types.py",
    "ir.op": "ir/op.py",
    "ir.graph": "ir/graph.py",
    "ir.ops": "ir/ops/__init__.py",
    "ir.ops.rules": "ir/ops/rules.py",
    "ir.ops.shared": "ir/ops/shared.py",
    "ir.ops.view": "ir/ops/view.py",
    "ir.ops.constants": "ir/ops/constants.py",
    "ir.ops.reduction": "ir/ops/reduction.py",
    "ir.ops.elementwise": "ir/ops/elementwise.py",
    "ir.ops.indexing": "ir/ops/indexing.py",
    "ir.ops.kernels": "ir/ops/kernels.py",
    "ir.ops.table": "ir/ops/table.py",
    "selection": "selection.py",
    "_version": "_version.py",
    "planning": "planning.py",
    "passes": "passes.py",
    "decompose": "decompose.py",
    "capture": "capture/__init__.py",
    "capture.tracer": "capture/tracer.py",
    "legalize": "legalize/__init__.py",
    "legalize.targets": "legalize/targets.py",
    "legalize.fx": "legalize/fx.py",
    "legalize.refs": "legalize/refs.py",
    "legalize.state": "legalize/state.py",
    "legalize.views": "legalize/views.py",
    "legalize.elementwise": "legalize/elementwise.py",
    "legalize.indexing": "legalize/indexing.py",
    "legalize.reductions": "legalize/reductions.py",
    "legalize.padding": "legalize/padding.py",
    "legalize.linalg": "legalize/linalg.py",
    "legalize.attention": "legalize/attention.py",
    "backends": "backends/__init__.py",
    "backends.apple_metal": "backends/apple_metal.py",
    "artifact": "artifact.py",
    "compiler": "compiler.py",
    "cli": "cli.py",
    "models.qwen2": "models/qwen2.py",
}


def _module_level_imports(relative_path: str) -> set[str]:
    """Intra-package modules imported at module scope, by dotted name."""

    tree = ast.parse((SOURCE_ROOT / relative_path).read_text())
    package = "aotolith." + ".".join(Path(relative_path).parts[:-1])
    package = package.rstrip(".")
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                target = node.module or ""
            else:
                # ``level`` counts from the module's own package upwards, so
                # ``from ..ir import X`` inside ``aotolith.backends`` resolves
                # against ``aotolith``.
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            if target == "aotolith" or target.startswith("aotolith."):
                prefix = target[len("aotolith.") :]
                if node.module is None and node.level:
                    # ``from . import views`` names submodules, not attributes.
                    found.update(f"{prefix}.{alias.name}" for alias in node.names)
                else:
                    found.add(prefix)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("aotolith."):
                    found.add(alias.name[len("aotolith.") :])
    return found


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_a_module_only_imports_from_a_lower_layer(module: str) -> None:
    layer = LAYERS[module]
    # A package's ``__init__`` re-exports its own submodules; that is the one
    # same-package edge allowed, and it is what makes ``from ..ir import
    # Graph`` work.
    package = (
        module
        if MODULE_PATHS[module].endswith("__init__.py")
        else module.rpartition(".")[0]
    )
    for imported in _module_level_imports(MODULE_PATHS[module]):
        if package and imported.startswith(package + "."):
            continue
        # A submodule of a known package counts as that package.
        known = next(
            (
                name
                for name in LAYERS
                if imported == name or imported.startswith(name + ".")
            ),
            None,
        )
        if known is None:
            continue
        assert LAYERS[known] < layer, (
            f"{module} (layer {layer}) imports {known} (layer {LAYERS[known]}); "
            "an import may only point at a strictly lower layer"
        )


def test_every_module_in_the_table_exists() -> None:
    for path in MODULE_PATHS.values():
        assert (SOURCE_ROOT / path).is_file(), path


def test_the_operator_registry_does_not_import_the_pass_pipeline() -> None:
    # This was a real backwards edge: the contract table read POINTWISE_OPS out
    # of passes.py to build its default contracts. Now no module under ir/ops
    # reaches for a rewrite.
    for module in ("ir.ops.table", "ir.ops.rules", "ir.ops.reduction"):
        imported = _module_level_imports(MODULE_PATHS[module])
        assert "passes" not in imported
        assert not any(
            name == "passes" or name.startswith("passes.") for name in imported
        )


def test_the_backend_does_not_import_the_pass_pipeline() -> None:
    # Same edge from the other side: the backend read LAYOUT_OPS out of
    # passes.py, which made codegen depend on what rewrites exist.
    imported = _module_level_imports(MODULE_PATHS["backends.apple_metal"])
    assert "passes" not in imported


def test_fusion_and_codegen_agree_on_what_is_pointwise() -> None:
    # Both sides import the one definition, so a new elementwise operator
    # cannot be fusable without also being emittable, or the reverse.
    from aotolith import passes
    from aotolith.backends import apple_metal

    assert passes.POINTWISE_OPS is apple_metal.POINTWISE_OPS
    assert passes.LAYOUT_OPS is apple_metal.LAYOUT_OPS
