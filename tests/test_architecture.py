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
    "selection": 0,
    "_version": 0,
    # The two lowest codegen pieces: what an MLX dtype maps to, and what a Metal
    # kernel header looks like. The target needs both to answer "can this
    # machine compile this at all", so they sit below it.
    "codegen.dtypes": 1,
    "codegen.msl": 1,
    "ir.op": 1,  # Operation, built from types
    "planning": 1,
    # Graph rewriting: knows operators and the IR, not the backend. The graph
    # helpers and the dumb predicates underpin everything above them; the
    # matchers read views, so they sit above `graph` but below the passes that
    # call them; the pipeline order sits at the top, below only its own package.
    "passes.graph": 1,
    "passes.checks": 1,
    "passes.matchers": 2,
    "passes.matchers.view": 2,
    "passes.matchers.matmul": 2,
    "passes.matchers.norm": 2,
    "passes.cse": 2,
    "passes.matchers.attention": 3,
    "passes.matchers.rope": 3,
    "passes.layout": 3,
    "passes.canonicalize": 4,
    "passes.fusion": 4,
    "passes": 5,
    "passes.driver": 5,
    # The decomposition table: a filter over Torch's rules. It reaches back for
    # nothing, so capture can depend on it rather than the reverse.
    "decompose": 1,
    # Tracing only: it knows the IR and takes the decomposition keep set as an
    # argument, so it stays below the legalizer.
    "capture.tracer": 2,
    "ir.graph": 2,  # Graph, built from both
    "legalize.targets": 2,
    # Device capability: what this machine can compile and how much it holds.
    # Everything below the backend may ask, and nothing tells it what to do.
    "target": 3,
    "target.apple": 3,
    # The contract table, over the whole IR.
    "ir.ops": 3,
    "ir.ops.rules": 3,
    "ir.ops.shared": 3,
    "ir.ops.view": 3,
    "ir.ops.constants": 3,
    "ir.ops.reduction": 3,
    "ir.ops.elementwise": 3,
    "ir.ops.indexing": 3,
    "ir.ops.kernels": 3,
    "legalize": 3,
    "legalize.fx": 3,
    "legalize.refs": 3,
    "legalize.state": 3,
    # Metal source emission. The families only need the IR and the target's
    # answers, so they line up here; `matmul` and `attention` reach for the norm
    # prologue and the reduction tree, so they sit above those.
    "ir.ops.table": 4,  # assembles every family
    "codegen.expr": 4,
    "codegen.reduction": 4,
    "codegen.pointwise": 4,
    "codegen.index": 4,
    "legalize.views": 4,
    "legalize.elementwise": 4,
    "legalize.indexing": 4,
    "legalize.reductions": 4,
    "legalize.padding": 4,
    "legalize.linalg": 4,
    "legalize.attention": 4,
    "legalize.kernels": 4,
    "codegen.norm": 5,
    "runtime.bounds": 5,
    # The composition point: trace, then legalize. It sits above both halves and
    # is imported by neither, which is what keeps the two one-way.
    "capture": 5,
    "codegen.matmul": 6,
    "codegen.attention": 7,
    # Execution: the MLX fallback, then the plan that runs it. The plan holds
    # compiled artifacts, so it sits above the whole of codegen.
    "runtime.mlx_ops": 7,
    "runtime.mlx_ops.pointwise": 8,
    "runtime.mlx_ops.layout": 8,
    "runtime.mlx_ops.scatter": 8,
    "runtime.mlx_ops.matmul": 8,
    "runtime.mlx_ops.attention": 8,
    "runtime.mlx_ops.kernels": 8,
    "runtime": 8,
    "runtime.plan": 9,
    # One target's implementation, composing the three above.
    "backends": 10,
    "backends.apple": 10,
    # Serialization of a finished plan.
    "artifact": 11,
    # Sequencing the whole pipeline.
    "compiler": 11,
    # Entry points.
    "cli": 12,
    "models.qwen2": 12,
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
    "passes": "passes/__init__.py",
    "passes.graph": "passes/graph.py",
    "passes.checks": "passes/checks.py",
    "passes.matchers": "passes/matchers/__init__.py",
    "passes.matchers.view": "passes/matchers/view.py",
    "passes.matchers.matmul": "passes/matchers/matmul.py",
    "passes.matchers.attention": "passes/matchers/attention.py",
    "passes.matchers.rope": "passes/matchers/rope.py",
    "passes.matchers.norm": "passes/matchers/norm.py",
    "passes.layout": "passes/layout.py",
    "passes.canonicalize": "passes/canonicalize.py",
    "passes.cse": "passes/cse.py",
    "passes.fusion": "passes/fusion.py",
    "passes.driver": "passes/driver.py",
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
    "legalize.kernels": "legalize/kernels.py",
    "target": "target/__init__.py",
    "target.apple": "target/apple.py",
    "codegen": "codegen/__init__.py",
    "codegen.dtypes": "codegen/dtypes.py",
    "codegen.msl": "codegen/msl.py",
    "codegen.expr": "codegen/expr.py",
    "codegen.reduction": "codegen/reduction.py",
    "codegen.pointwise": "codegen/pointwise.py",
    "codegen.index": "codegen/index.py",
    "codegen.norm": "codegen/norm.py",
    "codegen.matmul": "codegen/matmul.py",
    "codegen.attention": "codegen/attention.py",
    "runtime": "runtime/__init__.py",
    "runtime.bounds": "runtime/bounds.py",
    "runtime.mlx_ops": "runtime/mlx_ops/__init__.py",
    "runtime.mlx_ops.pointwise": "runtime/mlx_ops/pointwise.py",
    "runtime.mlx_ops.layout": "runtime/mlx_ops/layout.py",
    "runtime.mlx_ops.scatter": "runtime/mlx_ops/scatter.py",
    "runtime.mlx_ops.matmul": "runtime/mlx_ops/matmul.py",
    "runtime.mlx_ops.attention": "runtime/mlx_ops/attention.py",
    "runtime.mlx_ops.kernels": "runtime/mlx_ops/kernels.py",
    "runtime.plan": "runtime/plan.py",
    "backends": "backends/__init__.py",
    "backends.apple": "backends/apple.py",
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
    for module in ("backends.apple", "codegen.matmul", "runtime.plan"):
        imported = _module_level_imports(MODULE_PATHS[module])
        assert "passes" not in imported


def test_fusion_and_codegen_agree_on_what_is_pointwise() -> None:
    # Both sides import the one definition, so a new elementwise operator
    # cannot be fusable without also being emittable, or the reverse.
    from aotolith import op_sets, passes
    from aotolith.backends import apple

    assert passes.POINTWISE_OPS is op_sets.POINTWISE_OPS
    assert passes.LAYOUT_OPS is op_sets.LAYOUT_OPS
    assert apple.POINTWISE_OPS is op_sets.POINTWISE_OPS
    assert apple.LAYOUT_OPS is op_sets.LAYOUT_OPS
