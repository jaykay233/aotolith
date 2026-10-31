"""The rewrite pipeline: every pass, in the order it has to run.

The sequence used to be eleven assignments inside ``compiler.compile_graph``,
which meant the one place that knew why the order mattered sat above everything
and could not be tested on its own. It lives here instead, next to the passes it
orders, and the compiler asks for the finished pipeline rather than assembling
it.
"""

from __future__ import annotations

from collections.abc import Callable

from ..ir import Graph
from .canonicalize import canonicalize_batched_matmul
from .cse import eliminate_common_subexpressions
from .fusion import (
    fuse_activation_epilogue,
    fuse_attention,
    fuse_gate_up_projection,
    fuse_pointwise,
    fuse_residual_epilogue,
    fuse_rms_norm_epilogue,
    fuse_rope,
)
from .layout import fold_matmul_input_layouts, simplify_layouts
from .quantize import fold_quantized_weights

#: Names and functions, in order, with the reason each one sits where it does.
#: The names are kept so a test can assert the order without calling anything.
PIPELINE: tuple[tuple[str, Callable[[Graph], Graph]], ...] = (
    # Dead layout work first: every later matcher reads shapes through whatever
    # the frontend left behind, and would otherwise have to see through it.
    ("simplify_layouts", simplify_layouts),
    # PyTorch spells a batched matmul as fold-bmm-unfold. Undo that before
    # anything tries to fuse around a matmul, so the fusions see a real one.
    ("canonicalize_batched_matmul", canonicalize_batched_matmul),
    # Attention before the epilogue fusions. The attention region is large, and
    # fusing the smaller epilogue inside it first would cut it in half.
    ("fuse_attention", fuse_attention),
    # The layouts a matmul can absorb are cheaper folded than emitted, and the
    # rope matcher reads the matmul's operand order afterwards.
    ("fold_matmul_input_layouts", fold_matmul_input_layouts),
    # The packed weight replaces the weight operand itself, so it has to happen
    # once the operand layouts are final and before any matcher reads the
    # right operand as a matrix.
    ("fold_quantized_weights", fold_quantized_weights),
    ("fuse_rope", fuse_rope),
    # The epilogues, additive ones before the pointwise sweep that would
    # otherwise swallow the region they need.
    ("fuse_residual_epilogue", fuse_residual_epilogue),
    ("fuse_pointwise", fuse_pointwise),
    ("fuse_gate_up_projection", fuse_gate_up_projection),
    ("fuse_activation_epilogue", fuse_activation_epilogue),
    # CSE after fusion, so that two fused regions computing the same thing
    # collapse, and before rms norm, which looks for a single consumer.
    ("eliminate_common_subexpressions", eliminate_common_subexpressions),
    ("fuse_rms_norm_epilogue", fuse_rms_norm_epilogue),
)


def rewrite(graph: Graph) -> Graph:
    """Run the whole pipeline over ``graph``."""

    for _, step in PIPELINE:
        graph = step(graph)
    return graph
