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
from .cache import fuse_kv_cache_update
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
from .layout import (
    collapse_trailing_layouts,
    fold_matmul_input_layouts,
    simplify_layouts,
)
from .norm import fuse_norm_scale
from .quantize import fold_quantized_weights

#: Names and functions, in order, with the reason each one sits where it does.
#: The names are kept so a test can assert the order without calling anything.
PIPELINE: tuple[tuple[str, Callable[[Graph], Graph]], ...] = (
    # The KV-cache roll is matched against the frontend's own spelling - a
    # slice, a squeeze, two unsqueezes and a concatenate - so it has to run
    # before any pass that would tidy those views away.
    ("fuse_kv_cache_update", fuse_kv_cache_update),
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
    # The norm scale has to be inside the norm before anything looks for a
    # projection to absorb a normalization into, and before the pointwise
    # sweep would fuse the multiply into a standalone region.
    ("fuse_norm_scale", fuse_norm_scale),
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
    # Every fusion above emits views of its own, and none of them are the
    # frontend's to be cleaned up early. A second sweep at the end is the only
    # place that sees the final graph, and each view it removes is one node the
    # runner does not have to dispatch per decode step.
    ("collapse_trailing_layouts", collapse_trailing_layouts),
)


def rewrite(graph: Graph, *, native_matmul: bool = False) -> Graph:
    """Run the whole pipeline over ``graph``.

    ``native_matmul`` reports whether the backend will generate the matrix
    multiply kernels. Only ``fuse_rms_norm_epilogue`` needs it, and only because
    it moves work from a kernel it removes into a kernel it keeps: that is a
    saving on a generated GEMV and a loss on the MLX fallback.
    """

    for _, step in PIPELINE:
        if step is fuse_rms_norm_epilogue:
            graph = step(graph, native_matmul=native_matmul)
        else:
            graph = step(graph)
    return graph
