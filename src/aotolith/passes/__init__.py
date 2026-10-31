"""Graph rewrites: layout folding, fusion, and CSE.

A pass takes a graph and returns a graph, so the compiler can hand it a graph
and ask nothing more. The pipeline that says which ones run, and in what order,
is :mod:`aotolith.passes.driver`; the pattern recognisers the fusion passes lean
on live under :mod:`aotolith.passes.matchers`, one per family.

The operator sets are re-exported here because they are the language these
passes speak: what may be fused is defined once, in ``op_sets``, and both the
matchers and the backend read it from there.
"""

from __future__ import annotations

from ..op_sets import LAYOUT_OPS, POINTWISE_OPS
from .cache import fuse_kv_cache_update
from .canonicalize import canonicalize_batched_matmul
from .cse import eliminate_common_subexpressions
from .driver import PIPELINE, rewrite
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
from .norm import fuse_norm_scale
from .quantize import fold_quantized_weights

__all__ = [
    "LAYOUT_OPS",
    "PIPELINE",
    "POINTWISE_OPS",
    "canonicalize_batched_matmul",
    "eliminate_common_subexpressions",
    "fold_matmul_input_layouts",
    "fold_quantized_weights",
    "fuse_activation_epilogue",
    "fuse_attention",
    "fuse_gate_up_projection",
    "fuse_kv_cache_update",
    "fuse_norm_scale",
    "fuse_pointwise",
    "fuse_residual_epilogue",
    "fuse_rms_norm_epilogue",
    "fuse_rope",
    "rewrite",
    "simplify_layouts",
]
