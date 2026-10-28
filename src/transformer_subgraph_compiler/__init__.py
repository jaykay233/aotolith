"""Apple Metal-first Transformer subgraph compiler."""

from .compiler import CompilationResult, compile_callable, compile_graph
from .frontend import capture_fx
from .ir import DType, Graph, Operation, TensorType, Value
from .passes import (
    canonicalize_batched_matmul,
    eliminate_common_subexpressions,
    fuse_attention,
    fuse_pointwise,
    fold_matmul_input_layouts,
    simplify_layouts,
)

__all__ = [
    "CompilationResult",
    "DType",
    "Graph",
    "Operation",
    "TensorType",
    "Value",
    "capture_fx",
    "compile_callable",
    "compile_graph",
    "canonicalize_batched_matmul",
    "eliminate_common_subexpressions",
    "fuse_attention",
    "fuse_pointwise",
    "fold_matmul_input_layouts",
    "simplify_layouts",
]
