"""Apple Metal-first Transformer subgraph compiler."""

from .compiler import CompilationResult, compile_callable, compile_graph
from .frontend import capture_fx
from .ir import DType, Graph, Operation, TensorType, Value
from .passes import (
    canonicalize_batched_matmul,
    fuse_attention,
    fuse_pointwise,
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
    "fuse_attention",
    "fuse_pointwise",
    "simplify_layouts",
]
