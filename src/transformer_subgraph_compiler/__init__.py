"""Apple Metal-first Transformer subgraph compiler."""

from .compiler import (
    CompilationResult,
    CompiledModulePlan,
    compile_callable,
    compile_graph,
    compile_module,
)
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
    "CompiledModulePlan",
    "DType",
    "Graph",
    "Operation",
    "TensorType",
    "Value",
    "capture_fx",
    "compile_callable",
    "compile_graph",
    "compile_module",
    "canonicalize_batched_matmul",
    "eliminate_common_subexpressions",
    "fuse_attention",
    "fuse_pointwise",
    "fold_matmul_input_layouts",
    "simplify_layouts",
]
