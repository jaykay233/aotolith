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
from .planning import BufferSlot, StoragePlan, ValueLifetime, plan_storage
from .selection import KernelSelection
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
    "BufferSlot",
    "StoragePlan",
    "ValueLifetime",
    "DType",
    "Graph",
    "KernelSelection",
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
    "plan_storage",
    "simplify_layouts",
]
