"""Apple Metal-first Transformer subgraph compiler."""

from ._version import __version__
from .artifact import (
    ABI_VERSION,
    ArtifactCompatibilityError,
    ArtifactError,
    load_compiled_module,
    read_artifact_manifest,
    save_compiled,
)
from .backends import ArenaAssignment, ExecutionStats
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
from .selection import KernelBenchmark, KernelSelection
from .passes import (
    canonicalize_batched_matmul,
    eliminate_common_subexpressions,
    fuse_activation_epilogue,
    fuse_attention,
    fuse_gate_up_projection,
    fuse_pointwise,
    fuse_residual_epilogue,
    fuse_rms_norm_epilogue,
    fuse_rope,
    fold_matmul_input_layouts,
    simplify_layouts,
)

__all__ = [
    "ABI_VERSION",
    "ArtifactCompatibilityError",
    "ArtifactError",
    "ArenaAssignment",
    "CompilationResult",
    "CompiledModulePlan",
    "BufferSlot",
    "StoragePlan",
    "ValueLifetime",
    "DType",
    "ExecutionStats",
    "Graph",
    "KernelSelection",
    "KernelBenchmark",
    "Operation",
    "TensorType",
    "Value",
    "capture_fx",
    "compile_callable",
    "compile_graph",
    "compile_module",
    "load_compiled_module",
    "read_artifact_manifest",
    "save_compiled",
    "__version__",
    "canonicalize_batched_matmul",
    "eliminate_common_subexpressions",
    "fuse_activation_epilogue",
    "fuse_attention",
    "fuse_gate_up_projection",
    "fuse_residual_epilogue",
    "fuse_rms_norm_epilogue",
    "fuse_rope",
    "fuse_pointwise",
    "fold_matmul_input_layouts",
    "plan_storage",
    "simplify_layouts",
]
