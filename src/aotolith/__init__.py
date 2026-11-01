"""Apple Metal-first Transformer subgraph compiler."""

import platform as _platform
import sys as _sys

from ._version import __version__
from ._log import basic_config as _basic_config
from ._log import install_null_handler as _install_null_handler
from .errors import (
    AotolithError,
    BaselineError,
    BenchmarkError,
    BucketRangeError,
    CompilationError,
    DeviceError,
    DynamicShapeError,
    IRValidationError,
    IndexBoundsError,
    MemoryBudgetError,
    PlanDTypeError,
    PlanInputError,
    PlatformError,
    UnsupportedDTypeError,
    UnsupportedModelError,
    UnsupportedOperationError,
)


def _require_supported_host() -> None:
    """Fail on import, with an explanation, where the backend cannot exist.

    The wheel is pure Python, so ``pip install aotolith`` succeeds on a Linux
    box and then every call fails inside MLX with a message about Metal. The
    requirement is not really "macOS" - it is "Metal with a family this code
    generator emits for" - but the two checks below are the ones that can be
    answered before importing MLX, and they cover the install mistakes people
    actually make. The dependency declaration carries the same condition as a
    PEP 508 marker, so a normal install never reaches this path; it exists for
    the installations that bypass resolution (a copied wheel, a bundled venv, a
    container that installed the pure-Python artifact by hand).
    """

    system = _sys.platform
    machine = _platform.machine()
    if system != "darwin" or machine != "arm64":
        raise PlatformError(
            f"aotolith needs macOS on Apple Silicon, and this is "
            f"{system}/{machine}",
            hint=(
                "the backend emits Metal only. Install on a Darwin arm64 host, "
                "or import aotolith.errors alone to read the error contract"
            ),
        )


_require_supported_host()

try:
    import mlx.core as _mx  # noqa: F401 - the import itself is the check
except ImportError as exc:  # pragma: no cover - depends on the installation
    raise PlatformError(
        "mlx is not importable, and aotolith's execution layer is built on it",
        hint="pip install 'mlx>=0.32,<0.33' on macOS 14+ with an Apple GPU",
    ) from exc

# Attach the quiet placeholder handler every library should own. A host adds
# its own handler to the "aotolith" logger; importing this package never prints.
_install_null_handler()

#: Give the package logger a stderr handler at one level. For scripts and the
#: command line; a library host should configure ``logging.getLogger("aotolith")``
#: itself instead, so it keeps control of level, format and destination.
configure_logging = _basic_config

from .artifact import (
    ABI_VERSION,
    FORMAT_VERSION,
    ArtifactCompatibilityError,
    ArtifactError,
    load_compiled_module,
    read_artifact_manifest,
    save_compiled,
)
from .backends import (
    REFERENCE_TARGET,
    TARGETS,
    ArenaAssignment,
    ExecutionStats,
)
from .compiler import (
    CompilationResult,
    CompiledModulePlan,
    compile_callable,
    compile_graph,
    compile_module,
)
from .buckets import (
    BucketFootprint,
    BucketStats,
    BucketedPlans,
    ShapeBucket,
    compile_bucketed,
    load_bucketed,
    share_plan_state,
)
from .capture import capture_fx
from .ir import DType, Graph, Operation, TensorType, Value
from .planning import (
    BufferSlot,
    StoragePlan,
    ValueLifetime,
    plan_storage,
    tensor_nbytes,
)
from .runtime.plan import (
    DEFAULT_MEMORY,
    MEMORY_POLICIES,
    MemoryReport,
    PlanFootprint,
)
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
    "FORMAT_VERSION",
    "AotolithError",
    "ArtifactCompatibilityError",
    "ArtifactError",
    "BaselineError",
    "BenchmarkError",
    "ArenaAssignment",
    "BucketFootprint",
    "BucketRangeError",
    "BucketStats",
    "BucketedPlans",
    "CompilationError",
    "CompilationResult",
    "CompiledModulePlan",
    "BufferSlot",
    "DeviceError",
    "DynamicShapeError",
    "IRValidationError",
    "IndexBoundsError",
    "MemoryBudgetError",
    "MemoryReport",
    "MEMORY_POLICIES",
    "DEFAULT_MEMORY",
    "PlanFootprint",
    "PlanDTypeError",
    "PlanInputError",
    "PlatformError",
    "ShapeBucket",
    "StoragePlan",
    "UnsupportedDTypeError",
    "UnsupportedModelError",
    "UnsupportedOperationError",
    "ValueLifetime",
    "DType",
    "ExecutionStats",
    "REFERENCE_TARGET",
    "TARGETS",
    "Graph",
    "KernelSelection",
    "KernelBenchmark",
    "Operation",
    "TensorType",
    "Value",
    "capture_fx",
    "configure_logging",
    "compile_bucketed",
    "compile_callable",
    "compile_graph",
    "compile_module",
    "load_bucketed",
    "share_plan_state",
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
    "tensor_nbytes",
    "simplify_layouts",
]
