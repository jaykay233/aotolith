from __future__ import annotations

import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil, isfinite, isinf, isnan, prod
from statistics import median
from typing import Any

import mlx.core as mx

from ..ir import DType, Graph, Operation, TensorType
from ..planning import StoragePlan, plan_storage
from ..selection import KernelBenchmark, KernelSelection
from ..passes import LAYOUT_OPS, POINTWISE_OPS

_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
    DType.BOOL: mx.bool_,
    DType.I32: mx.int32,
    DType.I64: mx.int64,
}

_ELEMENTS_PER_THREAD = 4
_DTYPE_ITEMSIZE = {
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.BOOL: 1,
    DType.I32: 4,
    DType.I64: 8,
}
_COMPARISON_POINTWISE_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
_LOGICAL_POINTWISE_OPS = {
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
}
_BITWISE_POINTWISE_OPS = {
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
}
_NUMERIC_PREDICATE_POINTWISE_OPS = {"isnan", "isinf"}
_CONSTANT_POINTWISE_OPS = {
    "scalar_tensor",
    "zeros",
    "ones",
    "full",
    "new_zeros",
    "new_ones",
    "new_full",
    "zeros_like",
    "ones_like",
    "full_like",
}
_INTEGRAL_DTYPES = {DType.I32, DType.I64}
_INTEGER_POINTWISE_OPS = {
    "add",
    "sub",
    "rsub",
    "mul",
    "neg",
    "abs",
    "sign",
    "square",
    "maximum",
    "minimum",
    "floor_divide",
    "remainder",
    "clamp",
    "clamp_min",
    "clamp_max",
    *_BITWISE_POINTWISE_OPS,
}
_MATMUL_MICROTILE = (4, 4)
_MATMUL_LARGE_K_MICROTILE = (8, 4)
_MATMUL_LARGE_K_THRESHOLD = 1024
_MATMUL_BLOCK_ROWS = 64
_MATMUL_BLOCK_COLUMNS = 64
_MATMUL_BLOCK_REDUCTION = 32
_MATMUL_GEMV_THREADS = 256
# One warp owns one output column, so more columns means more blocks.  Wide
# outputs need narrow blocks to keep the grid large; narrow outputs need fat
# blocks so each one streams enough contiguous weights to saturate memory.
_MATMUL_GEMV_NARROW_COLUMNS = 8192
_MATMUL_GEMV_THREADS_NARROW = 1024
_MATMUL_GEMV_THREADS_WIDE = 128


def _gemv_threads(columns: int, transpose_right: bool) -> int:
    """Pick the GEMV threadgroup size for one output width.

    The transposed path runs one warp per output column, so the block size
    trades block count against per-block streaming width and the best value
    flips once the output is wide enough to fill the GPU on its own.
    """

    if transpose_right:
        if columns <= _MATMUL_GEMV_NARROW_COLUMNS:
            return _MATMUL_GEMV_THREADS_NARROW
        return _MATMUL_GEMV_THREADS_WIDE
    return _MATMUL_GEMV_THREADS
_METAL_KERNEL_CACHE_SIZE = 256
_ATTENTION_MAX_SHARED_BYTES = 31 * 1024
_ATTENTION_LONG_SEQUENCE_THRESHOLD = 768
_AUTOTUNE_WARMUP = 2
_AUTOTUNE_ITERATIONS = 7

_MSL_HEADER = """
inline float tsc_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = metal::abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - polynomial * metal::exp(-x * x));
}

inline float tsc_expm1(float value) {
    if (metal::abs(value) <= 0.5f) {
        return value * (1.0f + value * (0.5f + value * (
            0.16666666666666666f + value * (0.041666666666666664f + value * (
            0.008333333333333333f + value * (
            0.001388888888888889f + value * (
            0.0001984126984126984f + value * 0.0000248015873015873f)))))));
    }
    return metal::exp(value) - 1.0f;
}

inline float tsc_softplus(float value, float beta, float threshold) {
    float scaled = beta * value;
    return scaled > threshold
        ? value
        : log1p(metal::exp(scaled)) / beta;
}

inline float tsc_logsigmoid(float value) {
    if (value != value) {
        return value;
    }
    return metal::min(value, 0.0f)
        - log1p(metal::exp(-metal::abs(value)));
}

inline float tsc_logaddexp(float left, float right) {
    if (metal::isnan(left) || metal::isnan(right)) {
        return NAN;
    }
    float maximum = left < right ? right : left;
    if (metal::isinf(maximum)) {
        return maximum;
    }
    return maximum + log1p(metal::exp(-metal::abs(left - right)));
}

inline float tsc_atan2(float y, float x) {
    constexpr float pi = 3.14159265358979323846f;
    if (metal::isnan(y) || metal::isnan(x)) {
        return NAN;
    }
    bool x_negative = (as_type<uint>(x) & 0x80000000u) != 0u;
    if (metal::isinf(y)) {
        if (metal::isinf(x)) {
            float magnitude = x_negative ? 0.75f * pi : 0.25f * pi;
            return metal::copysign(magnitude, y);
        }
        return metal::copysign(0.5f * pi, y);
    }
    if (metal::isinf(x)) {
        return x_negative
            ? metal::copysign(pi, y)
            : metal::copysign(0.0f, y);
    }
    if (x == 0.0f) {
        if (y == 0.0f) {
            return x_negative
                ? metal::copysign(pi, y)
                : metal::copysign(0.0f, y);
        }
        return metal::copysign(0.5f * pi, y);
    }
    return atan2(y, x);
}

inline float tsc_mish(float value) {
    return value > 20.0f
        ? value
        : value * metal::tanh(tsc_softplus(value, 1.0f, 20.0f));
}

inline int tsc_neg_i32(int value) {
    return (int)(0u - (uint)value);
}

inline long tsc_neg_i64(long value) {
    return (long)((unsigned long)0 - (unsigned long)value);
}

inline int tsc_add_i32(int left, int right) {
    return (int)((uint)left + (uint)right);
}

inline long tsc_add_i64(long left, long right) {
    return (long)((unsigned long)left + (unsigned long)right);
}

inline int tsc_sub_i32(int left, int right) {
    return (int)((uint)left - (uint)right);
}

inline long tsc_sub_i64(long left, long right) {
    return (long)((unsigned long)left - (unsigned long)right);
}

inline int tsc_mul_i32(int left, int right) {
    return (int)((uint)left * (uint)right);
}

inline long tsc_mul_i64(long left, long right) {
    return (long)((unsigned long)left * (unsigned long)right);
}

inline int tsc_floor_div_i32(int dividend, int divisor) {
    if (divisor == -1) {
        return tsc_neg_i32(dividend);
    }
    int quotient = dividend / divisor;
    int remainder = dividend % divisor;
    bool round_down = remainder != 0 && ((remainder < 0) != (divisor < 0));
    return quotient - (round_down ? 1 : 0);
}

inline long tsc_floor_div_i64(long dividend, long divisor) {
    if (divisor == -1L) {
        return tsc_neg_i64(dividend);
    }
    long quotient = dividend / divisor;
    long remainder = dividend % divisor;
    bool round_down = remainder != 0 && ((remainder < 0) != (divisor < 0));
    return quotient - (round_down ? 1L : 0L);
}

inline int tsc_remainder_i32(int dividend, int divisor) {
    if (divisor == -1) {
        return 0;
    }
    int remainder = dividend % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        remainder += divisor;
    }
    return remainder;
}

inline long tsc_remainder_i64(long dividend, long divisor) {
    if (divisor == -1L) {
        return 0L;
    }
    long remainder = dividend % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        remainder += divisor;
    }
    return remainder;
}
"""


_MPP_HEADER = (
    _MSL_HEADER
    + "\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
)

# MetalPerformancePrimitives tensor ops exist only where the compiler defines
# `__HAVE_TENSOR__` (Apple GPU family 9 and later). Emitting them elsewhere
# still compiles but the metallib refuses to load, which reaches the caller as
# an opaque `RuntimeError` at the first matmul. Asking the compiler beats
# parsing `device_info()["architecture"]`, which would mean encoding a table of
# marketing names that Apple can extend or retire.
#
# This is a hardware-capability answer, not "these kernels will load": a
# paravirtualised GPU (hosted CI, cloud Macs) reports `__HAVE_TENSOR__` and
# still rejects them. That case is caught by loading a real kernel at compile
# time, which then sets `_TENSOR_CORES_BLOCKED` for the rest of the process.
_MPP_PROBE_SOURCE = """
        out[0] = 0;
#if __HAVE_TENSOR__
        out[0] = 1;
#endif
"""

_TENSOR_CORES_BLOCKED = False
_TENSOR_CORE_BLOCK_ERROR: str | None = None


def _block_tensor_cores(error: str) -> None:
    """Stop emitting tensor-core kernels after this device rejected one."""

    global _TENSOR_CORES_BLOCKED, _TENSOR_CORE_BLOCK_ERROR
    _TENSOR_CORES_BLOCKED = True
    _TENSOR_CORE_BLOCK_ERROR = error


def tensor_core_fallback_reason() -> str | None:
    """Return why tensor cores were abandoned, or None if they still apply."""

    return _TENSOR_CORE_BLOCK_ERROR


@lru_cache(maxsize=None)
def _compiler_reports_tensor_ops() -> bool:
    """Return whether this GPU has tensor ops the codegen may target."""

    try:
        kernel = _metal_kernel(
            "aotolith_mpp_probe",
            (),
            ("out",),
            _MPP_PROBE_SOURCE,
            _MPP_HEADER,
        )
        result = kernel(
            inputs=[],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.int32],
        )[0]
        # MLX is lazy: without this the kernel never compiles and the probe
        # answers yes no matter what the device thinks.
        mx.eval(result)
        return bool(result.item())
    except Exception:  # noqa: BLE001 - any failure means "do not use MPP".
        return False


def _mpp_supported() -> bool:
    """Return whether codegen may emit tensor-core kernels."""

    if _TENSOR_CORES_BLOCKED:
        return False
    return _compiler_reports_tensor_ops()


# Tensor-core (NAX / MXU) GEMM tiles, each
# (block_rows, block_columns, warp_rows, warp_columns, reduction_step).
# Swept on Apple M5 (applegpu_g17s) over a shape matrix in both operand
# layouts; the selection logic below encodes the measured regimes rather than a
# single compromise tile.  A transposed right operand is K-contiguous and
# rewards wide column blocks, while a row-major right operand strides through
# K and instead rewards a taller reduction step.
_MPP_GEMM_TILE_TINY = (32, 32, 1, 1, 32)
_MPP_GEMM_TILE_SMALL_K = (64, 64, 2, 2, 32)
_MPP_GEMM_TILE_SKINNY_LONG_K = (128, 32, 4, 1, 32)
# Right operand transposed (N x K) and right operand row-major (K x N).
_MPP_GEMM_TILE_TRANSPOSED_WIDE = (64, 128, 2, 4, 64)
_MPP_GEMM_TILE_ROW_MAJOR_WIDE = (64, 256, 1, 8, 32)
_MPP_GEMM_TILE_ROW_MAJOR = (128, 64, 2, 2, 32)
_MPP_GEMM_TILE_TALL = (128, 128, 2, 4, 32)
# fp32 accumulation over a deep reduction stays on the 2x2 warp tile.
_MPP_GEMM_TILE_F32 = _MPP_GEMM_TILE_ROW_MAJOR
# Progressively smaller tiles; the selector walks this list so that a block
# never exceeds the problem size (which would waste tensor-core work and force
# boundary predication on exact shapes).
_MPP_GEMM_TILE_LADDER = (
    _MPP_GEMM_TILE_ROW_MAJOR_WIDE,
    _MPP_GEMM_TILE_TALL,
    _MPP_GEMM_TILE_TRANSPOSED_WIDE,
    _MPP_GEMM_TILE_ROW_MAJOR,
    _MPP_GEMM_TILE_SKINNY_LONG_K,
    (64, 64, 2, 2, 64),
    _MPP_GEMM_TILE_SMALL_K,
    (64, 32, 2, 1, 64),
    _MPP_GEMM_TILE_TINY,
)
# Above this many output elements the wide tile wins on reuse and warp count.
_MPP_GEMM_WIDE_ELEMENTS = 3_000_000
# Fewer than this many columns per block makes the kernel re-stream the left
# operand once per column block, which dominates for tall-skinny products.
_MPP_GEMM_MIN_WIDE_COLUMNS = 8192
# Past this width a row-major right operand is re-read too many times per row
# block, so the tall square tile wins on left-operand reuse instead.
_MPP_GEMM_MAX_WIDE_COLUMNS = 32768
# Long reductions only favour the skinny tile while the row count stays small;
# past that the block count drops below the core count and the wide tile wins.
_MPP_GEMM_SKINNY_MAX_ROWS = 256
_MPP_GEMM_SKINNY_MAX_COLUMNS = 2048


@lru_cache(maxsize=_METAL_KERNEL_CACHE_SIZE)
def _metal_kernel(
    name: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    source: str,
    header: str,
) -> Any:
    """Reuse an mx.fast.metal_kernel for an identical static kernel spec."""

    return mx.fast.metal_kernel(
        name=name,
        input_names=list(input_names),
        output_names=list(output_names),
        source=source,
        header=header,
    )


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    name: str
    source: str
    header: str
    input_ids: tuple[str, ...]
    output_id: str
    tensor_type: TensorType
    kind: str


_PENDING_INDEX_CHECKS: ContextVar[list[tuple[mx.array, int, bool]] | None] = (
    ContextVar("aotolith_pending_index_checks", default=None)
)

# A bounds check needs the index values, so a device-resident index forces a
# sync. Callers that already know the indices are in range can turn the check
# off for one execution instead of paying for the drain every step.
_INDEX_CHECKS_ENABLED: ContextVar[bool] = ContextVar(
    "aotolith_index_checks_enabled", default=True
)


def _defer_index_check(
    indices: mx.array, upper_bound: int, *, allow_negative: bool
) -> bool:
    """Queue a bounds check so a plan run pays for at most one device sync.

    Returns ``True`` when the check was queued and the caller must not sync.
    """

    if not _INDEX_CHECKS_ENABLED.get():
        return True
    pending = _PENDING_INDEX_CHECKS.get()
    if pending is None:
        return False
    pending.append((indices, upper_bound, allow_negative))
    return True


def _flush_index_checks(
    pending: list[tuple[mx.array, int, bool]]
) -> None:
    """Collapse every queued bounds check into a single reduction and sync."""

    if not pending:
        return
    if not _INDEX_CHECKS_ENABLED.get():
        pending.clear()
        return
    violation: mx.array | None = None
    for indices, upper_bound, allow_negative in pending:
        low = -upper_bound if allow_negative else 0
        if indices.size == 0:
            continue
        bad = mx.logical_or(indices < low, indices >= upper_bound)
        violation = bad if violation is None else mx.logical_or(violation, bad)
    if violation is None:
        return
    if bool(mx.any(violation).item()):
        raise IndexError("index out of range in self")


def _check_take_indices(indices: mx.array, upper_bound: int) -> None:
    """Match PyTorch gather bounds instead of MLX's wrap-and-fill behavior."""

    if _defer_index_check(indices, upper_bound, allow_negative=False):
        return
    _flush_index_checks([(indices, upper_bound, False)])


def _mlx_take_checked(source: mx.array, indices: mx.array, axis: int) -> mx.array:
    _check_take_indices(indices, source.shape[axis])
    return mx.take(source, indices, axis=axis)


def _check_advanced_index_indices(indices: mx.array, upper_bound: int) -> None:
    if _defer_index_check(indices, upper_bound, allow_negative=True):
        return
    _flush_index_checks([(indices, upper_bound, True)])


def _tensor_nbytes(tensor_type: TensorType) -> int:
    itemsize = _DTYPE_ITEMSIZE[tensor_type.dtype]
    return prod(tensor_type.shape) * itemsize


def _evaluate_mlx_op(
    graph: Graph, op: Operation, env: dict[str, mx.array]
) -> mx.array:
    """Evaluate one IR operation with MLX fallback semantics."""

    if op.op == "fused_elementwise":
        body = op.attrs.get("body")
        if not isinstance(body, tuple) or not all(
            isinstance(item, Operation) for item in body
        ):
            raise ValueError(
                "fused_elementwise body must be a tuple of operations"
            )
        body_env = dict(env)
        for item in body:
            body_env[item.outputs[0]] = _evaluate_mlx_op(
                graph, item, body_env
            )
        return body_env[op.outputs[0]]
    if op.op == "attention":
        return _evaluate_attention_mlx(op, env)
    if op.op == "rope":
        value, cosine, sine = (env[value_id] for value_id in op.inputs)
        half = value.shape[-1] // 2
        rotated = mx.concatenate(
            (-value[..., half:], value[..., :half]), axis=-1
        )
        return value * cosine + rotated * sine
    if op.op == "embedding":
        weight, indices = (env[value_id] for value_id in op.inputs)
        return _mlx_take_checked(weight, indices, axis=0)
    if op.op == "index_select":
        source, indices = (env[value_id] for value_id in op.inputs)
        return _mlx_take_checked(source, indices, axis=int(op.attrs["dim"]))
    if op.op == "index":
        source = env[op.inputs[0]]
        indices = tuple(env[value_id] for value_id in op.inputs[1:])
        axes = tuple(int(axis) for axis in op.attrs["axes"])
        for axis, index in zip(axes, indices, strict=True):
            _check_advanced_index_indices(index, source.shape[axis])
        indexing = (slice(None),) * axes[0] + indices
        return source[indexing]
    if op.op == "gather":
        source, indices = (env[value_id] for value_id in op.inputs)
        axis = int(op.attrs["dim"])
        _check_take_indices(indices, source.shape[axis])
        source_slices = [
            slice(0, indices.shape[dim]) if dim != axis else slice(None)
            for dim in range(len(source.shape))
        ]
        return mx.take_along_axis(
            source[tuple(source_slices)], indices, axis=axis
        )
    if op.op == "scatter":
        source, indices = (env[value_id] for value_id in op.inputs[:2])
        axis = int(op.attrs["dim"])
        _check_take_indices(indices, source.shape[axis])
        values = (
            mx.array(op.attrs["value"], dtype=source.dtype)
            if len(op.inputs) == 2
            else env[op.inputs[2]]
        )
        if len(op.inputs) == 3:
            values = values[tuple(slice(0, size) for size in indices.shape)]
        source_slices = [
            slice(0, indices.shape[dim]) if dim != axis else slice(None)
            for dim in range(len(source.shape))
        ]
        slices = tuple(source_slices)
        updated = mx.put_along_axis(
            source[slices], indices, values, axis=axis
        )
        output = mx.array(source)
        output[slices] = updated
        return output
    if op.op == "cumsum":
        axis = int(op.attrs["dim"]) % len(env[op.inputs[0]].shape)
        value = mx.cumsum(env[op.inputs[0]], axis=axis)
        output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        return value.astype(output_dtype)
    if op.op == "cast":
        return env[op.inputs[0]].astype(
            _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        )
    if op.op in {"triu", "tril"}:
        operation = mx.triu if op.op == "triu" else mx.tril
        return operation(env[op.inputs[0]], int(op.attrs["diagonal"]))
    if op.op == "cat":
        rank = len(env[op.inputs[0]].shape)
        axis = int(op.attrs["dim"]) % rank
        return mx.concatenate(
            [env[value_id] for value_id in op.inputs], axis=axis
        )
    if op.op in LAYOUT_OPS:
        return _evaluate_layout_mlx(graph, op, env[op.inputs[0]])
    if op.op == "matmul":
        left, right = (env[value_id] for value_id in op.inputs[:2])
        left = _apply_mlx_matmul_norm(op, env, left)
        if op.attrs.get("transpose_a", False):
            left = mx.swapaxes(left, -1, -2)
        if op.attrs.get("transpose_b", False):
            right = mx.swapaxes(right, -1, -2)
        product = mx.matmul(left, right)
        if _matmul_epilogue_kind(op) == "silu_mul_paired":
            paired_position = _matmul_paired_weight_position(op, 1)
            assert paired_position is not None
            paired = env[op.inputs[paired_position]]
            if op.attrs.get("transpose_b", False):
                paired = mx.swapaxes(paired, -1, -2)
            return product * (
                mx.matmul(left, paired) / (1.0 + mx.exp(-product))
            )
        return _apply_mlx_matmul_epilogue(op, env, product)
    if op.op == "addmm":
        bias, left, right = (env[value_id] for value_id in op.inputs[:3])
        left = _apply_mlx_matmul_norm(op, env, left)
        if op.attrs.get("transpose_a", False):
            left = mx.swapaxes(left, -1, -2)
        if op.attrs.get("transpose_b", False):
            right = mx.swapaxes(right, -1, -2)
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        return _apply_mlx_matmul_epilogue(
            op, env, beta * bias + alpha * mx.matmul(left, right)
        )
    return _evaluate_pointwise_mlx(
        op, env, graph.value(op.outputs[0]).type
    )


def _apply_mlx_matmul_norm(
    op: Operation, env: dict[str, Any], left: Any
) -> Any:
    """Normalize a fused RMS-norm operand before the fallback matrix multiply."""

    position = _matmul_norm_position(op)
    if position is None:
        return left
    weight = env[op.inputs[position]]
    eps = float(op.attrs["rms_norm_eps"])
    normalized = left.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(normalized * normalized, axis=-1, keepdims=True) + eps)
    return (normalized * scale * weight.astype(mx.float32)).astype(left.dtype)


def _apply_mlx_matmul_epilogue(
    op: Operation, env: dict[str, Any], product: Any
) -> Any:
    kind = op.attrs.get("epilogue")
    if kind is None:
        return product
    if kind != "silu_mul":
        raise ValueError(f"unsupported matmul epilogue {kind!r}")
    gate = env[op.inputs[_matmul_epilogue_position(op)]]
    return product * (gate / (1.0 + mx.exp(-gate)))


def _device_profile() -> tuple[str | None, str | None]:
    try:
        info = mx.device_info()
    except (RuntimeError, ValueError):
        return None, None
    return (
        None if "device_name" not in info else str(info["device_name"]),
        None
        if "architecture" not in info
        else str(info["architecture"]),
    )


def _synthetic_tune_input(tensor_type: TensorType) -> mx.array:
    dtype = _MLX_DTYPES[tensor_type.dtype]
    if tensor_type.dtype in {DType.F16, DType.BF16, DType.F32}:
        return mx.full(tensor_type.shape, 0.25, dtype=dtype)
    if tensor_type.dtype == DType.BOOL:
        return mx.ones(tensor_type.shape, dtype=dtype)
    return mx.zeros(tensor_type.shape, dtype=dtype)


def _coerce_tune_input(value: Any, tensor_type: TensorType) -> mx.array:
    if isinstance(value, mx.array):
        array = value
    else:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            host_value = value.detach().cpu()
            if str(host_value.dtype) == "torch.bfloat16":
                host_value = host_value.float()
            array = mx.array(host_value.numpy())
        else:
            array = mx.array(value)
    array = array.astype(_MLX_DTYPES[tensor_type.dtype])
    if tuple(array.shape) != tensor_type.shape:
        raise ValueError(
            f"autotune input has shape {tuple(array.shape)}, "
            f"expected {tensor_type.shape}"
        )
    return array


def _outputs_match(
    actual: mx.array, expected: mx.array, dtype: DType
) -> bool:
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        return False
    if dtype in {DType.BOOL, DType.I32, DType.I64}:
        result = mx.array_equal(actual, expected)
    else:
        # MLX and native kernels can use different accumulation schedules.
        # Keep the gate aligned with the transformer parity tolerance rather
        # than rejecting a numerically valid candidate for a few ULPs.
        atol, rtol = (2e-2, 2e-2)
        result = mx.allclose(actual, expected, atol=atol, rtol=rtol)
    mx.eval(result)
    return bool(result.item())


def _time_runtime(
    runtime: Any,
    *,
    warmup_iterations: int,
    iterations: int,
) -> tuple[float, ...]:
    for _ in range(warmup_iterations):
        mx.eval(runtime())
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        mx.eval(runtime())
        samples.append(time.perf_counter() - start)
    return tuple(samples)


def _validate_plan_inputs(
    graph: Graph, inputs: tuple[Any, ...]
) -> list[mx.array]:
    """Validate the caller's arrays against the graph signature."""

    if len(inputs) != len(graph.inputs):
        raise ValueError(
            f"expected {len(graph.inputs)} inputs, received {len(inputs)}"
        )
    values: list[mx.array] = []
    for value_id, value in zip(graph.inputs, inputs, strict=True):
        array = value if isinstance(value, mx.array) else mx.array(value)
        expected = graph.value(value_id).type
        if tuple(array.shape) != expected.shape:
            raise ValueError(
                f"input {value_id} has shape {tuple(array.shape)}, "
                f"expected {expected.shape}"
            )
        if array.dtype != _MLX_DTYPES[expected.dtype]:
            raise TypeError(
                f"input {value_id} has dtype {array.dtype}, "
                f"expected {expected.dtype.value}"
            )
        values.append(array)
    return values


def _bind_plan_op(
    graph: Graph, op: Operation, runtime: Any
) -> Callable[..., mx.array] | None:
    """Return a positional evaluator for one operation of a compiled plan.

    The generic interpreter resolves operands through a value-id dictionary.
    A compiled runner already knows the operand order, so the frequently used
    fallbacks get a positional closure and only the rare kinds fall back to the
    dictionary adapter.
    """

    if runtime is not None:
        return runtime
    kind = op.op
    if kind in LAYOUT_OPS:
        return lambda value, _op=op, _graph=graph: _evaluate_layout_mlx(
            _graph, _op, value
        )
    if kind == "cat":
        rank = len(graph.value(op.inputs[0]).type.shape)
        axis = int(op.attrs["dim"]) % rank
        return lambda *values, _axis=axis: mx.concatenate(values, axis=_axis)
    if kind == "embedding":
        return lambda weight, indices: _mlx_take_checked(weight, indices, axis=0)
    operand_ids = op.inputs

    def generic(*values: mx.array) -> mx.array:
        return _evaluate_mlx_op(graph, op, dict(zip(operand_ids, values)))

    return generic


def _compile_plan_runner(plan: CompiledPlan) -> Callable[..., Any] | None:
    """Turn the plan interpreter into straight-line Python, or return ``None``.

    Interpreting a thousand operations costs more than the kernels they launch:
    a decode step spent about a millisecond per token resolving value ids.
    Unrolling the graph once at first use replaces that bookkeeping with local
    variable assignments, and the kernels keep their original call signatures.
    """

    graph = plan.graph
    ops = graph.ops
    if any(len(op.outputs) != 1 for op in ops):
        return None

    names: dict[str, str] = {
        value_id: f"i{position}"
        for position, value_id in enumerate(graph.inputs)
    }
    namespace: dict[str, Any] = {}
    body: list[str] = []
    for index, op in enumerate(ops):
        arguments: list[str] = []
        for value_id in op.inputs:
            name = names.get(value_id)
            if name is None:
                return None
            arguments.append(name)
        thunk = _bind_plan_op(graph, op, plan._callables.get(op.outputs[0]))
        if thunk is None:
            return None
        thunk_name = f"t{index}"
        namespace[thunk_name] = thunk
        target = f"v{index}"
        names[op.outputs[0]] = target
        body.append(f"    {target} = {thunk_name}({', '.join(arguments)})")

    returned: list[str] = []
    for value_id in graph.outputs:
        name = names.get(value_id)
        if name is None:
            return None
        returned.append(name)
    body.append(f"    return ({', '.join(returned)},)")

    parameters = ", ".join(f"i{position}" for position in range(len(graph.inputs)))
    source = f"def run({parameters}):\n" + "\n".join(body) + "\n"
    exec(compile(source, f"<aotolith-plan-{len(ops)}>", "exec"), namespace)
    generated = namespace["run"]

    def runner(
        inputs: tuple[Any, ...], *, verify_index_bounds: bool = True
    ) -> tuple[mx.array, ...]:
        values = _validate_plan_inputs(graph, inputs)
        # Unlike the interpreter, the unrolled body keeps every intermediate
        # alive until it returns. That is the lifetime the async executor needs
        # anyway, so the storage plan's early releases only apply to the
        # statistics path, which still runs through ``_execute``.
        pending: list[tuple[mx.array, int, bool]] = []
        check_token = _PENDING_INDEX_CHECKS.set(pending)
        enabled_token = _INDEX_CHECKS_ENABLED.set(verify_index_bounds)
        try:
            outputs = generated(*values)
        finally:
            _INDEX_CHECKS_ENABLED.reset(enabled_token)
            _PENDING_INDEX_CHECKS.reset(check_token)
        _flush_index_checks(pending)
        return outputs

    return runner


_NO_RUNNER = object()


@dataclass(slots=True)
class CompiledPlan:
    graph: Graph
    kernels: tuple[KernelArtifact, ...]
    selections: tuple[KernelSelection, ...]
    _callables: dict[str, Any]
    storage_plan: StoragePlan
    _runner: Any = None

    def __call__(
        self, *inputs: Any, verify_index_bounds: bool = True
    ) -> tuple[mx.array, ...]:
        runner = self._runner
        if runner is None:
            compiled = _compile_plan_runner(self)
            self._runner = _NO_RUNNER if compiled is None else compiled
            runner = compiled
        if runner is None or runner is _NO_RUNNER:
            outputs, _ = self._execute(
                inputs,
                collect_stats=False,
                verify_index_bounds=verify_index_bounds,
            )
            return outputs
        return runner(inputs, verify_index_bounds=verify_index_bounds)

    def call_with_stats(
        self, *inputs: Any
    ) -> tuple[tuple[mx.array, ...], "ExecutionStats"]:
        """Execute the plan and report how the planned arena was used."""

        return self._execute(inputs, collect_stats=True)

    def _execute(
        self,
        inputs: tuple[Any, ...],
        *,
        collect_stats: bool,
        verify_index_bounds: bool = True,
    ) -> tuple[tuple[mx.array, ...], "ExecutionStats"]:
        env = dict(
            zip(
                self.graph.inputs,
                _validate_plan_inputs(self.graph, inputs),
                strict=True,
            )
        )

        slot_by_value = (
            {
                value_id: slot
                for slot in self.storage_plan.slots
                for value_id in slot.value_ids
            }
            if collect_stats
            else {}
        )
        slot_by_id = {slot.slot_id: slot for slot in self.storage_plan.slots}
        arena = (
            mx.zeros((self.storage_plan.arena_size_bytes,), dtype=mx.uint8)
            if collect_stats and self.storage_plan.arena_size_bytes
            else None
        )
        slot_assignments: list[ArenaAssignment] = []
        assigned_values: dict[int, str] = {}
        pending_index_checks: list[tuple[mx.array, int, bool]] = []

        def store_result(value_id: str, result: mx.array) -> None:
            if not collect_stats:
                env[value_id] = result
                return
            slot = slot_by_value.get(value_id)
            if slot is None:
                env[value_id] = result
                return
            if arena is None:
                raise RuntimeError("storage plan assigned a value without an arena")

            value_type = self.graph.value(value_id).type
            nbytes = _tensor_nbytes(value_type)
            if nbytes > slot.size_bytes:
                raise RuntimeError(
                    f"value {value_id} needs {nbytes} bytes but slot "
                    f"{slot.slot_id} has {slot.size_bytes}"
                )
            raw = mx.view(result.reshape((-1,)), mx.uint8)
            start = slot.offset_bytes
            arena[start : start + nbytes] = raw
            storage = mx.view(
                arena[start : start + nbytes], _MLX_DTYPES[value_type.dtype]
            ).reshape(value_type.shape)
            env[value_id] = storage
            if collect_stats:
                previous = assigned_values.get(slot.slot_id)
                slot_assignments.append(
                    ArenaAssignment(
                        value_id=value_id,
                        slot_id=slot.slot_id,
                        offset_bytes=slot.offset_bytes,
                        size_bytes=nbytes,
                        reused=previous is not None and previous != value_id,
                    )
                )
            assigned_values[slot.slot_id] = value_id

        check_token = _PENDING_INDEX_CHECKS.set(pending_index_checks)
        enabled_token = _INDEX_CHECKS_ENABLED.set(verify_index_bounds)
        try:
            for op_index, op in enumerate(self.graph.ops):
                runtime = self._callables.get(op.outputs[0])
                if runtime is not None:
                    result = runtime(*(env[value_id] for value_id in op.inputs))
                    store_result(op.outputs[0], result)
                    for value_id in self.storage_plan.release_after_op[op_index]:
                        env.pop(value_id, None)
                    continue

                result = _evaluate_mlx_op(self.graph, op, env)
                store_result(op.outputs[0], result)
                for value_id in self.storage_plan.release_after_op[op_index]:
                    env.pop(value_id, None)
        finally:
            _INDEX_CHECKS_ENABLED.reset(enabled_token)
            _PENDING_INDEX_CHECKS.reset(check_token)

        _flush_index_checks(pending_index_checks)


        stats = ExecutionStats(
            arena_size_bytes=self.storage_plan.arena_size_bytes,
            arena_slot_count=len(slot_by_id),
            assignments=tuple(slot_assignments),
            reuse_count=sum(item.reused for item in slot_assignments),
        )
        return tuple(env[value_id] for value_id in self.graph.outputs), stats


@dataclass(frozen=True, slots=True)
class ArenaAssignment:
    value_id: str
    slot_id: int
    offset_bytes: int
    size_bytes: int
    reused: bool


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    arena_size_bytes: int
    arena_slot_count: int
    assignments: tuple[ArenaAssignment, ...]
    reuse_count: int


@dataclass(frozen=True, slots=True)
class _NativeCandidate:
    artifact: KernelArtifact
    runtime: Any


def _tensor_core_candidates_load(
    graph: Graph, candidates: Mapping[str, _NativeCandidate]
) -> bool:
    """Try to load the emitted tensor-core kernels and evaluate them.

    `__HAVE_TENSOR__` answers whether the GPU has tensor ops; it does not answer
    whether the metallib will load. A paravirtualised GPU reports the macro and
    then rejects the kernel with "unsupported deferred-static-alloca-size
    function body", so this asks the driver by running the smallest thing that
    goes through the real kernels. Doing it here costs a launch per compile
    instead of an opaque `RuntimeError` on a user's first matmul.

    One probe per kernel kind and input dtype set: support is not guaranteed to
    be uniform, so a device that loads FP16 tensor ops and rejects FP32 ones is
    caught too. Within a group the smallest shapes are probed, because the
    dummy inputs are allocated at full size. Nothing is checked for
    correctness, only that the kernel loads and dispatches; if this wrongly
    fails, the plan still computes the right answer through the simdgroup path,
    and the tests that assert tensor-core source catch the throughput loss.
    """

    probes: dict[tuple[Any, ...], tuple[int, _NativeCandidate]] = {}
    for candidate in candidates.values():
        artifact = candidate.artifact
        if "tensorcore" not in artifact.name:
            continue
        types = [graph.value(value_id).type for value_id in artifact.input_ids]
        key = (artifact.kind, tuple(item.dtype for item in types))
        elements = sum(prod(item.shape) for item in types)
        smallest = probes.get(key)
        if smallest is None or elements < smallest[0]:
            probes[key] = (elements, candidate)

    for _, candidate in probes.values():
        try:
            inputs = [
                mx.zeros(
                    graph.value(value_id).type.shape,
                    dtype=_MLX_DTYPES[graph.value(value_id).type.dtype],
                )
                for value_id in candidate.artifact.input_ids
            ]
            # MLX is lazy, so `runtime(...)` alone never loads the kernel and
            # would report success for one the device rejects.
            mx.eval(*_as_arrays(candidate.runtime(*inputs)))
        except Exception as error:  # noqa: BLE001 - any failure means "no MPP".
            _block_tensor_cores(str(error))
            return False
    return True




_TENSOR_CORE_NOTE = "tensor cores rejected by this device"


def _annotate_tensor_core_fallback(
    selections: tuple[KernelSelection, ...],
) -> tuple[KernelSelection, ...]:
    """Say so when the plan is native but is not using tensor cores."""

    if _TENSOR_CORE_BLOCK_ERROR is None:
        return selections
    return tuple(
        replace(selection, reason=f"{selection.reason} ({_TENSOR_CORE_NOTE})")
        if selection.is_native
        else selection
        for selection in selections
    )

def _as_arrays(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (list, tuple)):
        return tuple(outputs)
    return (outputs,)


class AppleMetalBackend:
    """Lower supported IR regions to Metal kernels with an MLX runtime."""

    def __init__(
        self,
        *,
        codegen_matmul: bool = False,
        codegen_attention: bool = False,
        autotune: bool = False,
        autotune_warmup: int = _AUTOTUNE_WARMUP,
        autotune_iterations: int = _AUTOTUNE_ITERATIONS,
        selection_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if autotune_warmup < 0:
            raise ValueError("autotune_warmup must be non-negative")
        if autotune_iterations <= 0:
            raise ValueError("autotune_iterations must be positive")
        self.codegen_matmul = codegen_matmul
        self.codegen_attention = codegen_attention
        self.autotune = autotune
        self.autotune_warmup = autotune_warmup
        self.autotune_iterations = autotune_iterations
        self.selection_overrides = dict(selection_overrides or {})

    def _lower_candidates(self, graph: Graph) -> dict[str, _NativeCandidate]:
        """Emit one native candidate per operation this backend can lower."""

        candidates: dict[str, _NativeCandidate] = {}
        for index, op in enumerate(graph.ops):
            compiled: tuple[KernelArtifact, Any] | None = None
            if op.op == "fused_elementwise":
                body = op.attrs.get("body")
                if not isinstance(body, tuple) or not all(
                    isinstance(item, Operation) for item in body
                ):
                    raise ValueError(
                        "fused_elementwise body must be a tuple of operations"
                    )
                compiled = self._compile_elementwise(
                    graph, op.inputs, body, op.outputs[0], index, "fused_elementwise"
                )
            elif op.op == "arange":
                compiled = self._compile_arange(graph, op, index)
            elif (
                op.op in _CONSTANT_POINTWISE_OPS
                and graph.value(op.outputs[0]).type.dtype in _INTEGRAL_DTYPES
            ):
                compiled = self._compile_integer_constant(graph, op, index)
            elif op.op in POINTWISE_OPS:
                compiled = self._compile_elementwise(
                    graph, op.inputs, (op,), op.outputs[0], index, op.op
                )
            elif op.op in {"softmax", "safe_softmax"}:
                compiled = self._compile_softmax(graph, op, index)
            elif op.op in {"sum", "mean"}:
                compiled = self._compile_reduction(graph, op, index)
            elif op.op == "layer_norm":
                compiled = self._compile_layer_norm(graph, op, index)
            elif op.op == "rms_norm":
                compiled = self._compile_rms_norm(graph, op, index)
            elif op.op == "rope":
                _validate_rope_op(graph, op)
                # The vectorized kernel owns four neighbouring columns, so both
                # the head dimension and its half must be multiples of four.
                if graph.value(op.outputs[0]).type.shape[-1] % 8 == 0:
                    compiled = self._compile_rope(graph, op, index)
            elif op.op == "attention":
                _validate_attention_op(graph, op)
                if self.codegen_attention or self.autotune or self._wants_native(op):
                    compiled = self._compile_attention(graph, op, index)
                else:
                    # A single-query attention needs one threadgroup per head
                    # and no online rescaling, so it beats the generic runtime
                    # kernel without the prefill-oriented codegen switch.
                    compiled = self._compile_decode_attention(
                        graph, op, index
                    )
            elif op.op == "cat":
                _validate_cat_op(graph, op)
            elif op.op == "embedding":
                _validate_embedding_op(graph, op)
            elif op.op == "index_select":
                _validate_index_select_op(graph, op)
            elif op.op == "index":
                _validate_index_op(graph, op)
            elif op.op == "gather":
                _validate_gather_op(graph, op)
            elif op.op == "scatter":
                _validate_scatter_op(graph, op)
            elif op.op == "cumsum":
                _validate_cumsum_op(graph, op)
            elif op.op in {"triu", "tril"}:
                _validate_triangular_op(graph, op)
            elif op.op == "cast":
                _validate_cast_op(graph, op)
            elif op.op in LAYOUT_OPS:
                _validate_layout_op(graph, op)
            elif (
                self.codegen_matmul or self.autotune or self._wants_native(op)
            ) and op.op in {"matmul", "addmm"}:
                left_position = 1 if op.op == "addmm" else 0
                right_position = 2 if op.op == "addmm" else 1
                left_type = graph.value(op.inputs[left_position]).type
                right_type = graph.value(op.inputs[right_position]).type
                output_type = graph.value(op.outputs[0]).type
                batch_shape = _matmul_batch_shape(
                    left_type.shape, right_type.shape
                )
                if (
                    batch_shape is not None
                    and output_type.shape[:-2] == batch_shape
                ):
                    compiled = self._compile_matmul(graph, op, index)

            if compiled is not None:
                artifact, runtime = compiled
                candidates[artifact.output_id] = _NativeCandidate(
                    artifact, runtime
                )
        return candidates

    def compile(
        self,
        graph: Graph,
        *,
        tune_inputs: Sequence[Any] | None = None,
    ) -> CompiledPlan:
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal is unavailable in this process")

        candidates = self._lower_candidates(graph)
        if _mpp_supported() and not _tensor_core_candidates_load(
            graph, candidates
        ):
            # This device compiled the tensor-core kernels and refused to
            # load them, so re-lower once without them rather than leaving
            # a plan that raises on its first matmul.
            candidates = self._lower_candidates(graph)

        if self.autotune and not self.selection_overrides:
            kernels, selections, callables = self._autotune(
                graph, candidates, tune_inputs
            )
        else:
            kernels, selections, callables = self._heuristic_select(
                graph, candidates
            )

        return CompiledPlan(
            graph,
            kernels,
            _annotate_tensor_core_fallback(selections),
            callables,
            plan_storage(graph),
        )

    def _wants_native(self, op: Operation) -> bool:
        return self.selection_overrides.get(op.outputs[0]) == "metal_kernel"

    def _heuristic_select(
        self,
        graph: Graph,
        candidates: Mapping[str, _NativeCandidate],
    ) -> tuple[
        tuple[KernelArtifact, ...], tuple[KernelSelection, ...], dict[str, Any]
    ]:
        kernels: list[KernelArtifact] = []
        selections: list[KernelSelection] = []
        callables: dict[str, Any] = {}
        device_name, device_architecture = _device_profile()
        for index, op in enumerate(graph.ops):
            output_id = op.outputs[0]
            candidate = candidates.get(output_id)
            override = self.selection_overrides.get(output_id)
            if override not in {None, "metal_kernel", "mlx"}:
                raise ValueError(
                    f"unknown kernel implementation override {override!r} "
                    f"for {output_id}"
                )
            if override == "metal_kernel" and candidate is None:
                raise ValueError(
                    f"artifact selected Metal for unsupported operation {output_id}"
                )
            use_native = override == "metal_kernel" or (
                override is None and candidate is not None
            )
            if use_native:
                assert candidate is not None
                kernels.append(candidate.artifact)
                callables[output_id] = candidate.runtime
                selection = KernelSelection(
                    op_index=index,
                    op=op.op,
                    output_id=output_id,
                    implementation="metal_kernel",
                    kind=candidate.artifact.kind,
                    reason=(
                        "native Metal kernel selected"
                        if override is None
                        else "serialized artifact selected native Metal kernel"
                    ),
                    artifact_name=candidate.artifact.name,
                    device_name=device_name,
                    device_architecture=device_architecture,
                )
            else:
                selection = KernelSelection(
                    op_index=index,
                    op=op.op,
                    output_id=output_id,
                    implementation="mlx",
                    kind=op.op,
                    reason=(
                        self._fallback_reason(graph, op)
                        if override is None
                        else "serialized artifact selected MLX fallback"
                    ),
                    device_name=device_name,
                    device_architecture=device_architecture,
                )
            selections.append(selection)
        return tuple(kernels), tuple(selections), callables

    def _autotune(
        self,
        graph: Graph,
        candidates: Mapping[str, _NativeCandidate],
        tune_inputs: Sequence[Any] | None,
    ) -> tuple[
        tuple[KernelArtifact, ...], tuple[KernelSelection, ...], dict[str, Any]
    ]:
        if tune_inputs is None:
            env = {
                value_id: _synthetic_tune_input(graph.value(value_id).type)
                for value_id in graph.inputs
            }
        else:
            if len(tune_inputs) != len(graph.inputs):
                raise ValueError(
                    f"autotune expected {len(graph.inputs)} inputs, "
                    f"received {len(tune_inputs)}"
                )
            env = {
                value_id: _coerce_tune_input(value, graph.value(value_id).type)
                for value_id, value in zip(
                    graph.inputs, tune_inputs, strict=True
                )
            }

        device_name, device_architecture = _device_profile()
        kernels: list[KernelArtifact] = []
        selections: list[KernelSelection] = []
        callables: dict[str, Any] = {}
        for index, op in enumerate(graph.ops):
            output_id = op.outputs[0]
            output_type = graph.value(output_id).type
            input_arrays = tuple(env[value_id] for value_id in op.inputs)

            def fallback_runtime() -> mx.array:
                return _evaluate_mlx_op(graph, op, env)

            candidate = candidates.get(output_id)
            try:
                fallback_result = fallback_runtime()
                mx.eval(fallback_result)
            except NotImplementedError:
                # Some ops only ever lower to a generated kernel: normalization,
                # rotary and attention have no MLX equivalent once they are
                # fused. There is nothing to compare against, so the kernel is
                # measured on its own and its own output feeds the next op.
                if candidate is None:
                    raise
                fallback_result = None

            if candidate is None:
                assert fallback_result is not None
                selections.append(
                    KernelSelection(
                        op_index=index,
                        op=op.op,
                        output_id=output_id,
                        implementation="mlx",
                        kind=op.op,
                        reason=self._fallback_reason(graph, op),
                        device_name=device_name,
                        device_architecture=device_architecture,
                    )
                )
                env[output_id] = fallback_result
                continue

            if fallback_result is None:
                native_result = candidate.runtime(*input_arrays)
                mx.eval(native_result)
                native_benchmark = self._benchmark(
                    "metal_kernel",
                    candidate.artifact.name,
                    lambda: candidate.runtime(*input_arrays),
                )
                mlx_benchmark = None
                env[output_id] = native_result
            else:
                mlx_benchmark = self._benchmark(
                    "mlx", None, fallback_runtime
                )
                try:
                    native_result = candidate.runtime(*input_arrays)
                    mx.eval(native_result)
                    if not _outputs_match(
                        native_result, fallback_result, output_type.dtype
                    ):
                        native_benchmark = KernelBenchmark(
                            candidate="metal_kernel",
                            artifact_name=candidate.artifact.name,
                            warmup_iterations=self.autotune_warmup,
                            iterations=self.autotune_iterations,
                            samples_seconds=(),
                            median_seconds=None,
                            min_seconds=None,
                            error="output differs from the MLX reference",
                        )
                    else:
                        native_benchmark = self._benchmark(
                            "metal_kernel",
                            candidate.artifact.name,
                            lambda: candidate.runtime(*input_arrays),
                        )
                except Exception as exc:
                    native_benchmark = KernelBenchmark(
                        candidate="metal_kernel",
                        artifact_name=candidate.artifact.name,
                        warmup_iterations=self.autotune_warmup,
                        iterations=self.autotune_iterations,
                        samples_seconds=(),
                        median_seconds=None,
                        min_seconds=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                env[output_id] = fallback_result

            native_seconds = native_benchmark.median_seconds
            mlx_seconds = (
                None if mlx_benchmark is None else mlx_benchmark.median_seconds
            )
            use_native = native_seconds is not None and (
                mlx_seconds is None or native_seconds <= mlx_seconds
            )
            if use_native:
                kernels.append(candidate.artifact)
                callables[output_id] = candidate.runtime
                implementation = "metal_kernel"
                kind = candidate.artifact.kind
                artifact_name: str | None = candidate.artifact.name
                reason = self._measured_reason(
                    "native Metal kernel", native_seconds, mlx_seconds
                )
            else:
                implementation = "mlx"
                kind = op.op
                artifact_name = None
                reason = self._measured_reason(
                    "MLX fallback", mlx_seconds, native_seconds
                )
                env[output_id] = fallback_result

            benchmarks = (native_benchmark,)
            if mlx_benchmark is not None:
                benchmarks = (native_benchmark, mlx_benchmark)
            selections.append(
                KernelSelection(
                    op_index=index,
                    op=op.op,
                    output_id=output_id,
                    implementation=implementation,
                    kind=kind,
                    reason=reason,
                    artifact_name=artifact_name,
                    benchmarks=benchmarks,
                    autotuned=True,
                    device_name=device_name,
                    device_architecture=device_architecture,
                )
            )

        return tuple(kernels), tuple(selections), callables

    def _benchmark(
        self,
        candidate: str,
        artifact_name: str | None,
        runtime: Any,
    ) -> KernelBenchmark:
        try:
            samples = _time_runtime(
                runtime,
                warmup_iterations=self.autotune_warmup,
                iterations=self.autotune_iterations,
            )
        except Exception as exc:
            return KernelBenchmark(
                candidate=candidate,
                artifact_name=artifact_name,
                warmup_iterations=self.autotune_warmup,
                iterations=self.autotune_iterations,
                samples_seconds=(),
                median_seconds=None,
                min_seconds=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        return KernelBenchmark(
            candidate=candidate,
            artifact_name=artifact_name,
            warmup_iterations=self.autotune_warmup,
            iterations=self.autotune_iterations,
            samples_seconds=samples,
            median_seconds=median(samples),
            min_seconds=min(samples),
        )

    @staticmethod
    def _measured_reason(
        winner: str,
        winner_seconds: float | None,
        alternative_seconds: float | None,
    ) -> str:
        if winner_seconds is None:
            return "autotune failed for every candidate; MLX fallback retained"
        if alternative_seconds is None:
            return f"autotuned: {winner} was the only valid candidate"
        return (
            f"autotuned: {winner} was faster "
            f"({winner_seconds * 1e3:.4f} ms vs "
            f"{alternative_seconds * 1e3:.4f} ms median)"
        )

    def _fallback_reason(self, graph: Graph, op: Operation) -> str:
        if op.op == "attention" and not self.codegen_attention:
            return "codegen_attention disabled; using MLX scaled dot-product attention"
        if op.op == "attention":
            return "attention shape or mask is outside native Metal constraints"
        if op.op in {"matmul", "addmm"} and not self.codegen_matmul:
            return "codegen_matmul disabled; using MLX matrix multiplication"
        if op.op in {"matmul", "addmm"}:
            return "matmul shape is outside native Metal constraints"
        if op.op == "cast":
            return "dtype conversion uses MLX astype"
        if op.op in LAYOUT_OPS:
            return "layout operation uses an MLX view or materialization"
        return "operation uses the MLX runtime implementation"

    def _compile_elementwise(
        self,
        graph: Graph,
        input_ids: tuple[str, ...],
        body: tuple[Operation, ...],
        output_id: str,
        index: int,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        output_type = graph.value(output_id).type
        _validate_pointwise_region(graph, body, output_id)
        for value_id in input_ids:
            input_type = graph.value(value_id).type
            if not _can_broadcast(input_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast pointwise input {input_type.shape} "
                    f"to {output_type.shape}"
                )

        source = _emit_pointwise_source(
            graph, input_ids, body, output_id, output_type.shape
        )
        name = f"{kind}_{index}"
        kernel = _metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(input_ids))),
            ("out",),
            source,
            _MSL_HEADER,
        )

        element_count = prod(output_type.shape)
        grid = (ceil(element_count / _ELEMENTS_PER_THREAD), 1, 1)
        threadgroup = (min(256, grid[0]), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MSL_HEADER,
            input_ids=input_ids,
            output_id=output_id,
            tensor_type=output_type,
            kind=kind,
        )
        return artifact, runtime

    def _compile_integer_constant(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        _validate_integer_constant(graph, op)
        output_type = graph.value(op.outputs[0]).type
        source = _emit_integer_constant_source(op, output_type)
        return self._compile_zero_input_kernel(op, index, source, output_type)

    def _compile_arange(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        _validate_arange(graph, op)
        output_type = graph.value(op.outputs[0]).type
        source = _emit_arange_source(op, output_type)
        return self._compile_zero_input_kernel(op, index, source, output_type)

    def _compile_zero_input_kernel(
        self,
        op: Operation,
        index: int,
        source: str,
        output_type: TensorType,
    ) -> tuple[KernelArtifact, Any]:
        name = f"{op.op}_{index}"
        kernel = _metal_kernel(name, (), ("out",), source, _MSL_HEADER)

        element_count = prod(output_type.shape)
        grid = (ceil(element_count / _ELEMENTS_PER_THREAD), 1, 1)
        threadgroup = (min(256, grid[0]), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime() -> mx.array:
            outputs = kernel(
                inputs=[],
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MSL_HEADER,
            input_ids=(),
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime

    def _compile_mpp_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        query_type = graph.value(op.inputs[0]).type
        key_type = graph.value(op.inputs[1]).type
        value_type = graph.value(op.inputs[2]).type
        output_type = graph.value(op.outputs[0]).type
        has_mask = bool(op.attrs.get("has_mask"))
        mask_type = graph.value(op.inputs[3]).type if has_mask else None
        mask_shape = mask_type.shape if mask_type is not None else None
        if not _supports_mpp_attention(
            query_type, key_type, value_type, output_type, mask_shape
        ):
            return None
        if not _mpp_supported():
            return None

        rank = len(query_type.shape)
        query_length = query_type.shape[-2]
        key_length = key_type.shape[-2]
        query_heads = query_type.shape[-3] if rank == 4 else 1
        key_value_heads = key_type.shape[-3] if rank == 4 else 1
        query_head_dim = query_type.shape[-1]
        value_head_dim = value_type.shape[-1]
        warps = _select_mpp_attention_warps(query_length)
        query_tile = warps * 16
        query_blocks = ceil(query_length / query_tile)
        batch = query_type.shape[0]
        threads = warps * 32
        mask_is_boolean = mask_type is not None and mask_type.dtype == DType.BOOL

        source = _emit_mpp_attention_source(
            query_length=query_length,
            key_length=key_length,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            query_head_dim=query_head_dim,
            value_head_dim=value_head_dim,
            warps=warps,
            scale=float(op.attrs.get("scale", 1.0)),
            has_mask=has_mask,
            mask_shape=mask_shape,
            mask_is_boolean=mask_is_boolean,
            causal=bool(op.attrs.get("causal")),
            rank=rank,
        )

        input_names = (
            ("in_0", "in_1", "in_2", "in_3")
            if has_mask
            else ("in_0", "in_1", "in_2")
        )
        name = (
            f"metal_attention_tensorcore_{query_tile}x{_MPP_ATTENTION_KEY_TILE}_"
            f"{threads}_{index}"
        )
        kernel = _metal_kernel(
            name,
            input_names,
            ("out",),
            source,
            _MPP_HEADER,
        )
        query_count = batch * query_heads * query_blocks
        grid = (query_count * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]
        template = [("T", dtype)]
        output_shapes = [output_type.shape]
        output_dtypes = [dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            return kernel(
                inputs=inputs,
                template=template,
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MPP_HEADER,
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind="attention",
        )
        return artifact, runtime

    def _compile_decode_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        """Emit the single-query attention kernel when the shape allows it."""

        config = _decode_attention_config(graph, op)
        if config is None:
            return None
        output_type = graph.value(op.outputs[0]).type
        source = _emit_decode_attention_source(config)
        name = (
            f"metal_decode_attention_{config.query_heads}"
            f"_{config.key_value_heads}_{config.key_length}"
            f"_{config.query_head_dim}_{config.value_head_dim}_{index}"
        )
        input_names = tuple(
            f"in_{position}" for position in range(len(op.inputs))
        )
        kernel = _metal_kernel(
            name,
            input_names,
            ("out",),
            source,
            _MSL_HEADER,
        )
        dtype = _MLX_DTYPES[config.dtype]
        # One threadgroup per (head, batch) pair: grid.y is the batch count, not
        # batch * threads, otherwise the runtime launches threads duplicate
        # threadgroups that write past the end of the output buffer.
        grid = (
            config.query_heads * config.threads,
            config.batch,
            1,
        )
        threadgroup = (config.threads, 1, 1)

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MSL_HEADER,
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind="attention",
        )
        return artifact, runtime

    def _compile_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        """Emit a FlashAttention-style tiled online-softmax kernel."""

        decode = self._compile_decode_attention(graph, op, index)
        if decode is not None:
            return decode

        tensor_core = self._compile_mpp_attention(graph, op, index)
        if tensor_core is not None:
            return tensor_core

        query_type = graph.value(op.inputs[0]).type
        key_type = graph.value(op.inputs[1]).type
        value_type = graph.value(op.inputs[2]).type
        output_type = graph.value(op.outputs[0]).type
        query_shape = query_type.shape
        key_shape = key_type.shape
        value_shape = value_type.shape
        rank = len(query_shape)
        query_head_dim = query_shape[-1]
        value_head_dim = value_shape[-1]
        if query_head_dim > 256 or value_head_dim > 256:
            return None

        has_mask = bool(op.attrs.get("has_mask"))
        mask_type = graph.value(op.inputs[3]).type if has_mask else None
        mask_shape = mask_type.shape if mask_type is not None else None
        mask_is_boolean = mask_type is not None and mask_type.dtype == DType.BOOL
        if has_mask and not _native_attention_mask_supported(rank, mask_shape):
            return None

        batch, query_length, key_length = query_shape[0], query_shape[-2], key_shape[-2]
        query_heads = query_shape[-3] if rank == 4 else 1
        key_value_heads = key_shape[-3] if rank == 4 else 1
        group_size = query_heads // key_value_heads
        key_tile = 32
        low_precision_attention = output_type.dtype in {DType.F16, DType.BF16}
        use_simdgroup_qk = (
            low_precision_attention
            and query_head_dim % 8 == 0
            and query_length >= 8
        )
        use_simdgroup_pv = (
            low_precision_attention
            and value_head_dim % 8 == 0
            and query_length >= 8
        )
        query_tile = 1 if query_length == 1 else 8
        if use_simdgroup_qk and use_simdgroup_pv and query_length > 8:
            candidates = ((16, 32),)
            if query_length >= _ATTENTION_LONG_SEQUENCE_THRESHOLD:
                candidates = ((24, 64), (16, 64), (24, 32), (16, 32))
            for candidate_query_tile, candidate_key_tile in candidates:
                candidate_pv_value_tile = (
                    min(value_head_dim, 64)
                    if candidate_key_tile == 64
                    else value_head_dim
                )
                shared_bytes = _attention_shared_memory_bytes(
                    candidate_query_tile,
                    candidate_key_tile,
                    query_head_dim,
                    value_head_dim,
                    candidate_pv_value_tile,
                )
                if shared_bytes <= _ATTENTION_MAX_SHARED_BYTES:
                    query_tile = candidate_query_tile
                    key_tile = candidate_key_tile
                    break
        pv_value_tile = min(value_head_dim, 64) if key_tile == 64 else value_head_dim
        threads = query_tile * 32
        query_blocks = ceil(query_length / query_tile)
        query_count = batch * query_heads * query_blocks

        if rank == 4:
            query_batch_head = f"(batch * {query_heads}u + head)"
            key_value_batch_head = (
                f"(batch * {key_value_heads}u + head / {group_size}u)"
            )
        else:
            query_batch_head = "batch"
            key_value_batch_head = "batch"

        if use_simdgroup_qk and use_simdgroup_pv and key_tile == 64:
            tile_alignment = "alignas(32) "
            load_vector_width = (
                16
                if query_head_dim % 16 == 0 and value_head_dim % 16 == 0
                else 8
            )
            query_load_source = f"""
        for (uint vector_index = tid;
             vector_index < {query_tile * query_head_dim // load_vector_width}u;
             vector_index += {threads}u) {{
            uint index = vector_index * {load_vector_width}u;
            uint query_row = index / {query_head_dim}u;
            uint dimension = index % {query_head_dim}u;
            uint position = query_block * {query_tile}u + query_row;
            threadgroup vec<T, {load_vector_width}>* destination =
                reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(
                    tile_q + index);
            if (position < {query_length}u) {{
                const device vec<T, {load_vector_width}>* source =
                    reinterpret_cast<const device vec<T, {load_vector_width}>*>(
                        in_0 + ({query_batch_head} * {query_length}u + position)
                            * {query_head_dim}u + dimension);
                *destination = *source;
            }} else {{
                *destination = vec<T, {load_vector_width}>(T(0));
            }}
        }}
"""
            key_load_source = f"""
            for (uint vector_index = tid;
                 vector_index < {key_tile * query_head_dim // load_vector_width}u;
                 vector_index += {threads}u) {{
                uint index = vector_index * {load_vector_width}u;
                uint key_row = index / {query_head_dim}u;
                uint dimension = index % {query_head_dim}u;
                uint key = key_base + key_row;
                threadgroup vec<T, {load_vector_width}>* destination =
                    reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(
                        tile_k + index);
                if (key < {key_length}u) {{
                    const device vec<T, {load_vector_width}>* source =
                        reinterpret_cast<const device vec<T, {load_vector_width}>*>(
                            in_1 + ({key_value_batch_head} * {key_length}u + key)
                                * {query_head_dim}u + dimension);
                    *destination = *source;
                }} else {{
                    *destination = vec<T, {load_vector_width}>(T(0));
                }}
            }}
"""
            value_load_source = f"""
            for (uint vector_index = tid;
                 vector_index < {key_tile * value_head_dim // load_vector_width}u;
                 vector_index += {threads}u) {{
                uint index = vector_index * {load_vector_width}u;
                uint key_row = index / {value_head_dim}u;
                uint dimension = index % {value_head_dim}u;
                uint global_key = key_base + key_row;
                threadgroup vec<T, {load_vector_width}>* destination =
                    reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(
                        tile_k + index);
                if (global_key < {key_length}u) {{
                    const device vec<T, {load_vector_width}>* source =
                        reinterpret_cast<const device vec<T, {load_vector_width}>*>(
                            in_2 + ({key_value_batch_head} * {key_length}u
                                + global_key) * {value_head_dim}u + dimension);
                    *destination = *source;
                }} else {{
                    *destination = vec<T, {load_vector_width}>(T(0));
                }}
            }}
"""
        else:
            tile_alignment = ""
            query_load_source = f"""
        for (uint index = tid; index < {query_tile * query_head_dim}u;
             index += {threads}u) {{
            uint query_row = index / {query_head_dim}u;
            uint dimension = index % {query_head_dim}u;
            uint position = query_block * {query_tile}u + query_row;
            tile_q[index] = position < {query_length}u
                ? in_0[({query_batch_head} * {query_length}u + position)
                    * {query_head_dim}u + dimension]
                : T(0);
        }}
"""
            key_load_source = f"""
            for (uint index = tid; index < {key_tile * query_head_dim}u;
                 index += {threads}u) {{
                uint key_row = index / {query_head_dim}u;
                uint dimension = index % {query_head_dim}u;
                uint key = key_base + key_row;
                tile_k[index] = key < {key_length}u
                    ? in_1[({key_value_batch_head} * {key_length}u + key)
                        * {query_head_dim}u + dimension]
                    : T(0);
            }}
"""
            value_load_source = f"""
            for (uint index = tid; index < {key_tile * value_head_dim}u;
                 index += {threads}u) {{
                uint key_row = index / {value_head_dim}u;
                uint dimension = index % {value_head_dim}u;
                uint global_key = key_base + key_row;
                tile_k[index] = global_key < {key_length}u
                    ? in_2[({key_value_batch_head} * {key_length}u + global_key)
                        * {value_head_dim}u + dimension]
                    : T(0);
            }}
"""

        score_expression = f"dot * {float(op.attrs['scale'])!r}"
        if has_mask:
            mask_offset = _attention_mask_offset_expression(rank, mask_shape)
            if mask_is_boolean:
                score_expression = (
                    f"in_3[{mask_offset}] ? ({score_expression}) : -INFINITY"
                )
            else:
                score_expression += f" + float(in_3[{mask_offset}])"
        if op.attrs.get("causal"):
            score_expression = (
                f"key > query_pos ? -INFINITY : ({score_expression})"
            )

        if use_simdgroup_qk:
            simdgroup_type = {
                DType.F16: "half",
                DType.BF16: "bfloat",
            }[output_type.dtype]
            query_fragments = query_tile // 8
            key_fragments = key_tile // 8
            qk_group_count = query_fragments * key_fragments
            qk_steps = "\n".join(
                f"""                simdgroup_load(qk_frag_q, tile_q
                    + query_fragment * 8u * {query_head_dim}u + {dimension}u,
                    {query_head_dim}, 0, false);
                simdgroup_load(qk_frag_k, tile_k
                    + key_fragment * 8u * {query_head_dim}u + {dimension}u,
                    {query_head_dim}, 0, true);
                simdgroup_multiply_accumulate(
                    qk_accumulator, qk_frag_q, qk_frag_k, qk_accumulator);"""
                for dimension in range(0, query_head_dim, 8)
            )
            if key_tile == 32:
                qk_score_read_source = f"""
            uint key = key_base + lane;
            float score = -INFINITY;
            if (query_pos < {query_length}u && key < {key_length}u) {{
                float dot = float(tile_scores[warp * {key_tile}u + lane]);
                score = {score_expression};
            }}
"""
            else:
                qk_score_read_source = f"""
            uint key0 = key_base + lane;
            float score0 = -INFINITY;
            if (query_pos < {query_length}u && key0 < {key_length}u) {{
                float dot = float(tile_scores[warp * {key_tile}u + lane]);
                uint key = key0;
                score0 = {score_expression};
            }}

            uint key1 = key0 + 32u;
            float score1 = -INFINITY;
            if (query_pos < {query_length}u && key1 < {key_length}u) {{
                float dot = float(tile_scores[warp * {key_tile}u + lane + 32u]);
                uint key = key1;
                score1 = {score_expression};
            }}
"""
            qk_score_source = f"""
            if (warp < {qk_group_count}u) {{
                uint query_fragment = warp / {key_fragments}u;
                uint key_fragment = warp % {key_fragments}u;
                simdgroup_float8x8 qk_accumulator =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_{simdgroup_type}8x8 qk_frag_q;
                simdgroup_{simdgroup_type}8x8 qk_frag_k;
{qk_steps}
                simdgroup_store(qk_accumulator,
                    tile_scores + query_fragment * 8u * {key_tile}u
                        + key_fragment * 8u,
                    {key_tile}, 0, false);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
{qk_score_read_source}
"""
        else:
            qk_score_source = f"""
            uint key = key_base + lane;
            float score = -INFINITY;
            if (query_pos < {query_length}u && key < {key_length}u) {{
                float dot = 0.0f;
                for (uint dimension = 0u; dimension < {query_head_dim}u;
                     ++dimension) {{
                    dot = fma(float(tile_q[warp * {query_head_dim}u + dimension]),
                        float(tile_k[lane * {query_head_dim}u + dimension]), dot);
                }}
                score = {score_expression};
            }}
"""
        if use_simdgroup_pv:
            simdgroup_type = {
                DType.F16: "half",
                DType.BF16: "bfloat",
            }[output_type.dtype]
            query_fragments = query_tile // 8
            value_fragments = value_head_dim // 8
            pv_probability_declaration = ""
            if key_tile == 64:
                probability_storage = "tile_probabilities_low"
                pv_shared_declaration = ""
                pv_probability_store = f"""
            tile_probabilities_low[warp * {key_tile}u + lane] =
                (T)probability0;
            tile_probabilities_low[warp * {key_tile}u + lane + 32u] =
                (T)probability1;"""
            else:
                probability_storage = "tile_probabilities_low"
                pv_shared_declaration = f"""
        threadgroup T tile_probabilities_low[{query_tile * key_tile}];
        threadgroup float tile_pv[{query_tile * pv_value_tile}];"""
                pv_probability_store = f"""
            tile_probabilities_low[warp * {key_tile}u + lane] = (T)probability;"""
            pv_preload_source = value_load_source
            if key_tile == 64:
                value_fragments_per_chunk = pv_value_tile // 8
                pv_steps = "\n".join(
                    f"""                    simdgroup_load(pv_frag_probability,
                        {probability_storage} + output_row_fragment * 8u
                            * {key_tile}u + {tile_key}u,
                        {key_tile}, 0, false);
                    simdgroup_load(pv_frag_value, tile_k
                        + {tile_key * value_head_dim}u
                        + value_chunk_base
                        + output_column_fragment * 8u,
                        {value_head_dim}, 0, false);
                    simdgroup_multiply_accumulate(
                        pv_accumulator, pv_frag_probability, pv_frag_value,
                        pv_accumulator);"""
                    for tile_key in range(0, key_tile, 8)
                )
                pv_accumulation_source = f"""
            for (uint value_chunk_base = 0u;
                 value_chunk_base < {value_head_dim}u;
                 value_chunk_base += {pv_value_tile}u) {{
                uint value_chunk_fragments = metal::min(
                    {value_fragments_per_chunk}u,
                    ({value_head_dim}u - value_chunk_base) / 8u);
                for (uint output_fragment = warp;
                     output_fragment <
                         {query_fragments}u * value_chunk_fragments;
                     output_fragment += {threads // 32}u) {{
                    uint output_row_fragment =
                        output_fragment / value_chunk_fragments;
                    uint output_column_fragment =
                        output_fragment % value_chunk_fragments;
                    simdgroup_float8x8 pv_accumulator =
                        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    simdgroup_{simdgroup_type}8x8 pv_frag_probability;
                    simdgroup_{simdgroup_type}8x8 pv_frag_value;
{pv_steps}
                    simdgroup_store(pv_accumulator,
                        tile_pv + output_row_fragment * 8u * {pv_value_tile}u
                            + output_column_fragment * 8u,
                        {pv_value_tile}, 0, false);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint component = 0u;
                     component < {ceil(value_head_dim / 32)}u; ++component) {{
                    uint dimension = lane + component * 32u;
                    if (query_pos < {query_length}u
                        && dimension >= value_chunk_base
                        && dimension < value_chunk_base + {pv_value_tile}u
                        && dimension < {value_head_dim}u) {{
                        accumulator[component] = accumulator[component] * alpha
                            + block_scale * tile_pv[warp * {pv_value_tile}u
                                + dimension - value_chunk_base];
                    }}
                }}
                if (value_chunk_base + {pv_value_tile}u < {value_head_dim}u) {{
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}
            }}
"""
            else:
                pv_fragment_count = query_fragments * value_fragments
                pv_steps = "\n".join(
                    f"""                simdgroup_load(pv_frag_probability,
                    {probability_storage} + output_row_fragment * 8u
                        * {key_tile}u + {tile_key}u, {key_tile}, 0, false);
                simdgroup_load(pv_frag_value, tile_k
                    + {tile_key * value_head_dim}u
                    + output_column_fragment * 8u,
                    {value_head_dim}, 0, false);
                simdgroup_multiply_accumulate(
                    pv_accumulator, pv_frag_probability, pv_frag_value,
                    pv_accumulator);"""
                    for tile_key in range(0, key_tile, 8)
                )
                pv_accumulation_source = f"""
            for (uint output_fragment = warp;
                 output_fragment < {pv_fragment_count}u;
                 output_fragment += {query_tile}u) {{
                uint output_row_fragment = output_fragment / {value_fragments}u;
                uint output_column_fragment = output_fragment % {value_fragments}u;
                simdgroup_float8x8 pv_accumulator =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_{simdgroup_type}8x8 pv_frag_probability;
                simdgroup_{simdgroup_type}8x8 pv_frag_value;
{pv_steps}
                simdgroup_store(pv_accumulator,
                    tile_pv + output_row_fragment * 8u * {pv_value_tile}u
                        + output_column_fragment * 8u,
                    {pv_value_tile}, 0, false);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint component = 0u;
                 component < {ceil(value_head_dim / 32)}u; ++component) {{
                uint dimension = lane + component * 32u;
                if (query_pos < {query_length}u && dimension < {value_head_dim}u) {{
                    accumulator[component] = accumulator[component] * alpha
                        + block_scale
                            * tile_pv[warp * {pv_value_tile}u + dimension];
                }}
            }}
"""
        else:
            pv_probability_declaration = f"""
        threadgroup float tile_probabilities[{query_tile * key_tile}];"""
            pv_shared_declaration = ""
            pv_probability_store = f"""
            tile_probabilities[warp * {key_tile}u + lane] = probability;"""
            pv_preload_source = ""
            pv_accumulation_source = f"""
            for (uint component = 0u;
                 component < {ceil(value_head_dim / 32)}u; ++component) {{
                uint dimension = lane + component * 32u;
                float tile_accumulator = 0.0f;
                if (query_pos < {query_length}u && dimension < {value_head_dim}u) {{
                    for (uint tile_key = 0u; tile_key < {key_tile}u; ++tile_key) {{
                        uint global_key = key_base + tile_key;
                        if (global_key < {key_length}u) {{
                            tile_accumulator = fma(
                                tile_probabilities[warp * {key_tile}u + tile_key],
                                float(in_2[({key_value_batch_head} * {key_length}u
                                    + global_key) * {value_head_dim}u + dimension]),
                                tile_accumulator);
                        }}
                    }}
                    accumulator[component] = accumulator[component] * alpha
                        + block_scale * tile_accumulator;
                }}
            }}
"""
        if key_tile == 64:
            probability_float_offset = query_tile * key_tile // 2
            workspace_floats = max(
                query_tile * key_tile,
                probability_float_offset + query_tile * pv_value_tile,
            )
            score_storage_declaration = f"""
        threadgroup float tile_workspace[{workspace_floats}];
        threadgroup float* tile_scores = tile_workspace;
        threadgroup T* tile_probabilities_low =
            (threadgroup T*)tile_workspace;
        threadgroup float* tile_pv =
            tile_workspace + {probability_float_offset}u;"""
        else:
            score_storage_declaration = (
                f"threadgroup float tile_scores[{query_tile * key_tile}];"
            )
        score_read_barrier = (
            "\n            threadgroup_barrier(mem_flags::mem_threadgroup);"
            if key_tile == 64
            else ""
        )
        if key_tile == 32:
            softmax_source = """
            float block_max = simd_max(score);
            float new_max = metal::max(running_max, block_max);
            float alpha = running_max == -INFINITY
                ? 0.0f
                : metal::exp(running_max - new_max);
            float block_scale = block_max == -INFINITY
                ? 0.0f
                : metal::exp(block_max - new_max);
            float probability = score == -INFINITY
                ? 0.0f
                : metal::exp(score - block_max);
            float block_sum = simd_sum(probability);
            running_sum = running_sum * alpha + block_scale * block_sum;
            running_max = new_max;
"""
        else:
            softmax_source = """
            float block_max = simd_max(metal::max(score0, score1));
            float new_max = metal::max(running_max, block_max);
            float alpha = running_max == -INFINITY
                ? 0.0f
                : metal::exp(running_max - new_max);
            float block_scale = block_max == -INFINITY
                ? 0.0f
                : metal::exp(block_max - new_max);
            float probability0 = score0 == -INFINITY
                ? 0.0f
                : metal::exp(score0 - block_max);
            float probability1 = score1 == -INFINITY
                ? 0.0f
                : metal::exp(score1 - block_max);
            float block_sum = simd_sum(probability0 + probability1);
            running_sum = running_sum * alpha + block_scale * block_sum;
            running_max = new_max;
"""
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid % 32u;
        uint warp = tid / 32u;
        uint query_index = threadgroup_position_in_grid.x;
        uint batch = query_index / {query_heads * query_blocks}u;
        uint query_remainder = query_index % {query_heads * query_blocks}u;
        uint head = query_remainder / {query_blocks}u;
        uint query_block = query_remainder % {query_blocks}u;
        uint query_pos = query_block * {query_tile}u + warp;
        {tile_alignment}threadgroup T tile_q[{query_tile * query_head_dim}];
        {tile_alignment}threadgroup T tile_k[{key_tile * max(query_head_dim, value_head_dim)}];
        {score_storage_declaration}{pv_probability_declaration}{pv_shared_declaration}

{query_load_source}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float accumulator[{ceil(value_head_dim / 32)}] = {{0.0f}};
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        for (uint key_base = 0u; key_base < {key_length}u;
             key_base += {key_tile}u) {{
{key_load_source}
            threadgroup_barrier(mem_flags::mem_threadgroup);
{qk_score_source}{pv_preload_source}
{softmax_source}
{score_read_barrier}
{pv_probability_store}
            threadgroup_barrier(mem_flags::mem_threadgroup);
{pv_accumulation_source}
        }}

        if (query_pos < {query_length}u) {{
            for (uint component = 0u;
                 component < {ceil(value_head_dim / 32)}u; ++component) {{
                uint dimension = lane + component * 32u;
                if (dimension < {value_head_dim}u) {{
                    uint output_offset = ({query_batch_head} * {query_length}u
                        + query_pos) * {value_head_dim}u + dimension;
                    out[output_offset] = (T)(running_sum == 0.0f
                        ? 0.0f
                        : accumulator[component] / running_sum);
                }}
            }}
        }}
        """
        input_names = ("in_0", "in_1", "in_2", "in_3") if has_mask else (
            "in_0",
            "in_1",
            "in_2",
        )
        name = (
            f"metal_attention_flash_{query_tile}x{key_tile}_"
            f"{threads}_{index}"
        )
        kernel = _metal_kernel(
            name,
            input_names,
            ("out",),
            source,
            "",
        )
        grid = (query_count * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind="attention",
        )
        return artifact, runtime

    def _compile_softmax(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        if len(op.inputs) != 1:
            raise ValueError(f"{op.op} expects one input")

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                f"Metal {op.op} requires identical input/output dtypes"
            )
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"{op.op} output shape {output_type.shape} != {input_type.shape}"
            )
        if not input_type.shape:
            raise NotImplementedError(f"Metal {op.op} requires a ranked tensor")

        rank = len(input_type.shape)
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"{op.op}.dim must be an integer, got {dim!r}")
        axis = dim % rank
        columns = input_type.shape[axis]
        outer_size = prod(input_type.shape[:axis]) or 1
        inner_size = prod(input_type.shape[axis + 1 :]) or 1
        softmax_count = outer_size * inner_size
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        if inner_size == 1:
            input_base = f"output_index * {columns}u"
        else:
            input_base = (
                f"(output_index / {inner_size}u) * {columns * inner_size}u "
                f"+ output_index % {inner_size}u"
            )
        axis_offset = "col" if inner_size == 1 else f"col * {inner_size}u"
        input_expression = f"in_0[offset + ({axis_offset})]"
        exp_expression = f"metal::exp(float({input_expression}) - max_value)"
        if op.op == "safe_softmax":
            exp_expression = (
                f"(float({input_expression}) == -INFINITY ? 0.0f : {exp_expression})"
            )
        normalized_expression = f"{exp_expression} / sum_value"
        if op.op == "safe_softmax":
            normalized_expression = (
                f"(sum_value == 0.0f ? 0.0f : {normalized_expression})"
            )
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint output_index = threadgroup_position_in_grid.x;
        uint offset = {input_base};
        threadgroup float scratch[{threads}];

        float local_max = -INFINITY;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_max = metal::max(
                local_max, float(in_0[offset + ({axis_offset})])
            );
        }}
        scratch[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] = metal::max(scratch[tid], scratch[tid + stride]);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_value = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_sum += {exp_expression};
        }}
        scratch[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float sum_value = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            out[offset + ({axis_offset})] = (T)({normalized_expression});
        }}
        """
        name = f"metal_{op.op}_{threads}_{index}"
        kernel = _metal_kernel(
            name,
            ("in_0",),
            ("out",),
            source,
            "",
        )
        grid = (softmax_count * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime

    def _compile_reduction(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        if len(op.inputs) != 1:
            raise ValueError(f"{op.op} expects one input")

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        requested_dtype = op.attrs.get("dtype")
        if requested_dtype is not None and requested_dtype != output_type.dtype:
            raise ValueError(
                f"{op.op}.dtype {requested_dtype} does not match "
                f"output dtype {output_type.dtype}"
            )
        if not input_type.shape:
            raise NotImplementedError(f"Metal {op.op} requires a ranked tensor")

        rank = len(input_type.shape)
        dims = op.attrs.get("dims")
        if not isinstance(dims, tuple) or not dims:
            raise ValueError(f"{op.op}.dims must be a non-empty tuple")
        normalized_dims: list[int] = []
        for dim in dims:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise ValueError(f"{op.op}.dims must contain integers")
            axis = dim % rank
            if axis in normalized_dims:
                raise ValueError(f"{op.op}.dims must not contain duplicates")
            normalized_dims.append(axis)
        normalized_dims.sort()

        keepdim = op.attrs.get("keepdim")
        if not isinstance(keepdim, bool):
            raise ValueError(f"{op.op}.keepdim must be a boolean")
        reduced_rank = len(normalized_dims)
        reduced_dims = set(normalized_dims)
        columns = prod(input_type.shape[axis] for axis in normalized_dims)
        if keepdim:
            expected_output_shape = tuple(
                1 if axis in reduced_dims else dim
                for axis, dim in enumerate(input_type.shape)
            )
        else:
            expected_output_shape = tuple(
                dim
                for axis, dim in enumerate(input_type.shape)
                if axis not in reduced_dims
            )
        if output_type.shape != expected_output_shape:
            raise ValueError(
                f"{op.op} output shape {output_type.shape} "
                f"!= {expected_output_shape}"
            )

        output_count = prod(expected_output_shape) or 1
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        result_expression = "scratch[0]"
        if op.op == "mean":
            result_expression = f"scratch[0] / {columns}.0f"

        first_reduced_dim = normalized_dims[0]
        is_contiguous = normalized_dims == list(
            range(first_reduced_dim, first_reduced_dim + reduced_rank)
        )
        if is_contiguous:
            inner_shape = input_type.shape[first_reduced_dim + reduced_rank :]
            inner_size = prod(inner_shape) or 1
            if inner_size == 1:
                input_base = f"output_index * {columns}u"
            else:
                input_base = (
                    f"(output_index / {inner_size}u) * {columns * inner_size}u "
                    f"+ output_index % {inner_size}u"
                )
            reduction_offset = (
                "col" if inner_size == 1 else f"col * {inner_size}u"
            )
        else:
            compact_output_shape = tuple(
                dim
                for axis, dim in enumerate(input_type.shape)
                if axis not in reduced_dims
            )
            input_base = _reduction_output_base_expression(
                input_type.shape, compact_output_shape, reduced_dims
            )
            reduction_offset = _reduction_offset_expression(
                input_type.shape, normalized_dims
            )
        input_index_source = f"        uint offset = {input_base};"
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint output_index = threadgroup_position_in_grid.x;
{input_index_source}
        threadgroup float scratch[{threads}];

        float local_value = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_value += float(in_0[offset + ({reduction_offset})]);
        }}
        scratch[tid] = local_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0u) {{
            out[output_index] = (O)({result_expression});
        }}
        """
        name = f"metal_{op.op}_{threads}_{index}"
        kernel = _metal_kernel(
            name,
            ("in_0",),
            ("out",),
            source,
            "",
        )
        grid = (output_count * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        input_dtype = _MLX_DTYPES[input_type.dtype]
        output_dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", input_dtype), ("O", output_dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[output_dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime

    def _compile_layer_norm(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        return self._compile_normalization(graph, op, index, kind="layer_norm")

    def _compile_rms_norm(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        return self._compile_normalization(graph, op, index, kind="rms_norm")

    def _compile_rope(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        _, cosine_id, sine_id = op.inputs
        output_type = graph.value(op.outputs[0]).type
        cosine_type = graph.value(cosine_id).type
        sine_type = graph.value(sine_id).type
        columns = output_type.shape[-1]
        half = columns // 2
        source = _emit_rope_source(
            columns,
            half,
            output_type.shape,
            cosine_type.shape,
            sine_type.shape,
        )
        name = f"metal_rope_{columns}_{index}"
        kernel = _metal_kernel(
            name,
            ("in_0", "in_1", "in_2"),
            ("out",),
            source,
            _MSL_HEADER,
        )
        element_count = prod(output_type.shape)
        groups = element_count // _ELEMENTS_PER_THREAD
        grid = (groups, 1, 1)
        threadgroup = (min(256, groups), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MSL_HEADER,
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind="rope",
        )
        return artifact, runtime

    def _compile_normalization(
        self,
        graph: Graph,
        op: Operation,
        index: int,
        *,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        normalized_shape = op.attrs.get("normalized_shape")
        if not isinstance(normalized_shape, tuple) or not normalized_shape:
            raise ValueError(f"{kind}.normalized_shape must be a non-empty tuple")
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
            for dim in normalized_shape
        ):
            raise ValueError(
                f"{kind}.normalized_shape must contain positive integers"
            )

        has_weight = op.attrs.get("has_weight")
        if not isinstance(has_weight, bool):
            raise ValueError(f"{kind}.has_weight must be a boolean")
        has_bias = False
        if kind == "layer_norm":
            has_bias = op.attrs.get("has_bias")
            if not isinstance(has_bias, bool):
                raise ValueError("layer_norm.has_bias must be a boolean")
        expected_inputs = 1 + int(has_weight) + int(has_bias)
        if len(op.inputs) != expected_inputs:
            raise ValueError(
                f"{kind} expects {expected_inputs} inputs, got {len(op.inputs)}"
            )

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                f"Metal {kind} requires identical input/output dtypes"
            )
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"{kind} output shape {output_type.shape} != {input_type.shape}"
            )
        normalized_rank = len(normalized_shape)
        if normalized_rank > len(input_type.shape):
            raise ValueError(
                f"{kind} normalized_shape {normalized_shape} exceeds input rank"
            )
        if input_type.shape[-normalized_rank:] != normalized_shape:
            raise ValueError(
                f"{kind} normalized_shape {normalized_shape} does not match "
                f"input shape {input_type.shape}"
            )

        input_position = 1
        weight_position: int | None = None
        bias_position: int | None = None
        if has_weight:
            weight_position = input_position
            input_position += 1
        if has_bias:
            bias_position = input_position
        for position in range(1, len(op.inputs)):
            parameter_type = graph.value(op.inputs[position]).type
            if parameter_type.dtype != input_type.dtype:
                raise NotImplementedError(
                    f"Metal {kind} requires identical parameter dtypes"
                )
            if parameter_type.shape != normalized_shape:
                raise ValueError(
                    f"{kind} parameter shape {parameter_type.shape} "
                    f"!= {normalized_shape}"
                )


        columns = prod(normalized_shape)
        rows = prod(input_type.shape[:-normalized_rank]) or 1
        threads = _reduction_thread_count(columns)
        eps = float(op.attrs.get("eps", 1e-5))

        if kind == "layer_norm":
            reduction_source = (
                textwrap.dedent(
                    f"""
        float local_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_sum += float(in_0[offset + col]);
        }}
        """
                ).strip()
                + "\n"
                + _emit_threadgroup_reduction(threads, "local_sum", "total_sum")
                + "\n"
                + textwrap.dedent(
                    f"""
        float mean_value = total_sum / {columns}.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_squared_deviation = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float centered = float(in_0[offset + col]) - mean_value;
            local_squared_deviation += centered * centered;
        }}
        """
                ).strip()
                + "\n"
                + _emit_threadgroup_reduction(
                    threads, "local_squared_deviation", "total_deviation"
                )
                + "\n"
                + f"        float variance = total_deviation / {columns}.0f;"
            )
            reduction_result = "variance"
            centered_expression = "current - mean_value"
        else:
            reduction_source = (
                textwrap.dedent(
                    f"""
        float local_square_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float current = float(in_0[offset + col]);
            local_square_sum += current * current;
        }}
        """
                ).strip()
                + "\n"
                + _emit_threadgroup_reduction(
                    threads, "local_square_sum", "total_square_sum"
                )
                + "\n"
                + f"        float mean_square = total_square_sum / {columns}.0f;"
            )
            reduction_result = "mean_square"
            centered_expression = "current"

        scale_expression = (
            f"float(in_{weight_position}[col])"
            if weight_position is not None
            else "1.0f"
        )
        normalized_expression = (
            f"({centered_expression}) * inv_std * {scale_expression}"
        )
        if bias_position is not None:
            normalized_expression += f" + float(in_{bias_position}[col])"

        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint offset = row * {columns}u;
        threadgroup float scratch[{threads}];

{textwrap.indent(reduction_source, '        ')}
        float inv_std = metal::rsqrt({reduction_result} + {eps!r}f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float current = float(in_0[offset + col]);
            out[offset + col] = (T)({normalized_expression});
        }}
        """
        name = f"metal_{kind}_{threads}_{index}"
        kernel = _metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            "",
        )
        grid = (rows * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=kind,
        )
        return artifact, runtime

    def _compile_matmul(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        expected_inputs = 3 if op.op == "addmm" else 2
        epilogue = _matmul_epilogue_kind(op)
        if epilogue is not None:
            # Either a gate value to scale by, or a second weight matrix.
            expected_inputs += 1
        norm_position = _matmul_norm_position(op)
        if norm_position is not None:
            expected_inputs += 1
        if len(op.inputs) != expected_inputs:
            raise ValueError(f"{op.op} expects {expected_inputs} inputs")

        if norm_position is not None:
            norm_type = graph.value(op.inputs[norm_position]).type
            left_position = 1 if op.op == "addmm" else 0
            left_type = graph.value(op.inputs[left_position]).type
            if norm_type.shape != (left_type.shape[-1],):
                raise ValueError(
                    f"fused rms_norm weight shape {norm_type.shape} does not match "
                    f"the reduction dimension {left_type.shape[-1]}"
                )

        if op.op == "addmm":
            bias_id, left_id, right_id = op.inputs[:3]
        else:
            bias_id = None
            left_id, right_id = op.inputs[:2]

        left_type = graph.value(left_id).type
        right_type = graph.value(right_id).type
        output_type = graph.value(op.outputs[0]).type
        if len(left_type.shape) < 2 or len(right_type.shape) < 2:
            raise NotImplementedError("Metal matmul requires tensors with rank >= 2")
        batch_shape = _matmul_batch_shape(left_type.shape, right_type.shape)
        if batch_shape is None:
            raise NotImplementedError(
                "Metal matmul batch dimensions are not broadcast-compatible"
            )

        transpose_left = _matmul_transpose_attribute(op, "transpose_a")
        transpose_right = _matmul_transpose_attribute(op, "transpose_b")
        logical_left = left_type.shape
        logical_right = right_type.shape
        if transpose_left:
            logical_left = logical_left[:-2] + logical_left[-2:][::-1]
        if transpose_right:
            logical_right = logical_right[:-2] + logical_right[-2:][::-1]
        rows, reduction = logical_left[-2:]
        right_reduction, columns = logical_right[-2:]
        if reduction != right_reduction:
            raise ValueError(
                f"matmul reduction dimensions differ: {reduction} and {right_reduction}"
            )
        expected_output = batch_shape + (rows, columns)
        if output_type.shape != expected_output:
            raise ValueError(
                f"matmul output shape {output_type.shape} != {expected_output}"
            )
        batch_count = prod(batch_shape)
        left_batch_offset = _broadcast_batch_offset_expression(
            left_type.shape[:-2], batch_shape, rows * reduction
        )
        right_batch_offset = _broadcast_batch_offset_expression(
            right_type.shape[:-2], batch_shape, reduction * columns
        )

        dtypes = {left_type.dtype, right_type.dtype, output_type.dtype}
        if bias_id is not None:
            bias_type = graph.value(bias_id).type
            dtypes.add(bias_type.dtype)
            if not _can_broadcast(bias_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast addmm bias {bias_type.shape} "
                    f"to {output_type.shape}"
                )
        if len(dtypes) != 1:
            raise NotImplementedError("Metal matmul requires identical input/output dtypes")

        row_blocks = ceil(rows / _MATMUL_BLOCK_ROWS)
        column_blocks = ceil(columns / _MATMUL_BLOCK_COLUMNS)
        header = ""
        mpp_grid: tuple[int, int, int] | None = None
        if norm_position is not None and not _supports_gemv_matmul(
            rows,
            output_type.dtype,
            transpose_left,
            transpose_right,
        ):
            raise NotImplementedError(
                "fused rms_norm requires the Metal GEMV matmul path"
            )
        if _supports_gemv_matmul(
            rows,
            output_type.dtype,
            transpose_left,
            transpose_right,
        ):
            source = _emit_gemv_matmul_source(
                op,
                reduction,
                columns,
                graph,
                batch_count,
                left_batch_offset,
                right_batch_offset,
                transpose_right,
            )
            threads = _gemv_threads(columns, transpose_right)
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            layout = "transposed" if transpose_right else "row_major"
            name = (
                f"metal_{op.op}_gemv_{layout}_{threads}"
                f"{batch_suffix}_{index}"
            )
        elif _supports_mpp_matmul(
            rows, columns, reduction, output_type.dtype
        ) and _mpp_supported():
            mpp_config = _select_mpp_gemm_config(
                rows, columns, reduction, output_type.dtype, transpose_right
            )
            (
                mpp_block_rows,
                mpp_block_columns,
                mpp_warp_rows,
                mpp_warp_columns,
                mpp_step,
            ) = mpp_config
            source = _emit_mpp_matmul_source(
                op,
                rows,
                reduction,
                columns,
                graph,
                output_type.dtype,
                transpose_left,
                transpose_right,
                batch_count,
                left_batch_offset,
                right_batch_offset,
                mpp_config,
            )
            header = _MPP_HEADER
            threads = mpp_warp_rows * mpp_warp_columns * 32
            mpp_grid = (
                ceil(columns / mpp_block_columns) * threads,
                ceil(rows / mpp_block_rows),
                batch_count,
            )
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            orientation = (
                ("t" if transpose_left else "n")
                + ("t" if transpose_right else "n")
            )
            name = (
                f"metal_{op.op}_tensorcore_{orientation}_"
                f"{mpp_block_rows}x{mpp_block_columns}x{mpp_step}"
                f"_{mpp_warp_rows}x{mpp_warp_columns}"
                f"{batch_suffix}_{index}"
            )
        elif _supports_simdgroup_matmul(rows, columns, output_type.dtype):
            source = _emit_simdgroup_matmul_source(
                op,
                rows,
                reduction,
                columns,
                graph,
                output_type.dtype,
                transpose_left,
                transpose_right,
                batch_count,
                left_batch_offset,
                right_batch_offset,
            )
            threads = 256
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            name = (
                f"metal_{op.op}_simdgroup_"
                f"{_MATMUL_BLOCK_ROWS}x{_MATMUL_BLOCK_COLUMNS}x"
                f"{_MATMUL_BLOCK_REDUCTION}{batch_suffix}_{index}"
            )
        else:
            micro_rows, micro_columns = _select_matmul_microtile(reduction)
            source = _emit_tiled_matmul_source(
                op,
                rows,
                reduction,
                columns,
                micro_rows,
                micro_columns,
                graph,
                transpose_left,
                transpose_right,
                batch_count,
                left_batch_offset,
                right_batch_offset,
            )
            threads = (
                (_MATMUL_BLOCK_ROWS // micro_rows)
                * (_MATMUL_BLOCK_COLUMNS // micro_columns)
            )
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            name = (
                f"metal_{op.op}_tiled_"
                f"{_MATMUL_BLOCK_ROWS}x{_MATMUL_BLOCK_COLUMNS}x"
                f"{_MATMUL_BLOCK_REDUCTION}_{micro_rows}x{micro_columns}"
                f"{batch_suffix}_{index}"
            )
        kernel = _metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            header,
        )
        if mpp_grid is not None:
            grid = mpp_grid
        elif _supports_gemv_matmul(
            rows,
            output_type.dtype,
            transpose_left,
            transpose_right,
        ):
            columns_per_group = (
                threads // 32 if transpose_right else threads
            )
            grid = (
                ceil(columns / columns_per_group) * threads,
                batch_count,
                1,
            )
        else:
            grid = (
                row_blocks * column_blocks * threads,
                batch_count,
                1,
            )
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=header,
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime


def _validate_layout_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 1:
        raise ValueError(f"{op.op} expects one input")

    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    if input_type.dtype != output_type.dtype:
        raise ValueError(f"{op.op} must preserve dtype")

    rank = len(input_type.shape)
    if op.op == "reshape":
        shape = op.attrs.get("shape")
        if not isinstance(shape, tuple) or not shape or not all(
            isinstance(dim, int) and not isinstance(dim, bool)
            for dim in shape
        ):
            raise ValueError("reshape.shape must be a non-empty tuple of integers")
        if any(dim == 0 or dim < -1 for dim in shape):
            raise ValueError(f"reshape.shape contains an invalid dimension: {shape}")
        if shape.count(-1) > 1:
            raise ValueError(f"reshape.shape has multiple inferred dimensions: {shape}")

        input_count = prod(input_type.shape) or 1
        known_count = prod(dim for dim in shape if dim != -1) or 1
        if -1 in shape:
            if input_count % known_count:
                raise ValueError(
                    f"reshape shape {shape} cannot infer a dimension for "
                    f"{input_count} elements"
                )
            inferred = input_count // known_count
            resolved_shape = tuple(inferred if dim == -1 else dim for dim in shape)
        else:
            resolved_shape = shape
        if resolved_shape != output_type.shape:
            raise ValueError(
                f"reshape output shape {output_type.shape} != {resolved_shape}"
            )
        return

    if op.op == "transpose":
        dim0 = op.attrs.get("dim0")
        dim1 = op.attrs.get("dim1")
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool)
            for dim in (dim0, dim1)
        ):
            raise ValueError("transpose dimensions must be integers")
        normalized = tuple(
            dim + rank if dim < 0 else dim for dim in (dim0, dim1)
        )
        if any(dim < 0 or dim >= rank for dim in normalized):
            raise ValueError(f"transpose dimensions {(dim0, dim1)} exceed rank {rank}")
        expected_shape = list(input_type.shape)
        expected_shape[normalized[0]], expected_shape[normalized[1]] = (
            expected_shape[normalized[1]],
            expected_shape[normalized[0]],
        )
        if tuple(expected_shape) != output_type.shape:
            raise ValueError(
                f"transpose output shape {output_type.shape} != {tuple(expected_shape)}"
            )
        return

    if op.op == "permute":
        dims = op.attrs.get("dims")
        if not isinstance(dims, tuple) or len(dims) != rank or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(
                f"permute.dims must contain {rank} integer dimensions"
            )
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        if sorted(normalized) != list(range(rank)):
            raise ValueError(f"permute dimensions {dims} are not a permutation of rank {rank}")
        expected_shape = tuple(input_type.shape[dim] for dim in normalized)
        if expected_shape != output_type.shape:
            raise ValueError(
                f"permute output shape {output_type.shape} != {expected_shape}"
            )
        return

    if op.op == "expand":
        shape = op.attrs.get("shape")
        if not isinstance(shape, tuple) or len(shape) != len(output_type.shape):
            raise ValueError(
                f"expand.shape must have rank {len(output_type.shape)}, got {shape!r}"
            )
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool) and (dim == -1 or dim > 0)
            for dim in shape
        ):
            raise ValueError(f"expand.shape contains invalid dimensions: {shape}")
        if not _can_broadcast(input_type.shape, output_type.shape):
            raise ValueError(
                f"cannot expand shape {input_type.shape} to {output_type.shape}"
            )
        resolved_shape = tuple(
            output_type.shape[axis] if dim == -1 else dim
            for axis, dim in enumerate(shape)
        )
        if resolved_shape != output_type.shape:
            raise ValueError(
                f"expand output shape {resolved_shape} != {output_type.shape}"
            )
        return

    if op.op == "slice":
        dim = op.attrs.get("dim")
        start = op.attrs.get("start")
        end = op.attrs.get("end")
        step = op.attrs.get("step")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"slice.dim must be an integer, got {dim!r}")
        if start is not None and (
            not isinstance(start, int) or isinstance(start, bool)
        ):
            raise ValueError(f"slice.start must be an integer or None, got {start!r}")
        if end is not None and (
            not isinstance(end, int) or isinstance(end, bool)
        ):
            raise ValueError(f"slice.end must be an integer or None, got {end!r}")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(f"slice.step must be a positive integer, got {step!r}")
        if dim < -rank or dim >= rank:
            raise ValueError(f"slice.dim {dim} exceeds rank {rank}")
        axis = dim + rank if dim < 0 else dim
        sliced_length = len(
            range(*slice(start, end, step).indices(input_type.shape[axis]))
        )
        expected_shape = list(input_type.shape)
        expected_shape[axis] = sliced_length
        if tuple(expected_shape) != output_type.shape:
            raise ValueError(
                f"slice output shape {output_type.shape} != {tuple(expected_shape)}"
            )
        return

    if op.op == "flip":
        dims = op.attrs.get("dims")
        if not isinstance(dims, (tuple, list)) or not dims or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(f"flip.dims must contain integers, got {dims!r}")
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        if any(dim < 0 or dim >= rank for dim in normalized):
            raise ValueError(f"flip.dims {dims} exceed rank {rank}")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"flip.dims contains duplicate axes: {dims}")
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"flip output shape {output_type.shape} != {input_type.shape}"
            )
        return

    if op.op == "roll":
        shifts = op.attrs.get("shifts")
        if not isinstance(shifts, (tuple, list)) or not shifts or not all(
            isinstance(shift, int) and not isinstance(shift, bool)
            for shift in shifts
        ):
            raise ValueError(f"roll.shifts must contain integers, got {shifts!r}")
        dims = op.attrs.get("dims")
        if dims is None:
            if len(shifts) != 1:
                raise ValueError(
                    "roll without dims requires exactly one shift, got "
                    f"{len(shifts)}"
                )
            if input_type.shape != output_type.shape:
                raise ValueError(
                    f"roll output shape {output_type.shape} != {input_type.shape}"
                )
            return
        if not isinstance(dims, (tuple, list)) or not dims or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(f"roll.dims must contain integers, got {dims!r}")
        if len(dims) != len(shifts):
            raise ValueError(
                "roll shifts and dims must have the same length, got "
                f"{len(shifts)} and {len(dims)}"
            )
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        if any(dim < 0 or dim >= rank for dim in normalized):
            raise ValueError(f"roll.dims {dims} exceed rank {rank}")
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"roll output shape {output_type.shape} != {input_type.shape}"
            )
        return

    if op.op == "unsqueeze":
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"unsqueeze.dim must be an integer, got {dim!r}")
        output_rank = rank + 1
        axis = dim + output_rank if dim < 0 else dim
        if axis < 0 or axis >= output_rank:
            raise ValueError(f"unsqueeze.dim {dim} exceeds output rank {output_rank}")
        expected_shape = input_type.shape[:axis] + (1,) + input_type.shape[axis:]
        if expected_shape != output_type.shape:
            raise ValueError(
                f"unsqueeze output shape {output_type.shape} != {expected_shape}"
            )
        return

    if op.op == "squeeze":
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"squeeze.dim must be an integer, got {dim!r}")
        if rank == 0 or dim < -rank or dim >= rank:
            raise ValueError(f"squeeze.dim {dim} exceeds rank {rank}")
        axis = dim + rank if dim < 0 else dim
        expected_shape = input_type.shape
        if input_type.shape[axis] == 1:
            expected_shape = (
                input_type.shape[:axis] + input_type.shape[axis + 1 :]
            )
        if expected_shape != output_type.shape:
            raise ValueError(
                f"squeeze output shape {output_type.shape} != {expected_shape}"
            )
        return

    if input_type.shape != output_type.shape:
        raise ValueError(
            f"{op.op} output shape {output_type.shape} != {input_type.shape}"
        )


def _can_broadcast(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
    if len(input_shape) > len(output_shape):
        return False
    offset = len(output_shape) - len(input_shape)
    return all(
        input_dim in {1, output_shape[offset + axis]}
        for axis, input_dim in enumerate(input_shape)
    )


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError(f"expected a positive integer, got {value}")
    return 1 << (value - 1).bit_length()


def _reduction_output_base_expression(
    input_shape: tuple[int, ...],
    compact_output_shape: tuple[int, ...],
    reduced_dims: set[int],
) -> str:
    """Map a flattened output index to the first input element it reduces."""

    non_reduced_axes = [
        axis for axis in range(len(input_shape)) if axis not in reduced_dims
    ]
    terms: list[str] = []
    for output_axis, input_axis in enumerate(non_reduced_axes):
        output_stride = prod(compact_output_shape[output_axis + 1 :])
        output_dim = compact_output_shape[output_axis]
        coordinate = (
            f"((output_index / {output_stride}u) % {output_dim}u)"
            if output_stride > 1
            else f"(output_index % {output_dim}u)"
        )
        input_stride = prod(input_shape[input_axis + 1 :])
        terms.append(
            coordinate if input_stride == 1 else f"{coordinate} * {input_stride}u"
        )
    return " + ".join(terms) if terms else "0u"


def _reduction_offset_expression(
    input_shape: tuple[int, ...], reduced_dims: list[int]
) -> str:
    """Map a flattened reduction index to its input offset."""

    terms: list[str] = []
    reduction_stride = 1
    for axis in reversed(reduced_dims):
        dimension = input_shape[axis]
        coordinate = (
            f"(col % {dimension}u)"
            if reduction_stride == 1
            else f"((col / {reduction_stride}u) % {dimension}u)"
        )
        input_stride = prod(input_shape[axis + 1 :])
        terms.append(
            coordinate if input_stride == 1 else f"{coordinate} * {input_stride}u"
        )
        reduction_stride *= dimension
    return " + ".join(reversed(terms))


def _reduction_thread_count(columns: int) -> int:
    return min(256, _next_power_of_two(columns))


def _emit_threadgroup_reduction(
    threads: int,
    accumulator: str,
    label: str,
) -> str:
    """Reduce ``accumulator`` across the threadgroup and publish it in ``label``.

    A normalization row is often shorter than the threadgroup, so the classic
    barrier tree spends its time in synchronization rather than arithmetic. When
    the threadgroup covers whole SIMD groups the partial sums collapse with one
    shuffle reduction per group and a single cross-group step, which needs two
    barriers instead of ``log2(threads)``.
    """

    if threads >= 32:
        groups = threads // 32
        return textwrap.dedent(
            f"""
        {accumulator} = metal::simd_sum({accumulator});
        if ((tid & 31u) == 0u) {{
            scratch[tid >> 5u] = {accumulator};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {accumulator} = (tid < {groups}u) ? scratch[tid] : 0.0f;
        {accumulator} = metal::simd_sum({accumulator});
        if (tid == 0u) {{
            scratch[0] = {accumulator};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float {label} = scratch[0];
        """
        ).strip()

    return textwrap.dedent(
        f"""
        scratch[tid] = {accumulator};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = {threads // 2}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float {label} = scratch[0];
        """
    ).strip()


def _emit_rope_source(
    columns: int,
    half: int,
    output_shape: tuple[int, ...],
    cosine_shape: tuple[int, ...],
    sine_shape: tuple[int, ...],
) -> str:
    """Emit the rotary embedding kernel for one contiguous output tensor.

    ``out = x * cos + rotate_half(x) * sin`` and ``rotate_half`` pairs element
    ``d`` of the head with element ``d ^ half``, negating the first half. One
    thread owns four neighbouring columns; because the head dimension and the
    half both divide by four, a group never straddles that boundary, so the
    partner loads stay contiguous and vectorizable.
    """

    vectorized = columns % 4 == 0 and half % 4 == 0
    if not vectorized:
        raise NotImplementedError(
            "Metal rope requires a head dimension divisible by four"
        )
    lines = [
        "uint group = thread_position_in_grid.x;",
        "uint base = group * 4u;",
        f"uint column = base % {columns}u;",
        "uint row = base - column;",
        f"uint partner = row + (column < {half}u ? column + {half}u : column - {half}u);",
        f"float sign = column < {half}u ? -1.0f : 1.0f;",
        "vec<T, 4> current = *(const device vec<T, 4>*)(in_0 + base);",
        "vec<T, 4> rotated = *(const device vec<T, 4>*)(in_0 + partner);",
        "vec<T, 4> result;",
    ]
    for offset in range(4):
        element = f"(base + {offset}u)"
        cosine_offset = _offset_expression(cosine_shape, output_shape, element)
        sine_offset = _offset_expression(sine_shape, output_shape, element)
        lines.append(
            f"result[{offset}] = (T)(float(current[{offset}]) "
            f"* float(in_1[{cosine_offset}]) "
            f"+ sign * float(rotated[{offset}]) * float(in_2[{sine_offset}]));"
        )
    lines.append("*(device vec<T, 4>*)(out + base) = result;")
    return "\n".join(f"        {line}" for line in lines)


def _matmul_epilogue_kind(op: Operation) -> str | None:
    """Return the fused activation epilogue attached to a matrix multiply."""

    kind = op.attrs.get("epilogue")
    if kind is None:
        return None
    if kind not in {"silu_mul", "silu_mul_paired"}:
        raise ValueError(f"unsupported matmul epilogue {kind!r}")
    return kind


def _matmul_paired_weight_position(op: Operation, right_position: int) -> int | None:
    """Return the input slot holding the second projection of a paired epilogue."""

    if _matmul_epilogue_kind(op) != "silu_mul_paired":
        return None
    return right_position + 1


def _matmul_norm_position(op: Operation) -> int | None:
    """Return the input slot holding a fused RMS-norm weight, if any."""

    if op.attrs.get("rms_norm_eps") is None:
        return None
    return len(op.inputs) - 1


def _matmul_epilogue_position(op: Operation) -> int | None:
    """Return the input slot holding the fused epilogue operand, if any."""

    kind = _matmul_epilogue_kind(op)
    if kind is None:
        return None
    if kind == "silu_mul_paired":
        raise ValueError(
            "the paired gate/up epilogue is only lowered by the transposed GEMV"
        )
    return len(op.inputs) - 1 - int(_matmul_norm_position(op) is not None)


def _matmul_epilogue_value(
    value: str,
    op: Operation,
    graph: Graph,
    output_shape: tuple[int, ...],
    linear: str,
) -> str:
    """Apply the fused activation epilogue to one accumulated output element.

    ``silu_mul`` multiplies the accumulation by ``silu`` of the gate tensor that
    the activation would have consumed, so the multiply that followed the
    projection disappears into this kernel.
    """

    position = _matmul_epilogue_position(op)
    if position is None:
        return value
    gate_id = op.inputs[position]
    gate_type = graph.value(gate_id).type
    gate_offset = _offset_expression(gate_type.shape, output_shape, linear)
    gate = f"float(in_{position}[{gate_offset}])"
    return f"({value}) * ({gate} / (1.0f + metal::exp(-{gate})))"


def _offset_expression(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    index_expression: str,
) -> str:
    if input_shape == output_shape:
        return index_expression

    offset = len(output_shape) - len(input_shape)
    terms: list[str] = []
    for output_axis, output_dim in enumerate(output_shape):
        input_axis = output_axis - offset
        if input_axis < 0 or input_shape[input_axis] == 1:
            continue
        output_stride = prod(output_shape[output_axis + 1 :])
        input_stride = prod(input_shape[input_axis + 1 :])
        coordinate = f"((({index_expression}) / {output_stride}) % {output_dim})"
        terms.append(f"{coordinate} * {input_stride}")
    return " + ".join(terms) if terms else "0"


def _validate_pointwise_region(
    graph: Graph, body: tuple[Operation, ...], output_id: str
) -> None:
    for op in body:
        output_type = graph.value(op.outputs[0]).type
        input_types = tuple(graph.value(value_id).type for value_id in op.inputs)

        if op.op in _CONSTANT_POINTWISE_OPS:
            if op.inputs:
                raise ValueError(f"{op.op} must not have tensor inputs")
            value = op.attrs.get("value")
            if not isinstance(value, (bool, int, float)):
                raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
            if output_type.dtype in _INTEGRAL_DTYPES:
                _integer_constant_value(op, output_type.dtype)
            continue

        if op.op in _COMPARISON_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL:
                raise ValueError(f"{op.op} must produce a boolean tensor")
            if not (
                len(input_types) == 2
                or (len(input_types) == 1 and "scalar" in op.attrs)
            ):
                raise ValueError(f"malformed comparison op: {op.op}")
            continue

        if op.op in _LOGICAL_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL:
                raise ValueError(f"{op.op} must produce a boolean tensor")
            if op.op == "logical_not":
                expected_inputs = len(input_types) == 1
            else:
                expected_inputs = len(input_types) == 2 or (
                    len(input_types) == 1
                    and isinstance(op.attrs.get("scalar"), bool)
                )
            if not expected_inputs:
                raise ValueError(f"malformed logical op: {op.op}")
            if any(input_type.dtype != DType.BOOL for input_type in input_types):
                raise ValueError(f"{op.op} requires boolean tensor inputs")
            continue

        if op.op in _BITWISE_POINTWISE_OPS:
            if output_type.dtype == DType.BOOL:
                if op.op == "bitwise_not":
                    expected_inputs = len(input_types) == 1
                else:
                    expected_inputs = len(input_types) == 2 or (
                        len(input_types) == 1
                        and isinstance(op.attrs.get("scalar"), bool)
                    )
                if not expected_inputs:
                    raise ValueError(f"malformed bitwise op: {op.op}")
                if any(input_type.dtype != DType.BOOL for input_type in input_types):
                    raise ValueError(
                        f"boolean {op.op} requires boolean tensor inputs"
                    )
                continue
            _validate_integer_pointwise_op(op, output_type, input_types)
            continue

        if op.op in _NUMERIC_PREDICATE_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL or len(input_types) != 1:
                raise ValueError(f"malformed predicate op: {op.op}")
            if input_types[0].dtype == DType.BOOL:
                raise ValueError(f"{op.op} requires a numeric tensor input")
            if input_types[0].dtype in _INTEGRAL_DTYPES:
                raise ValueError(f"{op.op} does not accept integer tensor inputs")
            continue

        if op.op == "where":
            if len(input_types) != 3:
                raise ValueError("where requires condition, input and other tensors")
            if input_types[0].dtype != DType.BOOL:
                raise ValueError("where condition must be a boolean tensor")
            if output_type.dtype in _INTEGRAL_DTYPES:
                if any(
                    input_type.dtype != output_type.dtype
                    for input_type in input_types[1:]
                ):
                    raise ValueError(
                        "integer where requires matching integer branch dtypes"
                    )
            continue

        if op.op == "masked_fill":
            if len(input_types) != 2:
                raise ValueError("masked_fill requires input and mask tensors")
            if input_types[1].dtype != DType.BOOL:
                raise ValueError("masked_fill mask must be a boolean tensor")
            if output_type != input_types[0]:
                raise ValueError("masked_fill must preserve the input tensor type")
            value = op.attrs.get("value")
            if output_type.dtype == DType.BOOL:
                if not isinstance(value, bool):
                    raise ValueError("boolean masked_fill requires a boolean value")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("numeric masked_fill requires a numeric value")
            continue

        if output_type.dtype == DType.BOOL or any(
            input_type.dtype == DType.BOOL for input_type in input_types
        ):
            raise ValueError(f"{op.op} does not accept boolean tensor operands")
        if output_type.dtype in _INTEGRAL_DTYPES or any(
            input_type.dtype in _INTEGRAL_DTYPES for input_type in input_types
        ):
            _validate_integer_pointwise_op(op, output_type, input_types)
            continue

    if graph.value(output_id).type.dtype == DType.BOOL:
        producer = next(
            (op for op in reversed(body) if op.outputs[0] == output_id), None
        )
        if producer is None or producer.op not in (
            _COMPARISON_POINTWISE_OPS
            | _LOGICAL_POINTWISE_OPS
            | _BITWISE_POINTWISE_OPS
            | _NUMERIC_PREDICATE_POINTWISE_OPS
            | {"where", "masked_fill"}
            | _CONSTANT_POINTWISE_OPS
        ):
            raise ValueError("boolean pointwise region has an unsupported root op")


def _validate_integer_pointwise_op(
    op: Operation,
    output_type: TensorType,
    input_types: tuple[TensorType, ...],
) -> None:
    if op.op not in _INTEGER_POINTWISE_OPS:
        raise ValueError(f"{op.op} does not accept integer tensor operands")
    if output_type.dtype not in _INTEGRAL_DTYPES:
        raise ValueError(f"{op.op} does not support integer tensor operands")
    if any(input_type.dtype not in _INTEGRAL_DTYPES for input_type in input_types):
        raise ValueError(f"{op.op} requires matching integer tensor operands")

    if op.op in {"neg", "abs", "sign", "square", "bitwise_not"}:
        if len(input_types) != 1 or input_types[0].dtype != output_type.dtype:
            raise ValueError(f"malformed integer unary op: {op.op}")
        return

    if op.op in {
        "add",
        "sub",
        "rsub",
        "mul",
        "maximum",
        "minimum",
        "floor_divide",
        "remainder",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    }:
        if len(input_types) == 2:
            if any(
                input_type.dtype != output_type.dtype for input_type in input_types
            ):
                raise ValueError(f"integer {op.op} requires matching input dtypes")
        elif len(input_types) == 1:
            scalar = op.attrs.get("scalar")
            if not isinstance(scalar, int) or isinstance(scalar, bool):
                raise ValueError(
                    f"integer {op.op} scalar must be an integer, got {scalar!r}"
                )
        else:
            raise ValueError(f"malformed integer binary op: {op.op}")
        return

    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        if not input_types or input_types[0].dtype != output_type.dtype:
            raise ValueError(f"malformed integer {op.op} op")
        for input_type in input_types[1:]:
            if input_type.dtype != output_type.dtype:
                raise ValueError(f"integer {op.op} requires matching bound dtypes")
        for name in ("min", "max"):
            bound = op.attrs.get(name)
            if bound is not None and (
                not isinstance(bound, int) or isinstance(bound, bool)
            ):
                raise ValueError(
                    f"integer {op.op}.{name} must be an integer, got {bound!r}"
                )
        return

    raise AssertionError(f"unclassified integer pointwise op: {op.op}")


def _validate_integer_constant(graph: Graph, op: Operation) -> None:
    if op.inputs:
        raise ValueError(f"{op.op} must not have tensor inputs")
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype not in _INTEGRAL_DTYPES:
        raise ValueError(f"{op.op} is not an integer constant")
    _integer_constant_value(op, output_type.dtype)


def _integer_constant_value(op: Operation, dtype: DType) -> int:
    value = op.attrs.get("value")
    if not isinstance(value, (bool, int, float)):
        raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{op.op}.value cannot be converted to an integer")

    converted = int(value)
    bits = 32 if dtype == DType.I32 else 64
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if not lower <= converted <= upper:
        raise ValueError(
            f"{op.op}.value {value!r} overflows {dtype.value}"
        )
    return converted


def _emit_integer_constant_source(op: Operation, output_type: TensorType) -> str:
    value = _integer_constant_value(op, output_type.dtype)
    return _emit_zero_input_expression(
        _msl_integer_literal(value, output_type.dtype), output_type.shape
    )


def _validate_arange(graph: Graph, op: Operation) -> None:
    if op.inputs:
        raise ValueError("arange must not have tensor inputs")
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype not in _INTEGRAL_DTYPES:
        raise NotImplementedError(
            "arange currently supports int32 and int64 outputs only"
        )
    if len(output_type.shape) != 1:
        raise ValueError(f"arange must produce a rank-1 tensor, got {output_type.shape}")

    values: list[int] = []
    for name in ("start", "end", "step"):
        value = op.attrs.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise NotImplementedError(
                "arange currently requires integer start, end and step values"
            )
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError(f"arange.{name} {value} exceeds signed int64")
        values.append(value)
    start, end, step = values
    if step == 0:
        raise ValueError("arange.step must be nonzero")

    expected_length = _arange_length(start, end, step)
    if output_type.shape[0] != expected_length:
        raise ValueError(
            f"arange output shape {output_type.shape} does not match expected "
            f"length {expected_length}"
        )


def _arange_length(start: int, end: int, step: int) -> int:
    if step > 0:
        if start >= end:
            return 0
        return (end - start + step - 1) // step
    if start <= end:
        return 0
    stride = -step
    return (start - end + stride - 1) // stride


def _emit_arange_source(op: Operation, output_type: TensorType) -> str:
    start = _msl_integer_literal(int(op.attrs["start"]), DType.I64)
    step = _msl_integer_literal(int(op.attrs["step"]), DType.I64)
    expression = f"(long)({start}) + (long)elem * (long)({step})"
    return _emit_zero_input_expression(expression, output_type.shape)


def _emit_zero_input_expression(
    expression: str, output_shape: tuple[int, ...]
) -> str:
    element_count = prod(output_shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
        f"    out[elem] = (T)({expression});",
        "}",
    ]
    return "\n".join(f"        {line}" for line in lines)


def _msl_integer_literal(value: int, dtype: DType) -> str:
    if dtype == DType.I64:
        if value == -(1 << 63):
            return "(-9223372036854775807LL - 1LL)"
        return f"{value}LL"
    if dtype == DType.I32:
        if value == -(1 << 31):
            return "(-2147483647 - 1)"
        return str(value)
    raise ValueError(f"{dtype.value} is not an integer dtype")


def _emit_pointwise_source(
    graph: Graph,
    input_ids: tuple[str, ...],
    body: tuple[Operation, ...],
    output_id: str,
    output_shape: tuple[int, ...],
) -> str:
    element_count = prod(output_shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
    ]
    env: dict[str, str] = {}
    common: dict[tuple[object, ...], str] = {}

    for position, value_id in enumerate(input_ids):
        variable = f"input_{position}"
        input_type = graph.value(value_id).type
        ctype = _pointwise_ctype(input_type.dtype)
        if input_type.shape:
            offset = _offset_expression(
                input_type.shape, output_shape, "elem"
            )
            lines.append(f"    {ctype} {variable} = in_{position}[{offset}];")
        else:
            lines.append(f"    {ctype} {variable} = ({ctype})in_{position};")
        env[value_id] = variable

    for op_index, op in enumerate(body):
        arguments = [env[value_id] for value_id in op.inputs]
        key = (
            op.op,
            op.inputs,
            graph.value(op.outputs[0]).type.dtype.value,
            tuple((name, repr(value)) for name, value in sorted(op.attrs.items())),
        )
        existing = common.get(key)
        if existing is not None:
            env[op.outputs[0]] = existing
            continue
        output = f"temp_{op_index}"
        input_types = tuple(graph.value(value_id).type for value_id in op.inputs)
        output_type = graph.value(op.outputs[0]).type
        expression = _emit_pointwise_expression(
            op, arguments, input_types, output_type.dtype
        )
        ctype = _pointwise_ctype(output_type.dtype)
        lines.append(f"    {ctype} {output} = {expression};")
        env[op.outputs[0]] = output
        common[key] = output

    if output_id not in env:
        raise ValueError("pointwise region does not produce its declared output")
    lines.extend(
        [
            f"    out[elem] = (T)({env[output_id]});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


def _pointwise_ctype(dtype: DType) -> str:
    if dtype == DType.BOOL:
        return "bool"
    if dtype == DType.I32:
        return "int"
    if dtype == DType.I64:
        return "long"
    return "float"


def _wrap_integer_scalar(value: int, dtype: DType) -> int:
    bits = 32 if dtype == DType.I32 else 64
    modulus = 1 << bits
    wrapped = value % modulus
    if wrapped >= (1 << (bits - 1)):
        wrapped -= modulus
    return wrapped


def _checked_integer_scalar(value: int | float, dtype: DType) -> int:
    converted = int(value)
    bits = 32 if dtype == DType.I32 else 64
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if not lower <= converted <= upper:
        raise ValueError(f"integer scalar {value!r} overflows {dtype.value}")
    return converted


def _binary_operands(
    op: Operation,
    arguments: list[str],
    input_types: tuple[TensorType, ...],
) -> tuple[str, str]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar_value = op.attrs["scalar"]
    if isinstance(scalar_value, bool):
        scalar = "true" if scalar_value else "false"
    elif input_types[0].dtype in _INTEGRAL_DTYPES and isinstance(scalar_value, int):
        scalar = _msl_integer_literal(
            _wrap_integer_scalar(scalar_value, input_types[0].dtype),
            input_types[0].dtype,
        )
    else:
        scalar = _msl_float_literal(float(scalar_value))
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _msl_float_literal(value: float) -> str:
    if isnan(value):
        return "NAN"
    if isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)


def _emit_pointwise_expression(
    op: Operation,
    arguments: list[str],
    input_types: tuple[TensorType, ...],
    output_dtype: DType,
) -> str:
    if op.op in _CONSTANT_POINTWISE_OPS:
        value = op.attrs.get("value")
        if isinstance(value, bool):
            return "true" if value else "false"
        if output_dtype in _INTEGRAL_DTYPES:
            return _msl_integer_literal(
                _integer_constant_value(op, output_dtype), output_dtype
            )
        if isinstance(value, (int, float)):
            return _msl_float_literal(float(value))
        raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
    if op.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }[op.op]
        return f"({left} {operator} {right})"
    if op.op == "logical_and":
        left, right = _binary_operands(op, arguments, input_types)
        return f"({left} && {right})"
    if op.op == "logical_or":
        left, right = _binary_operands(op, arguments, input_types)
        return f"({left} || {right})"
    if op.op == "logical_xor":
        left, right = _binary_operands(op, arguments, input_types)
        return f"({left} != {right})"
    if op.op == "logical_not":
        return f"(!{arguments[0]})"
    if op.op in _BITWISE_POINTWISE_OPS:
        if output_dtype == DType.BOOL:
            if op.op == "bitwise_not":
                return f"(!{arguments[0]})"
            left, right = _binary_operands(op, arguments, input_types)
            operator = {
                "bitwise_and": "&",
                "bitwise_or": "|",
                "bitwise_xor": "^",
            }[op.op]
            return f"({left} {operator} {right})"
        return _emit_integer_pointwise_expression(
            op, arguments, input_types, output_dtype
        )
    if op.op == "isnan":
        return f"metal::isnan({arguments[0]})"
    if op.op == "isinf":
        return f"metal::isinf({arguments[0]})"
    if op.op == "where":
        return f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
    if op.op == "masked_fill":
        value = op.attrs.get("value")
        if isinstance(value, bool):
            fill = "true" if value else "false"
        elif output_dtype in _INTEGRAL_DTYPES:
            fill = _msl_integer_literal(
                _checked_integer_scalar(value, output_dtype), output_dtype
            )
        else:
            fill = _msl_float_literal(float(value))
        return f"({arguments[1]} ? {fill} : {arguments[0]})"
    if output_dtype in _INTEGRAL_DTYPES:
        return _emit_integer_pointwise_expression(
            op, arguments, input_types, output_dtype
        )
    if op.op == "silu":
        value = arguments[0]
        return f"{value} / (1.0f + metal::exp(-{value}))"
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            return (
                f"0.5f * {value} * (1.0f + metal::tanh("
                f"0.7978845608028654f * ({value} + "
                f"0.044715f * {value} * {value} * {value})))"
            )
        return (
            f"0.5f * {value} * (1.0f + tsc_erf("
            f"{value} * 0.7071067811865476f))"
        )
    if op.op == "softplus":
        beta = _msl_float_literal(float(op.attrs["beta"]))
        threshold = _msl_float_literal(float(op.attrs["threshold"]))
        return f"tsc_softplus({arguments[0]}, {beta}, {threshold})"
    if op.op == "mish":
        return f"tsc_mish({arguments[0]})"
    if op.op == "logsigmoid":
        return f"tsc_logsigmoid({arguments[0]})"
    if op.op in {"mul", "div"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = "*" if op.op == "mul" else "/"
        return f"{left} {operator} {right}"
    if op.op == "pow":
        left, right = _binary_operands(op, arguments, input_types)
        return f"metal::pow({left}, {right})"
    if op.op in {"maximum", "minimum"}:
        left, right = _binary_operands(op, arguments, input_types)
        comparison = "<" if op.op == "maximum" else ">"
        selected = f"(({left}) {comparison} ({right}) ? ({right}) : ({left}))"
        return (
            f"((metal::isnan({left}) || metal::isnan({right})) "
            f"? NAN : {selected})"
        )
    if op.op == "logaddexp":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_logaddexp({left}, {right})"
    if op.op == "atan2":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_atan2({left}, {right})"
    if op.op in {"add", "sub", "rsub"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = "+" if op.op == "add" else "-"
        alpha = float(op.attrs.get("alpha", 1.0))
        return f"{left} {operator} {alpha!r} * {right}"
    if op.op == "neg":
        return f"-{arguments[0]}"
    if op.op == "abs":
        return f"metal::abs({arguments[0]})"
    if op.op == "sigmoid":
        return f"1.0f / (1.0f + metal::exp(-{arguments[0]}))"
    if op.op == "tanh":
        return f"metal::tanh({arguments[0]})"
    if op.op == "relu":
        return f"metal::max(0.0f, {arguments[0]})"
    if op.op == "leaky_relu":
        value = arguments[0]
        slope = _msl_float_literal(float(op.attrs.get("negative_slope", 0.01)))
        return f"(({value}) >= 0.0f ? ({value}) : {slope} * ({value}))"
    if op.op == "elu":
        value = arguments[0]
        alpha = _msl_float_literal(float(op.attrs.get("alpha", 1.0)))
        return (
            f"(({value}) >= 0.0f ? ({value}) : "
            f"{alpha} * (metal::exp({value}) - 1.0f))"
        )
    if op.op == "hardswish":
        value = arguments[0]
        body = (
            f"({value}) * metal::min(metal::max(({value}) + 3.0f, 0.0f), 6.0f) "
            "/ 6.0f"
        )
        return f"(metal::isnan({value}) ? NAN : {body})"
    if op.op == "hardsigmoid":
        value = arguments[0]
        body = (
            "metal::min(metal::max("
            f"({value}) + 3.0f, 0.0f), 6.0f) / 6.0f"
        )
        return f"(metal::isnan({value}) ? NAN : {body})"
    if op.op == "threshold":
        value = arguments[0]
        threshold = _msl_float_literal(float(op.attrs["threshold"]))
        fill_value = _msl_float_literal(float(op.attrs["value"]))
        return (
            f"(metal::isnan({value}) ? {value} : "
            f"(({value}) > {threshold} ? ({value}) : {fill_value}))"
        )
    if op.op == "exp":
        return f"metal::exp({arguments[0]})"
    if op.op == "expm1":
        return f"tsc_expm1({arguments[0]})"
    if op.op == "exp2":
        return f"metal::exp2({arguments[0]})"
    if op.op == "log":
        return f"metal::log({arguments[0]})"
    if op.op == "log2":
        return f"metal::log2({arguments[0]})"
    if op.op == "log10":
        return f"metal::log10({arguments[0]})"
    if op.op == "log1p":
        return f"log1p({arguments[0]})"
    if op.op == "erf":
        return f"tsc_erf({arguments[0]})"
    if op.op == "sin":
        return f"metal::sin({arguments[0]})"
    if op.op == "cos":
        return f"metal::cos({arguments[0]})"
    if op.op == "floor":
        return f"metal::floor({arguments[0]})"
    if op.op == "ceil":
        return f"metal::ceil({arguments[0]})"
    if op.op == "round":
        return f"metal::rint({arguments[0]})"
    if op.op == "trunc":
        return f"metal::trunc({arguments[0]})"
    if op.op == "sign":
        return f"metal::sign({arguments[0]})"
    if op.op == "square":
        value = arguments[0]
        return f"({value}) * ({value})"
    if op.op == "sqrt":
        return f"metal::sqrt({arguments[0]})"
    if op.op == "rsqrt":
        return f"metal::rsqrt({arguments[0]})"
    if op.op == "reciprocal":
        return f"1.0f / {arguments[0]}"
    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        lower = _clamp_bound_expression(op, arguments, "min", output_dtype)
        upper = _clamp_bound_expression(op, arguments, "max", output_dtype)
        if lower is not None and upper is not None:
            clipped = f"metal::min(metal::max({value}, {lower}), {upper})"
        elif lower is not None:
            clipped = f"metal::max({value}, {lower})"
        elif upper is not None:
            clipped = f"metal::min({value}, {upper})"
        else:
            return value
        invalid_inputs = [value]
        if lower is not None:
            invalid_inputs.append(lower)
        if upper is not None:
            invalid_inputs.append(upper)
        nan_check = " || ".join(
            f"metal::isnan({expression})" for expression in invalid_inputs
        )
        return f"(({nan_check}) ? NAN : {clipped})"
    raise NotImplementedError(f"cannot emit fused pointwise op: {op.op}")


def _emit_integer_pointwise_expression(
    op: Operation,
    arguments: list[str],
    input_types: tuple[TensorType, ...],
    output_dtype: DType,
) -> str:
    suffix = "i32" if output_dtype == DType.I32 else "i64"
    ctype = _pointwise_ctype(output_dtype)
    unsigned_ctype = "uint" if output_dtype == DType.I32 else "unsigned long"

    if op.op == "neg":
        return f"tsc_neg_{suffix}({arguments[0]})"
    if op.op == "abs":
        value = arguments[0]
        return f"(({value}) < 0 ? tsc_neg_{suffix}({value}) : ({value}))"
    if op.op == "sign":
        value = arguments[0]
        return f"(({value}) > 0 ? 1 : (({value}) < 0 ? -1 : 0))"
    if op.op == "square":
        value = arguments[0]
        return f"tsc_mul_{suffix}({value}, {value})"
    if op.op == "bitwise_not":
        value = arguments[0]
        return f"({ctype})(~({unsigned_ctype})({value}))"

    if op.op in {"add", "sub", "rsub", "mul"}:
        left, right = _binary_operands(op, arguments, input_types)
        if op.op == "mul":
            return f"tsc_mul_{suffix}({left}, {right})"
        alpha = op.attrs.get("alpha", 1)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(f"integer {op.op}.alpha must be an integer")
        if isinstance(alpha, float):
            if not isfinite(alpha) or not alpha.is_integer():
                raise ValueError(f"integer {op.op}.alpha must be an integer")
            alpha = int(alpha)
        if alpha != 1:
            alpha_literal = _msl_integer_literal(
                _wrap_integer_scalar(int(alpha), output_dtype), output_dtype
            )
            right = f"tsc_mul_{suffix}({right}, {alpha_literal})"
        function = "tsc_add" if op.op == "add" else "tsc_sub"
        return f"{function}_{suffix}({left}, {right})"

    if op.op in {"maximum", "minimum"}:
        left, right = _binary_operands(op, arguments, input_types)
        comparison = "<" if op.op == "maximum" else ">"
        return f"(({left}) {comparison} ({right}) ? ({right}) : ({left}))"

    if op.op in {"floor_divide", "remainder"}:
        left, right = _binary_operands(op, arguments, input_types)
        function = "floor_div" if op.op == "floor_divide" else "remainder"
        return f"tsc_{function}_{suffix}({left}, {right})"

    if op.op in {"bitwise_and", "bitwise_or", "bitwise_xor"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = {
            "bitwise_and": "&",
            "bitwise_or": "|",
            "bitwise_xor": "^",
        }[op.op]
        return (
            f"({ctype})(({unsigned_ctype})({left}) {operator} "
            f"({unsigned_ctype})({right}))"
        )

    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        lower = _clamp_bound_expression(op, arguments, "min", output_dtype)
        upper = _clamp_bound_expression(op, arguments, "max", output_dtype)
        if lower is not None and upper is not None:
            return (
                f"(({value}) > ({upper}) ? ({upper}) : "
                f"(({value}) < ({lower}) ? ({lower}) : ({value})))"
            )
        if lower is not None:
            return f"(({value}) < ({lower}) ? ({lower}) : ({value}))"
        if upper is not None:
            return f"(({value}) > ({upper}) ? ({upper}) : ({value}))"
        return value

    raise NotImplementedError(f"cannot emit integer pointwise op: {op.op}")


def _clamp_bound_expression(
    op: Operation,
    arguments: list[str],
    name: str,
    output_dtype: DType,
) -> str | None:
    input_position = op.attrs.get(f"{name}_input")
    if input_position is not None:
        if (
            not isinstance(input_position, int)
            or isinstance(input_position, bool)
            or input_position < 1
            or input_position >= len(arguments)
        ):
            raise ValueError(f"clamp.{name}_input is malformed")
        return arguments[input_position]
    bound = op.attrs.get(name)
    if bound is None:
        return None
    if not isinstance(bound, (int, float)) or isinstance(bound, bool):
        raise ValueError(f"clamp.{name} must be a scalar number")
    if output_dtype in _INTEGRAL_DTYPES:
        return _msl_integer_literal(
            _checked_integer_scalar(bound, output_dtype), output_dtype
        )
    value = float(bound)
    if isnan(value):
        return "NAN"
    if isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)


def _select_matmul_microtile(reduction: int) -> tuple[int, int]:
    if reduction >= _MATMUL_LARGE_K_THRESHOLD:
        return _MATMUL_LARGE_K_MICROTILE
    return _MATMUL_MICROTILE


def _supports_simdgroup_matmul(rows: int, columns: int, dtype: DType) -> bool:
    return (
        dtype in {DType.F16, DType.BF16}
        and rows % _MATMUL_BLOCK_ROWS == 0
        and columns % _MATMUL_BLOCK_COLUMNS == 0
    )


def _supports_gemv_matmul(
    rows: int,
    dtype: DType,
    transpose_left: bool,
    transpose_right: bool,
) -> bool:
    """Return whether the one-row matmul can use the bandwidth-oriented GEMV."""

    return (
        rows == 1
        and dtype in {DType.F16, DType.BF16, DType.F32}
        and not transpose_left
    )


def _emit_thread_rms_norm_prologue(
    op: Operation,
    norm_position: int,
    left_position: int,
    reduction: int,
    left_batch_prefix: str,
) -> list[str]:
    """Emit an RMS scale computed by one thread over the whole reduction."""

    eps = float(op.attrs["rms_norm_eps"])
    return [
        "float norm_square_sum = 0.0f;",
        f"for (uint norm_index = 0u; norm_index < {reduction}u; norm_index += 1u) {{",
        f"    float norm_value = float(in_{left_position}"
        f"[{left_batch_prefix}norm_index]);",
        "    norm_square_sum = fma(norm_value, norm_value, norm_square_sum);",
        "}",
        f"float norm_scale = metal::rsqrt(norm_square_sum / {reduction}.0f + {eps!r});",
    ]


def _emit_warp_rms_norm_prologue(
    op: Operation,
    norm_position: int,
    left_position: int,
    reduction: int,
    left_batch_prefix: str,
) -> list[str]:
    """Emit a warp-wide RMS scale that every lane of the warp can read back."""

    eps = float(op.attrs["rms_norm_eps"])
    vectors = reduction // 4
    return [
        "const device vec<T, 4>* norm_source = "
        "reinterpret_cast<const device vec<T, 4>*>("
        f"in_{left_position} + {left_batch_prefix}0u);",
        "const device vec<T, 4>* norm_weight_values = "
        "reinterpret_cast<const device vec<T, 4>*>("
        f"in_{norm_position});",
        "float norm_square_sum = 0.0f;",
        f"for (uint norm_index = lane; norm_index < {vectors}u; norm_index += 32u) {{",
        "    vec<T, 4> current = norm_source[norm_index];",
        "    norm_square_sum = fma(float(current[0]), float(current[0]), norm_square_sum);",
        "    norm_square_sum = fma(float(current[1]), float(current[1]), norm_square_sum);",
        "    norm_square_sum = fma(float(current[2]), float(current[2]), norm_square_sum);",
        "    norm_square_sum = fma(float(current[3]), float(current[3]), norm_square_sum);",
        "}",
        f"float norm_scale = metal::rsqrt(simd_sum(norm_square_sum) / {reduction}.0f + {eps!r});",
    ]


def _emit_gemv_matmul_source(
    op: Operation,
    reduction: int,
    columns: int,
    graph: Graph,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
    transpose_right: bool,
) -> str:
    """Emit a GEMV with coalesced weight loads for either right-hand layout."""

    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    threads = _gemv_threads(columns, False)
    block_columns = threads
    exact_columns = columns % block_columns == 0
    left_batch_prefix = f"{left_batch_offset} + " if left_batch_offset else ""
    right_batch_prefix = f"{right_batch_offset} + " if right_batch_offset else ""
    if transpose_right:
        return _emit_transposed_gemv_matmul_source(
            op,
            graph,
            reduction,
            columns,
            bias_id,
            bias_type,
            output_shape,
            alpha,
            beta,
            left_position,
            right_position,
            batch_count,
            left_batch_prefix,
            right_batch_prefix,
        )

    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    norm_slot = _matmul_norm_position(op)
    if norm_slot is not None:
        lines.extend(
            _emit_thread_rms_norm_prologue(
                op, norm_slot, left_position, reduction, left_batch_prefix
            )
        )
    lines.extend(
        [
            f"uint column = threadgroup_index * {block_columns}u + tid;",
            "float acc0 = 0.0f;",
            "float acc1 = 0.0f;",
            "float acc2 = 0.0f;",
            "float acc3 = 0.0f;",
            "uint k = 0u;",
            f"for (; k + 3u < {reduction}u; k += 4u) {{",
        ]
    )
    if norm_slot is None:
        lines.extend(
            [
                f"    float x0 = float(in_{left_position}[{left_batch_prefix}k]);",
                f"    float x1 = float(in_{left_position}[{left_batch_prefix}k + 1u]);",
                f"    float x2 = float(in_{left_position}[{left_batch_prefix}k + 2u]);",
                f"    float x3 = float(in_{left_position}[{left_batch_prefix}k + 3u]);",
            ]
        )
    else:
        for offset in range(4):
            lines.append(
                f"    float x{offset} = float((T)(norm_scale * "
                f"float(in_{left_position}[{left_batch_prefix}k + {offset}u]) * "
                f"float(in_{norm_slot}[k + {offset}u])));"
            )
    if not exact_columns:
        lines.append(f"    if (column < {columns}u) {{")
    indent = "        " if not exact_columns else "    "
    lines.extend(
        [
            f"{indent}acc0 = fma(x0, float(in_{right_position}["
            f"{right_batch_prefix}k * {columns}u + column]), acc0);",
            f"{indent}acc1 = fma(x1, float(in_{right_position}["
            f"{right_batch_prefix}(k + 1u) * {columns}u + column]), acc1);",
            f"{indent}acc2 = fma(x2, float(in_{right_position}["
            f"{right_batch_prefix}(k + 2u) * {columns}u + column]), acc2);",
            f"{indent}acc3 = fma(x3, float(in_{right_position}["
            f"{right_batch_prefix}(k + 3u) * {columns}u + column]), acc3);",
        ]
    )
    if not exact_columns:
        lines.append("    }")
    lines.extend(
        [
            "}",
            f"for (; k < {reduction}u; ++k) {{",
        ]
    )
    if norm_slot is None:
        lines.append(f"    float x = float(in_{left_position}[{left_batch_prefix}k]);")
    else:
        lines.append(
            f"    float x = float((T)(norm_scale * "
            f"float(in_{left_position}[{left_batch_prefix}k]) * "
            f"float(in_{norm_slot}[k])));"
        )
    if not exact_columns:
        lines.append(f"    if (column < {columns}u) {{")
    indent = "        " if not exact_columns else "    "
    lines.append(
        f"{indent}acc0 = fma(x, float(in_{right_position}["
        f"{right_batch_prefix}k * {columns}u + column]), acc0);"
    )
    if not exact_columns:
        lines.append("    }")
    lines.append("}")
    if not exact_columns:
        lines.append(f"if (column < {columns}u) {{")
    indent = "    " if not exact_columns else ""
    output_batch_offset = f"batch * {columns}u + " if batch_count > 1 else ""
    output_linear = f"{output_batch_offset}column"
    value = "((acc0 + acc1) + (acc2 + acc3))"
    if bias_id is not None:
        assert bias_type is not None
        bias_offset = _offset_expression(
            bias_type.shape,
            output_shape,
            output_linear,
        )
        value = f"{beta!r} * float(in_0[{bias_offset}]) + {alpha!r} * {value}"
    value = _matmul_epilogue_value(value, op, graph, output_shape, output_linear)
    lines.append(
        f"{indent}out[{output_linear}] = (T)({value});"
    )
    if not exact_columns:
        lines.append("}")
    return "\n".join(f"        {line}" for line in lines)


def _emit_transposed_gemv_matmul_source(
    op: Operation,
    graph: Graph,
    reduction: int,
    columns: int,
    bias_id: str | None,
    bias_type: TensorType | None,
    output_shape: tuple[int, ...],
    alpha: float,
    beta: float,
    left_position: int,
    right_position: int,
    batch_count: int,
    left_batch_prefix: str,
    right_batch_prefix: str,
) -> str:
    """Emit warp-per-column GEMV for a physical ``[columns, reduction]`` weight."""

    threads = _gemv_threads(columns, True)
    warps = threads // 32
    exact_columns = columns % warps == 0
    output_batch_offset = f"batch * {columns}u + " if batch_count > 1 else ""
    output_linear = f"{output_batch_offset}column"
    paired_position = _matmul_paired_weight_position(op, right_position)
    if paired_position is not None and reduction % 4 != 0:
        raise ValueError(
            "the paired gate/up epilogue needs a reduction that is a multiple of 4"
        )
    value = "total"
    if bias_id is not None:
        assert bias_type is not None
        bias_offset = _offset_expression(
            bias_type.shape,
            output_shape,
            output_linear,
        )
        value = f"{beta!r} * float(in_0[{bias_offset}]) + {alpha!r} * total"
    if paired_position is None:
        value = _matmul_epilogue_value(value, op, graph, output_shape, output_linear)
    else:
        # ``silu`` of the gate row scaled by the paired up row, both accumulated
        # from the same activation in one pass over the two weight rows. The gate
        # is rounded to the stored element type first, which is exactly the value
        # the separate projection would have written, so the result is unchanged.
        value = "(total_up) * (paired_gate / (1.0f + metal::exp(-paired_gate)))"

    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = tid % 32u;",
        "uint warp = tid / 32u;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint column = threadgroup_index * {warps}u + warp;",
        ]
    )
    norm_slot = _matmul_norm_position(op)
    if norm_slot is not None:
        lines.extend(
            _emit_warp_rms_norm_prologue(
                op, norm_slot, left_position, reduction, left_batch_prefix
            )
        )
    if not exact_columns:
        lines.append(f"if (column < {columns}u) {{")
    indent = "    " if not exact_columns else ""
    # Vectorized fast path: one lane owns four consecutive weights of its row,
    # so a warp moves 256 bytes per load instruction instead of 64.  Pure
    # scalar loads cap the streaming rate well below what the GPU can sustain.
    vectorized = reduction % 4 == 0
    lines.extend(
        [
            f"{indent}float acc0 = 0.0f;",
            f"{indent}float acc1 = 0.0f;",
            f"{indent}float acc2 = 0.0f;",
            f"{indent}float acc3 = 0.0f;",
        ]
    )
    if paired_position is not None:
        lines.extend(
            [
                f"{indent}float acc4 = 0.0f;",
                f"{indent}float acc5 = 0.0f;",
                f"{indent}float acc6 = 0.0f;",
                f"{indent}float acc7 = 0.0f;",
            ]
        )
    if vectorized:
        lines.extend(
            [
                f"{indent}const device vec<T, 4>* row = "
                "reinterpret_cast<const device vec<T, 4>*>("
                f"in_{right_position} + {right_batch_prefix}"
                f"column * {reduction}u);",
                f"{indent}constexpr uint vectors = {reduction // 4}u;",
                f"{indent}for (uint index = lane; index < vectors; index += 32u) {{",
                f"{indent}    vec<T, 4> weight = row[index];",
                f"{indent}    uint base = index * 4u;",
            ]
        )
        if paired_position is not None:
            lines.extend(
                [
                    f"{indent}    const device vec<T, 4>* paired_row = "
                    "reinterpret_cast<const device vec<T, 4>*>("
                    f"in_{paired_position} + {right_batch_prefix}"
                    f"column * {reduction}u);",
                    f"{indent}    vec<T, 4> paired_weight = paired_row[index];",
                ]
            )
        if norm_slot is None:
            for offset in range(4):
                lines.append(
                    f"{indent}    float x{offset} = float(in_{left_position}"
                    f"[{left_batch_prefix}base + {offset}u]);"
                )
        else:
            lines.append(f"{indent}    vec<T, 4> norm_source_value = norm_source[index];")
            lines.append(f"{indent}    vec<T, 4> norm_weight_value = norm_weight_values[index];")
            for offset in range(4):
                lines.append(
                    f"{indent}    float x{offset} = float((T)(norm_scale * "
                    f"float(norm_source_value[{offset}]) * "
                    f"float(norm_weight_value[{offset}])));"
                )
        for offset in range(4):
            lines.append(
                f"{indent}    acc{offset} = fma(x{offset}, float(weight[{offset}]), "
                f"acc{offset});"
            )
        if paired_position is not None:
            for offset in range(4):
                lines.append(
                    f"{indent}    acc{offset + 4} = fma(x{offset}, "
                    f"float(paired_weight[{offset}]), acc{offset + 4});"
                )
        lines.append(f"{indent}}}")
    else:
        def left_term(index: str) -> str:
            if norm_slot is None:
                return f"float(in_{left_position}[{left_batch_prefix}{index}])"
            return (
                f"float((T)(norm_scale * float(in_{left_position}"
                f"[{left_batch_prefix}{index}]) * float(in_{norm_slot}[{index}])))"
            )

        lines.extend(
            [
                f"{indent}uint k = lane;",
                f"{indent}for (; k + 96u < {reduction}u; k += 128u) {{",
                f"{indent}    acc0 = fma({left_term('k')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k]), acc0);",
                f"{indent}    acc1 = fma({left_term('k + 32u')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k + 32u]), acc1);",
                f"{indent}    acc2 = fma({left_term('k + 64u')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k + 64u]), acc2);",
                f"{indent}    acc3 = fma({left_term('k + 96u')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k + 96u]), acc3);",
                f"{indent}}}",
                f"{indent}for (; k < {reduction}u; k += 32u) {{",
                f"{indent}    acc0 = fma({left_term('k')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k]), acc0);",
                f"{indent}}}",
            ]
        )
    lines.append(f"{indent}float total = simd_sum((acc0 + acc1) + (acc2 + acc3));")
    if paired_position is not None:
        lines.append(
            f"{indent}float total_up = simd_sum((acc4 + acc5) + (acc6 + acc7));"
        )
        lines.append(f"{indent}float paired_gate = float((T)total);")
    lines.extend(
        [
            f"{indent}if (lane == 0u) {{",
            f"{indent}    out[{output_linear}] = (T)({value});",
            f"{indent}}}",
        ]
    )
    if not exact_columns:
        lines.append("}")
    return "\n".join(f"        {line}" for line in lines)


def _matmul_batch_shape(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    if len(left_shape) < 2 or len(right_shape) < 2:
        return None

    left_batch = left_shape[:-2]
    right_batch = right_shape[:-2]
    batch_rank = max(len(left_batch), len(right_batch))
    batch_shape: list[int] = []
    for axis in range(batch_rank):
        left_dim = (
            left_batch[axis - (batch_rank - len(left_batch))]
            if axis >= batch_rank - len(left_batch)
            else 1
        )
        right_dim = (
            right_batch[axis - (batch_rank - len(right_batch))]
            if axis >= batch_rank - len(right_batch)
            else 1
        )
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            return None
        batch_shape.append(max(left_dim, right_dim))
    return tuple(batch_shape)


def _broadcast_batch_offset_expression(
    input_batch_shape: tuple[int, ...],
    output_batch_shape: tuple[int, ...],
    matrix_elements: int,
) -> str:
    """Map a flat output batch index to the source operand's matrix base."""

    if not output_batch_shape or prod(output_batch_shape) == 1:
        return ""
    if input_batch_shape == output_batch_shape:
        return f"batch * {matrix_elements}u"

    batch_offset = len(output_batch_shape) - len(input_batch_shape)
    terms: list[str] = []
    for input_axis, input_dim in enumerate(input_batch_shape):
        output_axis = input_axis + batch_offset
        if input_dim == 1:
            continue
        output_stride = prod(output_batch_shape[output_axis + 1 :])
        input_stride = prod(input_batch_shape[input_axis + 1 :]) * matrix_elements
        coordinate = (
            f"batch % {output_batch_shape[output_axis]}u"
            if output_stride == 1
            else f"((batch / {output_stride}u) % {output_batch_shape[output_axis]}u)"
        )
        terms.append(f"{coordinate} * {input_stride}u")
    return " + ".join(terms)


def _matmul_transpose_attribute(op: Operation, name: str) -> bool:
    value = op.attrs.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"matmul {name} must be a boolean")
    return value


def _emit_simdgroup_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    graph: Graph,
    dtype: DType,
    transpose_left: bool,
    transpose_right: bool,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
) -> str:
    if rows % _MATMUL_BLOCK_ROWS != 0 or columns % _MATMUL_BLOCK_COLUMNS != 0:
        raise ValueError("simdgroup matmul requires complete 64x64 output blocks")

    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    block_rows = _MATMUL_BLOCK_ROWS
    block_columns = _MATMUL_BLOCK_COLUMNS
    block_reduction = _MATMUL_BLOCK_REDUCTION
    threads = 256
    column_blocks = columns // block_columns
    exact_reduction = reduction % block_reduction == 0
    left_batch_prefix = f"{left_batch_offset} + " if left_batch_offset else ""
    right_batch_prefix = f"{right_batch_offset} + " if right_batch_offset else ""
    output_batch_offset = (
        f"batch * {rows * columns}u + " if batch_count > 1 else ""
    )
    load_vector_width = 8
    vector_loads = (
        exact_reduction
        and reduction % load_vector_width == 0
        and columns % load_vector_width == 0
        and not transpose_left
        and not transpose_right
    )
    tile_alignment = "alignas(16) " if vector_loads else ""
    simdgroup_type = {
        DType.F16: "half",
        DType.BF16: "bfloat",
    }[dtype]
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint warp = tid / 32u;",
        "uint warp_row = warp / 2u;",
        "uint warp_column = warp % 2u;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint block_row = threadgroup_index / {column_blocks}u;",
            f"uint block_column = threadgroup_index % {column_blocks}u;",
            f"uint row_base = block_row * {block_rows}u;",
            f"uint col_base = block_column * {block_columns}u;",
            f"{tile_alignment}threadgroup T tile_a[{block_rows * block_reduction}];",
            f"{tile_alignment}threadgroup T tile_b[{block_reduction * block_columns}];",
            f"threadgroup float tile_c[{block_rows * block_columns}];",
            "simdgroup_float8x8 acc[8];",
            f"simdgroup_{simdgroup_type}8x8 frag_a[2];",
            f"simdgroup_{simdgroup_type}8x8 frag_b[4];",
            "for (uint i = 0u; i < 8u; ++i) {",
            "    acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);",
            "}",
            f"for (uint k_base = 0u; k_base < {reduction}u; "
            f"k_base += {block_reduction}u) {{",
        ]
    )
    if vector_loads:
        lines.extend(
            [
                f"    for (uint vector_index = tid; "
                f"vector_index < {block_rows * block_reduction // load_vector_width}u; "
                f"vector_index += {threads}u) {{",
                f"        uint index = vector_index * {load_vector_width}u;",
                f"        uint local_row = index / {block_reduction}u;",
                f"        uint local_k = index % {block_reduction}u;",
                f"        threadgroup vec<T, {load_vector_width}>* destination =",
                f"            reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(",
                "                tile_a + index);",
                f"        const device vec<T, {load_vector_width}>* source =",
                f"            reinterpret_cast<const device vec<T, {load_vector_width}>*>(",
                f"                in_{left_position} + {left_batch_prefix}"
                f"(row_base + local_row) * {reduction}u",
                "                    + k_base + local_k);",
                "        *destination = *source;",
                "    }",
                f"    for (uint vector_index = tid; "
                f"vector_index < {block_reduction * block_columns // load_vector_width}u; "
                f"vector_index += {threads}u) {{",
                f"        uint index = vector_index * {load_vector_width}u;",
                f"        uint local_k = index / {block_columns}u;",
                f"        uint local_column = index % {block_columns}u;",
                f"        threadgroup vec<T, {load_vector_width}>* destination =",
                f"            reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(",
                "                tile_b + index);",
                f"        const device vec<T, {load_vector_width}>* source =",
                f"            reinterpret_cast<const device vec<T, {load_vector_width}>*>(",
                f"                in_{right_position} + {right_batch_prefix}"
                f"(k_base + local_k) * {columns}u",
                "                    + col_base + local_column);",
                "        *destination = *source;",
                "    }",
            ]
        )
    else:
        lines.extend(
            [
                f"    for (uint index = tid; "
                f"index < {block_rows * block_reduction}u; "
                f"index += {threads}u) {{",
                f"        uint local_row = index / {block_reduction}u;",
                f"        uint local_k = index % {block_reduction}u;",
            ]
        )
        if transpose_left:
            left_index = (
                f"(k_base + local_k) * {rows}u + row_base + local_row"
            )
        else:
            left_index = (
                f"(row_base + local_row) * {reduction}u + k_base + local_k"
            )
        if left_batch_offset:
            left_index = f"{left_batch_offset} + {left_index}"
        load_a = f"in_{left_position}[{left_index}]"
        if not exact_reduction:
            load_a = f"k_base + local_k < {reduction}u ? {load_a} : T(0.0)"
        lines.extend(
            [
                f"        tile_a[index] = {load_a};",
                "    }",
                f"    for (uint index = tid; "
                f"index < {block_reduction * block_columns}u; "
                f"index += {threads}u) {{",
                f"        uint local_k = index / {block_columns}u;",
                f"        uint local_column = index % {block_columns}u;",
            ]
        )
        if transpose_right:
            right_index = (
                f"(col_base + local_column) * {reduction}u + k_base + local_k"
            )
        else:
            right_index = (
                f"(k_base + local_k) * {columns}u + col_base + local_column"
            )
        if right_batch_offset:
            right_index = f"{right_batch_offset} + {right_index}"
        load_b = f"in_{right_position}[{right_index}]"
        if not exact_reduction:
            load_b = f"k_base + local_k < {reduction}u ? {load_b} : T(0.0)"
        lines.extend(
            [
                f"        tile_b[index] = {load_b};",
                "    }",
            ]
        )
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")

    for k_offset in range(0, block_reduction, 8):
        lines.extend(
            [
                f"    simdgroup_load(frag_a[0], tile_a + "
                f"(warp_row * 16u) * {block_reduction}u + {k_offset}u, "
                f"{block_reduction}, 0, false);",
                f"    simdgroup_load(frag_a[1], tile_a + "
                f"(warp_row * 16u + 8u) * {block_reduction}u + {k_offset}u, "
                f"{block_reduction}, 0, false);",
            ]
        )
        for column_fragment in range(4):
            lines.append(
                f"    simdgroup_load(frag_b[{column_fragment}], tile_b + "
                f"{k_offset}u * {block_columns}u + warp_column * 32u + "
                f"{column_fragment * 8}u, {block_columns}, 0, false);"
            )
        for row_fragment in range(2):
            for column_fragment in range(4):
                accumulator = row_fragment * 4 + column_fragment
                lines.append(
                    f"    simdgroup_multiply_accumulate(acc[{accumulator}], "
                    f"frag_a[{row_fragment}], frag_b[{column_fragment}], "
                    f"acc[{accumulator}]);"
                )

    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append("}")

    for row_fragment in range(2):
        row_offset = row_fragment * 8
        for column_fragment in range(4):
            accumulator = row_fragment * 4 + column_fragment
            column_offset = column_fragment * 8
            lines.append(
                f"simdgroup_store(acc[{accumulator}], tile_c + "
                f"(warp_row * 16u + {row_offset}u) * {block_columns}u + "
                f"warp_column * 32u + {column_offset}u, "
                f"{block_columns}, 0, false);"
            )
    lines.extend(
        [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"for (uint index = tid; index < {block_rows * block_columns}u; "
            f"index += {threads}u) {{",
            f"    uint local_row = index / {block_columns}u;",
            f"    uint local_column = index % {block_columns}u;",
            f"    uint linear = {output_batch_offset}(row_base + local_row) * {columns}u "
            "+ col_base + local_column;",
        ]
    )
    value = "tile_c[index]"
    if bias_id is not None:
        assert bias_type is not None
        bias_offset = _offset_expression(bias_type.shape, output_shape, "linear")
        value = (
            f"{beta!r} * float(in_0[{bias_offset}]) "
            f"+ {alpha!r} * {value}"
        )
    value = _matmul_epilogue_value(value, op, graph, output_shape, "linear")
    lines.extend(
        [
            f"    out[linear] = (T)({value});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


def _supports_mpp_matmul(
    rows: int,
    columns: int,
    reduction: int,
    dtype: DType,
) -> bool:
    """Return whether the tensor-core (MPP ``matmul2d``) path applies.

    The reduction must be a multiple of the 16-wide MMA fragment so that every
    cooperative-tensor load stays fully in bounds; everything else is handled
    by predicated stores and clamped input reads.
    """

    return (
        dtype in {DType.F16, DType.BF16, DType.F32}
        and rows >= 16
        and columns >= 16
        and reduction >= 16
        and reduction % 16 == 0
    )


def _select_mpp_gemm_config(
    rows: int,
    columns: int,
    reduction: int,
    dtype: DType,
    transposed_right: bool = True,
) -> tuple[int, int, int, int, int]:
    """Pick the tensor-core block tile for one concrete matmul shape."""

    wide = (
        _MPP_GEMM_TILE_TRANSPOSED_WIDE
        if transposed_right
        else _MPP_GEMM_TILE_ROW_MAJOR_WIDE
    )
    if dtype == DType.F32:
        preferred = _MPP_GEMM_TILE_F32
    elif not transposed_right and columns >= _MPP_GEMM_MAX_WIDE_COLUMNS:
        preferred = _MPP_GEMM_TILE_TALL
    elif reduction >= 2048 and rows <= _MPP_GEMM_SKINNY_MAX_ROWS:
        preferred = (
            _MPP_GEMM_TILE_SKINNY_LONG_K
            if columns <= _MPP_GEMM_SKINNY_MAX_COLUMNS
            else wide
        )
    elif rows * columns >= _MPP_GEMM_WIDE_ELEMENTS or (
        rows <= _MPP_GEMM_SKINNY_MAX_ROWS
        and columns >= _MPP_GEMM_MIN_WIDE_COLUMNS
    ):
        preferred = wide
    elif transposed_right or rows < _MPP_GEMM_TILE_ROW_MAJOR[0]:
        preferred = _MPP_GEMM_TILE_SMALL_K
    else:
        preferred = _MPP_GEMM_TILE_ROW_MAJOR
    row_bound = max(16, ceil(rows / 16) * 16)
    column_bound = max(32, ceil(columns / 32) * 32)
    for config in _MPP_GEMM_TILE_LADDER[
        _MPP_GEMM_TILE_LADDER.index(preferred) :
    ]:
        if config[0] <= row_bound and config[1] <= column_bound:
            return config
    return _MPP_GEMM_TILE_TINY


def _emit_mpp_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    graph: Graph,
    dtype: DType,
    transpose_left: bool,
    transpose_right: bool,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
    config: tuple[int, int, int, int, int],
) -> str:
    """Emit a ``mpp::tensor_ops::matmul2d`` GEMM for Apple tensor cores."""

    block_rows, block_columns, warp_rows, warp_columns, step = config
    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    warp_block_rows = block_rows // warp_rows
    warp_block_columns = block_columns // warp_columns
    row_fragments = warp_block_rows // 16
    column_pairs = warp_block_columns // 32
    step_fragments = step // 16
    if row_fragments < 1 or column_pairs < 1:
        raise ValueError("MPP matmul requires at least one 16x32 warp tile")

    full_steps, tail_fragments = divmod(reduction, step)
    tail_fragments //= 16
    guard_rows = rows % block_rows != 0
    guard_columns = columns % block_columns != 0
    left_batch_prefix = f"{left_batch_offset} + " if left_batch_offset else ""
    right_batch_prefix = f"{right_batch_offset} + " if right_batch_offset else ""
    output_batch_prefix = f"batch * {rows * columns}u + " if batch_count > 1 else ""

    lines: list[str] = []
    append = lines.append
    append("uint tid = thread_position_in_threadgroup.x;")
    append("uint lane = tid % 32u;")
    append("uint sg = tid / 32u;")
    append("uint qid = lane >> 2u;")
    append("uint fm = (qid & 4u) | ((lane >> 1u) & 3u);")
    append("uint fn = ((qid & 2u) | (lane & 1u)) * 4u;")
    if batch_count > 1:
        append("uint batch = threadgroup_position_in_grid.z;")
    append(
        f"uint row_base = threadgroup_position_in_grid.y * {block_rows}u"
        f" + (sg / {warp_columns}u) * {warp_block_rows}u;"
    )
    append(
        f"uint col_base = threadgroup_position_in_grid.x * {block_columns}u"
        f" + (sg % {warp_columns}u) * {warp_block_columns}u;"
    )
    append(
        "constexpr auto tsc_gemm_descriptor = "
        "mpp::tensor_ops::matmul2d_descriptor("
    )
    append("    16, 32, 16, false, true, true,")
    append(
        "    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);"
    )
    append(
        "mpp::tensor_ops::matmul2d<tsc_gemm_descriptor, "
        "metal::execution_simdgroup> tsc_gemm;"
    )
    append(
        "using tsc_at = decltype("
        "tsc_gemm.get_left_input_cooperative_tensor<T, T, float>());"
    )
    append(
        "using tsc_bt = decltype("
        "tsc_gemm.get_right_input_cooperative_tensor<T, T, float>());"
    )
    append(
        "using tsc_ct = decltype(tsc_gemm.get_destination_cooperative_tensor<"
        "metal::remove_addrspace_t<tsc_at>, "
        "metal::remove_addrspace_t<tsc_bt>, float>());"
    )
    for row_fragment in range(row_fragments):
        for column_pair in range(column_pairs):
            name = f"tsc_acc_{row_fragment}_{column_pair}"
            append(f"tsc_ct {name} = tsc_gemm.get_destination_cooperative_tensor<")
            append(
                "    metal::remove_addrspace_t<tsc_at>, "
                "metal::remove_addrspace_t<tsc_bt>, float>();"
            )
            append(f"for (uint e = 0u; e < 16u; ++e) {name}[e] = 0.0f;")

    def row_expression(row_fragment: int, row_offset: int) -> str:
        expression = f"row_base + {row_fragment * 16 + row_offset}u + fm"
        if guard_rows:
            return f"metal::min({expression}, {rows - 1}u)"
        return expression

    def column_load_expression(column_fragment: int, column_offset: int) -> str:
        # The right cooperative tensor indexes its ``n`` rows with ``fm``.
        expression = f"col_base + {column_fragment * 16 + column_offset}u + fm"
        if guard_columns:
            return f"metal::min({expression}, {columns - 1}u)"
        return expression

    def column_store_expression(column_fragment: int, column_offset: int) -> str:
        # The destination cooperative tensor indexes its columns with ``fn``.
        expression = f"col_base + {column_fragment * 16 + column_offset}u + fn"
        return expression

    def k_expression(origin: str | int, offset: int) -> str:
        # Every lane owns ``fn`` of the 16-wide reduction fragment.
        if isinstance(origin, int):
            return f"{origin + offset}u + fn"
        return f"{origin} + {offset}u + fn"

    def emit_declarations(
        fragments: int, prefix: str, indent: str
    ) -> list[str]:
        emitted: list[str] = []
        for row_fragment in range(row_fragments):
            for fragment in range(fragments):
                emitted.append(
                    f"{indent}tsc_at {prefix}ca_{row_fragment}_{fragment} ="
                    " tsc_gemm.get_left_input_cooperative_tensor<T, T, float>();"
                )
        for column_pair in range(column_pairs):
            for fragment in range(fragments):
                emitted.append(
                    f"{indent}tsc_bt {prefix}cb_{fragment}_{column_pair} ="
                    " tsc_gemm.get_right_input_cooperative_tensor<T, T, float>();"
                )
        return emitted

    def emit_body(
        origin: str | int, fragments: int, prefix: str, indent: str
    ) -> list[str]:
        emitted: list[str] = []
        for row_fragment in range(row_fragments):
            for fragment in range(fragments):
                for half in range(2):
                    for element in range(4):
                        k = k_expression(origin, fragment * 16 + element)
                        target = f"{prefix}ca_{row_fragment}_{fragment}"
                        if transpose_left:
                            index = (
                                f"{left_batch_prefix}({k})"
                                f" * {rows}u + ({row_expression(row_fragment, half * 8)})"
                            )
                        else:
                            index = (
                                f"{left_batch_prefix}"
                                f"({row_expression(row_fragment, half * 8)})"
                                f" * {reduction}u + ({k})"
                            )
                        emitted.append(
                            f"{indent}{target}[{half * 4 + element}] ="
                            f" in_{left_position}[{index}];"
                        )
        for column_pair in range(column_pairs):
            for fragment in range(fragments):
                for half in range(2):
                    for nf in range(2):
                        column_fragment = column_pair * 2 + nf
                        for element in range(4):
                            k = k_expression(origin, fragment * 16 + element)
                            target = f"{prefix}cb_{fragment}_{column_pair}"
                            if transpose_right:
                                index = (
                                    f"{right_batch_prefix}"
                                    f"({column_load_expression(column_fragment, half * 8)})"
                                    f" * {reduction}u + ({k})"
                                )
                            else:
                                index = (
                                    f"{right_batch_prefix}({k})"
                                    f" * {columns}u"
                                    f" + ({column_load_expression(column_fragment, half * 8)})"
                                )
                            emitted.append(
                                f"{indent}{target}[{nf * 8 + half * 4 + element}] ="
                                f" in_{right_position}[{index}];"
                            )
        for row_fragment in range(row_fragments):
            for column_pair in range(column_pairs):
                for fragment in range(fragments):
                    emitted.append(
                        f"{indent}tsc_gemm.run({prefix}ca_{row_fragment}_{fragment},"
                        f" {prefix}cb_{fragment}_{column_pair},"
                        f" tsc_acc_{row_fragment}_{column_pair});"
                    )
        return emitted

    if full_steps > 0:
        lines.extend(emit_declarations(step_fragments, "loop_", ""))
        append(
            f"for (uint k0 = 0u; k0 < {full_steps * step}u; k0 += {step}u) {{"
        )
        lines.extend(emit_body("k0", step_fragments, "loop_", "    "))
        append("}")
    if tail_fragments > 0:
        lines.extend(emit_declarations(tail_fragments, "tail_", ""))
        lines.extend(emit_body(full_steps * step, tail_fragments, "tail_", ""))

    for row_fragment in range(row_fragments):
        for column_pair in range(column_pairs):
            for nf in range(2):
                column_fragment = column_pair * 2 + nf
                for half in range(2):
                    for element in range(4):
                        row = row_expression(row_fragment, half * 8)
                        column = column_store_expression(column_fragment, element)
                        linear = f"({row}) * {columns}u + ({column})"
                        value = f"float(tsc_acc_{row_fragment}_{column_pair}[{nf * 8 + half * 4 + element}])"
                        if bias_id is not None:
                            assert bias_type is not None
                            bias_index = _offset_expression(
                                bias_type.shape, output_shape, linear
                            )
                            value = (
                                f"{beta!r} * float(in_0[{bias_index}])"
                                f" + {alpha!r} * {value}"
                            )
                        elif alpha != 1.0:
                            value = f"{alpha!r} * {value}"
                        value = _matmul_epilogue_value(
                            value, op, graph, output_shape, linear
                        )
                        conditions = []
                        if guard_rows:
                            conditions.append(
                                f"row_base + {row_fragment * 16 + half * 8}u"
                                f" + fm < {rows}u"
                            )
                        if guard_columns:
                            conditions.append(
                                f"col_base + {column_fragment * 16 + element}u"
                                f" + fn < {columns}u"
                            )
                        guard = (
                            f"if ({' && '.join(conditions)}) " if conditions else ""
                        )
                        append(
                            f"{guard}out[{output_batch_prefix}{linear}] = (T)({value});"
                        )
    return "\n".join(lines)


_MPP_ATTENTION_KEY_TILE = 32
_MPP_ATTENTION_MAX_HEAD_DIM = 128
_MPP_ATTENTION_MIN_QUERY_LENGTH = 16
_MPP_ATTENTION_MAX_WARPS = 2


def _supports_mpp_attention(
    query_type: TensorType,
    key_type: TensorType,
    value_type: TensorType,
    output_type: TensorType,
    mask_shape: tuple[int, ...] | None,
) -> bool:
    """Tensor-core attention needs 16-row query fragments and 32-wide values."""

    if query_type.dtype not in {DType.F16, DType.BF16}:
        return False
    for tensor_type in (key_type, value_type, output_type):
        if tensor_type.dtype != query_type.dtype:
            return False
    rank = len(query_type.shape)
    if rank not in {3, 4}:
        return False
    if len(key_type.shape) != rank or len(value_type.shape) != rank:
        return False
    query_head_dim = query_type.shape[-1]
    value_head_dim = value_type.shape[-1]
    if (
        query_head_dim % 16 != 0
        or not 16 <= query_head_dim <= _MPP_ATTENTION_MAX_HEAD_DIM
    ):
        return False
    if (
        value_head_dim % 32 != 0
        or not 32 <= value_head_dim <= _MPP_ATTENTION_MAX_HEAD_DIM
    ):
        return False
    if output_type.shape[-1] != value_head_dim:
        return False
    if query_type.shape[-2] < _MPP_ATTENTION_MIN_QUERY_LENGTH:
        return False
    if key_type.shape[-2] <= 0:
        return False
    if mask_shape is not None and not _native_attention_mask_supported(
        rank, mask_shape
    ):
        return False
    batch_axis = -3 if rank == 4 else -2
    batch_shape = query_type.shape[:batch_axis]
    if key_type.shape[:batch_axis] != batch_shape:
        return False
    if value_type.shape[:batch_axis] != batch_shape:
        return False
    query_heads = query_type.shape[-3] if rank == 4 else 1
    key_value_heads = key_type.shape[-3] if rank == 4 else 1
    if key_value_heads <= 0 or query_heads % key_value_heads != 0:
        return False
    return True


def _select_mpp_attention_warps(query_length: int) -> int:
    return max(
        1,
        min(_MPP_ATTENTION_MAX_WARPS, query_length // 16),
    )


def _emit_mpp_attention_source(
    *,
    query_length: int,
    key_length: int,
    query_heads: int,
    key_value_heads: int,
    query_head_dim: int,
    value_head_dim: int,
    warps: int,
    scale: float,
    has_mask: bool,
    mask_shape: tuple[int, ...] | None,
    mask_is_boolean: bool,
    causal: bool,
    rank: int,
) -> str:
    """Emit a register-resident FlashAttention pass on Apple tensor cores.

    Each simdgroup owns a 16-row query fragment and the full value width, so the
    online-softmax accumulators never leave registers. ``Q`` is staged once in
    threadgroup memory; ``K``/``V`` stream straight from device memory.
    """

    query_tile = warps * 16
    key_tile = _MPP_ATTENTION_KEY_TILE
    dimension_fragments = query_head_dim // 16
    value_pairs = value_head_dim // 32
    query_blocks = ceil(query_length / query_tile)
    group_size = query_heads // key_value_heads
    threads = warps * 32
    scores_per_half = 4

    if rank == 4:
        query_batch_head = f"(batch * {query_heads}u + head)"
        key_value_batch_head = (
            f"(batch * {key_value_heads}u + head / {group_size}u)"
        )
        output_batch_head = query_batch_head
    else:
        query_batch_head = "batch"
        key_value_batch_head = "batch"
        output_batch_head = "batch"

    lines: list[str] = []
    append = lines.append
    append("uint tid = thread_position_in_threadgroup.x;")
    append("uint lane = tid % 32u;")
    append("uint sg = tid / 32u;")
    append("uint qid = lane >> 2u;")
    append("uint fm = (qid & 4u) | ((lane >> 1u) & 3u);")
    append("uint fn = ((qid & 2u) | (lane & 1u)) * 4u;")
    append("uint query_index = threadgroup_position_in_grid.x;")
    append(
        f"uint batch = query_index / {query_heads * query_blocks}u;"
    )
    append(
        f"uint batch_index = query_index % {query_heads * query_blocks}u;"
    )
    append(f"uint head = batch_index / {query_blocks}u;")
    append(f"uint query_block = batch_index % {query_blocks}u;")
    append(
        f"threadgroup T tile_q[{query_tile * query_head_dim}];"
    )
    append(
        f"for (uint index = tid; index < {query_tile * query_head_dim}u;"
        f" index += {threads}u) {{"
    )
    append(f"    uint row = index / {query_head_dim}u;")
    append(f"    uint dimension = index % {query_head_dim}u;")
    append(
        f"    uint position = metal::min(query_block * {query_tile}u + row,"
        f" {query_length - 1}u);"
    )
    append(
        f"    tile_q[index] = in_0[({query_batch_head} * {query_length}u"
        f" + position) * {query_head_dim}u + dimension];"
    )
    append("}")
    append("threadgroup_barrier(mem_flags::mem_threadgroup);")

    # The descriptor always declares the transposed right operand; the physical
    # row-major layout is absorbed by the per-operand index expressions.
    for tag in ("qk", "pv"):
        append(
            f"constexpr auto tsc_{tag}_descriptor = "
            "mpp::tensor_ops::matmul2d_descriptor("
        )
        append("    16, 32, 16, false, true, true,")
        append(
            "    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);"
        )
        append(
            f"mpp::tensor_ops::matmul2d<tsc_{tag}_descriptor,"
            " metal::execution_simdgroup> tsc_"
            f"{tag};"
        )
        append(
            f"using tsc_{tag}_at = decltype("
            f"tsc_{tag}.get_left_input_cooperative_tensor<T, T, float>());"
        )
        append(
            f"using tsc_{tag}_bt = decltype("
            f"tsc_{tag}.get_right_input_cooperative_tensor<T, T, float>());"
        )
        append(
            f"using tsc_{tag}_ct = decltype(tsc_{tag}."
            f"get_destination_cooperative_tensor<metal::remove_addrspace_t<tsc_{tag}_at>,"
            f" metal::remove_addrspace_t<tsc_{tag}_bt>, float>());"
        )

    for pair in range(value_pairs):
        append(
            f"tsc_pv_ct tsc_acc_{pair} = tsc_pv."
            "get_destination_cooperative_tensor<"
            "metal::remove_addrspace_t<tsc_pv_at>,"
            " metal::remove_addrspace_t<tsc_pv_bt>, float>();"
        )
        append(f"for (uint e = 0u; e < 16u; ++e) {{ tsc_acc_{pair}[e] = 0.0f; }}")

    for half in range(2):
        append(f"float running_max_{half} = -INFINITY;")
        append(f"float running_sum_{half} = 0.0f;")

    if causal:
        append(
            f"uint key_limit = metal::min({key_length}u,"
            f" query_block * {query_tile}u + {query_tile}u);"
        )
    else:
        append(f"uint key_limit = {key_length}u;")

    append(f"for (uint key_base = 0u; key_base < key_limit; key_base += {key_tile}u) {{")
    append(
        f"    const device T* key_head = in_1 + {key_value_batch_head}"
        f" * {key_length * query_head_dim}u;"
    )
    append(
        f"    const device T* value_head = in_2 + {key_value_batch_head}"
        f" * {key_length * value_head_dim}u;"
    )

    # --- Q @ K^T ---------------------------------------------------------
    append(
        "    tsc_qk_ct tsc_scores = tsc_qk."
        "get_destination_cooperative_tensor<"
        "metal::remove_addrspace_t<tsc_qk_at>,"
        " metal::remove_addrspace_t<tsc_qk_bt>, float>();"
    )
    append("    for (uint e = 0u; e < 16u; ++e) { tsc_scores[e] = 0.0f; }")
    for half in range(2):
        append(
            f"    uint query_row_{half} = sg * 16u + {half * 8}u + fm;"
        )
        append(
            f"    uint query_pos_{half} = query_block * {query_tile}u"
            f" + query_row_{half};"
        )
    for row in range(4):
        expression = (
            f"metal::min(key_base + {row * 8}u + fm, {key_length - 1}u)"
        )
        if key_length % key_tile == 0:
            expression = f"key_base + {row * 8}u + fm"
        append(f"    uint key_row_{row} = {expression};")
    append("    vec<T, 4> tsc_vector;")
    for fragment in range(dimension_fragments):
        append("    {")
        append("        tsc_qk_at tsc_q; tsc_qk_bt tsc_k;")
        for half in range(2):
            append(
                "        tsc_vector = *reinterpret_cast<"
                f"const threadgroup vec<T, 4>*>(tile_q + query_row_{half}"
                f" * {query_head_dim}u + {fragment * 16}u + fn);"
            )
            for element in range(4):
                append(
                    f"        tsc_q[{half * 4 + element}] = tsc_vector[{element}];"
                )
        for nf in range(2):
            for half in range(2):
                row = nf * 2 + half
                append(
                    "        tsc_vector = *reinterpret_cast<"
                    f"const device vec<T, 4>*>(key_head + key_row_{row}"
                    f" * {query_head_dim}u + {fragment * 16}u + fn);"
                )
                for element in range(4):
                    append(
                        f"        tsc_k[{nf * 8 + half * 4 + element}]"
                        f" = tsc_vector[{element}];"
                    )
        append("        tsc_qk.run(tsc_q, tsc_k, tsc_scores);")
        append("    }")

    # --- scale, mask, causal --------------------------------------------
    mask_offset_expression = (
        _attention_mask_offset_expression(rank, mask_shape)
        if has_mask and mask_shape is not None
        else ""
    )
    append("    float scores[16];")
    for half in range(2):
        if has_mask and mask_shape is not None:
            append(
                f"    uint mask_row_{half} = metal::min(query_pos_{half},"
                f" {query_length - 1}u);"
            )
    for element in range(16):
        nf = element >> 3
        half = (element >> 2) & 1
        offset = element & 3
        append(
            f"    scores[{element}] = tsc_scores[{element}] * {float(scale)!r};"
        )
        if has_mask and mask_shape is not None:
            mask_offset = mask_offset_expression.replace(
                "query_pos", f"mask_row_{half}"
            )
            append("    {")
            append(f"        uint key = key_base + {nf * 16 + offset}u + fn;")
            if mask_is_boolean:
                append(
                    f"        if (!in_3[{mask_offset}])"
                    f" {{ scores[{element}] = -INFINITY; }}"
                )
            else:
                append(f"        scores[{element}] += float(in_3[{mask_offset}]);")
            append("    }")
    if key_length % key_tile != 0:
        append(f"    if (key_base + {key_tile}u > {key_length}u) {{")
        for element in range(16):
            nf = element >> 3
            offset = element & 3
            append(
                f"        if (key_base + {nf * 16 + offset}u + fn"
                f" >= {key_length}u) {{ scores[{element}] = -INFINITY; }}"
            )
        append("    }")
    if causal:
        append(
            f"    if (key_base + {key_tile}u > query_block * {query_tile}u"
            f" + sg * 16u) {{"
        )
        for element in range(16):
            nf = element >> 3
            half = (element >> 2) & 1
            offset = element & 3
            append(
                f"        if (key_base + {nf * 16 + offset}u + fn"
                f" > query_pos_{half}) {{ scores[{element}] = -INFINITY; }}"
            )
        append("    }")

    # --- online softmax --------------------------------------------------
    for half in range(2):
        elements = [
            nf * 8 + half * scores_per_half + offset
            for nf in range(2)
            for offset in range(scores_per_half)
        ]
        local_max = f"scores[{elements[0]}]"
        for element in elements[1:]:
            local_max = f"metal::max({local_max}, scores[{element}])"
        append(f"    float block_max_{half} = {local_max};")
        append(
            f"    block_max_{half} = metal::max(block_max_{half},"
            f" simd_shuffle(block_max_{half}, (uint)(lane ^ 1u)));"
        )
        append(
            f"    block_max_{half} = metal::max(block_max_{half},"
            f" simd_shuffle(block_max_{half}, (uint)(lane ^ 8u)));"
        )
        append(
            f"    float new_max_{half} ="
            f" metal::max(running_max_{half}, block_max_{half});"
        )
        append(
            f"    float alpha_{half} = running_max_{half} == -INFINITY"
            f" ? 0.0f : metal::fast::exp(running_max_{half} - new_max_{half});"
        )
        append(f"    float block_sum_{half} = 0.0f;")
        for element in elements:
            append(
                f"    {{ float p = scores[{element}] == -INFINITY ? 0.0f"
                f" : metal::fast::exp(scores[{element}] - new_max_{half});"
                f" scores[{element}] = p; block_sum_{half} += p; }}"
            )
        append(
            f"    block_sum_{half} += simd_shuffle(block_sum_{half},"
            f" (uint)(lane ^ 1u));"
        )
        append(
            f"    block_sum_{half} += simd_shuffle(block_sum_{half},"
            f" (uint)(lane ^ 8u));"
        )
        append(
            f"    running_sum_{half} = running_sum_{half} * alpha_{half}"
            f" + block_sum_{half};"
        )
        append(f"    running_max_{half} = new_max_{half};")

    append("    if (alpha_0 != 1.0f || alpha_1 != 1.0f) {")
    for pair in range(value_pairs):
        for element in range(16):
            half = (element >> 2) & 1
            append(f"        tsc_acc_{pair}[{element}] *= alpha_{half};")
    append("    }")

    # --- P @ V -----------------------------------------------------------
    for fragment in range(2):
        append("    {")
        append("        tsc_pv_at tsc_p;")
        for half in range(2):
            for element in range(4):
                append(
                    f"        tsc_p[{half * 4 + element}] = (T)scores["
                    f"{fragment * 8 + half * 4 + element}];"
                )
        for pair in range(value_pairs):
            append("        {")
            append("            tsc_pv_bt tsc_v;")
            for half in range(2):
                for nf in range(2):
                    for element in range(4):
                        append(
                            f"            tsc_v[{nf * 8 + half * 4 + element}]"
                            f" = value_head[metal::min(key_base +"
                            f" {fragment * 16 + element}u + fn,"
                            f" {key_length - 1}u) * {value_head_dim}u"
                            f" + {pair * 32 + nf * 16 + half * 8}u + fm];"
                        )
            append(
                f"            tsc_pv.run(tsc_p, tsc_v, tsc_acc_{pair});"
            )
            append("        }")
        append("    }")

    append("}")

    # --- normalize and store --------------------------------------------
    for half in range(2):
        append("if (")
        append(
            f"    query_block * {query_tile}u + sg * 16u + {half * 8}u + fm"
            f" < {query_length}u) {{"
        )
        append(
            f"    float inverse_sum = running_sum_{half} == 0.0f"
            " ? 0.0f : 1.0f / running_sum_"
            f"{half};"
        )
        append(
            f"    uint output_row = (({output_batch_head} * {query_length}u"
            f" + query_block * {query_tile}u + sg * 16u + {half * 8}u + fm)"
            f" * {value_head_dim}u);"
        )
        for pair in range(value_pairs):
            for nf in range(2):
                for element in range(4):
                    index = nf * 8 + half * 4 + element
                    append(
                        f"    out[output_row + {pair * 32 + nf * 16 + element}u"
                        f" + fn] = (T)(tsc_acc_{pair}[{index}] * inverse_sum);"
                    )
        append("}")
    return "\n".join(lines)


def _emit_tiled_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    micro_rows: int,
    micro_columns: int,
    graph: Graph,
    transpose_left: bool,
    transpose_right: bool,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
) -> str:
    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    block_rows = _MATMUL_BLOCK_ROWS
    block_columns = _MATMUL_BLOCK_COLUMNS
    block_reduction = _MATMUL_BLOCK_REDUCTION
    micro_rows_per_block = block_rows // micro_rows
    micro_columns_per_block = block_columns // micro_columns
    threads = micro_rows_per_block * micro_columns_per_block
    column_blocks = ceil(columns / block_columns)
    exact_rows = rows % block_rows == 0
    exact_columns = columns % block_columns == 0
    exact_reduction = reduction % block_reduction == 0
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint block_row = threadgroup_index / {column_blocks}u;",
            f"uint block_column = threadgroup_index % {column_blocks}u;",
            f"uint row_base = block_row * {block_rows}u;",
            f"uint col_base = block_column * {block_columns}u;",
            f"uint micro_row = tid / {micro_columns_per_block}u;",
            f"uint micro_column = tid % {micro_columns_per_block}u;",
            f"threadgroup float tile_a[{block_rows * block_reduction}];",
            f"threadgroup float tile_b[{block_reduction * block_columns}];",
        ]
    )
    for row in range(micro_rows):
        for column in range(micro_columns):
            lines.append(f"float acc_{row}_{column} = 0.0f;")
    lines.append(
        f"for (uint k_base = 0u; k_base < {reduction}u; "
        f"k_base += {block_reduction}u) {{"
    )

    lines.append(
        f"    for (uint index = tid; index < {block_rows * block_reduction}u; "
        f"index += {threads}u) {{"
    )
    lines.append(f"        uint local_row = index / {block_reduction}u;")
    lines.append(f"        uint local_k = index % {block_reduction}u;")
    if transpose_left:
        left_index = f"(k_base + local_k) * {rows}u + row_base + local_row"
    else:
        left_index = f"(row_base + local_row) * {reduction}u + k_base + local_k"
    if left_batch_offset:
        left_index = f"{left_batch_offset} + {left_index}"
    load_a = f"float(in_{left_position}[{left_index}])"
    a_guards = []
    if not exact_rows:
        a_guards.append(f"row_base + local_row < {rows}u")
    if not exact_reduction:
        a_guards.append(f"k_base + local_k < {reduction}u")
    if a_guards:
        load_a = f"{' && '.join(a_guards)} ? {load_a} : 0.0f"
    lines.append(f"        tile_a[index] = {load_a};")
    lines.append("    }")

    lines.append(
        f"    for (uint index = tid; index < {block_reduction * block_columns}u; "
        f"index += {threads}u) {{"
    )
    lines.append(f"        uint local_k = index / {block_columns}u;")
    lines.append(f"        uint local_column = index % {block_columns}u;")
    if transpose_right:
        right_index = (
            f"(col_base + local_column) * {reduction}u + k_base + local_k"
        )
    else:
        right_index = f"(k_base + local_k) * {columns}u + col_base + local_column"
    if right_batch_offset:
        right_index = f"{right_batch_offset} + {right_index}"
    load_b = f"float(in_{right_position}[{right_index}])"
    b_guards = []
    if not exact_columns:
        b_guards.append(f"col_base + local_column < {columns}u")
    if not exact_reduction:
        b_guards.append(f"k_base + local_k < {reduction}u")
    if b_guards:
        load_b = f"{' && '.join(b_guards)} ? {load_b} : 0.0f"
    lines.append(f"        tile_b[index] = {load_b};")
    lines.append("    }")
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")

    lines.append(f"    for (uint local_k = 0u; local_k < {block_reduction}u; ++local_k) {{")
    for row in range(micro_rows):
        lines.append(
            f"        float av_{row} = tile_a[(micro_row * {micro_rows}u + {row}u) "
            f"* {block_reduction}u + local_k];"
        )
    for column in range(micro_columns):
        lines.append(
            f"        float bv_{column} = tile_b[local_k * {block_columns}u "
            f"+ micro_column * {micro_columns}u + {column}u];"
        )
    for row in range(micro_rows):
        for column in range(micro_columns):
            lines.append(
                f"        acc_{row}_{column} = fma("
                f"av_{row}, bv_{column}, acc_{row}_{column});"
            )
    lines.append("    }")
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append("}")

    exact_tiles = exact_rows and exact_columns
    if not exact_tiles:
        for row in range(micro_rows):
            lines.append(
                f"bool row_ok_{row} = "
                f"row_base + micro_row * {micro_rows}u + {row}u < {rows}u;"
            )
        for column in range(micro_columns):
            lines.append(
                f"bool col_ok_{column} = "
                f"col_base + micro_column * {micro_columns}u + {column}u < {columns}u;"
            )

    for row in range(micro_rows):
        for column in range(micro_columns):
            linear = f"linear_{row}_{column}"
            if not exact_tiles:
                lines.append(f"if (row_ok_{row} && col_ok_{column}) {{")
            indent = "    " if not exact_tiles else ""
            batch_offset = (
                f"batch * {rows * columns}u + " if batch_count > 1 else ""
            )
            lines.append(
                f"{indent}uint {linear} = "
                f"{batch_offset}"
                f"(row_base + micro_row * {micro_rows}u + {row}u) * {columns}u "
                f"+ col_base + micro_column * {micro_columns}u + {column}u;"
            )
            accumulator = f"acc_{row}_{column}"
            if bias_id is not None:
                assert bias_type is not None
                bias_offset = _offset_expression(bias_type.shape, output_shape, linear)
                value = (
                    f"{beta!r} * float(in_0[{bias_offset}]) "
                    f"+ {alpha!r} * {accumulator}"
                )
            else:
                value = accumulator
            value = _matmul_epilogue_value(value, op, graph, output_shape, linear)
            lines.append(f"{indent}out[{linear}] = (T)({value});")
            if not exact_tiles:
                lines.append("}")

    return "\n".join(f"        {line}" for line in lines)


def _evaluate_layout_mlx(
    graph: Graph, op: Operation, value: mx.array
) -> mx.array:
    if op.op == "reshape":
        return mx.reshape(value, graph.value(op.outputs[0]).type.shape)

    rank = len(value.shape)
    if op.op == "transpose":
        dim0 = int(op.attrs["dim0"])
        dim1 = int(op.attrs["dim1"])
        dim0 = dim0 + rank if dim0 < 0 else dim0
        dim1 = dim1 + rank if dim1 < 0 else dim1
        return mx.swapaxes(value, dim0, dim1)

    if op.op == "permute":
        dims = tuple(int(dim) for dim in op.attrs["dims"])
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        return mx.transpose(value, normalized)

    if op.op == "expand":
        return mx.broadcast_to(value, graph.value(op.outputs[0]).type.shape)

    if op.op == "slice":
        index: list[slice] = [slice(None)] * rank
        axis = int(op.attrs["dim"]) % rank
        start = (
            int(op.attrs["start"]) if op.attrs.get("start") is not None else None
        )
        end = int(op.attrs["end"]) if op.attrs.get("end") is not None else None
        step = int(op.attrs["step"])
        start, end, step = slice(start, end, step).indices(value.shape[axis])
        index[axis] = slice(start, end, step)
        return value[tuple(index)]

    if op.op == "flip":
        dims = tuple(int(dim) for dim in op.attrs["dims"])
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        return mx.flip(value, axis=normalized)

    if op.op == "roll":
        shifts = tuple(int(shift) for shift in op.attrs["shifts"])
        dims = op.attrs.get("dims")
        if dims is None:
            return mx.roll(value, shift=shifts[0], axis=None)
        normalized = tuple(
            int(dim) + rank if int(dim) < 0 else int(dim) for dim in dims
        )
        return mx.roll(value, shift=shifts, axis=normalized)

    if op.op == "unsqueeze":
        output_rank = rank + 1
        dim = int(op.attrs["dim"])
        axis = dim + output_rank if dim < 0 else dim
        return mx.expand_dims(value, axis=axis)

    if op.op == "squeeze":
        dim = int(op.attrs["dim"])
        axis = dim + rank if dim < 0 else dim
        if value.shape[axis] != 1:
            return value
        return mx.squeeze(value, axis=axis)

    if op.op == "contiguous":
        return mx.contiguous(value)

    if op.op == "alias":
        return value

    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _evaluate_pointwise_mlx(
    op: Operation, env: dict[str, mx.array], output_type: TensorType
) -> mx.array:
    arguments = [env[value_id] for value_id in op.inputs]
    if op.op == "scalar_tensor":
        value = op.attrs.get("value")
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"scalar_tensor.value must be scalar, got {value!r}")
        return mx.array(value, dtype=_MLX_DTYPES[output_type.dtype])
    if op.op in {
        "zeros",
        "ones",
        "full",
        "new_zeros",
        "new_ones",
        "new_full",
        "zeros_like",
        "ones_like",
        "full_like",
    }:
        value = op.attrs.get("value")
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
        fill = mx.array(value, dtype=_MLX_DTYPES[output_type.dtype])
        return mx.full(output_type.shape, fill)
    if op.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "eq":
            return left == right
        if op.op == "ne":
            return left != right
        if op.op == "lt":
            return left < right
        if op.op == "le":
            return left <= right
        if op.op == "gt":
            return left > right
        return left >= right
    if op.op == "logical_and":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_and(left, right)
    if op.op == "logical_or":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_or(left, right)
    if op.op == "logical_xor":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_xor(left, right)
    if op.op == "logical_not":
        return mx.logical_not(arguments[0])
    if op.op == "bitwise_and":
        left, right = _binary_mx_operands(op, arguments)
        return mx.bitwise_and(left, right)
    if op.op == "bitwise_or":
        left, right = _binary_mx_operands(op, arguments)
        return mx.bitwise_or(left, right)
    if op.op == "bitwise_xor":
        left, right = _binary_mx_operands(op, arguments)
        return mx.bitwise_xor(left, right)
    if op.op == "bitwise_not":
        return ~arguments[0]
    if op.op == "isnan":
        return mx.isnan(arguments[0])
    if op.op == "isinf":
        return mx.isinf(arguments[0])
    if op.op == "where":
        return mx.where(arguments[0], arguments[1], arguments[2])
    if op.op == "masked_fill":
        fill = mx.array(op.attrs["value"], dtype=arguments[0].dtype)
        return mx.where(arguments[1], fill, arguments[0])
    if op.op == "silu":
        value = arguments[0]
        return value / (1.0 + mx.exp(-value))
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            inner = 0.7978845608028654 * (
                value + 0.044715 * value * value * value
            )
            return 0.5 * value * (1.0 + mx.tanh(inner))
        return 0.5 * value * (1.0 + mx.erf(value * 0.7071067811865476))
    if op.op == "softplus":
        value = arguments[0]
        beta = float(op.attrs["beta"])
        threshold = float(op.attrs["threshold"])
        scaled = beta * value
        return mx.where(
            scaled > threshold,
            value,
            mx.log1p(mx.exp(scaled)) / beta,
        )
    if op.op == "mish":
        value = arguments[0]
        softplus = mx.where(
            value > 20.0,
            value,
            mx.log1p(mx.exp(value)),
        )
        return value * mx.tanh(softplus)
    if op.op == "logsigmoid":
        value = arguments[0]
        body = mx.minimum(value, 0.0) - mx.log1p(mx.exp(-mx.abs(value)))
        return mx.where(mx.isnan(value), value, body)
    if op.op in {
        "mul",
        "div",
        "floor_divide",
        "remainder",
        "add",
        "sub",
        "rsub",
        "pow",
    }:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "mul":
            return left * right
        if op.op == "div":
            return left / right
        if op.op == "floor_divide":
            return mx.floor_divide(left, right)
        if op.op == "remainder":
            return mx.remainder(left, right)
        if op.op == "pow":
            return mx.power(left, right)
        alpha = float(op.attrs.get("alpha", 1.0))
        if op.op == "add":
            return left + alpha * right
        return left - alpha * right
    if op.op in {"maximum", "minimum"}:
        left, right = _binary_mx_operands(op, arguments)
        comparison = left < right if op.op == "maximum" else left > right
        selected = mx.where(comparison, right, left)
        return mx.where(mx.isnan(left) | mx.isnan(right), float("nan"), selected)
    if op.op == "logaddexp":
        left, right = _binary_mx_operands(op, arguments)
        maximum = mx.where(left < right, right, left)
        body = maximum + mx.log1p(mx.exp(-mx.abs(left - right)))
        body = mx.where(mx.isinf(maximum), maximum, body)
        return mx.where(mx.isnan(left) | mx.isnan(right), float("nan"), body)
    if op.op == "atan2":
        left, right = _binary_mx_operands(op, arguments)
        return mx.arctan2(left, right)
    if op.op == "neg":
        return -arguments[0]
    if op.op == "abs":
        return mx.abs(arguments[0])
    if op.op == "sigmoid":
        return 1.0 / (1.0 + mx.exp(-arguments[0]))
    if op.op == "tanh":
        return mx.tanh(arguments[0])
    if op.op == "relu":
        return mx.maximum(arguments[0], 0.0)
    if op.op == "leaky_relu":
        value = arguments[0]
        slope = float(op.attrs.get("negative_slope", 0.01))
        return mx.where(value >= 0.0, value, slope * value)
    if op.op == "elu":
        value = arguments[0]
        alpha = float(op.attrs.get("alpha", 1.0))
        return mx.where(value >= 0.0, value, alpha * (mx.exp(value) - 1.0))
    if op.op == "hardswish":
        value = arguments[0]
        body = value * mx.minimum(mx.maximum(value + 3.0, 0.0), 6.0) / 6.0
        return mx.where(mx.isnan(value), float("nan"), body)
    if op.op == "hardsigmoid":
        value = arguments[0]
        body = mx.minimum(mx.maximum(value + 3.0, 0.0), 6.0) / 6.0
        return mx.where(mx.isnan(value), float("nan"), body)
    if op.op == "threshold":
        value = arguments[0]
        threshold = float(op.attrs["threshold"])
        fill_value = float(op.attrs["value"])
        return mx.where(
            mx.isnan(value),
            value,
            mx.where(value > threshold, value, fill_value),
        )
    if op.op == "exp":
        return mx.exp(arguments[0])
    if op.op == "expm1":
        return mx.expm1(arguments[0])
    if op.op == "exp2":
        return mx.exp2(arguments[0])
    if op.op == "log":
        return mx.log(arguments[0])
    if op.op == "log2":
        return mx.log2(arguments[0])
    if op.op == "log10":
        return mx.log10(arguments[0])
    if op.op == "log1p":
        return mx.log1p(arguments[0])
    if op.op == "erf":
        return mx.erf(arguments[0])
    if op.op == "sin":
        return mx.sin(arguments[0])
    if op.op == "cos":
        return mx.cos(arguments[0])
    if op.op == "floor":
        return mx.floor(arguments[0])
    if op.op == "ceil":
        return mx.ceil(arguments[0])
    if op.op == "round":
        return mx.round(arguments[0])
    if op.op == "trunc":
        return mx.trunc(arguments[0])
    if op.op == "sign":
        return mx.sign(arguments[0])
    if op.op == "square":
        return mx.square(arguments[0])
    if op.op == "sqrt":
        return mx.sqrt(arguments[0])
    if op.op == "rsqrt":
        return 1.0 / mx.sqrt(arguments[0])
    if op.op == "reciprocal":
        return 1.0 / arguments[0]
    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        integral = value.dtype in {mx.int32, mx.int64}
        integer_min = -(1 << (31 if value.dtype == mx.int32 else 63))
        integer_max = (1 << (31 if value.dtype == mx.int32 else 63)) - 1
        lower: mx.array | float = (
            mx.array(integer_min, dtype=value.dtype)
            if integral
            else -float("inf")
        )
        upper: mx.array | float = (
            mx.array(integer_max, dtype=value.dtype)
            if integral
            else float("inf")
        )
        if "min_input" in op.attrs:
            lower = arguments[int(op.attrs["min_input"])]
        elif "min" in op.attrs:
            lower = (
                mx.array(op.attrs["min"], dtype=value.dtype)
                if integral
                else float(op.attrs["min"])
            )
        if "max_input" in op.attrs:
            upper = arguments[int(op.attrs["max_input"])]
        elif "max" in op.attrs:
            upper = (
                mx.array(op.attrs["max"], dtype=value.dtype)
                if integral
                else float(op.attrs["max"])
            )
        clipped = mx.minimum(mx.maximum(value, lower), upper)
        if integral:
            return clipped
        invalid = mx.isnan(value) | mx.isnan(lower) | mx.isnan(upper)
        return mx.where(invalid, float("nan"), clipped)
    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _binary_mx_operands(
    op: Operation, arguments: list[mx.array]
) -> tuple[mx.array | float, mx.array | float]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar_value = op.attrs["scalar"]
    if isinstance(scalar_value, bool):
        scalar = mx.array(scalar_value)
    elif isinstance(scalar_value, int) and arguments[0].dtype in {
        mx.int32,
        mx.int64,
    }:
        dtype = DType.I32 if arguments[0].dtype == mx.int32 else DType.I64
        scalar = mx.array(
            _wrap_integer_scalar(scalar_value, dtype), dtype=_MLX_DTYPES[dtype]
        )
    else:
        scalar = float(scalar_value)
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _native_attention_mask_supported(
    query_rank: int, mask_shape: tuple[int, ...] | None
) -> bool:
    if mask_shape is None:
        return False
    if query_rank == 4:
        return len(mask_shape) == 4
    if query_rank == 3:
        return len(mask_shape) in {2, 3}
    return False


def _attention_shared_memory_bytes(
    query_tile: int,
    key_tile: int,
    query_head_dim: int,
    value_head_dim: int,
    pv_value_tile: int,
) -> int:
    element_bytes = 2
    if key_tile == 64:
        score_workspace_bytes = query_tile * key_tile * 4
        probability_and_pv_bytes = (
            query_tile * key_tile * element_bytes
            + query_tile * pv_value_tile * 4
        )
        score_and_pv_bytes = max(
            score_workspace_bytes, probability_and_pv_bytes
        )
        probability_bytes = 0
        pv_bytes = 0
    else:
        score_and_pv_bytes = query_tile * key_tile * 4
        probability_bytes = query_tile * key_tile * element_bytes
        pv_bytes = query_tile * pv_value_tile * 4
    return (
        query_tile * query_head_dim * element_bytes
        + key_tile * max(query_head_dim, value_head_dim) * element_bytes
        + score_and_pv_bytes
        + probability_bytes
        + pv_bytes
    )


def _attention_mask_offset_expression(
    query_rank: int, mask_shape: tuple[int, ...]
) -> str:
    if query_rank == 4:
        coordinates = ("batch", "head", "query_pos", "key")
    else:
        coordinates = ("batch", "query_pos", "key")
    offset = len(coordinates) - len(mask_shape)
    if offset < 0:
        raise ValueError(f"attention mask rank exceeds query rank: {mask_shape}")

    terms: list[str] = []
    for axis, dimension in enumerate(mask_shape):
        if dimension == 1:
            continue
        coordinate = coordinates[axis + offset]
        stride = prod(mask_shape[axis + 1 :])
        terms.append(coordinate if stride == 1 else f"{coordinate} * {stride}u")
    return " + ".join(terms) if terms else "0u"


_DECODE_ATTENTION_THREADS = 128
_DECODE_ATTENTION_MAX_KEYS = 4096
_DECODE_ATTENTION_DTYPES = frozenset({DType.F16, DType.BF16, DType.F32})


@dataclass(frozen=True, slots=True)
class _DecodeAttentionConfig:
    """Static description of one single-query attention launch."""

    threads: int
    batch: int
    query_heads: int
    key_value_heads: int
    group_size: int
    key_length: int
    query_head_dim: int
    value_head_dim: int
    scale: float
    mask_shape: tuple[int, ...] | None
    mask_is_boolean: bool
    dtype: DType


def _decode_attention_config(
    graph: Graph, op: Operation
) -> _DecodeAttentionConfig | None:
    """Return a launch plan when a one-query attention can use one kernel.

    Prefill attention needs the tiled online-softmax path. A decode step asks
    every head for a single output row, so the whole head fits in one
    threadgroup: scores live in threadgroup memory, the softmax is a two-level
    reduction, and the value accumulation has no online rescaling. That removes
    the per-layer gather/spread work the generic kernels need for a one-row
    query.
    """

    query_type = graph.value(op.inputs[0]).type
    key_type = graph.value(op.inputs[1]).type
    value_type = graph.value(op.inputs[2]).type
    output_type = graph.value(op.outputs[0]).type
    if len(query_type.shape) != 4 or output_type.dtype not in _DECODE_ATTENTION_DTYPES:
        return None
    batch, query_heads, query_length, query_head_dim = query_type.shape
    key_value_heads, key_length = key_type.shape[1], key_type.shape[2]
    value_head_dim = value_type.shape[-1]
    if query_length != 1 or op.attrs.get("causal"):
        return None
    if key_length < 1 or key_length > _DECODE_ATTENTION_MAX_KEYS:
        return None
    if query_head_dim > 256 or value_head_dim > 256:
        return None
    if batch < 1 or query_heads < 1 or key_value_heads < 1:
        return None
    if query_heads % key_value_heads != 0:
        return None
    has_mask = bool(op.attrs.get("has_mask"))
    mask_shape: tuple[int, ...] | None = None
    mask_is_boolean = False
    if has_mask:
        mask_type = graph.value(op.inputs[3]).type
        mask_shape = mask_type.shape
        mask_is_boolean = mask_type.dtype == DType.BOOL
        if not mask_is_boolean and mask_type.dtype != query_type.dtype:
            return None
        if len(mask_shape) > 4:
            return None
    threads = _DECODE_ATTENTION_THREADS
    shared_bytes = (
        4 * key_length + 4 * threads + 4 * query_head_dim
    )
    if shared_bytes > _ATTENTION_MAX_SHARED_BYTES:
        return None
    return _DecodeAttentionConfig(
        threads=threads,
        batch=batch,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        group_size=query_heads // key_value_heads,
        key_length=key_length,
        query_head_dim=query_head_dim,
        value_head_dim=value_head_dim,
        scale=float(op.attrs["scale"]),
        mask_shape=mask_shape,
        mask_is_boolean=mask_is_boolean,
        dtype=output_type.dtype,
    )


def _emit_decode_attention_source(config: _DecodeAttentionConfig) -> str:
    """Emit the one-query attention kernel body for a static head shape."""

    threads = config.threads
    key_length = config.key_length
    query_dim = config.query_head_dim
    value_dim = config.value_head_dim
    query_heads = config.query_heads
    key_value_heads = config.key_value_heads
    splits = threads // value_dim if threads % value_dim == 0 else 1
    indent = "    "
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint head = threadgroup_position_in_grid.x;",
        "uint batch = threadgroup_position_in_grid.y;",
        "uint query_pos = 0u;",
        f"uint key_value_head = head / {config.group_size}u;",
        f"const device T* query = in_0 "
        f"+ (batch * {query_heads}u + head) * {query_dim}u;",
        f"const device T* key_base = in_1 "
        f"+ (batch * {key_value_heads}u + key_value_head) "
        f"* {key_length * query_dim}u;",
        f"const device T* value_base = in_2 "
        f"+ (batch * {key_value_heads}u + key_value_head) "
        f"* {key_length * value_dim}u;",
        f"threadgroup float scores[{key_length}];",
        f"threadgroup float scratch[{threads}];",
        f"threadgroup float query_shared[{query_dim}];",
        f"for (uint dimension = tid; dimension < {query_dim}u; "
        f"dimension += {threads}u) {{",
        f"{indent}query_shared[dimension] = float(query[dimension]);",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
        f"{indent}const device T* row = key_base + key * {query_dim}u;",
        f"{indent}float dot = 0.0f;",
        f"{indent}for (uint dimension = 0u; dimension < {query_dim}u; "
        "++dimension) {",
        f"{indent * 2}dot = fma(query_shared[dimension], "
        "float(row[dimension]), dot);",
        f"{indent}}}",
        f"{indent}float score = dot * float({config.scale!r});",
    ]
    if config.mask_shape is not None:
        mask_offset = _attention_mask_offset_expression(4, config.mask_shape)
        if config.mask_is_boolean:
            lines.append(
                f"{indent}score = in_3[{mask_offset}] ? score : -INFINITY;"
            )
        else:
            lines.append(f"{indent}score += float(in_3[{mask_offset}]);")
    lines.extend(
        [
            f"{indent}scores[key] = score;",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "float local_max = -INFINITY;",
            f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
            f"{indent}local_max = metal::max(local_max, scores[key]);",
            "}",
            "scratch[tid] = local_max;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"for (uint stride = {threads // 2}u; stride > 0u; stride >>= 1) {{",
            f"{indent}if (tid < stride) {{",
            f"{indent * 2}scratch[tid] = metal::max(scratch[tid], "
            "scratch[tid + stride]);",
            f"{indent}}}",
            f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
            # A fully masked head has no finite maximum; shifting by zero keeps
            # every exponential underflowing to zero instead of producing NaN.
            "float row_max = metal::isinf(scratch[0]) ? 0.0f : scratch[0];",
            # Every thread must read the reduction result before the scratch
            # buffer is reused for the softmax sum, otherwise a leading thread
            # can overwrite it while its peers are still loading the maximum.
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "float local_sum = 0.0f;",
            f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
            f"{indent}local_sum += metal::exp(scores[key] - row_max);",
            "}",
            "scratch[tid] = local_sum;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"for (uint stride = {threads // 2}u; stride > 0u; stride >>= 1) {{",
            f"{indent}if (tid < stride) {{",
            f"{indent * 2}scratch[tid] += scratch[tid + stride];",
            f"{indent}}}",
            f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
            "float inv_total = scratch[0] > 0.0f ? 1.0f / scratch[0] : 0.0f;",
            f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
            f"{indent}scores[key] = metal::exp(scores[key] - row_max) "
            "* inv_total;",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]
    )
    if splits > 1:
        lines.extend(
            [
                f"uint part = tid / {value_dim}u;",
                f"uint dimension = tid - part * {value_dim}u;",
                f"uint begin = ({key_length}u * part) / {splits}u;",
                f"uint end = ({key_length}u * (part + 1u)) / {splits}u;",
                "float accumulator = 0.0f;",
                "for (uint key = begin; key < end; ++key) {",
                f"{indent}accumulator = fma(scores[key], float(value_base[key * "
                f"{value_dim}u + dimension]), accumulator);",
                "}",
                "scratch[tid] = accumulator;",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                f"if (tid < {value_dim}u) {{",
                f"{indent}float total = 0.0f;",
                f"{indent}for (uint split = 0u; split < {splits}u; ++split) {{",
                f"{indent * 2}total += scratch[split * {value_dim}u + tid];",
                f"{indent}}}",
                f"{indent}out[(batch * {query_heads}u + head) * {value_dim}u "
                "+ tid] = (T)total;",
                "}",
            ]
        )
    else:
        lines.extend(
            [
                f"for (uint dimension = tid; dimension < {value_dim}u; "
                f"dimension += {threads}u) {{",
                f"{indent}float accumulator = 0.0f;",
                f"{indent}for (uint key = 0u; key < {key_length}u; ++key) {{",
                f"{indent * 2}accumulator = fma(scores[key], float(value_base[key "
                f"* {value_dim}u + dimension]), accumulator);",
                f"{indent}}}",
                f"{indent}out[(batch * {query_heads}u + head) * {value_dim}u "
                "+ dimension] = (T)accumulator;",
                "}",
            ]
        )
    return "\n".join(f"        {line}" for line in lines)


def _validate_cat_op(graph: Graph, op: Operation) -> None:
    if not op.inputs:
        raise ValueError("cat expects at least one input tensor")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"cat.dim must be an integer, got {dim!r}")

    input_types = [graph.value(value_id).type for value_id in op.inputs]
    output_type = graph.value(op.outputs[0]).type
    rank = len(input_types[0].shape)
    if rank == 0:
        raise NotImplementedError("cat requires ranked tensors")
    axis = dim % rank
    if any(len(input_type.shape) != rank for input_type in input_types):
        raise ValueError("cat inputs must have the same rank")
    if any(input_type.dtype != output_type.dtype for input_type in input_types):
        raise ValueError("cat must preserve input dtype")
    for position, input_type in enumerate(input_types):
        for input_axis, (input_dim, output_dim) in enumerate(
            zip(input_type.shape, output_type.shape, strict=True)
        ):
            if input_axis != axis and input_dim != output_dim:
                raise ValueError(
                    f"cat input {position} shape {input_type.shape} is incompatible "
                    f"with output shape {output_type.shape} on axis {input_axis}"
                )

    expected_shape = list(output_type.shape)
    expected_shape[axis] = sum(input_type.shape[axis] for input_type in input_types)
    if tuple(expected_shape) != output_type.shape:
        raise ValueError(
            f"cat output shape {output_type.shape} != {tuple(expected_shape)}"
        )


def _validate_rope_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 3:
        raise ValueError(f"rope expects three inputs, got {len(op.inputs)}")
    source_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    if source_type != output_type:
        raise ValueError(
            f"rope must preserve shape and dtype, got {source_type} -> {output_type}"
        )
    if output_type.dtype not in {DType.F16, DType.BF16, DType.F32}:
        raise ValueError(
            f"rope requires floating point, got {output_type.dtype.value}"
        )
    if output_type.shape[-1] % 2 != 0:
        raise ValueError(
            f"rope head dimension must be even, got {output_type.shape[-1]}"
        )
    columns = output_type.shape[-1]
    for value_id in op.inputs[1:]:
        table_type = graph.value(value_id).type
        if table_type.dtype != output_type.dtype:
            raise ValueError("rope table dtype must match the source dtype")
        if not _can_broadcast(table_type.shape, output_type.shape):
            raise ValueError(
                f"rope table {table_type.shape} cannot broadcast "
                f"to {output_type.shape}"
            )
        if table_type.shape[-1] not in (1, columns):
            raise ValueError(
                f"rope table last dimension must be 1 or {columns}, "
                f"got {table_type.shape[-1]}"
            )


def _validate_embedding_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 2:
        raise ValueError(f"embedding expects two inputs, got {len(op.inputs)}")
    weight_type = graph.value(op.inputs[0]).type
    indices_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    if len(weight_type.shape) != 2:
        raise ValueError(
            f"embedding weight must have rank 2, got {weight_type.shape}"
        )
    if weight_type.dtype not in {DType.F16, DType.BF16, DType.F32}:
        raise ValueError(
            f"embedding weight must be floating point, got {weight_type.dtype.value}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "embedding indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    expected_shape = indices_type.shape + (weight_type.shape[1],)
    if output_type.shape != expected_shape or output_type.dtype != weight_type.dtype:
        raise ValueError(
            f"embedding output metadata {(output_type.shape, output_type.dtype)} does "
            f"not match expected {(expected_shape, weight_type.dtype)}"
        )

    allowed_attrs = {"padding_idx", "scale_grad_by_freq", "sparse"}
    unknown_attrs = set(op.attrs) - allowed_attrs
    if unknown_attrs:
        raise ValueError(
            f"embedding has unsupported attributes: {sorted(unknown_attrs)}"
        )
    padding_idx = op.attrs.get("padding_idx", -1)
    if padding_idx is not None:
        if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
            raise ValueError(
                f"embedding.padding_idx must be an integer, got {padding_idx!r}"
            )
        if not -weight_type.shape[0] <= padding_idx < weight_type.shape[0]:
            raise ValueError(
                f"embedding.padding_idx {padding_idx} is outside the weight table"
            )
    for name in ("scale_grad_by_freq", "sparse"):
        if not isinstance(op.attrs.get(name, False), bool):
            raise ValueError(f"embedding.{name} must be a boolean")


def _validate_index_select_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 2:
        raise ValueError(
            f"index_select expects two inputs, got {len(op.inputs)}"
        )
    source_type = graph.value(op.inputs[0]).type
    indices_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("index_select requires a tensor with rank >= 1")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"index_select.dim must be an integer, got {dim!r}")
    if dim < 0 or dim >= rank:
        raise ValueError(f"index_select.dim {dim} is outside rank {rank}")
    if len(indices_type.shape) != 1:
        raise ValueError(
            "index_select indices must have rank 1, "
            f"got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "index_select indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    expected_shape = (
        source_type.shape[:dim]
        + (indices_type.shape[0],)
        + source_type.shape[dim + 1 :]
    )
    if output_type != TensorType(expected_shape, source_type.dtype):
        raise ValueError(
            f"index_select output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(expected_shape, source_type.dtype)}"
        )
    if set(op.attrs) != {"dim"}:
        raise ValueError(
            f"index_select has unsupported attributes: {sorted(set(op.attrs) - {'dim'})}"
        )


def _validate_index_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) < 2:
        raise ValueError(f"index expects at least two inputs, got {len(op.inputs)}")
    source_type = graph.value(op.inputs[0]).type
    axes_value = op.attrs.get("axes")
    if not isinstance(axes_value, tuple) or not axes_value:
        raise ValueError(f"index.axes must be a non-empty tuple, got {axes_value!r}")
    if not all(
        isinstance(axis, int) and not isinstance(axis, bool) for axis in axes_value
    ):
        raise ValueError(f"index.axes must contain integers, got {axes_value!r}")
    if len(op.inputs) != len(axes_value) + 1:
        raise ValueError(
            f"index has {len(op.inputs) - 1} indices for axes {axes_value}"
        )

    first_axis = axes_value[0]
    last_axis = axes_value[-1]
    if axes_value != tuple(range(first_axis, last_axis + 1)):
        raise NotImplementedError(
            "non-contiguous advanced indexing is not supported: "
            f"indexed axes {axes_value}"
        )
    if first_axis < 0 or last_axis >= len(source_type.shape):
        raise ValueError(
            f"index axes {axes_value} exceed source rank {len(source_type.shape)}"
        )

    index_shapes: list[tuple[int, ...]] = []
    for index_id, axis in zip(op.inputs[1:], axes_value, strict=True):
        index_type = graph.value(index_id).type
        if index_type.dtype not in {DType.I32, DType.I64}:
            raise ValueError(
                "index indices must be int32 or int64, "
                f"got {index_type.dtype.value}"
            )
        index_shapes.append(index_type.shape)

    broadcast_shape = _broadcast_shapes(tuple(index_shapes))
    expected_shape = (
        source_type.shape[:first_axis]
        + broadcast_shape
        + source_type.shape[last_axis + 1 :]
    )
    output_type = graph.value(op.outputs[0]).type
    expected_type = TensorType(expected_shape, source_type.dtype)
    if output_type != expected_type:
        raise ValueError(
            f"index output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(expected_shape, expected_type.dtype)}"
        )
    if set(op.attrs) != {"axes"}:
        raise ValueError(
            f"index has unsupported attributes: {sorted(set(op.attrs) - {'axes'})}"
        )


def _broadcast_shapes(
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    rank = max(len(shape) for shape in shapes)
    result = [1] * rank
    for shape in shapes:
        offset = rank - len(shape)
        for axis, dim in enumerate(shape):
            current = result[offset + axis]
            if current != dim and current != 1 and dim != 1:
                raise ValueError(
                    f"index shapes cannot broadcast: {shapes}"
                )
            result[offset + axis] = max(current, dim)
    return tuple(result)


def _validate_gather_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 2:
        raise ValueError(f"gather expects two inputs, got {len(op.inputs)}")
    source_type = graph.value(op.inputs[0]).type
    indices_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("gather requires a tensor with rank >= 1")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"gather.dim must be an integer, got {dim!r}")
    if dim < 0 or dim >= rank:
        raise ValueError(f"gather.dim {dim} is outside rank {rank}")
    if len(indices_type.shape) != rank:
        raise ValueError(
            f"gather indices must have rank {rank}, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "gather indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != dim and index_dim > source_dim:
            raise ValueError(
                f"gather indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-gather axis {index_axis}"
            )
    if output_type != TensorType(indices_type.shape, source_type.dtype):
        raise ValueError(
            f"gather output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(indices_type.shape, source_type.dtype)}"
        )
    if set(op.attrs) != {"dim"}:
        raise ValueError(
            f"gather has unsupported attributes: {sorted(set(op.attrs) - {'dim'})}"
        )


def _validate_scatter_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) not in {2, 3}:
        raise ValueError(f"scatter expects two or three inputs, got {len(op.inputs)}")
    source_type = graph.value(op.inputs[0]).type
    indices_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("scatter requires a tensor with rank >= 1")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"scatter.dim must be an integer, got {dim!r}")
    if dim < 0 or dim >= rank:
        raise ValueError(f"scatter.dim {dim} is outside rank {rank}")
    if len(indices_type.shape) != rank:
        raise ValueError(
            f"scatter indices must have rank {rank}, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "scatter indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != dim and index_dim > source_dim:
            raise ValueError(
                f"scatter indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-scatter axis {index_axis}"
            )
    if output_type != source_type:
        raise ValueError(
            f"scatter output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )

    if len(op.inputs) == 2:
        if set(op.attrs) != {"dim", "value"}:
            raise ValueError(
                f"scatter value op has unexpected attributes: {sorted(op.attrs)}"
            )
        value = op.attrs["value"]
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"scatter.value must be a scalar number, got {value!r}")
    else:
        if set(op.attrs) != {"dim"}:
            raise ValueError(
                f"scatter has unsupported attributes: {sorted(set(op.attrs) - {'dim'})}"
            )
        update_type = graph.value(op.inputs[2]).type
        if update_type.dtype != source_type.dtype:
            raise ValueError(
                "scatter source dtype must match the input dtype, "
                f"got {update_type.dtype.value} and {source_type.dtype.value}"
            )
        if len(update_type.shape) != rank:
            raise ValueError(
                f"scatter source must have rank {rank}, got {update_type.shape}"
            )
        for update_axis, (update_dim, index_dim) in enumerate(
            zip(update_type.shape, indices_type.shape, strict=True)
        ):
            if update_dim < index_dim:
                raise ValueError(
                    f"scatter source shape {update_type.shape} is smaller than "
                    f"indices shape {indices_type.shape} on axis {update_axis}"
                )


def _validate_cumsum_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 1:
        raise ValueError(f"cumsum expects one input, got {len(op.inputs)}")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"cumsum.dim must be an integer, got {dim!r}")
    input_type = graph.value(op.inputs[0]).type
    if not input_type.shape:
        raise ValueError("cumsum does not support rank-0 tensors")
    axis = dim + len(input_type.shape) if dim < 0 else dim
    if axis < 0 or axis >= len(input_type.shape):
        raise ValueError(f"cumsum.dim {dim} exceeds rank {len(input_type.shape)}")
    output_type = graph.value(op.outputs[0]).type
    if output_type.shape != input_type.shape:
        raise ValueError(
            f"cumsum output shape {output_type.shape} != {input_type.shape}"
        )


def _validate_triangular_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 1:
        raise ValueError(f"{op.op} expects one input, got {len(op.inputs)}")
    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    if len(input_type.shape) < 2:
        raise NotImplementedError(f"{op.op} requires a tensor with rank >= 2")
    if input_type != output_type:
        raise ValueError(f"{op.op} must preserve the input tensor type")
    diagonal = op.attrs.get("diagonal")
    if not isinstance(diagonal, int) or isinstance(diagonal, bool):
        raise ValueError(f"{op.op}.diagonal must be an integer, got {diagonal!r}")


def _validate_cast_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 1:
        raise ValueError(f"cast expects one input, got {len(op.inputs)}")
    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    if input_shape != output_shape:
        raise ValueError(f"cast shape {input_shape} != {output_shape}")


def _validate_attention_op(graph: Graph, op: Operation) -> None:
    expected_inputs = 4 if op.attrs.get("has_mask") else 3
    if len(op.inputs) != expected_inputs:
        raise ValueError(
            f"attention expects {expected_inputs} inputs, got {len(op.inputs)}"
        )

    query_type = graph.value(op.inputs[0]).type
    key_type = graph.value(op.inputs[1]).type
    value_type = graph.value(op.inputs[2]).type
    output_type = graph.value(op.outputs[0]).type
    query_shape = query_type.shape
    key_shape = key_type.shape
    value_shape = value_type.shape
    if len(query_shape) not in {3, 4}:
        raise NotImplementedError(
            f"attention requires rank-3 or rank-4 inputs, got {query_shape}"
        )
    if len(key_shape) != len(query_shape) or len(value_shape) != len(query_shape):
        raise ValueError("attention Q/K/V ranks must match")
    if query_shape[:1] != key_shape[:1]:
        raise ValueError("attention Q/K batch dimensions must match")
    if key_shape[:1] != value_shape[:1]:
        raise ValueError("attention K/V batch dimensions must match")
    if query_shape[-1] != key_shape[-1]:
        raise ValueError("attention Q/K head dimensions must match")
    if key_shape[-2] != value_shape[-2]:
        raise ValueError("attention K/V sequence dimensions must match")
    if len(query_shape) == 4:
        query_heads, key_value_heads = query_shape[-3], key_shape[-3]
        if query_heads < key_value_heads or query_heads % key_value_heads != 0:
            raise ValueError(
                "attention query heads must be a multiple of key/value heads, "
                f"got {query_heads} and {key_value_heads}"
            )
    expected_output_shape = query_shape[:-1] + (value_shape[-1],)
    if output_type.shape != expected_output_shape:
        raise ValueError(
            f"attention output shape {output_type.shape} != {expected_output_shape}"
        )
    if not (query_type.dtype == key_type.dtype == value_type.dtype == output_type.dtype):
        raise NotImplementedError(
            "attention requires identical Q/K/V/output dtypes"
        )

    scale = op.attrs.get("scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError(f"attention.scale must be a scalar number, got {scale!r}")
    causal = op.attrs.get("causal", False)
    if not isinstance(causal, bool):
        raise ValueError(f"attention.causal must be a boolean, got {causal!r}")
    safe_softmax = op.attrs.get("safe_softmax", False)
    if not isinstance(safe_softmax, bool):
        raise ValueError(
            f"attention.safe_softmax must be a boolean, got {safe_softmax!r}"
        )
    if causal and expected_inputs == 4:
        raise ValueError("attention cannot combine an explicit mask with causal mode")

    if expected_inputs == 4:
        mask_type = graph.value(op.inputs[3]).type
        score_shape = (
            (query_shape[0], query_shape[-2], key_shape[-2])
            if len(query_shape) == 3
            else (
                query_shape[:1]
                + (query_shape[-3], query_shape[-2], key_shape[-2])
            )
        )
        if mask_type.dtype not in {query_type.dtype, DType.BOOL}:
            raise NotImplementedError(
                "attention mask must be boolean or match the Q/K/V dtype"
            )
        if not _can_broadcast(mask_type.shape, score_shape):
            raise ValueError(
                f"attention mask shape {mask_type.shape} cannot broadcast "
                f"to {score_shape}"
            )


def _evaluate_attention_mlx(op: Operation, env: dict[str, mx.array]) -> mx.array:
    query, key, value = (env[value_id] for value_id in op.inputs[:3])
    explicit_mask = env[op.inputs[3]] if len(op.inputs) == 4 else None
    mask = explicit_mask
    if op.attrs.get("causal"):
        mask = "causal"
    scale = float(op.attrs["scale"])

    if query.ndim == 4:
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )
        if op.attrs.get("safe_softmax") and explicit_mask is not None:
            output = _zero_fully_masked_rows(explicit_mask, output)
        return output
    if query.ndim != 3:
        raise NotImplementedError(
            f"attention requires rank-3 or rank-4 inputs, got {query.shape}"
        )

    query = mx.expand_dims(query, axis=1)
    key = mx.expand_dims(key, axis=1)
    value = mx.expand_dims(value, axis=1)
    if mask is not None and mask.ndim == 3:
        mask = mx.expand_dims(mask, axis=1)
    output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask=mask
    )
    output = mx.squeeze(output, axis=1)
    if op.attrs.get("safe_softmax") and explicit_mask is not None:
        output = _zero_fully_masked_rows(
            mx.squeeze(explicit_mask, axis=1), output
        )
    return output


def _zero_fully_masked_rows(mask: mx.array, output: mx.array) -> mx.array:
    score_shape = output.shape[:-1] + (mask.shape[-1],)
    broadcast_mask = mx.broadcast_to(mask, score_shape)
    if broadcast_mask.dtype == mx.bool_:
        all_masked = mx.all(mx.logical_not(broadcast_mask), axis=-1, keepdims=True)
    else:
        all_masked = mx.all(broadcast_mask == -mx.inf, axis=-1, keepdims=True)
    return mx.where(all_masked, mx.array(0, dtype=output.dtype), output)
