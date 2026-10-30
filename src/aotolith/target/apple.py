"""What this Apple GPU can do, answered once and cached.

The backend needs three facts about the machine before it can choose a kernel:
whether the device can run the tensor-core (MPP) path at all, whether the
compiler actually emits those instructions, and how much threadgroup memory an
attention kernel may claim. All three are properties of the device and the OS,
not of the program, so they live here rather than in codegen.

Two of them are not simple yes/no answers. ``_mpp_supported`` gates on the chip
family, but a device can advertise the extension and still refuse to load the
kernel, which is why ``_block_tensor_cores`` exists: the first concrete failure
disables the path for the rest of the process and ``tensor_core_fallback_reason``
reports it instead of retrying every compilation. The tile ladders are here for
the same reason - they are calibrated to a threadgroup memory budget, so the
choice of tile is a device fact.
"""

from __future__ import annotations

import mlx.core as mx
from functools import lru_cache
from math import ceil
from ..ir import DType
from ..ir import TensorType
from ..codegen.msl import _MPP_HEADER, _metal_kernel

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


_ATTENTION_MAX_SHARED_BYTES = 31 * 1024


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
            query_tile * key_tile * element_bytes + query_tile * pv_value_tile * 4
        )
        score_and_pv_bytes = max(score_workspace_bytes, probability_and_pv_bytes)
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


_MPP_GEMM_TILE_TINY = (32, 32, 1, 1, 32)


_MPP_GEMM_TILE_SMALL_K = (64, 64, 2, 2, 32)


_MPP_GEMM_TILE_SKINNY_LONG_K = (128, 32, 4, 1, 32)


_MPP_GEMM_TILE_TRANSPOSED_WIDE = (64, 128, 2, 4, 64)


_MPP_GEMM_TILE_ROW_MAJOR_WIDE = (64, 256, 1, 8, 32)


_MPP_GEMM_TILE_ROW_MAJOR = (128, 64, 2, 2, 32)


_MPP_GEMM_TILE_TALL = (128, 128, 2, 4, 32)


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


_MPP_GEMM_WIDE_ELEMENTS = 3_000_000


_MPP_GEMM_MIN_WIDE_COLUMNS = 8192


_MPP_GEMM_MAX_WIDE_COLUMNS = 32768


_MPP_GEMM_SKINNY_MAX_ROWS = 256


_MPP_GEMM_SKINNY_MAX_COLUMNS = 2048


def _select_mpp_gemm_config(
    rows: int,
    columns: int,
    reduction: int,
    transposed_right: bool = True,
) -> tuple[int, int, int, int, int]:
    """Pick the tensor-core block tile for one concrete matmul shape."""

    wide = (
        _MPP_GEMM_TILE_TRANSPOSED_WIDE
        if transposed_right
        else _MPP_GEMM_TILE_ROW_MAJOR_WIDE
    )
    if not transposed_right and columns >= _MPP_GEMM_MAX_WIDE_COLUMNS:
        preferred = _MPP_GEMM_TILE_TALL
    elif reduction >= 2048 and rows <= _MPP_GEMM_SKINNY_MAX_ROWS:
        preferred = (
            _MPP_GEMM_TILE_SKINNY_LONG_K
            if columns <= _MPP_GEMM_SKINNY_MAX_COLUMNS
            else wide
        )
    elif rows * columns >= _MPP_GEMM_WIDE_ELEMENTS or (
        rows <= _MPP_GEMM_SKINNY_MAX_ROWS and columns >= _MPP_GEMM_MIN_WIDE_COLUMNS
    ):
        preferred = wide
    elif transposed_right or rows < _MPP_GEMM_TILE_ROW_MAJOR[0]:
        preferred = _MPP_GEMM_TILE_SMALL_K
    else:
        preferred = _MPP_GEMM_TILE_ROW_MAJOR
    row_bound = max(16, ceil(rows / 16) * 16)
    column_bound = max(32, ceil(columns / 32) * 32)
    for config in _MPP_GEMM_TILE_LADDER[_MPP_GEMM_TILE_LADDER.index(preferred) :]:
        if config[0] <= row_bound and config[1] <= column_bound:
            return config
    return _MPP_GEMM_TILE_TINY


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

    Half and bfloat16 only. The hardware has no float32 tensor cores, so a
    `T = float` `matmul2d` is executed at bf16 precision: on a 512x2048x512
    f32 GEMM it lands 1.6e-1 away from PyTorch, the same distance MLX's own
    f32 matmul lands, while the scalar tiled kernel matches bit for bit. That
    is a silent accuracy loss in exchange for 1.6x, so f32 stays on the exact
    path.
    """

    return (
        dtype in {DType.F16, DType.BF16}
        and rows >= 16
        and columns >= 16
        and reduction >= 16
        and reduction % 16 == 0
    )


_MPP_ATTENTION_KEY_TILE = 32


_MPP_ATTENTION_MAX_HEAD_DIM = 128


_MPP_ATTENTION_MIN_QUERY_LENGTH = 16


_MPP_ATTENTION_MAX_WARPS = 2


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


_DECODE_ATTENTION_THREADS = 128


_DECODE_ATTENTION_MAX_KEYS = 4096


_DECODE_ATTENTION_DTYPES = frozenset({DType.F16, DType.BF16, DType.F32})


def _device_profile() -> tuple[str | None, str | None]:
    try:
        info = mx.device_info()
    except (RuntimeError, ValueError):
        return None, None
    return (
        None if "device_name" not in info else str(info["device_name"]),
        None if "architecture" not in info else str(info["architecture"]),
    )
