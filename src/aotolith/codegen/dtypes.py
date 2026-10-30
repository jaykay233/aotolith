"""The IR dtype to MLX dtype map, shared by codegen and the runtime.

A ``_compile_*`` method returns the kernel it built together with the MLX
callable it falls back to, and that fallback has to allocate arrays of the right
dtype. The runtime needs the reverse map to report a result. Both are pure
lookups with no dependencies, so they sit below both readers instead of in the
runtime, where codegen would have to reach upwards for them.
"""

from __future__ import annotations

import mlx.core as mx
from ..ir import DType

_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
    DType.BOOL: mx.bool_,
    DType.I32: mx.int32,
    DType.I64: mx.int64,
}


_MLX_DTYPE_TO_DTYPE = {mlx_dtype: dtype for dtype, mlx_dtype in _MLX_DTYPES.items()}
