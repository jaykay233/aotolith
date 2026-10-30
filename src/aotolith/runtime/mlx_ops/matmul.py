"""Matmul evaluation when no Metal kernel was selected, including its epilogues.

``addmm`` and the fused normalizations read as a matmul plus a post-pass, and
running them through MLX means applying that post-pass here rather than in a
kernel. The epilogue helpers are shared with the fused-elementwise path so the
two cannot disagree about what a fused matmul means.
"""

from __future__ import annotations

import mlx.core as mx
from typing import Any
from ...ir import Operation
from ...codegen.matmul import _matmul_epilogue_position, _matmul_norm_position


def _apply_mlx_matmul_norm(op: Operation, env: dict[str, Any], left: Any) -> Any:
    """Normalize a fused RMS-norm operand before the fallback matrix multiply."""

    position = _matmul_norm_position(op)
    if position is None:
        return left
    weight = env[op.inputs[position]]
    eps = float(op.attrs["rms_norm_eps"])
    normalized = left.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(normalized * normalized, axis=-1, keepdims=True) + eps)
    return (normalized * scale * weight.astype(mx.float32)).astype(left.dtype)


def _apply_mlx_matmul_epilogue(op: Operation, env: dict[str, Any], product: Any) -> Any:
    kind = op.attrs.get("epilogue")
    if kind is None:
        return product
    if kind != "silu_mul":
        raise ValueError(f"unsupported matmul epilogue {kind!r}")
    gate = env[op.inputs[_matmul_epilogue_position(op)]]
    return product * (gate / (1.0 + mx.exp(-gate)))
