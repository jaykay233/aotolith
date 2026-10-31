"""Weight-only int4: the packed layout, the quantizer, and the unpack.

A 4-bit weight is only worth carrying if the *format* is stated once, because
three different pieces of code have to agree on it: the host code that packs a
checkpoint, the MSL kernel that reads the words while streaming them, and the
MLX fallback that unpacks them when no native kernel applies. So the layout is
defined here, in one place, in terms of bit positions:

* A weight matrix is stored the way a ``Linear`` stores it: ``[out, in]``, so
  one *row* of the packed tensor is one output unit's weights.
* The reduction axis is cut into groups of ``group_size`` elements. A group
  shares one scale, which is what keeps 4 bits usable: the width of a group
  follows the local range of the weights instead of the global one.
* Quantization is symmetric and scaled by ``max|w| / 7``, so the largest
  magnitude in a group lands exactly on ``+7`` (and ``-7``), and the value
  ``-8`` is representable but never produced. Rounding to nearest.
* Nibbles are packed eight to an ``int32`` word. Word ``j`` holds weights
  ``8j .. 8j+7`` and weight ``8j + k`` sits in nibble ``k`` - nibble 0 in the
  low four bits. That is little-endian byte order, so a word is exactly the
  ``uchar4`` a kernel would load, read as one integer.
* Values are two's complement: a nibble in ``8..15`` is a negative weight in
  ``-8..-1``.

The scales live beside the words, ``[out, in / group_size]``, in the model's
compute dtype rather than float32: the 4-bit rounding error is three orders of
magnitude larger than the difference between a bf16 and an f32 scale, so widening
the scale buys nothing and would force the kernel to carry a second dtype.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

#: The representable data range, before the scale is applied.
INT4_MIN = -8
INT4_MAX = 7

#: Symmetric quantization reserves the sign bit's spare code, so the widest
#: magnitude maps onto ``7`` and negating a weight stays exact.
INT4_MAGNITUDE = 7

#: Nibbles per packed word.
NIBBLES_PER_WORD = 8


def check_group_size(group_size: int, inner: int) -> int:
    """Validate one group size against a reduction length.

    A group has to divide the reduction (a partial group would need a scale
    that covers it and a kernel branch that reads it), and it has to be a
    multiple of the eight weights one packed word spans (otherwise a group
    boundary falls inside a word and the kernel cannot pick a scale from the
    word index alone).
    """

    if group_size <= 0:
        raise ValueError(f"quantization group size must be positive, got {group_size}")
    if group_size % NIBBLES_PER_WORD != 0:
        raise ValueError(
            f"quantization group size must be a multiple of {NIBBLES_PER_WORD}, "
            f"got {group_size}"
        )
    if inner % group_size != 0:
        raise ValueError(
            f"quantization group size {group_size} does not divide the reduction "
            f"length {inner}"
        )
    return group_size


def quantize_int4(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a ``[out, in]`` weight into packed words and group scales.

    Returns ``(packed, scales)`` where ``packed`` is ``int32 [out, in / 8]`` and
    ``scales`` is ``float32 [out, in / group_size]``.
    """

    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"quantize_int4 expects a 2-D weight, got {matrix.shape}")
    rows, inner = matrix.shape
    check_group_size(group_size, inner)
    groups = inner // group_size

    blocks = matrix.reshape(rows, groups, group_size)
    widest = np.abs(blocks).max(axis=-1)
    # An all-zero group would divide by zero. Giving it unit scale keeps the
    # arithmetic finite and still rounds every weight in it to zero.
    scale = np.where(widest > 0.0, widest, 1.0) / INT4_MAGNITUDE
    codes = np.clip(np.rint(blocks / scale[:, :, None]), INT4_MIN, INT4_MAX)
    codes = codes.astype(np.int32).reshape(rows, inner)

    packed = np.zeros((rows, inner // NIBBLES_PER_WORD), dtype=np.int32)
    for nibble in range(NIBBLES_PER_WORD):
        field = (codes[:, nibble::NIBBLES_PER_WORD] & 0xF).astype(np.int32)
        packed |= field << (4 * nibble)
    return packed, scale.astype(np.float32)


def unpack_codes(packed: np.ndarray) -> np.ndarray:
    """Expand packed words back into signed 4-bit codes, ``[out, in]``."""

    words = np.asarray(packed, dtype=np.int64)
    if words.dtype != np.int64:
        raise TypeError("unpack_codes needs integer words")
    rows, word_count = words.shape
    codes = np.empty((rows, word_count * NIBBLES_PER_WORD), dtype=np.int8)
    for nibble in range(NIBBLES_PER_WORD):
        field = ((words >> (4 * nibble)) & 0xF).astype(np.int8)
        codes[:, nibble::NIBBLES_PER_WORD] = np.where(field >= 8, field - 16, field)
    return codes


def dequantize_int4(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """The float32 weights a packed pair stands for. The reference unpack."""

    codes = unpack_codes(packed)
    return codes.astype(np.float32) * np.repeat(
        np.asarray(scales, dtype=np.float32), group_size, axis=-1
    )


def mx_dequantize_int4(
    packed: mx.array,
    scales: mx.array,
    group_size: int,
) -> mx.array:
    """The MLX unpack, used when no native kernel applies to a matmul.

    Same bits as :func:`dequantize_int4`, in the scales' dtype: the multiply
    happens in that dtype so the result matches what the kernel accumulates
    from, and a wider intermediate would only be rounded back down.
    """

    words = packed.astype(mx.int32)
    fields = []
    for nibble in range(NIBBLES_PER_WORD):
        field = (words >> (4 * nibble)) & 0xF
        fields.append(mx.where(field >= 8, field - 16, field).astype(scales.dtype))
    codes = mx.stack(fields, axis=-1).reshape(packed.shape[0], -1)
    return codes * mx.repeat(scales, group_size, axis=-1)
