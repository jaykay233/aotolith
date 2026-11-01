"""Weight-only quantization: the packed layouts, the quantizers, the unpacks.

Two widths are supported, 4 and 8 bits, and they differ only in how many codes
one 32-bit word holds: eight nibbles, or four bytes. Everything else - the
group scales, the symmetric rounding, the ``[out, in]`` orientation - is the
same, so the two share the document below and the constants that describe it.

A quantized weight is only worth carrying if the *format* is stated once,
because three different pieces of code have to agree on it: the host code that
packs a checkpoint, the MSL kernel that reads the words while streaming them,
and the MLX fallback that unpacks them when no native kernel applies. So the
layouts are defined here, in one place, in terms of bit positions:

* A weight matrix is stored the way a ``Linear`` stores it: ``[out, in]``, so
  one *row* of the packed tensor is one output unit's weights.
* The reduction axis is cut into groups of ``group_size`` elements. A group
  shares one scale, which is what keeps the narrower widths usable: the width
  of a group follows the local range of the weights instead of the global one.
* Quantization is symmetric and scaled by ``max|w| / magnitude``, where the
  magnitude is ``7`` for four bits and ``127`` for eight, so the largest
  magnitude in a group lands exactly on ``+magnitude`` (and ``-magnitude``).
  The most negative code is representable but never produced. Rounding to
  nearest.
* Codes are packed little-endian into an ``int32`` word: eight nibbles for four
  bits, four bytes for eight. Word ``j`` holds codes ``8j .. 8j+7`` (four bits)
  or ``4j .. 4j+3`` (eight), code ``k`` inside the word sits at shift
  ``4k`` or ``8k``. That makes a word exactly the ``uchar4`` a kernel loads
  and reads as one integer.
* Values are two's complement: a code at the top of its range is negative - a
  nibble in ``8..15`` is a weight in ``-8..-1``, a byte in ``128..255`` one in
  ``-128..-1``.

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

#: The same three facts for eight bits. A group of int8 weights needs no
#: unusual rounding to survive: 127 levels either side of zero is wider than
#: most weights use, so at a group size that suits four bits the error is
#: below the bf16 step and eight bits is effectively lossless.
INT8_MIN = -128
INT8_MAX = 127
INT8_MAGNITUDE = 127

#: Bytes per packed word. The two widths exist so that one word is one load.
BYTES_PER_WORD = 4


def check_group_size(group_size: int, inner: int, packed_width: int = NIBBLES_PER_WORD) -> int:
    """Validate one group size against a reduction length.

    A group has to divide the reduction (a partial group would need a scale
    that covers it and a kernel branch that reads it), and it has to be a
    multiple of the codes one packed word spans (otherwise a group boundary
    falls inside a word and the kernel cannot pick a scale from the word index
    alone).

    ``packed_width`` is the codes per word: eight for four bits, four for
    eight. It defaults to eight so every existing caller keeps its meaning.
    """

    if group_size <= 0:
        raise ValueError(f"quantization group size must be positive, got {group_size}")
    if group_size % packed_width != 0:
        raise ValueError(
            f"quantization group size must be a multiple of {packed_width}, "
            f"got {group_size}"
        )
    if inner % group_size != 0:
        raise ValueError(
            f"quantization group size {group_size} does not divide the reduction "
            f"length {inner}"
        )
    return group_size


def _quantize_codes(
    weight: np.ndarray,
    group_size: int,
    packed_width: int,
    lowest: int,
    magnitude: int,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The quantizer both widths share: group scales, round, pack.

    ``packed_width`` codes fit one word and each takes ``shift`` bits, so four
    bits packs eight to a word and eight bits packs four. The two paths differ
    in nothing else, which is why they are one function and not two.
    """

    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"quantization expects a 2-D weight, got {matrix.shape}"
        )
    rows, inner = matrix.shape
    check_group_size(group_size, inner, packed_width)
    groups = inner // group_size

    blocks = matrix.reshape(rows, groups, group_size)
    widest = np.abs(blocks).max(axis=-1)
    # An all-zero group would divide by zero. Giving it unit scale keeps the
    # arithmetic finite and still rounds every weight in it to zero.
    scale = np.where(widest > 0.0, widest, 1.0) / magnitude
    codes = np.clip(np.rint(blocks / scale[:, :, None]), lowest, magnitude)
    codes = codes.astype(np.int32).reshape(rows, inner)

    packed = np.zeros((rows, inner // packed_width), dtype=np.int32)
    mask = (1 << shift) - 1
    for position in range(packed_width):
        field = (codes[:, position::packed_width] & mask).astype(np.int32)
        packed |= field << (shift * position)
    return packed, scale.astype(np.float32)


def quantize_int4(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a ``[out, in]`` weight into packed words and group scales.

    Returns ``(packed, scales)`` where ``packed`` is ``int32 [out, in / 8]`` and
    ``scales`` is ``float32 [out, in / group_size]``.
    """

    return _quantize_codes(
        weight,
        group_size,
        NIBBLES_PER_WORD,
        INT4_MIN,
        INT4_MAGNITUDE,
        4,
    )


def quantize_int8(
    weight: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a ``[out, in]`` weight to eight bits.

    Returns ``(packed, scales)`` where ``packed`` is ``int32 [out, in / 4]`` and
    ``scales`` is ``float32 [out, in / group_size]``.
    """

    return _quantize_codes(
        weight,
        group_size,
        BYTES_PER_WORD,
        INT8_MIN,
        INT8_MAGNITUDE,
        8,
    )


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


def unpack_codes_int8(packed: np.ndarray) -> np.ndarray:
    """Expand packed words back into signed 8-bit codes, ``[out, in]``."""

    words = np.asarray(packed, dtype=np.int64)
    rows, word_count = words.shape
    codes = np.empty((rows, word_count * BYTES_PER_WORD), dtype=np.int8)
    for byte in range(BYTES_PER_WORD):
        field = ((words >> (8 * byte)) & 0xFF).astype(np.int8)
        # ``astype(np.int8)`` already wrapped the unsigned byte into two's
        # complement, so the sign is correct without a second correction.
        codes[:, byte::BYTES_PER_WORD] = field
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


def dequantize_int8(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """The float32 weights an eight-bit packed pair stands for."""

    codes = unpack_codes_int8(packed)
    return codes.astype(np.float32) * np.repeat(
        np.asarray(scales, dtype=np.float32), group_size, axis=-1
    )


def mx_dequantize_int8(
    packed: mx.array,
    scales: mx.array,
    group_size: int,
) -> mx.array:
    """The MLX unpack for eight-bit words, in the scales' dtype.

    The byte is moved into the sign position and shifted back down with an
    arithmetic shift, which is the same two-instruction sign extension the
    kernel uses; masking and subtracting would be a third.
    """

    words = packed.astype(mx.int32)
    fields = []
    for byte in range(BYTES_PER_WORD):
        field = (words >> (8 * byte)) & 0xFF
        fields.append(mx.where(field >= 128, field - 256, field).astype(scales.dtype))
    codes = mx.stack(fields, axis=-1).reshape(packed.shape[0], -1)
    return codes * mx.repeat(scales, group_size, axis=-1)
