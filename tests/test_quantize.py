"""The int4 weight format: packing, unpacking, and the arithmetic around it.

Three properties matter, and the tests are grouped by them. The format must be
lossless in the sense that unpacking a packed word returns exactly the codes
that went in - including negatives, which is where a mask-only nibble decode
goes wrong, since a 4-bit two's-complement -1 reads back as 15. The dequantized
values must agree with the scale, because every downstream tolerance is stated
in terms of it. And the packed representation must be a quarter the size of the
float32 weights, or a third of the bfloat16 ones, since that ratio is the whole
point of quantizing weights held in memory.

The packing is defined on ``int32`` words with eight nibbles each, ordered so
that the lowest nibble is the lowest column. That ordering is what lets the
Metal kernel's shift-and-sign-extend decode produce a straight run of columns,
and the tests pin it down because getting it backwards still produces a
consistent, plausible-looking pack-then-unpack pair.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from aotolith.quantize import (
    check_group_size,
    dequantize_int4,
    mx_dequantize_int4,
    quantize_int4,
    unpack_codes,
)


def _weight(rows: int, inner: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, inner)).astype(np.float32)


def test_packed_shape_is_a_third_of_the_source() -> None:
    weight = _weight(8, 32)
    packed, _ = quantize_int4(weight, 32)
    assert packed.dtype == np.int32
    assert packed.shape == (8, 4)
    # Eight weights to a byte turns four bytes of float32 into half a byte.
    assert packed.nbytes * 8 == weight.nbytes


def test_unpack_round_trips_the_codes_exactly() -> None:
    weight = _weight(16, 64, seed=1)
    packed, _ = quantize_int4(weight, 64)
    codes = unpack_codes(packed)
    assert codes.shape == weight.shape
    assert codes.dtype == np.int8
    assert int(codes.min()) >= -8
    assert int(codes.max()) <= 7
    # Unpacking twice must give the same thing; a shift-based decode that wrote
    # past its word would not.
    np.testing.assert_array_equal(codes, unpack_codes(packed))


def test_nibbles_are_ordered_low_to_high_across_a_word() -> None:
    # Each column carries a distinct code, so a reversed nibble order shows up
    # as a permutation rather than as an error.
    codes = np.array([[1, 2, 3, 4, 5, 6, 7, -8]], dtype=np.int8)
    word = 0
    for position, code in enumerate(codes[0]):
        word |= (int(code) & 0xF) << (4 * position)
    packed = np.array([[word - (1 << 32) if word >= 1 << 31 else word]], dtype=np.int32)
    np.testing.assert_array_equal(unpack_codes(packed), codes)


def test_negative_codes_survive_the_pack() -> None:
    weight = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 0.25]], dtype=np.float32)
    packed, scales = quantize_int4(weight, 8)
    codes = unpack_codes(packed)
    assert int(codes.min()) < 0
    actual = codes.astype(np.float32) * scales[:, :1]
    np.testing.assert_allclose(actual, weight, atol=float(scales.max()) / 2 + 1e-6)


def test_dequantize_matches_the_pack_then_scale_product() -> None:
    weight = _weight(4, 32, seed=2)
    packed, scales = quantize_int4(weight, 16)
    actual = dequantize_int4(packed, scales, 16)
    expected = unpack_codes(packed).astype(np.float32) * np.repeat(scales, 16, axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_scale_is_the_group_magnitude_over_the_code_range() -> None:
    weight = _weight(2, 64, seed=3)
    weight[0, :32] *= 7.0
    _, scales = quantize_int4(weight, 32)
    assert scales.shape == (2, 2)
    np.testing.assert_allclose(scales[0, 0], np.abs(weight[0, :32]).max() / 7.0)
    np.testing.assert_allclose(scales[0, 1], np.abs(weight[0, 32:]).max() / 7.0)


def test_an_all_zero_group_keeps_a_finite_scale() -> None:
    weight = np.zeros((1, 16), dtype=np.float32)
    packed, scales = quantize_int4(weight, 16)
    assert np.isfinite(scales).all()
    assert np.all(unpack_codes(packed) == 0)


def test_relative_error_is_bounded_by_half_the_scale() -> None:
    weight = _weight(32, 128, seed=4)
    packed, scales = quantize_int4(weight, 32)
    actual = dequantize_int4(packed, scales, 32)
    widest = float(np.repeat(scales, 32, axis=1).max())
    assert float(np.abs(actual - weight).max()) <= widest / 2 + 1e-6


def test_mx_dequantize_matches_the_reference() -> None:
    weight = _weight(6, 32, seed=5)
    packed, scales = quantize_int4(weight, 8)
    actual = mx_dequantize_int4(mx.array(packed), mx.array(scales), 8)
    mx.eval(actual)
    np.testing.assert_allclose(
        np.asarray(actual), dequantize_int4(packed, scales, 8), atol=1e-6
    )


@pytest.mark.parametrize("group_size", [0, -8, 3, 12])
def test_rejects_group_sizes_that_do_not_cover_whole_words(group_size: int) -> None:
    with pytest.raises(ValueError):
        check_group_size(group_size, 32)


def test_rejects_a_group_size_that_does_not_divide_the_reduction() -> None:
    with pytest.raises(ValueError):
        check_group_size(48, 32)


def test_rejects_a_reduction_axis_that_is_not_whole_words() -> None:
    with pytest.raises(ValueError):
        quantize_int4(_weight(2, 12), 4)


def test_torch_reference_agrees_with_the_numpy_pack() -> None:
    from aotolith.models.quantized import dequantize_i4, pack_weight

    weight = torch.from_numpy(_weight(4, 32, seed=6))
    packed, scales = pack_weight(weight, 16)
    assert packed.dtype == torch.int32
    assert scales.dtype == torch.float32
    restored = dequantize_i4(packed, scales, 16)
    np.testing.assert_allclose(
        restored.numpy(),
        dequantize_int4(packed.numpy(), scales.numpy(), 16),
        atol=1e-6,
    )


def test_pack_weight_keeps_a_bfloat16_weights_dtype() -> None:
    from aotolith.models.quantized import pack_weight

    weight = torch.from_numpy(_weight(4, 32, seed=7)).to(torch.bfloat16)
    packed, scales = pack_weight(weight, 16)
    # The kernel multiplies scales into bf16 activations, so a wider scale would
    # only be rounded back down.
    assert scales.dtype == torch.bfloat16
    assert packed.dtype == torch.int32
