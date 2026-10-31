"""The torch side of weight-only int4: the custom operator and the packer.

The compiler lowers a matmul over a *packed* weight only if the traced graph
says one thing: "this projection reads a 4-bit weight". Written as arithmetic,
that statement would be eight bit-twiddling ATen calls, and the pass that folds
it into the matmul would have to re-recognise them. So the frontend expresses it
as one custom operator, ``aotolith::dequantize_i4``, registered here:

* ``CompositeExplicitAutograd`` gives eager execution the real unpack, so a
  reference run, a test, or a CPU fallback all work without the compiler;
* ``register_fake`` gives ``make_fx`` the shape rule it needs in fake mode,
  which is how capture works;
* the tracer sees one node, which is what :mod:`aotolith.passes.quantize` folds.

Packing happens on the host, once, when the adapter is built. It is a
checkpoint-time cost, so it runs in NumPy through :mod:`aotolith.quantize` -
the same code the tests check the kernel against - rather than in torch.
"""

from __future__ import annotations

import numpy as np

from ..quantize import check_group_size, quantize_int4

#: The library name every op in this build is registered under.
QUANTIZE_LIBRARY = "aotolith"

#: The operator a quantized projection calls. Kept as a constant so the tracer,
#: the legalizer, and the docs name it once.
DEQUANTIZE_I4 = f"{QUANTIZE_LIBRARY}::dequantize_i4"

_registered = False


def _register() -> None:
    """Define the operator, once per process.

    Torch refuses a second definition of the same schema, and a test suite
    imports this module from several fixtures, so the flag is what makes the
    registration idempotent rather than an import-time side effect that only
    works once.
    """

    global _registered
    if _registered:
        return

    import torch
    from torch import Tensor

    torch.library.define(
        DEQUANTIZE_I4,
        "(Tensor packed, Tensor scales, int group_size) -> Tensor",
    )

    @torch.library.impl(DEQUANTIZE_I4, "CompositeExplicitAutograd")
    def _dequantize_i4(packed: Tensor, scales: Tensor, group_size: int) -> Tensor:
        if packed.dtype != torch.int32:
            raise TypeError(f"packed int4 weights must be int32, got {packed.dtype}")
        if packed.dim() != 2:
            raise ValueError(f"packed int4 weights must be 2-D, got {packed.shape}")
        # The unpack is elementwise over the words, so it runs as the same
        # shift-and-mask the MLX fallback uses, in the scales' dtype.
        words = packed.to(torch.int64) & 0xFFFFFFFF
        fields = []
        for nibble in range(8):
            field = (words >> (4 * nibble)) & 0xF
            fields.append(torch.where(field >= 8, field - 16, field))
        codes = torch.stack(fields, dim=-1).reshape(packed.shape[0], -1)
        codes = codes.to(scales.dtype)
        return codes * scales.repeat_interleave(group_size, dim=-1)

    @torch.library.register_fake(DEQUANTIZE_I4)
    def _dequantize_i4_fake(packed: Tensor, scales: Tensor, group_size: int) -> Tensor:
        return packed.new_empty(
            (packed.shape[0], packed.shape[1] * 8), dtype=scales.dtype
        )

    _registered = True


def dequantize_i4(packed, scales, group_size: int):
    """Call the registered operator, defining it first if this is the first use."""

    _register()
    import torch

    return torch.ops.aotolith.dequantize_i4(packed, scales, group_size)


def pack_weight(weight, group_size: int):
    """Pack a ``[out, in]`` weight into int32 words and group scales.

    ``weight`` is a torch tensor or anything NumPy can read. The scales keep the
    weight's dtype, because the kernel multiplies them into activations of that
    dtype and a wider scale would only be rounded back down.
    """

    import torch

    if isinstance(weight, torch.Tensor):
        matrix = weight.detach().float().cpu().numpy()
        output_dtype = weight.dtype
    else:
        matrix = np.asarray(weight, dtype=np.float32)
        output_dtype = torch.float32
    check_group_size(group_size, matrix.shape[-1])
    packed, scales = quantize_int4(matrix, group_size)
    return (
        torch.from_numpy(packed),
        torch.from_numpy(scales.astype(np.float32)).to(output_dtype),
    )
