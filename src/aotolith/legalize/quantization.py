"""Lowering for the weight-only int4 unpack.

There is only one operator here, and it exists for a reason worth stating: the
unpack is *not* expressible as a handful of ATen calls the legalizer already
knows, because the packing is a bit layout rather than an arithmetic one. A
shift-and-mask chain would lower to eight elementwise regions, and every one of
them would have to be fused back together before the result was a weight again.

So the frontend calls ``aotolith.dequantize_i4`` as one custom operator, and it
arrives here as one node. What the graph then holds is a single
``dequantize_i4`` whose result is the weight matrix, which is exactly what the
folding pass needs to find: it looks for a matmul reading one, and replaces both
with a matmul that streams the packed words directly. A graph that keeps the
unpacked weight for any other consumer still works - the MLX fallback unpacks it
- it just pays for the full matrix.
"""

from __future__ import annotations

from typing import Any

from .fx import _node_type
from .refs import _argument_ids
from .state import Handler, Lowering

#: The custom operator the quantized frontend emits. It is a PyTorch
#: ``library`` op rather than an ATen one, so the target carries our namespace.
DEQUANTIZE_I4_TARGET = "aotolith.dequantize_i4.default"


def _lower_dequantize_i4(lowering: Lowering, node: Any, target: str) -> bool:
    args = getattr(node, "args", ())
    if len(args) != 3:
        raise ValueError(f"{target} expects three arguments, got {len(args)}")
    packed_node, scales_node, group_size = args
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise ValueError(f"{target}.group_size must be an integer, got {group_size!r}")
    inputs = (
        _argument_ids(packed_node, lowering.node_ids),
        _argument_ids(scales_node, lowering.node_ids),
    )
    lowering.bind(
        node,
        lowering.emit(
            "dequantize_i4",
            inputs,
            {"group_size": group_size},
            _node_type(node),
        ),
    )
    return True


HANDLERS: dict[str, tuple[Handler, ...]] = {
    DEQUANTIZE_I4_TARGET: (_lower_dequantize_i4,),
}
