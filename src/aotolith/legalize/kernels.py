"""The spatial operators: convolution and the pooling windows.

Neither has a hand-written source here. A convolution is one MLX kernel, which
already handles groups, strides, dilation and every rank from one dimension to
three, and writing a second one would only be slower. So the whole job in this
module is reading ATen's spelling of the geometry - a list of per-axis strides,
a symmetric padding, a dilation - and handing it over as typed attributes.

The one spelling that does not survive the trip is a transposed convolution:
Torch states it as the same operator with a flag, while the MLX kernel wants its
weight in a different layout and, on the Metal path, computes it at a lower
precision than the forward direction. It stays an error rather than a silent
approximation.
"""

from __future__ import annotations

from typing import Any

from ..ir import TensorType
from .fx import _node_type
from .refs import _argument_ids
from .state import Handler, Lowering
from .targets import _CONV_TARGET


def _int_tuple(value: object, name: str, expected: int) -> tuple[int, ...]:
    """Read one per-axis geometry argument ATen passes as a list."""

    if isinstance(value, int) and not isinstance(value, bool):
        items = (value,) * expected
    elif isinstance(value, (tuple, list)):
        items = tuple(value)
    else:
        raise ValueError(f"convolution.{name} must be an integer tuple, got {value!r}")
    if len(items) != expected:
        raise ValueError(
            f"convolution.{name} expects {expected} entries, got {len(items)}"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise ValueError(f"convolution.{name} must contain integers, got {value!r}")
    return items


def _conv_operation(
    node: Any, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    if len(args) != 9:
        raise ValueError(f"{_CONV_TARGET} expects nine arguments, got {len(args)}")
    input_node, weight_node, bias_node = args[:3]
    transposed = args[6]
    if not isinstance(transposed, bool):
        raise ValueError(
            f"convolution.transposed must be a boolean, got {transposed!r}"
        )
    if transposed:
        raise NotImplementedError(
            "a transposed convolution is not lowered: MLX computes it in a "
            "different weight layout and, on the Metal path, at a lower "
            "precision than the forward direction"
        )
    output_padding = args[7]
    if output_padding is not None and any(int(item) for item in output_padding):
        raise NotImplementedError(
            "convolution.output_padding only grows the output of a transposed "
            "convolution, which is not lowered"
        )
    groups = args[8]
    if not isinstance(groups, int) or isinstance(groups, bool):
        raise ValueError(f"convolution.groups must be an integer, got {groups!r}")

    input_type = _node_type(input_node)
    spatial = len(input_type.shape) - 2
    if spatial < 1:
        raise NotImplementedError(
            "convolution requires a channel axis and a spatial axis, got "
            f"{input_type.shape}"
        )
    attrs: dict[str, object] = {
        "stride": _int_tuple(args[3], "stride", spatial),
        "padding": _int_tuple(args[4], "padding", spatial),
        "dilation": _int_tuple(args[5], "dilation", spatial),
        "groups": groups,
    }
    inputs = [
        _argument_ids(input_node, node_ids),
        _argument_ids(weight_node, node_ids),
    ]
    if bias_node is not None:
        inputs.append(_argument_ids(bias_node, node_ids))
    return tuple(inputs), attrs, _node_type(node)


def _lower_conv(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _conv_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("conv", inputs, attrs, output_type))
    return True


HANDLERS: dict[str, tuple[Handler, ...]] = {
    _CONV_TARGET: (_lower_conv,),
}
