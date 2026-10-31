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

import operator
from typing import Any

from ..ir import TensorType
from .fx import _node_type
from .refs import _argument_ids
from .state import Handler, Lowering
from .targets import (
    _ADAPTIVE_AVG_POOL_TARGETS,
    _ADAPTIVE_MAX_POOL_TARGETS,
    _AVG_POOL_TARGETS,
    _CONV_TARGET,
    _MAX_POOL_TARGETS,
)


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


def _per_axis(
    value: object, name: str, spatial: int, default: object
) -> tuple[int, ...]:
    """A pooling geometry argument, with ATen's empty-list default filled in."""

    if isinstance(value, (tuple, list)) and not value:
        value = default
    return _int_tuple(value, name, spatial)


def _pool_selections(
    node: Any, lowering: Lowering
) -> tuple[dict[int, Any], TensorType]:
    """The selections a multi-output pooling reads, and the value's type.

    A pooling reports its values and, for a maximum, where they came from. Only
    the selections the graph reads are emitted: a graph that keeps the values
    pays nothing for the indices, which is why the type comes from the selection
    rather than from the node's own (tuple) meta.
    """

    # An average pooling reports values only, so there is nothing to select.
    if not isinstance(getattr(node, "meta", {}).get("val"), (tuple, list)):
        return {}, _node_type(node)

    selections: dict[int, Any] = {}
    output_type: TensorType | None = None
    for user in node.users:
        if getattr(user, "target", None) is not operator.getitem:
            raise NotImplementedError(
                f"{_POOL_TARGETS} must be consumed through getitem"
            )
        index = user.args[1]
        if index not in {0, 1}:
            raise ValueError(f"pooling output index must be 0 or 1, got {index}")
        if not user.users:
            continue
        selections[index] = user
        if index == 0:
            output_type = _node_type(user)
    if output_type is None:
        output_type = _node_type(node.args[0])
    return selections, output_type


def _pool_shape(value: object) -> object:
    """A pooling size argument, where a missing or ``None`` value means zero."""

    return [] if value is None else value


def _pool_argument(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    position: int,
    name: str,
    default: object = None,
) -> object:
    """One pooling argument, which ATen may leave out when it is the default."""

    if position < len(args):
        return args[position]
    return kwargs.get(name, default)


def _pool_operation(
    node: Any, target: str, lowering: Lowering
) -> tuple[tuple[str, ...], dict[str, object], tuple[TensorType, ...], dict[int, Any]]:
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    average = target in _AVG_POOL_TARGETS
    input_type = _node_type(args[0])
    spatial = len(input_type.shape) - 2
    kernel_size = _pool_argument(args, kwargs, 1, "kernel_size")
    if kernel_size is None:
        raise ValueError(f"{target} requires a kernel size")
    kernel = _per_axis(_pool_shape(kernel_size), "kernel", spatial, None)
    attrs: dict[str, object] = {
        "mode": "avg" if average else "max",
        "kernel": kernel,
        # An omitted stride means the window does not overlap, so it is the
        # kernel itself rather than one.
        "stride": _per_axis(
            _pool_shape(_pool_argument(args, kwargs, 2, "stride")),
            "stride",
            spatial,
            kernel,
        ),
        "padding": _per_axis(
            _pool_shape(_pool_argument(args, kwargs, 3, "padding")),
            "padding",
            spatial,
            0,
        ),
    }
    if average:
        attrs["dilation"] = (1,) * spatial
        attrs["ceil_mode"] = _pool_argument(args, kwargs, 4, "ceil_mode", False)
        attrs["count_include_pad"] = _pool_argument(
            args, kwargs, 5, "count_include_pad", True
        )
        override = _pool_argument(args, kwargs, 6, "divisor_override")
        attrs["divisor_override"] = None if override is None else int(override)
    else:
        attrs["dilation"] = _per_axis(
            _pool_shape(_pool_argument(args, kwargs, 4, "dilation")),
            "dilation",
            spatial,
            1,
        )
        attrs["ceil_mode"] = _pool_argument(args, kwargs, 5, "ceil_mode", False)
    if not isinstance(attrs["ceil_mode"], bool):
        raise ValueError(
            f"{target}.ceil_mode must be a boolean, got {attrs['ceil_mode']!r}"
        )
    selections, output_type = _pool_selections(node, lowering)
    if average and 1 in selections:
        raise ValueError(f"{target} does not report indices")
    output_types = [output_type]
    if 1 in selections:
        output_types.append(_node_type(selections[1]))
    return (
        (_argument_ids(args[0], lowering.node_ids),),
        attrs,
        tuple(output_types),
        selections,
    )


def _lower_pool(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_types, selections = _pool_operation(node, target, lowering)
    output_ids = lowering.emit_multi("pool", inputs, attrs, output_types)
    produced = dict(enumerate(output_ids))
    for index, user in selections.items():
        lowering.bind(user, produced[index])
    lowering.bind(node, produced[0])
    return True


def _adaptive_pool_operation(
    node: Any, target: str, lowering: Lowering
) -> tuple[tuple[str, ...], dict[str, object], tuple[TensorType, ...], dict[int, Any]]:
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    input_type = _node_type(args[0])
    spatial = len(input_type.shape) - 2
    output_size = _per_axis(
        _pool_shape(_pool_argument(args, kwargs, 1, "output_size")),
        "output_size",
        spatial,
        None,
    )
    mode = "max" if target in _ADAPTIVE_MAX_POOL_TARGETS else "avg"
    selections, output_type = _pool_selections(node, lowering)
    output_types = [output_type]
    if 1 in selections:
        output_types.append(_node_type(selections[1]))
    return (
        (_argument_ids(args[0], lowering.node_ids),),
        {"mode": mode, "output_size": output_size},
        tuple(output_types),
        selections,
    )


def _lower_adaptive_pool(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_types, selections = _adaptive_pool_operation(
        node, target, lowering
    )
    output_ids = lowering.emit_multi("adaptive_pool", inputs, attrs, output_types)
    produced = dict(enumerate(output_ids))
    for index, user in selections.items():
        lowering.bind(user, produced[index])
    lowering.bind(node, produced[0])
    return True


# The spellings that report their values and indices as a tuple are consumed
# through getitem; the ones that report one value bind directly.
_POOL_TARGETS = _MAX_POOL_TARGETS | _AVG_POOL_TARGETS
_ADAPTIVE_POOL_TARGETS = _ADAPTIVE_MAX_POOL_TARGETS | _ADAPTIVE_AVG_POOL_TARGETS

HANDLERS: dict[str, tuple[Handler, ...]] = {
    _CONV_TARGET: (_lower_conv,),
    **{target: (_lower_pool,) for target in _POOL_TARGETS},
    **{target: (_lower_adaptive_pool,) for target in _ADAPTIVE_POOL_TARGETS},
}
