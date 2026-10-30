"""The operators the backend implements as one fused kernel: attention, layer
norm, RMS norm, and log-sigmoid.

Recognising them is the whole job here. Each builder verifies the surrounding
subgraph really is the pattern it claims - an RMS norm is only an ``rms_norm``
if the square, mean, epsilon and reciprocal multiply all line up - and returns
``None`` otherwise, so an operator that merely resembles one stays decomposed.
The builder also reports which FX nodes it consumed, so the loop can bind them
all to the single output it produced."""

from __future__ import annotations

import operator
from typing import Any
from ..ir import DType, TensorType
from .fx import _node_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .refs import _single_user
from .targets import (
    _DIRECT_ATTENTION_TARGETS,
    _LAYER_NORM_TARGET,
    _LOGSIGMOID_TARGET,
    _TUPLE_ATTENTION_TARGETS,
)


def _attention_call_arguments(
    target: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[object, object, object, object, object]:
    def argument(
        name: str, position: int | None = None, default: object = None
    ) -> object:
        if name in kwargs:
            return kwargs[name]
        if position is not None and position < len(args):
            return args[position]
        return default

    if target == "aten.scaled_dot_product_attention.default":
        return (
            argument("attn_mask", 3),
            argument("dropout_p", 4, 0.0),
            argument("is_causal", 5, False),
            argument("enable_gqa", default=False),
            argument("scale"),
        )
    if target == "aten._scaled_dot_product_flash_attention_for_cpu.default":
        return (
            argument("attn_mask"),
            argument("dropout_p", 3, 0.0),
            argument("is_causal", 4, False),
            False,
            argument("scale"),
        )
    if target == "aten._scaled_dot_product_flash_attention.default":
        if argument("return_debug_mask", 5, False):
            raise NotImplementedError(
                "scaled_dot_product_attention debug masks are not supported"
            )
        return (
            None,
            argument("dropout_p", 3, 0.0),
            argument("is_causal", 4, False),
            False,
            argument("scale"),
        )
    if target == "aten._scaled_dot_product_efficient_attention.default":
        return (
            argument("attn_bias", 3),
            argument("dropout_p", 5, 0.0),
            argument("is_causal", 6, False),
            False,
            argument("scale"),
        )
    raise ValueError(f"unsupported attention target: {target}")


def _attention_output_node(node: object, target: str) -> object:
    if target == "aten.scaled_dot_product_attention.default":
        return node

    output_node = None
    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(
                "scaled_dot_product_attention outputs must be selected with getitem"
            )
        if len(user.args) != 2 or not isinstance(user.args[1], int):
            raise ValueError(f"malformed attention output selection: {user!r}")
        if user.args[1] == 0 and user.users:
            output_node = user
        elif user.args[1] != 0 and user.users:
            raise NotImplementedError(
                "scaled_dot_product_attention auxiliary outputs are not supported"
            )
    if output_node is None:
        raise ValueError("scaled_dot_product_attention result is not used")
    return output_node


def _attention_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], object]:
    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if len(args) < 3:
        raise ValueError("scaled_dot_product_attention expects Q, K and V")

    query, key, value = args[:3]
    query_type = _node_type(query)
    key_type = _node_type(key)
    value_type = _node_type(value)
    if len(query_type.shape) not in {3, 4}:
        raise NotImplementedError(
            "scaled_dot_product_attention requires rank-3 or rank-4 inputs, "
            f"got {query_type.shape}"
        )
    if not (query_type.dtype == key_type.dtype == value_type.dtype):
        raise NotImplementedError(
            "scaled_dot_product_attention requires identical Q/K/V dtypes"
        )
    if not (len(query_type.shape) == len(key_type.shape) == len(value_type.shape)):
        raise NotImplementedError(
            "scaled_dot_product_attention requires matching Q/K/V ranks"
        )

    attn_mask, dropout_p, is_causal, enable_gqa, scale = _attention_call_arguments(
        target, args, kwargs
    )
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float)):
        raise ValueError(f"dropout_p must be a scalar number, got {dropout_p!r}")
    if float(dropout_p) != 0.0:
        raise NotImplementedError(
            "scaled_dot_product_attention dropout is not supported"
        )
    if not isinstance(is_causal, bool):
        raise ValueError(f"is_causal must be a boolean, got {is_causal!r}")
    if not isinstance(enable_gqa, bool):
        raise ValueError(f"enable_gqa must be a boolean, got {enable_gqa!r}")
    if scale is None:
        scale = query_type.shape[-1] ** -0.5
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"attention scale must be a scalar number, got {scale!r}")
    if attn_mask is not None and is_causal:
        raise ValueError("attn_mask and is_causal=True cannot both be specified")

    inputs = [
        _argument_ids(query, node_ids),
        _argument_ids(key, node_ids),
        _argument_ids(value, node_ids),
    ]
    if attn_mask is not None:
        inputs.append(_argument_ids(attn_mask, node_ids))

    output_node = _attention_output_node(node, target)

    attrs: dict[str, object] = {
        "scale": float(scale),
        "has_mask": attn_mask is not None,
        "causal": is_causal,
        "safe_softmax": True,
    }
    return tuple(inputs), attrs, output_node


def _layer_norm_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], TensorType, TensorType, frozenset[int]]:
    """Read one ``native_layer_norm`` call and the selections the graph reads.

    The op writes the normalized value, and - because the kernel reduces the
    moments anyway - the per-row mean and reciprocal standard deviation as two
    more outputs, so a graph that reads them pays no second reduction.
    """
    args = getattr(node, "args", ())
    if len(args) != 5:
        raise ValueError(f"native_layer_norm expects five arguments, got {len(args)}")
    input_value, normalized_shape, weight, bias, eps = args

    if not isinstance(normalized_shape, (tuple, list)) or not normalized_shape:
        raise ValueError(
            f"layer_norm normalized_shape must be a non-empty sequence, got {normalized_shape!r}"
        )
    shape: list[int] = []
    for dim in normalized_shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(
                "layer_norm normalized_shape must contain positive integers, "
                f"got {normalized_shape!r}"
            )
        shape.append(dim)
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise ValueError(f"layer_norm.eps must be a scalar number, got {eps!r}")

    output_node = None
    wanted: set[int] = set()
    for user in getattr(node, "users", {}):
        if user.target is not operator.getitem:
            raise NotImplementedError(
                "native_layer_norm must be consumed through getitem"
            )
        if len(user.args) != 2 or not isinstance(user.args[1], int):
            raise ValueError(f"malformed layer_norm getitem: {user!r}")
        index = user.args[1]
        if index not in {0, 1, 2}:
            raise ValueError(
                f"native_layer_norm output index must be 0, 1, or 2, got {index}"
            )
        if not user.users:
            continue
        wanted.add(index)
        if index == 0:
            output_node = user

    if output_node is None and not wanted:
        raise ValueError("native_layer_norm result is not used")

    inputs = [_argument_ids(input_value, node_ids)]
    has_weight = weight is not None
    has_bias = bias is not None
    if has_weight:
        inputs.append(_argument_ids(weight, node_ids))
    if has_bias:
        inputs.append(_argument_ids(bias, node_ids))

    attrs: dict[str, object] = {
        "normalized_shape": tuple(shape),
        "eps": float(eps),
        "has_weight": has_weight,
        "has_bias": has_bias,
    }
    input_type = _node_type(node.args[0])
    output_type = TensorType(input_type.shape, input_type.dtype)
    # ATen reports the statistics with the normalized axes collapsed to one, so
    # they broadcast against the normalized value.
    stat_shape = input_type.shape[: len(input_type.shape) - len(shape)] + (1,) * len(
        shape
    )
    stat_type = TensorType(stat_shape, input_type.dtype)
    return (
        tuple(inputs),
        attrs,
        output_type,
        stat_type,
        frozenset(wanted),
    )


def _logsigmoid_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], object]:
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if len(args) != 1 or kwargs:
        raise ValueError(
            "log_sigmoid_forward expects one positional argument without keywords"
        )

    output_node = None
    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise NotImplementedError(
                "log_sigmoid_forward outputs must be selected with getitem"
            )
        if (
            len(user.args) != 2
            or not isinstance(user.args[1], int)
            or isinstance(user.args[1], bool)
        ):
            raise ValueError(f"malformed logsigmoid output selection: {user!r}")
        index = user.args[1]
        if index == 0:
            output_node = user
        elif index == 1 and user.users:
            raise NotImplementedError(
                "log_sigmoid_forward backward buffer is not supported"
            )
        elif index not in {0, 1}:
            raise ValueError(
                f"log_sigmoid_forward output index must be 0 or 1, got {index}"
            )

    if output_node is None:
        raise ValueError("log_sigmoid_forward result is not used")
    return (_argument_ids(args[0], node_ids),), {}, output_node


def _rms_norm_square_base(square_node: object) -> object | None:
    """Return the tensor RMSNorm squares, or ``None`` if it does not square."""

    args = getattr(square_node, "args", ())
    if str(getattr(square_node, "target", "")) == "aten.pow.Tensor_Scalar":
        if len(args) != 2:
            return None
        exponent = args[1]
        if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
            return None
        if exponent != 2:
            return None
        return args[0]
    if len(args) != 2 or args[0] is not args[1]:
        return None
    return args[0]


def _rms_norm_normalized_input(
    normalized: object,
    compute_input: object,
    scale: object,
    scale_op: str,
) -> object | None:
    """Match ``x * rsqrt(var)`` and ``x / sqrt(var)`` spellings."""

    args = getattr(normalized, "args", ())
    if len(args) != 2 or args[0] is not compute_input or args[1] is not scale:
        return None
    target = str(getattr(normalized, "target", ""))
    expected = (
        "aten.mul.Tensor" if scale_op == "aten.rsqrt.default" else "aten.div.Tensor"
    )
    return compute_input if target == expected else None


def _rms_norm_operation(
    node: object,
    node_ids: dict[object, str],
) -> (
    tuple[
        tuple[str, ...],
        dict[str, object],
        TensorType,
        tuple[object, ...],
    ]
    | None
):
    target = str(getattr(node, "target", ""))
    cast_input = None
    square_targets = {"aten.pow.Tensor_Scalar", "aten.mul.Tensor"}
    if target in square_targets:
        square_node = node
    elif target == "aten._to_copy.default":
        square_users = [
            user
            for user in getattr(node, "users", {})
            if str(user.target) in square_targets
        ]
        if len(square_users) != 1:
            return None
        square_node = square_users[0]
        cast_input = node
    else:
        return None

    compute_input = _rms_norm_square_base(square_node)
    if compute_input is None:
        return None
    if cast_input is not None:
        cast_args = cast_input.args
        if len(cast_args) != 1 or compute_input is not cast_input:
            return None
        input_value = cast_args[0]
        try:
            input_type = _node_type(input_value)
            cast_type = _node_type(cast_input)
        except ValueError:
            return None
        if input_type.dtype == DType.F32 or cast_type.dtype != DType.F32:
            return None
    else:
        input_value = compute_input

    mean = _single_user(square_node)
    if mean is None or str(mean.target) != "aten.mean.dim":
        return None
    mean_args = mean.args
    if len(mean_args) != 3 or mean_args[0] is not square_node:
        return None
    if mean_args[2] is not True:
        return None
    axes = mean_args[1]
    if not isinstance(axes, (tuple, list)) or not axes:
        return None
    if not all(isinstance(axis, int) and not isinstance(axis, bool) for axis in axes):
        return None

    add_eps = _single_user(mean)
    if add_eps is None or str(add_eps.target) not in {
        "aten.add_.Scalar",
        "aten.add.Scalar",
        "aten.add.Tensor",
    }:
        return None
    add_args = add_eps.args
    if len(add_args) != 2 or add_args[0] is not mean:
        return None
    eps = add_args[1]
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        return None

    scale = _single_user(add_eps)
    if scale is None or str(scale.target) not in {
        "aten.rsqrt.default",
        "aten.sqrt.default",
    }:
        return None
    if len(scale.args) != 1 or scale.args[0] is not add_eps:
        return None
    scale_op = str(scale.target)

    normalized = _single_user(scale)
    if (
        normalized is None
        or _rms_norm_normalized_input(normalized, compute_input, scale, scale_op)
        is None
    ):
        return None
    if cast_input is not None and set(cast_input.users) != {square_node, normalized}:
        return None

    input_type = _node_type(input_value)
    rank = len(input_type.shape)
    if rank == 0:
        return None
    normalized_axes = sorted({axis % rank for axis in axes})
    if len(normalized_axes) != len(axes):
        return None
    if normalized_axes != list(range(rank - len(normalized_axes), rank)):
        return None
    normalized_shape = tuple(input_type.shape[axis] for axis in normalized_axes)
    expected_mean_shape = input_type.shape[: rank - len(normalized_axes)]
    expected_mean_shape += (1,) * len(normalized_axes)
    if _node_type(mean).shape != expected_mean_shape:
        return None

    weight = None
    output_node = normalized
    weighted_output = _single_user(normalized)
    if weighted_output is not None and str(weighted_output.target) == "aten.mul.Tensor":
        weighted_args = weighted_output.args
        position = (
            0
            if len(weighted_args) == 2 and weighted_args[0] is normalized
            else 1
            if len(weighted_args) == 2 and weighted_args[1] is normalized
            else None
        )
        if position is not None:
            candidate_weight = weighted_args[1 - position]
            try:
                weight_type = _node_type(candidate_weight)
            except ValueError:
                weight_type = None
            if (
                weight_type is not None
                and weight_type.shape == normalized_shape
                and weight_type.dtype == input_type.dtype
            ):
                weight = candidate_weight
                output_node = weighted_output

    if cast_input is not None:
        cast_output = _single_user(output_node)
        if (
            cast_output is None
            or str(cast_output.target) != "aten._to_copy.default"
            or len(cast_output.args) != 1
            or cast_output.args[0] is not output_node
            or _node_type(cast_output).dtype != input_type.dtype
        ):
            return None
        output_node = cast_output

    inputs = [_argument_ids(input_value, node_ids)]
    if weight is not None:
        inputs.append(_argument_ids(weight, node_ids))

    attrs: dict[str, object] = {
        "normalized_shape": normalized_shape,
        "eps": float(eps),
        "has_weight": weight is not None,
    }
    folded_nodes = [square_node, mean, add_eps, scale, normalized]
    if cast_input is not None:
        folded_nodes.insert(0, cast_input)
    if weight is not None:
        folded_nodes.append(weighted_output)
    if output_node is not normalized and output_node is not weighted_output:
        folded_nodes.append(output_node)
    return tuple(inputs), attrs, _node_type(output_node), tuple(folded_nodes)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _lower_rms_norm(lowering: Lowering, node: Any, target: str) -> bool:
    """Recognise an RMS norm, which no target name announces.

    ``rms_norm`` is decomposed by the time it reaches the graph, so the only way
    to get the fused operator back is to match the pattern. This handler is
    therefore tried for *every* node, not only for a known target, and it
    returns False for the overwhelming majority of them.
    """

    rms_norm = _rms_norm_operation(node, lowering.node_ids)
    if rms_norm is None:
        return False
    inputs, attrs, output_type, folded_nodes = rms_norm
    output_id = lowering.emit("rms_norm", inputs, attrs, output_type)
    # Every intermediate the pattern consumed is that one output now; binding
    # them keeps a later node that still refers to the square from emitting a
    # second, separate computation.
    for folded_node in folded_nodes:
        lowering.bind(folded_node, output_id)
    return True


def _lower_attention(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_node = _attention_operation(node, lowering.node_ids)
    output_id = lowering.emit("attention", inputs, attrs, _node_type(output_node))
    lowering.bind(node, output_id)
    if target in _TUPLE_ATTENTION_TARGETS:
        # The tuple form returns (output, weights); the weights are only ever
        # discarded by the decompositions we support, so every user resolves to
        # the same value.
        for user in node.users:
            lowering.bind(user, output_id)
    return True


def _lower_layer_norm(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type, stat_type, wanted = _layer_norm_operation(
        node, lowering.node_ids
    )
    # One kernel writes the normalized value and, when the graph reads them, the
    # two statistics it already reduced. Reading only a statistic still needs the
    # op, so the normalized value is written and simply left unused.
    # A graph that reads only the normalized value keeps the single-output form;
    # any other selection needs the full tuple.
    output_types = (
        (output_type, stat_type, stat_type) if wanted - {0} else (output_type,)
    )
    output_ids = lowering.emit_multi("layer_norm", inputs, attrs, output_types)
    produced = dict(enumerate(output_ids))
    for user in node.users:
        index = user.args[1]
        if index in produced:
            lowering.bind(user, produced[index])
        else:
            lowering.dead_nodes.add(user)
    return True


def _lower_logsigmoid(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_node = _logsigmoid_operation(node, lowering.node_ids)
    output_id = lowering.emit("logsigmoid", inputs, attrs, _node_type(output_node))
    lowering.bind(node, output_id)
    for user in node.users:
        lowering.bind(user, output_id)
    return True


# Tried before any target-specific handler, because recognition depends on the
# node's shape rather than on its name.
PRE_HANDLERS: tuple[Handler, ...] = (_lower_rms_norm,)

HANDLERS: dict[str, tuple[Handler, ...]] = {
    **{target: (_lower_attention,) for target in _DIRECT_ATTENTION_TARGETS},
    _LAYER_NORM_TARGET: (_lower_layer_norm,),
    _LOGSIGMOID_TARGET: (_lower_logsigmoid,),
}
