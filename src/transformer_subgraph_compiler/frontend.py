from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

from .ir import DType, Graph, Operation, TensorType, Value

_SUPPORTED_OPS = {
    "aten.mm.default": "matmul",
    "aten.addmm.default": "addmm",
    "aten._softmax.default": "softmax",
    "aten.silu.default": "silu",
    "aten.gelu.default": "gelu",
    "aten.mul.Tensor": "mul",
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "sub",
    "aten.rsub.Scalar": "rsub",
    "aten.div.Tensor": "div",
    "aten.neg.default": "neg",
    "aten.abs.default": "abs",
    "aten.sigmoid.default": "sigmoid",
    "aten.tanh.default": "tanh",
    "aten.relu.default": "relu",
    "aten.exp.default": "exp",
    "aten.erf.default": "erf",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.reciprocal.default": "reciprocal",
}

_TORCH_DTYPES = {
    "torch.float16": DType.F16,
    "torch.bfloat16": DType.BF16,
    "torch.float32": DType.F32,
}

_NUMERIC_ATTRS = {
    "addmm": {"alpha": 1.0, "beta": 1.0},
    "add": {"alpha": 1.0},
    "sub": {"alpha": 1.0},
    "rsub": {"alpha": 1.0},
}

_BINARY_OPS = {"mul", "add", "sub", "rsub", "div"}


def capture_fx(fn: Callable[..., Any], example_inputs: Sequence[Any]) -> Graph:
    """Capture a supported callable with make_fx and build the compiler IR."""

    try:
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the FX frontend") from exc

    traced = make_fx(fn)(*example_inputs)
    values: dict[str, Value] = {}
    input_ids: list[str] = []
    node_ids: dict[object, str] = {}
    ops: list[Operation] = []
    output_ids: tuple[str, ...] = ()

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            value_id = f"arg{len(input_ids)}"
            node_ids[node] = value_id
            input_ids.append(value_id)
            values[value_id] = Value(value_id, _node_type(node))
        elif node.op == "call_function":
            if node in node_ids:
                continue
            rms_norm = _rms_norm_operation(node, node_ids)
            if rms_norm is not None:
                inputs, attrs, output_type, folded_nodes = rms_norm
                output_id = f"v{len(ops)}"
                ops.append(Operation("rms_norm", inputs, (output_id,), attrs))
                values[output_id] = Value(output_id, output_type)
                for folded_node in folded_nodes:
                    node_ids[folded_node] = output_id
                continue
            target = str(node.target)
            if target == "aten.native_layer_norm.default":
                output_id = f"v{len(ops)}"
                inputs, attrs, output_type = _layer_norm_operation(
                    node, node_ids
                )
                ops.append(Operation("layer_norm", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                for user in node.users:
                    node_ids[user] = output_id
                continue
            if target not in _SUPPORTED_OPS:
                raise ValueError(f"unsupported FX target: {target}")
            op_name = _SUPPORTED_OPS[target]
            inputs, scalar_attrs = _operation_inputs(op_name, node.args, node_ids)
            output_id = f"v{len(ops)}"
            attrs = _operation_attrs(op_name, node.kwargs)
            attrs.update(scalar_attrs)
            op = Operation(op_name, inputs, (output_id,), attrs)
            ops.append(op)
            node_ids[node] = output_id
            values[output_id] = Value(output_id, _node_type(node))
        elif node.op == "output":
            output_ids = _output_ids(node.args[0], node_ids)
        else:
            raise ValueError(f"unsupported FX node kind: {node.op}")

    if not output_ids:
        raise ValueError("captured graph has no outputs")

    graph = Graph(tuple(input_ids), values, tuple(ops), output_ids)
    graph.validate()
    return graph


def _node_type(node: object) -> TensorType:
    meta = getattr(node, "meta", {})
    tensor = meta.get("val")
    if tensor is None or not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
        raise ValueError(f"missing concrete shape/dtype metadata for FX node: {node}")

    dtype = _TORCH_DTYPES.get(str(tensor.dtype))
    if dtype is None:
        raise ValueError(f"unsupported dtype: {tensor.dtype}")
    if any(dim <= 0 for dim in tensor.shape):
        raise ValueError(f"dynamic or empty shapes are not supported yet: {tuple(tensor.shape)}")
    return TensorType(tuple(int(dim) for dim in tensor.shape), dtype)


def _layer_norm_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
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
    for user in getattr(node, "users", {}):
        if user.target is not operator.getitem:
            raise NotImplementedError(
                "native_layer_norm must be consumed through getitem"
            )
        if len(user.args) != 2 or not isinstance(user.args[1], int):
            raise ValueError(f"malformed layer_norm getitem: {user!r}")
        index = user.args[1]
        if index == 0 and user.users:
            output_node = user
        elif index in {1, 2} and user.users:
            raise NotImplementedError(
                "layer_norm mean/rstd outputs are not supported yet"
            )
        elif index not in {0, 1, 2}:
            raise ValueError(f"native_layer_norm output index must be 0, 1, or 2, got {index}")

    if output_node is None:
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
    return tuple(inputs), attrs, _node_type(output_node)


def _rms_norm_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[
    tuple[str, ...],
    dict[str, object],
    TensorType,
    tuple[object, ...],
] | None:
    target = str(getattr(node, "target", ""))
    cast_input = None
    if target == "aten.pow.Tensor_Scalar":
        pow_node = node
    elif target == "aten._to_copy.default":
        pow_users = [
            user
            for user in getattr(node, "users", {})
            if str(user.target) == "aten.pow.Tensor_Scalar"
        ]
        if len(pow_users) != 1:
            return None
        pow_node = pow_users[0]
        cast_input = node
    else:
        return None
    if pow_node is None or str(pow_node.target) != "aten.pow.Tensor_Scalar":
        return None

    pow_args = pow_node.args
    if len(pow_args) != 2 or pow_args[1] != 2:
        return None
    compute_input = pow_args[0]
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

    mean = _single_user(pow_node)
    if mean is None or str(mean.target) != "aten.mean.dim":
        return None
    mean_args = mean.args
    if len(mean_args) != 3 or mean_args[0] is not pow_node:
        return None
    if mean_args[2] is not True:
        return None
    axes = mean_args[1]
    if not isinstance(axes, (tuple, list)) or not axes:
        return None
    if not all(
        isinstance(axis, int) and not isinstance(axis, bool) for axis in axes
    ):
        return None

    add_eps = _single_user(mean)
    if add_eps is None or str(add_eps.target) != "aten.add_.Scalar":
        return None
    add_args = add_eps.args
    if len(add_args) != 2 or add_args[0] is not mean:
        return None
    eps = add_args[1]
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        return None

    rsqrt = _single_user(add_eps)
    if rsqrt is None or str(rsqrt.target) != "aten.rsqrt.default":
        return None
    if len(rsqrt.args) != 1 or rsqrt.args[0] is not add_eps:
        return None

    normalized = _single_user(rsqrt)
    if normalized is None or str(normalized.target) != "aten.mul.Tensor":
        return None
    normalized_args = normalized.args
    if len(normalized_args) != 2:
        return None
    if normalized_args[0] is not compute_input or normalized_args[1] is not rsqrt:
        return None
    if cast_input is not None and set(cast_input.users) != {pow_node, normalized}:
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
    if (
        weighted_output is not None
        and str(weighted_output.target) == "aten.mul.Tensor"
    ):
        weighted_args = weighted_output.args
        if len(weighted_args) == 2 and weighted_args[0] is normalized:
            candidate_weight = weighted_args[1]
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
    folded_nodes = [pow_node, mean, add_eps, rsqrt, normalized]
    if cast_input is not None:
        folded_nodes.insert(0, cast_input)
    if weight is not None:
        folded_nodes.append(weighted_output)
    if output_node is not normalized and output_node is not weighted_output:
        folded_nodes.append(output_node)
    return tuple(inputs), attrs, _node_type(output_node), tuple(folded_nodes)


def _single_user(node: object) -> Any | None:
    users = tuple(getattr(node, "users", {}))
    return users[0] if len(users) == 1 else None


def _argument_ids(argument: object, node_ids: dict[object, str]) -> str:
    try:
        return node_ids[argument]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported FX argument: {argument!r}") from exc


def _operation_inputs(
    op: str,
    args: tuple[object, ...],
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object]]:
    if op == "softmax":
        if len(args) != 3:
            raise ValueError(f"softmax expects three arguments, got {len(args)}")
        input_id, dim, half_to_float = args
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"softmax.dim must be an integer, got {dim!r}")
        if not isinstance(half_to_float, bool):
            raise ValueError(
                f"softmax.half_to_float must be a boolean, got {half_to_float!r}"
            )
        return ((_argument_ids(input_id, node_ids),), {"dim": dim})

    if op not in _BINARY_OPS:
        return tuple(_argument_ids(arg, node_ids) for arg in args), {}

    if len(args) != 2:
        raise ValueError(f"{op} expects two arguments, got {len(args)}")

    tensor_inputs: list[str] = []
    scalars: list[tuple[int, float]] = []
    for position, argument in enumerate(args):
        try:
            tensor_inputs.append(_argument_ids(argument, node_ids))
        except ValueError:
            if (
                not isinstance(argument, (int, float))
                or isinstance(argument, bool)
            ):
                raise ValueError(
                    f"unsupported {op} argument at position {position}: {argument!r}"
                ) from None
            scalars.append((position, float(argument)))

    if len(tensor_inputs) == 2:
        return tuple(tensor_inputs), {}
    if len(tensor_inputs) == 1 and len(scalars) == 1:
        position, value = scalars[0]
        return (
            tuple(tensor_inputs),
            {
                "scalar": value,
                "scalar_position": (
                    "lhs" if position == 0 or op == "rsub" else "rhs"
                ),
            },
        )
    raise ValueError(f"{op} requires at least one tensor input")


def _operation_attrs(op: str, kwargs: dict[str, object]) -> dict[str, object]:
    numeric_attrs = _NUMERIC_ATTRS.get(op, {})
    allowed = set(numeric_attrs)
    if op == "gelu":
        allowed.add("approximate")
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"unsupported keyword arguments for {op}: {sorted(unknown)}")

    attrs: dict[str, object] = {}
    for name, default in numeric_attrs.items():
        value = kwargs.get(name, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{op}.{name} must be a scalar number, got {value!r}")
        attrs[name] = float(value)

    if op == "gelu":
        approximate = kwargs.get("approximate", "none")
        if approximate not in {"none", "tanh"}:
            raise ValueError(
                "gelu.approximate must be 'none' or 'tanh', "
                f"got {approximate!r}"
            )
        attrs["approximate"] = approximate
    return attrs


def _output_ids(output: object, node_ids: dict[object, str]) -> tuple[str, ...]:
    try:
        return (_argument_ids(output, node_ids),)
    except ValueError:
        pass
    if isinstance(output, (tuple, list)):
        return tuple(_argument_ids(item, node_ids) for item in output)
    raise ValueError(f"unsupported FX output: {output!r}")
