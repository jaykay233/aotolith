from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

from .ir import DType, Graph, Operation, TensorType, Value

_SUPPORTED_OPS = {
    "aten.mm.default": "matmul",
    "aten.bmm.default": "matmul",
    "aten.addmm.default": "addmm",
    "aten.baddbmm.default": "addmm",
    "aten.view.default": "reshape",
    "aten._unsafe_view.default": "reshape",
    "aten.reshape.default": "reshape",
    "aten.transpose.int": "transpose",
    "aten.permute.default": "permute",
    "aten.expand.default": "expand",
    "aten.slice.Tensor": "slice",
    "aten.flip.default": "flip",
    "aten.roll.default": "roll",
    "aten.triu.default": "triu",
    "aten.tril.default": "tril",
    "aten.unsqueeze.default": "unsqueeze",
    "aten.squeeze.dim": "squeeze",
    "aten.clone.default": "contiguous",
    "aten.alias.default": "alias",
    "aten._to_copy.default": "cast",
    "aten._softmax.default": "softmax",
    "aten._safe_softmax.default": "safe_softmax",
    "aten.sum.default": "sum",
    "aten.sum.dim_IntList": "sum",
    "aten.mean.default": "mean",
    "aten.mean.dim": "mean",
    "aten.silu.default": "silu",
    "aten.gelu.default": "gelu",
    "aten.softplus.default": "softplus",
    "aten.mish.default": "mish",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "sub",
    "aten.rsub.Scalar": "rsub",
    "aten.div.Tensor": "div",
    "aten.pow.Tensor_Tensor": "pow",
    "aten.pow.Tensor_Scalar": "pow",
    "aten.pow.Scalar": "pow",
    "aten.neg.default": "neg",
    "aten.abs.default": "abs",
    "aten.sigmoid.default": "sigmoid",
    "aten.tanh.default": "tanh",
    "aten.relu.default": "relu",
    "aten.exp.default": "exp",
    "aten.expm1.default": "expm1",
    "aten.exp2.default": "exp2",
    "aten.log.default": "log",
    "aten.log2.default": "log2",
    "aten.log10.default": "log10",
    "aten.log1p.default": "log1p",
    "aten.erf.default": "erf",
    "aten.sin.default": "sin",
    "aten.cos.default": "cos",
    "aten.floor.default": "floor",
    "aten.ceil.default": "ceil",
    "aten.round.default": "round",
    "aten.trunc.default": "trunc",
    "aten.sign.default": "sign",
    "aten.square.default": "square",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.reciprocal.default": "reciprocal",
    "aten.clamp.default": "clamp",
    "aten.clamp.Tensor": "clamp",
    "aten.clamp_min.default": "clamp_min",
    "aten.clamp_min.Tensor": "clamp_min",
    "aten.clamp_max.default": "clamp_max",
    "aten.clamp_max.Tensor": "clamp_max",
    "aten.hardtanh.default": "clamp",
    "aten.leaky_relu.default": "leaky_relu",
    "aten.elu.default": "elu",
    "aten.hardswish.default": "hardswish",
    "aten.hardsigmoid.default": "hardsigmoid",
    "aten.threshold.default": "threshold",
    "aten.eq.Tensor": "eq",
    "aten.eq.Scalar": "eq",
    "aten.ne.Tensor": "ne",
    "aten.ne.Scalar": "ne",
    "aten.lt.Tensor": "lt",
    "aten.lt.Scalar": "lt",
    "aten.le.Tensor": "le",
    "aten.le.Scalar": "le",
    "aten.gt.Tensor": "gt",
    "aten.gt.Scalar": "gt",
    "aten.ge.Tensor": "ge",
    "aten.ge.Scalar": "ge",
    "aten.where.self": "where",
    "aten.logical_and.default": "logical_and",
    "aten.logical_or.default": "logical_or",
    "aten.logical_xor.default": "logical_xor",
    "aten.logical_not.default": "logical_not",
    "aten.bitwise_and.Tensor": "logical_and",
    "aten.bitwise_and.Scalar": "logical_and",
    "aten.bitwise_or.Tensor": "logical_or",
    "aten.bitwise_or.Scalar": "logical_or",
    "aten.bitwise_xor.Tensor": "logical_xor",
    "aten.bitwise_xor.Scalar": "logical_xor",
    "aten.bitwise_not.default": "logical_not",
    "aten.isnan.default": "isnan",
    "aten.isinf.default": "isinf",
    "aten.masked_fill.Scalar": "masked_fill",
    "aten.maximum.default": "maximum",
    "aten.minimum.default": "minimum",
    "aten.logaddexp.default": "logaddexp",
    "aten.atan2.default": "atan2",
    "aten.scalar_tensor.default": "scalar_tensor",
    "aten.zeros.default": "zeros",
    "aten.ones.default": "ones",
    "aten.full.default": "full",
    "aten.new_zeros.default": "new_zeros",
    "aten.new_ones.default": "new_ones",
    "aten.new_full.default": "new_full",
    "aten.zeros_like.default": "zeros_like",
    "aten.ones_like.default": "ones_like",
    "aten.full_like.default": "full_like",
    "aten.cat.default": "cat",
    "aten.embedding.default": "embedding",
    "aten.index_select.default": "index_select",
    "aten.gather.default": "gather",
    "aten.scatter.src": "scatter",
    "aten.scatter.value": "scatter",
}

_TORCH_DTYPES = {
    "torch.float16": DType.F16,
    "torch.bfloat16": DType.BF16,
    "torch.float32": DType.F32,
    "torch.bool": DType.BOOL,
    "torch.int32": DType.I32,
    "torch.int64": DType.I64,
}

_NUMERIC_ATTRS = {
    "addmm": {"alpha": 1.0, "beta": 1.0},
    "add": {"alpha": 1.0},
    "sub": {"alpha": 1.0},
    "rsub": {"alpha": 1.0},
    "leaky_relu": {"negative_slope": 0.01},
    "elu": {"alpha": 1.0},
    "softplus": {"beta": 1.0, "threshold": 20.0},
    "threshold": {"threshold": 0.0, "value": 0.0},
}

_COMPARISON_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
_LOGICAL_BINARY_OPS = {"logical_and", "logical_or", "logical_xor"}
_BINARY_OPS = _COMPARISON_OPS | {
    "mul",
    "add",
    "sub",
    "rsub",
    "div",
    "pow",
    "maximum",
    "minimum",
    "logaddexp",
    "atan2",
    "logical_and",
    "logical_or",
    "logical_xor",
}
_REDUCTION_OPS = {"sum", "mean"}
_DIRECT_ATTENTION_TARGETS = {
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten.scaled_dot_product_attention.default",
}

_TUPLE_ATTENTION_TARGETS = {
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
}

_SEQUENCE_VIEW_TARGETS = {
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
}

_SELECT_TARGET = "aten.select.int"
_STACK_TARGET = "aten.stack.default"
_SCATTER_TARGETS = {"aten.scatter.src", "aten.scatter.value"}
_LOGSIGMOID_TARGET = "aten.log_sigmoid_forward.default"


def capture_fx(fn: Callable[..., Any], example_inputs: Sequence[Any]) -> Graph:
    """Capture a supported callable with make_fx and build the compiler IR."""

    try:
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the FX frontend") from exc

    traced = make_fx(fn)(*example_inputs)
    _reject_unsupported_attention_dropout(traced.graph)
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
            target = str(node.target)
            if target == _SELECT_TARGET:
                select_ops = _select_operations(node, node_ids, len(ops))
                for operation, value in select_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = select_ops[-1][0].outputs[0]
                continue
            if target in _SEQUENCE_VIEW_TARGETS:
                sequence_ops = _sequence_view_operations(node, node_ids, len(ops))
                for operation, value in sequence_ops:
                    ops.append(operation)
                    values[value.id] = value
                continue
            if target == _STACK_TARGET:
                stack_ops, output_id, output_type = _stack_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in stack_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target == "aten.embedding.default":
                inputs, attrs, output_type = _embedding_operation(
                    node, node_ids
                )
                output_id = f"v{len(ops)}"
                ops.append(Operation("embedding", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target == "aten.index_select.default":
                inputs, attrs, output_type = _index_select_operation(
                    node, node_ids
                )
                output_id = f"v{len(ops)}"
                ops.append(Operation("index_select", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target == "aten.gather.default":
                inputs, attrs, output_type = _gather_operation(node, node_ids)
                output_id = f"v{len(ops)}"
                ops.append(Operation("gather", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target in _SCATTER_TARGETS:
                inputs, attrs, output_type = _scatter_operation(node, node_ids)
                output_id = f"v{len(ops)}"
                ops.append(Operation("scatter", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
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
            if target in _DIRECT_ATTENTION_TARGETS:
                inputs, attrs, output_node = _attention_operation(
                    node, node_ids
                )
                output_id = f"v{len(ops)}"
                ops.append(Operation("attention", inputs, (output_id,), attrs))
                values[output_id] = Value(output_id, _node_type(output_node))
                node_ids[node] = output_id
                if target in _TUPLE_ATTENTION_TARGETS:
                    for user in node.users:
                        node_ids[user] = output_id
                continue
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
            if target == _LOGSIGMOID_TARGET:
                output_id = f"v{len(ops)}"
                inputs, attrs, output_node = _logsigmoid_operation(node, node_ids)
                ops.append(Operation("logsigmoid", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                for user in node.users:
                    node_ids[user] = output_id
                values[output_id] = Value(output_id, _node_type(output_node))
                continue
            if target not in _SUPPORTED_OPS:
                raise ValueError(f"unsupported FX target: {target}")
            op_name = _SUPPORTED_OPS[target]
            inputs, scalar_attrs = _operation_inputs(
                op_name, node.args, node_ids, node.kwargs
            )
            output_id = f"v{len(ops)}"
            attrs = _operation_attrs(op_name, node.kwargs)
            attrs.update(scalar_attrs)
            if op_name in _REDUCTION_OPS and "dims" not in attrs:
                input_type = values[inputs[0]].type
                attrs["dims"] = tuple(range(len(input_type.shape)))
                attrs["keepdim"] = False
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

    needed = set(output_ids)
    live_ops: list[Operation] = []
    for op in reversed(ops):
        if needed.intersection(op.outputs):
            live_ops.append(op)
            needed.update(op.inputs)
    ops = list(reversed(live_ops))

    graph = Graph(tuple(input_ids), values, tuple(ops), output_ids)
    graph.validate()
    return graph


def _reject_unsupported_attention_dropout(graph: object) -> None:
    for node in getattr(graph, "nodes", ()):
        if str(getattr(node, "target", "")) == "aten.bernoulli_.float":
            raise NotImplementedError(
                "scaled_dot_product_attention dropout is not supported"
            )


def _node_type(node: object) -> TensorType:
    meta = getattr(node, "meta", {})
    tensor = meta.get("val")
    if tensor is None:
        raise ValueError(f"missing concrete shape/dtype metadata for FX node: {node}")
    return _tensor_type(tensor, f"FX node {node}")


def _tensor_type(tensor: object, context: str) -> TensorType:
    if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
        raise ValueError(f"missing concrete shape/dtype metadata for {context}")

    dtype = _TORCH_DTYPES.get(str(tensor.dtype))
    if dtype is None:
        raise ValueError(f"unsupported dtype: {tensor.dtype}")
    if any(dim <= 0 for dim in tensor.shape):
        raise ValueError(f"dynamic or empty shapes are not supported yet: {tuple(tensor.shape)}")
    return TensorType(tuple(int(dim) for dim in tensor.shape), dtype)


def _sequence_view_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[Operation, Value], ...]:
    """Expand split/chunk/unbind into single-output slice or squeeze ops."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if kwargs:
        raise ValueError(f"{target} received unsupported keyword arguments: {kwargs!r}")

    if target in {"aten.split.Tensor", "aten.split_with_sizes.default"}:
        if len(args) != 3:
            raise ValueError(f"{target} expects three arguments, got {len(args)}")
        input_node, _, dim = args
    else:
        if len(args) != 2:
            raise ValueError(f"{target} expects two arguments, got {len(args)}")
        input_node, dim = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")

    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")
    axis = dim + rank if dim < 0 else dim
    if axis < 0 or axis >= rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")

    metadata = getattr(node, "meta", {}).get("val")
    if not isinstance(metadata, (tuple, list)) or not metadata:
        raise ValueError(f"missing concrete split output metadata for FX node: {node}")
    output_types = tuple(
        _tensor_type(tensor, f"output {index} of FX node {node}")
        for index, tensor in enumerate(metadata)
    )
    if target == "aten.unbind.int":
        expected_shape = input_type.shape[:axis] + input_type.shape[axis + 1 :]
        if any(output_type.shape != expected_shape for output_type in output_types):
            raise ValueError(
                f"unbind output shapes must all be {expected_shape}, got "
                f"{[output_type.shape for output_type in output_types]}"
            )
    else:
        if sum(output_type.shape[axis] for output_type in output_types) != input_type.shape[axis]:
            raise ValueError(
                f"split output lengths do not preserve input axis {axis} of shape "
                f"{input_type.shape}"
            )
        for output_type in output_types:
            if (
                output_type.shape[:axis] != input_type.shape[:axis]
                or output_type.shape[axis + 1 :] != input_type.shape[axis + 1 :]
            ):
                raise ValueError(
                    f"split output shape {output_type.shape} is incompatible with "
                    f"input shape {input_type.shape} on axis {axis}"
                )

    input_id = _argument_ids(input_node, node_ids)
    operations: list[tuple[Operation, Value]] = []
    output_ids: list[str] = []
    start = 0
    next_op_index = first_op_index
    for output_type in output_types:
        if target == "aten.unbind.int":
            sliced_id = f"v{next_op_index}"
            next_op_index += 1
            sliced_shape = (
                input_type.shape[:axis] + (1,) + input_type.shape[axis + 1 :]
            )
            operations.append(
                (
                    Operation(
                        "slice",
                        (input_id,),
                        (sliced_id,),
                        {
                            "dim": axis,
                            "start": start,
                            "end": start + 1,
                            "step": 1,
                        },
                    ),
                    Value(sliced_id, TensorType(sliced_shape, input_type.dtype)),
                )
            )
            start += 1
            output_id = f"v{next_op_index}"
            next_op_index += 1
            operation = Operation(
                "squeeze", (sliced_id,), (output_id,), {"dim": axis}
            )
        else:
            output_id = f"v{next_op_index}"
            next_op_index += 1
            end = start + output_type.shape[axis]
            operation = Operation(
                "slice",
                (input_id,),
                (output_id,),
                {"dim": axis, "start": start, "end": end, "step": 1},
            )
            start = end
        output_ids.append(output_id)
        operations.append((operation, Value(output_id, output_type)))

    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(f"{target} outputs must be selected with getitem")
        user_args = getattr(user, "args", ())
        if len(user_args) != 2 or not isinstance(user_args[1], int):
            raise ValueError(f"malformed sequence output selection: {user!r}")
        index = user_args[1]
        if index < 0 or index >= len(output_ids):
            raise ValueError(f"{target} output index {index} is out of range")
        node_ids[user] = output_ids[index]

    return tuple(operations)


def _select_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[Operation, Value], ...]:
    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if kwargs:
        raise ValueError(f"{target} received unsupported keyword arguments: {kwargs!r}")
    if len(args) != 3:
        raise ValueError(f"{target} expects three arguments, got {len(args)}")

    input_node, dim, index = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"{target}.index must be an integer, got {index!r}")

    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")
    axis = dim + rank if dim < 0 else dim
    if axis < 0 or axis >= rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")
    resolved_index = index + input_type.shape[axis] if index < 0 else index
    if resolved_index < 0 or resolved_index >= input_type.shape[axis]:
        raise ValueError(
            f"{target}.index {index} is outside axis {axis} with length "
            f"{input_type.shape[axis]}"
        )

    sliced_shape = (
        input_type.shape[:axis] + (1,) + input_type.shape[axis + 1 :]
    )
    sliced_id = f"v{first_op_index}"
    output_id = f"v{first_op_index + 1}"
    sliced_type = TensorType(sliced_shape, input_type.dtype)
    output_type = _node_type(node)
    expected_shape = input_type.shape[:axis] + input_type.shape[axis + 1 :]
    if output_type.shape != expected_shape or output_type.dtype != input_type.dtype:
        raise ValueError(
            f"{target} output metadata {(output_type.shape, output_type.dtype)} does "
            f"not match expected {(expected_shape, input_type.dtype)}"
        )

    return (
        (
            Operation(
                "slice",
                (_argument_ids(input_node, node_ids),),
                (sliced_id,),
                {
                    "dim": axis,
                    "start": resolved_index,
                    "end": resolved_index + 1,
                    "step": 1,
                },
            ),
            Value(sliced_id, sliced_type),
        ),
        (
            Operation(
                "squeeze",
                (sliced_id,),
                (output_id,),
                {"dim": axis},
            ),
            Value(output_id, output_type),
        ),
    )


def _stack_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str, TensorType]:
    """Expand stack into unsqueeze ops followed by one cat op."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if len(args) not in {1, 2}:
        raise ValueError(f"stack expects one or two arguments, got {len(args)}")
    tensors = args[0]
    if not isinstance(tensors, (tuple, list)) or not tensors:
        raise ValueError("stack expects a non-empty tensor sequence")
    if len(args) == 2:
        if "dim" in kwargs:
            raise ValueError("stack received duplicate dim arguments")
        dim = args[1]
    else:
        dim = kwargs.get("dim", 0)
    unknown_kwargs = set(kwargs) - {"dim"}
    if unknown_kwargs:
        raise ValueError(
            f"stack received unsupported keyword arguments: {sorted(unknown_kwargs)}"
        )
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"stack.dim must be an integer, got {dim!r}")

    input_types = tuple(_node_type(tensor) for tensor in tensors)
    first_type = input_types[0]
    if any(input_type != first_type for input_type in input_types[1:]):
        raise ValueError(
            "stack inputs must have identical shapes and dtypes, got "
            f"{[input_type.shape for input_type in input_types]}"
        )

    output_type = _node_type(node)
    output_rank = len(first_type.shape) + 1
    axis = dim + output_rank if dim < 0 else dim
    if axis < 0 or axis >= output_rank:
        raise ValueError(f"stack.dim {dim} exceeds output rank {output_rank}")
    expected_shape = (
        first_type.shape[:axis] + (len(tensors),) + first_type.shape[axis:]
    )
    if output_type.shape != expected_shape or output_type.dtype != first_type.dtype:
        raise ValueError(
            f"stack output metadata {(output_type.shape, output_type.dtype)} does not "
            f"match expected {(expected_shape, first_type.dtype)}"
        )

    operations: list[tuple[Operation, Value]] = []
    expanded_ids: list[str] = []
    expanded_shape = first_type.shape[:axis] + (1,) + first_type.shape[axis:]
    for offset, tensor in enumerate(tensors):
        expanded_id = f"v{first_op_index + offset}"
        expanded_ids.append(expanded_id)
        operations.append(
            (
                Operation(
                    "unsqueeze",
                    (_argument_ids(tensor, node_ids),),
                    (expanded_id,),
                    {"dim": axis},
                ),
                Value(expanded_id, TensorType(expanded_shape, first_type.dtype)),
            )
        )

    output_id = f"v{first_op_index + len(tensors)}"
    operations.append(
        (
            Operation("cat", tuple(expanded_ids), (output_id,), {"dim": axis}),
            Value(output_id, output_type),
        )
    )
    return tuple(operations), output_id, output_type


def _embedding_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, str], dict[str, object], TensorType]:
    """Validate an embedding gather while ignoring inference-only gradient flags."""

    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    parameter_names = (
        "weight",
        "indices",
        "padding_idx",
        "scale_grad_by_freq",
        "sparse",
    )
    if len(args) < 2 or len(args) > len(parameter_names):
        raise ValueError(
            f"embedding expects two to five arguments, got {len(args)}"
        )
    for position, name in enumerate(parameter_names[: len(args)]):
        if name in kwargs:
            raise ValueError(f"embedding received duplicate {name} argument")
        kwargs[name] = args[position]

    unknown = set(kwargs) - set(parameter_names)
    if unknown:
        raise ValueError(
            f"embedding received unsupported keyword arguments: {sorted(unknown)}"
        )
    missing = set(parameter_names[:2]) - set(kwargs)
    if missing:
        raise ValueError(f"embedding is missing arguments: {sorted(missing)}")

    padding_idx = kwargs.get("padding_idx", -1)
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq", False)
    sparse = kwargs.get("sparse", False)
    if padding_idx is not None and (
        not isinstance(padding_idx, int) or isinstance(padding_idx, bool)
    ):
        raise ValueError(f"embedding.padding_idx must be an integer, got {padding_idx!r}")
    if not isinstance(scale_grad_by_freq, bool):
        raise ValueError(
            "embedding.scale_grad_by_freq must be a boolean, "
            f"got {scale_grad_by_freq!r}"
        )
    if not isinstance(sparse, bool):
        raise ValueError(f"embedding.sparse must be a boolean, got {sparse!r}")

    weight_id = _argument_ids(kwargs["weight"], node_ids)
    indices_id = _argument_ids(kwargs["indices"], node_ids)
    weight_type = _node_type(kwargs["weight"])
    indices_type = _node_type(kwargs["indices"])
    if len(weight_type.shape) != 2:
        raise NotImplementedError(
            f"embedding weight must have rank 2, got {weight_type.shape}"
        )
    if weight_type.dtype not in {DType.F16, DType.BF16, DType.F32}:
        raise ValueError(
            f"embedding weight must be floating point, got {weight_type.dtype.value}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "embedding indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    if padding_idx is not None and not (
        -weight_type.shape[0] <= padding_idx < weight_type.shape[0]
    ):
        raise ValueError(
            f"embedding.padding_idx {padding_idx} is outside the weight table"
        )

    output_type = _node_type(node)
    expected_shape = indices_type.shape + (weight_type.shape[1],)
    if output_type.shape != expected_shape or output_type.dtype != weight_type.dtype:
        raise ValueError(
            f"embedding output metadata {(output_type.shape, output_type.dtype)} does "
            f"not match expected {(expected_shape, weight_type.dtype)}"
        )
    attrs = {
        "padding_idx": padding_idx,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
    }
    return (weight_id, indices_id), attrs, output_type


def _index_select_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, str], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if kwargs:
        raise ValueError(
            "index_select received unsupported keyword arguments: "
            f"{sorted(kwargs)}"
        )
    if len(args) != 3:
        raise ValueError(f"index_select expects three arguments, got {len(args)}")

    source, dim, indices = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"index_select.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("index_select requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"index_select.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != 1:
        raise ValueError(
            "index_select indices must have rank 1, "
            f"got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "index_select indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )

    output_type = _node_type(node)
    expected_shape = (
        source_type.shape[:axis]
        + (indices_type.shape[0],)
        + source_type.shape[axis + 1 :]
    )
    if output_type != TensorType(expected_shape, source_type.dtype):
        raise ValueError(
            f"index_select output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(expected_shape, source_type.dtype)}"
        )
    return (
        (_argument_ids(source, node_ids), _argument_ids(indices, node_ids)),
        {"dim": axis},
        output_type,
    )


def _gather_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, str], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    sparse_grad = kwargs.pop("sparse_grad", False)
    if kwargs:
        raise ValueError(
            "gather received unsupported keyword arguments: " f"{sorted(kwargs)}"
        )
    if not isinstance(sparse_grad, bool):
        raise ValueError(f"gather.sparse_grad must be a boolean, got {sparse_grad!r}")
    if len(args) != 3:
        raise ValueError(f"gather expects three arguments, got {len(args)}")

    source, dim, indices = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"gather.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("gather requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"gather.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != rank:
        raise ValueError(
            f"gather indices must have rank {rank}, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "gather indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != axis and index_dim > source_dim:
            raise ValueError(
                f"gather indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-gather axis {index_axis}"
            )

    output_type = _node_type(node)
    if output_type != TensorType(indices_type.shape, source_type.dtype):
        raise ValueError(
            f"gather output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(indices_type.shape, source_type.dtype)}"
        )
    return (
        (_argument_ids(source, node_ids), _argument_ids(indices, node_ids)),
        {"dim": axis},
        output_type,
    )


def _scatter_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if kwargs:
        raise ValueError(
            f"scatter received unsupported keyword arguments: {sorted(kwargs)}"
        )
    if len(args) != 4:
        raise ValueError(f"scatter expects four arguments, got {len(args)}")

    source, dim, indices, update = args
    target = str(getattr(node, "target", ""))
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"scatter.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("scatter requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"scatter.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != rank:
        raise ValueError(
            f"scatter indices must have rank {rank}, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "scatter indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != axis and index_dim > source_dim:
            raise ValueError(
                f"scatter indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-scatter axis {index_axis}"
            )

    attrs: dict[str, object] = {"dim": axis}
    if target == "aten.scatter.value":
        if not isinstance(update, (bool, int, float)):
            raise ValueError(
                f"scatter.value must be a scalar number, got {update!r}"
            )
        attrs["value"] = update
        inputs = (_argument_ids(source, node_ids), _argument_ids(indices, node_ids))
    else:
        update_type = _node_type(update)
        if update_type.dtype != source_type.dtype:
            raise ValueError(
                "scatter source dtype must match the input dtype, "
                f"got {update_type.dtype.value} and {source_type.dtype.value}"
            )
        if len(update_type.shape) != rank:
            raise ValueError(
                f"scatter source must have rank {rank}, got {update_type.shape}"
            )
        for update_axis, (update_dim, index_dim) in enumerate(
            zip(update_type.shape, indices_type.shape, strict=True)
        ):
            if update_dim < index_dim:
                raise ValueError(
                    f"scatter source shape {update_type.shape} is smaller than "
                    f"indices shape {indices_type.shape} on axis {update_axis}"
                )
        inputs = (
            _argument_ids(source, node_ids),
            _argument_ids(indices, node_ids),
            _argument_ids(update, node_ids),
        )

    output_type = _node_type(node)
    if output_type != source_type:
        raise ValueError(
            f"scatter output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )
    return inputs, attrs, output_type


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
    if not (
        len(query_type.shape)
        == len(key_type.shape)
        == len(value_type.shape)
    ):
        raise NotImplementedError(
            "scaled_dot_product_attention requires matching Q/K/V ranks"
        )

    attn_mask, dropout_p, is_causal, enable_gqa, scale = (
        _attention_call_arguments(target, args, kwargs)
    )
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float)):
        raise ValueError(f"dropout_p must be a scalar number, got {dropout_p!r}")
    if float(dropout_p) != 0.0:
        raise NotImplementedError("scaled_dot_product_attention dropout is not supported")
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
    kwargs: dict[str, object] | None = None,
) -> tuple[tuple[str, ...], dict[str, object]]:
    kwargs = {} if kwargs is None else kwargs
    if op == "cat":
        if len(args) not in {1, 2}:
            raise ValueError(f"cat expects one or two arguments, got {len(args)}")
        tensors = args[0]
        if not isinstance(tensors, (tuple, list)) or not tensors:
            raise ValueError("cat expects a non-empty tensor sequence")
        if len(args) == 2:
            if "dim" in kwargs:
                raise ValueError("cat received duplicate dim arguments")
            dim = args[1]
        else:
            dim = kwargs.get("dim", 0)
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"cat.dim must be an integer, got {dim!r}")
        return tuple(_argument_ids(tensor, node_ids) for tensor in tensors), {
            "dim": dim
        }

    if op == "scalar_tensor":
        if len(args) != 1:
            raise ValueError(f"scalar_tensor expects one argument, got {len(args)}")
        value = args[0]
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"scalar_tensor value must be scalar, got {value!r}")
        return (), {"value": value}

    if op in {"zeros", "ones", "full", "new_zeros", "new_ones", "new_full"}:
        has_template = op.startswith("new_")
        base_op = op.removeprefix("new_")
        expected_args = 2 if has_template else 1
        if base_op == "full":
            expected_args += 1
        if len(args) != expected_args:
            raise ValueError(
                f"{op} expects {expected_args} positional arguments, got {len(args)}"
            )

        offset = 0
        if has_template:
            _argument_ids(args[0], node_ids)
            offset = 1
        shape = args[offset]
        if not isinstance(shape, (tuple, list)) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
        ):
            raise ValueError(f"{op}.size must contain integers, got {shape!r}")

        value = (
            args[offset + 1]
            if base_op == "full"
            else (0.0 if base_op == "zeros" else 1.0)
        )
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"{op} fill value must be scalar, got {value!r}")
        return (), {"value": value}

    if op in {"zeros_like", "ones_like", "full_like"}:
        expected_args = 2 if op == "full_like" else 1
        if len(args) != expected_args:
            raise ValueError(
                f"{op} expects {expected_args} positional arguments, got {len(args)}"
            )
        _argument_ids(args[0], node_ids)
        value = (
            args[1]
            if op == "full_like"
            else (0.0 if op == "zeros_like" else 1.0)
        )
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"{op} fill value must be scalar, got {value!r}")
        return (), {"value": value}

    if op == "softplus":
        if len(args) not in {1, 2, 3}:
            raise ValueError(
                f"softplus expects one to three arguments, got {len(args)}"
            )
        if len(args) >= 2 and "beta" in kwargs:
            raise ValueError("softplus received duplicate beta arguments")
        if len(args) >= 3 and "threshold" in kwargs:
            raise ValueError("softplus received duplicate threshold arguments")
        beta = args[1] if len(args) >= 2 else kwargs.get("beta", 1.0)
        threshold = args[2] if len(args) >= 3 else kwargs.get("threshold", 20.0)
        if (
            not isinstance(beta, (int, float))
            or isinstance(beta, bool)
            or beta <= 0
        ):
            raise ValueError(f"softplus.beta must be positive, got {beta!r}")
        if not isinstance(threshold, (int, float)) or isinstance(
            threshold, bool
        ):
            raise ValueError(
                f"softplus.threshold must be a scalar number, got {threshold!r}"
            )
        return (
            (_argument_ids(args[0], node_ids),),
            {"beta": float(beta), "threshold": float(threshold)},
        )

    if op == "threshold":
        if len(args) not in {1, 2, 3}:
            raise ValueError(
                f"threshold expects one to three arguments, got {len(args)}"
            )
        if len(args) >= 2 and "threshold" in kwargs:
            raise ValueError("threshold received duplicate threshold arguments")
        if len(args) >= 3 and "value" in kwargs:
            raise ValueError("threshold received duplicate value arguments")
        threshold = args[1] if len(args) >= 2 else kwargs.get("threshold", 0.0)
        fill_value = args[2] if len(args) >= 3 else kwargs.get("value", 0.0)
        for name, value in (("threshold", threshold), ("value", fill_value)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"threshold.{name} must be a scalar number, got {value!r}"
                )
        return (
            (_argument_ids(args[0], node_ids),),
            {"threshold": float(threshold), "value": float(fill_value)},
        )

    if op == "cast":
        if len(args) != 1:
            raise ValueError(f"cast expects one argument, got {len(args)}")
        return ((_argument_ids(args[0], node_ids),), {})

    if op in {"softmax", "safe_softmax"}:
        if len(args) not in ({2, 3} if op == "safe_softmax" else {3}):
            raise ValueError(f"{op} received unexpected arguments: {len(args)}")
        input_id, dim = args[:2]
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"{op}.dim must be an integer, got {dim!r}")
        if len(args) == 3 and not isinstance(args[2], bool):
            raise ValueError(
                f"{op}.half_to_float must be a boolean, got {args[2]!r}"
            )
        return ((_argument_ids(input_id, node_ids),), {"dim": dim})

    if op == "reshape":
        if len(args) != 2:
            raise ValueError(f"reshape expects two arguments, got {len(args)}")
        input_id, shape = args
        if not isinstance(shape, (tuple, list)) or not shape or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
        ):
            raise ValueError(f"reshape.shape must contain integers, got {shape!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"shape": tuple(shape)},
        )

    if op == "transpose":
        if len(args) != 3:
            raise ValueError(f"transpose expects three arguments, got {len(args)}")
        input_id, dim0, dim1 = args
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool)
            for dim in (dim0, dim1)
        ):
            raise ValueError(
                f"transpose dimensions must be integers, got {(dim0, dim1)!r}"
            )
        return (
            (_argument_ids(input_id, node_ids),),
            {"dim0": dim0, "dim1": dim1},
        )

    if op == "permute":
        if len(args) != 2:
            raise ValueError(f"permute expects two arguments, got {len(args)}")
        input_id, dims = args
        if not isinstance(dims, (tuple, list)) or not dims or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(f"permute.dims must contain integers, got {dims!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"dims": tuple(dims)},
        )

    if op == "expand":
        if len(args) != 2:
            raise ValueError(f"expand expects two arguments, got {len(args)}")
        input_id, shape = args
        if not isinstance(shape, (tuple, list)) or not shape or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
        ):
            raise ValueError(f"expand.shape must contain integers, got {shape!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"shape": tuple(shape)},
        )

    if op == "slice":
        if len(args) not in {4, 5}:
            raise ValueError(f"slice expects four or five arguments, got {len(args)}")
        input_id, dim, start, end = args[:4]
        step = args[4] if len(args) == 5 else 1
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"slice.dim must be an integer, got {dim!r}")
        if start is not None and (
            not isinstance(start, int) or isinstance(start, bool)
        ):
            raise ValueError(f"slice.start must be an integer or None, got {start!r}")
        if end is not None and (
            not isinstance(end, int) or isinstance(end, bool)
        ):
            raise ValueError(f"slice.end must be an integer or None, got {end!r}")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(f"slice.step must be a positive integer, got {step!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"dim": dim, "start": start, "end": end, "step": step},
        )

    if op == "flip":
        if len(args) != 2:
            raise ValueError(f"flip expects two arguments, got {len(args)}")
        input_id, dims = args
        if not isinstance(dims, (tuple, list)) or not dims or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(f"flip.dims must contain integers, got {dims!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"dims": tuple(dims)},
        )

    if op == "roll":
        if len(args) not in {2, 3}:
            raise ValueError(f"roll expects two or three arguments, got {len(args)}")
        input_id, shifts = args[:2]
        dims = args[2] if len(args) == 3 else kwargs.get("dims")
        if isinstance(shifts, int) and not isinstance(shifts, bool):
            normalized_shifts = (shifts,)
        elif isinstance(shifts, (tuple, list)) and shifts and all(
            isinstance(shift, int) and not isinstance(shift, bool)
            for shift in shifts
        ):
            normalized_shifts = tuple(shifts)
        else:
            raise ValueError(f"roll.shifts must contain integers, got {shifts!r}")

        if dims is None:
            if len(normalized_shifts) != 1:
                raise ValueError(
                    "roll without dims requires exactly one shift, got "
                    f"{len(normalized_shifts)}"
                )
            normalized_dims = None
        elif isinstance(dims, (tuple, list)) and dims and all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            normalized_dims = tuple(dims)
            if len(normalized_dims) != len(normalized_shifts):
                raise ValueError(
                    "roll shifts and dims must have the same length, got "
                    f"{len(normalized_shifts)} and {len(normalized_dims)}"
                )
        else:
            raise ValueError(f"roll.dims must contain integers, got {dims!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"shifts": normalized_shifts, "dims": normalized_dims},
        )

    if op in {"triu", "tril"}:
        if len(args) not in {1, 2}:
            raise ValueError(f"{op} expects one or two arguments, got {len(args)}")
        diagonal = args[1] if len(args) == 2 else kwargs.get("diagonal", 0)
        if len(args) == 2 and "diagonal" in kwargs:
            raise ValueError(f"{op} received duplicate diagonal arguments")
        if not isinstance(diagonal, int) or isinstance(diagonal, bool):
            raise ValueError(f"{op}.diagonal must be an integer, got {diagonal!r}")
        return ((_argument_ids(args[0], node_ids),), {"diagonal": diagonal})

    if op == "unsqueeze":
        if len(args) != 2:
            raise ValueError(f"unsqueeze expects two arguments, got {len(args)}")
        input_id, dim = args
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"unsqueeze.dim must be an integer, got {dim!r}")
        return ((_argument_ids(input_id, node_ids),), {"dim": dim})

    if op == "squeeze":
        if len(args) != 2:
            raise ValueError(f"squeeze expects two arguments, got {len(args)}")
        input_id, dim = args
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"squeeze.dim must be an integer, got {dim!r}")
        return ((_argument_ids(input_id, node_ids),), {"dim": dim})

    if op in {"contiguous", "alias"}:
        if len(args) != 1:
            raise ValueError(f"{op} expects one argument, got {len(args)}")
        return ((_argument_ids(args[0], node_ids),), {})

    if op == "where":
        if len(args) != 3:
            raise ValueError(f"where expects three arguments, got {len(args)}")
        return tuple(_argument_ids(argument, node_ids) for argument in args), {}

    if op == "masked_fill":
        if len(args) != 3:
            raise ValueError(f"masked_fill expects three arguments, got {len(args)}")
        input_id, mask_id, value = args
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"masked_fill.value must be scalar, got {value!r}")
        return (
            (_argument_ids(input_id, node_ids), _argument_ids(mask_id, node_ids)),
            {"value": value},
        )

    if op == "clamp":
        if len(args) not in {1, 2, 3}:
            raise ValueError(
                f"clamp expects one to three positional arguments, got {len(args)}"
            )
        inputs = [_argument_ids(args[0], node_ids)]
        attrs: dict[str, object] = {}
        for position, name in ((1, "min"), (2, "max")):
            if position >= len(args) or args[position] is None:
                continue
            bound = args[position]
            try:
                bound_id = _argument_ids(bound, node_ids)
            except ValueError:
                if not isinstance(bound, (int, float)) or isinstance(bound, bool):
                    raise ValueError(
                        f"unsupported clamp {name} at position {position}: {bound!r}"
                    ) from None
                attrs[name] = float(bound)
            else:
                attrs[f"{name}_input"] = len(inputs)
                inputs.append(bound_id)
        return tuple(inputs), attrs

    if op in {"clamp_min", "clamp_max"}:
        if len(args) != 2:
            raise ValueError(f"{op} expects two arguments, got {len(args)}")
        inputs = [_argument_ids(args[0], node_ids)]
        attrs: dict[str, object] = {}
        bound = args[1]
        name = "min" if op == "clamp_min" else "max"
        try:
            bound_id = _argument_ids(bound, node_ids)
        except ValueError:
            if not isinstance(bound, (int, float)) or isinstance(bound, bool):
                raise ValueError(
                    f"unsupported {op} {name}: {bound!r}"
                ) from None
            attrs[name] = float(bound)
        else:
            attrs[f"{name}_input"] = len(inputs)
            inputs.append(bound_id)
        return tuple(inputs), attrs

    if op in {"leaky_relu", "elu"}:
        if len(args) != 2:
            raise ValueError(f"{op} expects two arguments, got {len(args)}")
        argument = args[1]
        if not isinstance(argument, (int, float)) or isinstance(argument, bool):
            raise ValueError(f"{op} scalar must be a number, got {argument!r}")
        name = "negative_slope" if op == "leaky_relu" else "alpha"
        return ((_argument_ids(args[0], node_ids),), {name: float(argument)})

    if op in _REDUCTION_OPS:
        if len(args) == 1:
            return ((_argument_ids(args[0], node_ids),), {})
        if len(args) not in {2, 3}:
            raise ValueError(
                f"{op} expects one to three positional arguments, got {len(args)}"
            )
        input_id = args[0]
        dims = args[1]
        keepdim = args[2] if len(args) == 3 else False
        if isinstance(dims, int) and not isinstance(dims, bool):
            normalized_dims = (dims,)
        elif isinstance(dims, (tuple, list)) and all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            normalized_dims = tuple(dims)
        else:
            raise ValueError(f"{op}.dim must contain integers, got {dims!r}")
        if not normalized_dims:
            raise ValueError(f"{op}.dim must not be empty")
        if not isinstance(keepdim, bool):
            raise ValueError(f"{op}.keepdim must be a boolean, got {keepdim!r}")
        return (
            (_argument_ids(input_id, node_ids),),
            {"dims": normalized_dims, "keepdim": keepdim},
        )

    if op not in _BINARY_OPS:
        return tuple(_argument_ids(arg, node_ids) for arg in args), {}

    if len(args) != 2:
        raise ValueError(f"{op} expects two arguments, got {len(args)}")

    tensor_inputs: list[str] = []
    scalars: list[tuple[int, bool | float]] = []
    for position, argument in enumerate(args):
        try:
            tensor_inputs.append(_argument_ids(argument, node_ids))
        except ValueError:
            if isinstance(argument, bool) and op in (
                _COMPARISON_OPS | _LOGICAL_BINARY_OPS
            ):
                scalars.append((position, argument))
                continue
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
    if op in _REDUCTION_OPS:
        allowed.add("dtype")
    if op == "gelu":
        allowed.add("approximate")
    if op == "contiguous":
        allowed.add("memory_format")
    if op == "cast":
        allowed.update(
            {"dtype", "layout", "device", "pin_memory", "non_blocking", "memory_format"}
        )
    if op in {
        "scalar_tensor",
        "zeros",
        "ones",
        "full",
        "new_zeros",
        "new_ones",
        "new_full",
        "zeros_like",
        "ones_like",
        "full_like",
    }:
        allowed.update(
            {
                "dtype",
                "layout",
                "device",
                "pin_memory",
                "requires_grad",
                "memory_format",
            }
        )
    if op == "cat":
        allowed.add("dim")
    if op == "flip":
        allowed.add("dims")
    if op == "roll":
        allowed.update({"shifts", "dims"})
    if op in {"triu", "tril"}:
        allowed.add("diagonal")
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"unsupported keyword arguments for {op}: {sorted(unknown)}")
    if op in {
        "cast",
        "scalar_tensor",
        "zeros",
        "ones",
        "full",
        "new_zeros",
        "new_ones",
        "new_full",
        "zeros_like",
        "ones_like",
        "full_like",
        "cat",
        "flip",
        "roll",
        "triu",
        "tril",
    }:
        return {}

    attrs: dict[str, object] = {}
    for name, default in numeric_attrs.items():
        value = kwargs.get(name, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{op}.{name} must be a scalar number, got {value!r}")
        attrs[name] = float(value)

    if op in _REDUCTION_OPS and kwargs.get("dtype") is not None:
        dtype_name = str(kwargs["dtype"])
        reduction_dtype = _TORCH_DTYPES.get(dtype_name)
        if reduction_dtype is None:
            raise ValueError(f"unsupported reduction dtype: {dtype_name}")
        attrs["dtype"] = reduction_dtype

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
