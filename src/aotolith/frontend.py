from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from math import isfinite
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
    "aten.constant_pad_nd.default": "pad",
    "aten.triu.default": "triu",
    "aten.tril.default": "tril",
    "aten.unsqueeze.default": "unsqueeze",
    "aten.squeeze.dim": "squeeze",
    "aten.clone.default": "contiguous",
    "aten.alias.default": "alias",
    "aten._to_copy.default": "cast",
    "aten._softmax.default": "softmax",
    "aten._safe_softmax.default": "safe_softmax",
    "aten._log_softmax.default": "log_softmax",
    "aten.sum.default": "sum",
    "aten.sum.dim_IntList": "sum",
    "aten.prod.dim_int": "prod",
    "aten.prod.default": "prod",
    "aten.amax.default": "amax",
    "aten.amin.default": "amin",
    "aten.cumsum.default": "cumsum",
    "aten.cumprod.default": "cumprod",
    "aten.argmax.default": "argmax",
    "aten.argmin.default": "argmin",
    "aten.mean.default": "mean",
    "aten.mean.dim": "mean",
    "aten.silu.default": "silu",
    "aten.gelu.default": "gelu",
    "aten.softplus.default": "softplus",
    "aten.mish.default": "mish",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.add.Tensor": "add",
    "aten.add_.Tensor": "add",
    "aten.sub.Tensor": "sub",
    "aten.rsub.Scalar": "rsub",
    "aten.div.Tensor": "div",
    "aten.floor_divide.default": "floor_divide",
    "aten.floor_divide.Tensor": "floor_divide",
    "aten.floor_divide.Scalar": "floor_divide",
    "aten.remainder.Tensor": "remainder",
    "aten.remainder.Scalar": "remainder",
    "aten.fmod.Tensor": "fmod",
    "aten.fmod.Scalar": "fmod",
    "aten.lerp.Tensor": "lerp",
    "aten.lerp.Scalar": "lerp",
    "aten.addcmul.default": "addcmul",
    "aten.addcdiv.default": "addcdiv",
    "aten.pow.Tensor_Tensor": "pow",
    "aten.pow.Tensor_Scalar": "pow",
    "aten.logit.default": "logit",
    "aten.erfc.default": "erfc",
    "aten.frac.default": "frac",
    "aten.sinc.default": "sinc",
    "aten.hypot.default": "hypot",
    "aten.copysign.Tensor": "copysign",
    "aten.copysign.Scalar": "copysign",
    "aten.copysign.Scalar_Scalar": "copysign",
    "aten.xlogy.Tensor": "xlogy",
    "aten.xlogy.Scalar_Self": "xlogy",
    "aten.xlogy.Scalar_Other": "xlogy",
    "aten.nan_to_num.default": "nan_to_num",
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
    "aten.bitwise_and.Tensor": "bitwise_and",
    "aten.bitwise_and.Scalar": "bitwise_and",
    "aten.bitwise_or.Tensor": "bitwise_or",
    "aten.bitwise_or.Scalar": "bitwise_or",
    "aten.bitwise_xor.Tensor": "bitwise_xor",
    "aten.bitwise_xor.Scalar": "bitwise_xor",
    "aten.bitwise_not.default": "bitwise_not",
    "aten.isnan.default": "isnan",
    "aten.isinf.default": "isinf",
    "aten.masked_fill.Scalar": "masked_fill",
    "aten.masked_fill.Tensor": "where",
    "aten.maximum.default": "maximum",
    "aten.minimum.default": "minimum",
    "aten.logaddexp.default": "logaddexp",
    "aten.atan2.default": "atan2",
    "aten.scalar_tensor.default": "scalar_tensor",
    "aten.arange.default": "arange",
    "aten.arange.start": "arange",
    "aten.arange.start_step": "arange",
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
    "aten.scatter_add.default": "scatter",
    "aten.index_add.default": "index_add",
}

# Copies that carry no arithmetic. When one of these produces an empty tensor it
# is only ever a KV cache placeholder waiting to be concatenated away, so the
# frontend skips it rather than requiring an op that cannot be typed.
_EMPTY_COPY_TARGETS = frozenset(
    {
        "aten.lift_fresh_copy.default",
        "aten.clone.default",
        "aten.detach.default",
        "aten.alias.default",
        "aten._to_copy.default",
        "aten.contiguous.default",
        "aten.lift_fresh.default",
    }
)

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
    "elu": {"alpha": 1.0, "scale": 1.0, "input_scale": 1.0},
    "softplus": {"beta": 1.0, "threshold": 20.0},
    "threshold": {"threshold": 0.0, "value": 0.0},
    "addcmul": {"value": 1.0},
    "addcdiv": {"value": 1.0},
}

_COMPARISON_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
# Pointwise ops whose kernels only accept floating-point operands. PyTorch
# promotes integer inputs to the operator's floating-point opmath dtype, so the
# frontend inserts the same cast instead of rejecting the graph.
_FLOAT_ONLY_POINTWISE_OPS = {
    "logit",
    "erfc",
    "frac",
    "sinc",
    "hypot",
    "copysign",
    "xlogy",
    "lerp",
    "addcmul",
    "addcdiv",
    "nan_to_num",
}
_LOGICAL_BINARY_OPS = {"logical_and", "logical_or", "logical_xor"}
_BITWISE_BINARY_OPS = {"bitwise_and", "bitwise_or", "bitwise_xor"}
_BINARY_OPS = _COMPARISON_OPS | {
    "mul",
    "add",
    "sub",
    "rsub",
    "div",
    "floor_divide",
    "remainder",
    "fmod",
    "hypot",
    "copysign",
    "xlogy",
    "pow",
    "maximum",
    "minimum",
    "logaddexp",
    "atan2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
}
_REDUCTION_OPS = {"sum", "mean", "prod", "amax", "amin"}
_INTEGRAL_DTYPES = {DType.I32, DType.I64}
_FLOATING_DTYPES = {DType.F16, DType.BF16, DType.F32}
# Binary pointwise ops whose operands PyTorch would promote before applying
# the operator. `pow` stays out because integer exponentiation has no exact
# float equivalent.
_PROMOTABLE_POINTWISE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "floor_divide",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
    "logaddexp",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
} | _COMPARISON_OPS
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
_INDEX_TARGET = "aten.index.Tensor"
_MATRIX_TRANSPOSE_TARGET = "aten.t.default"
_STACK_TARGET = "aten.stack.default"
_SCATTER_TARGETS = {"aten.scatter.src", "aten.scatter.value"}
_SCATTER_ADD_TARGET = "aten.scatter_add.default"
_SCATTER_REDUCE_TARGET = "aten.scatter_reduce.two"
_INDEX_ADD_TARGET = "aten.index_add.default"
# `scatter_reduce` renames the two reductions whose accumulation is not an
# addition; `mean` accumulates counts alongside the values.
_SCATTER_REDUCE_NAMES = {
    "sum": "add",
    "prod": "prod",
    "amax": "amax",
    "amin": "amin",
    "mean": "mean",
}
_LOGSIGMOID_TARGET = "aten.log_sigmoid_forward.default"
# `sort` and `topk` return (values, indices); the IR has single-output
# operations, so each selection becomes its own operation and only the ones the
# graph actually reads are emitted.
_SORT_TARGETS = {"aten.sort.default", "aten.topk.default"}
# PyTorch spells variance two ways depending on version and call style; both
# normalize to one IR shape (dims, keepdim, correction).
_VARIANCE_TARGETS = {
    "aten.var.correction": "var",
    "aten.var.dim": "var",
    "aten.var.default": "var",
    "aten.std.correction": "std",
    "aten.std.dim": "std",
    "aten.std.default": "std",
}
# ``repeat`` and ``tile`` share this ATen target; it lowers to layout ops.
_REPEAT_TARGET = "aten.repeat.default"
# ``diagonal`` is a strided view: once the two axes are trailing and flat, the
# diagonal is every ``m + 1``-th element.
_DIAGONAL_TARGET = "aten.diagonal.default"
_DIAGONAL_COPY_TARGET = "aten.diagonal_copy.default"
_VECTOR_NORM_TARGET = "aten.linalg_vector_norm.default"
# Reflect and replicate padding have no IR op: they are separable, so each axis
# becomes a slice/flip (reflect) or slice/expand (replicate) plus a concat.
_PAD_MODE_TARGETS = {
    "aten.reflection_pad1d.default": ("reflect", 1),
    "aten.reflection_pad2d.default": ("reflect", 2),
    "aten.replication_pad1d.default": ("replicate", 1),
    "aten.replication_pad2d.default": ("replicate", 2),
}
_INPLACE_TARGETS = {"aten.add_.Tensor"}
# Constant padding accepts negative entries the same way the mirror modes do:
# they crop the padded extent, so the frontend slices before it pads.
_CONSTANT_PAD_TARGET = "aten.constant_pad_nd.default"
# `logsumexp` is a composite in ATen: a stable max shift plus a log-sum, with
# infinite maxima replaced by zero so a saturated row keeps its limit.
_LOGSUMEXP_TARGET = "aten.logsumexp.default"
# `make_fx` keeps the in-place spelling of every scatter family operator, so
# each one is legalized into the functional op with the same lowering.
_INPLACE_SCATTER_TARGETS = {
    "aten.scatter_.src": "aten.scatter.src",
    "aten.scatter_.value": "aten.scatter.value",
    "aten.scatter_add_.default": "aten.scatter_add.default",
    "aten.scatter_reduce_.two": "aten.scatter_reduce.two",
    "aten.index_add_.default": "aten.index_add.default",
}


def capture_fx(fn: Callable[..., Any], example_inputs: Sequence[Any]) -> Graph:
    """Capture a supported callable with make_fx and build the compiler IR."""

    try:
        from torch.fx.experimental.proxy_tensor import make_fx
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the FX frontend") from exc

    # Fake tracing keeps capture shape-only: the default "real" mode executes
    # every operator on the example inputs, so tracing a large matmul costs a
    # full CPU matmul and compile time scales with the problem size.
    traced = make_fx(fn, tracing_mode="fake")(*example_inputs)
    _reject_unsupported_attention_dropout(traced.graph)
    # ``use_cache=True`` models start their KV cache from zero-length tensors,
    # which make_fx bakes into the graph as constants plus identity copies of
    # them. They are legal only as concatenation operands, where a zero-length
    # piece is a no-op, so they are recorded here and folded away below.
    empty_nodes = _empty_value_nodes(traced)
    values: dict[str, Value] = {}
    input_ids: list[str] = []
    node_ids: dict[object, str] = {}
    ops: list[Operation] = []
    output_ids: tuple[str, ...] = ()
    # Multi-output nodes whose unused selections never need a value.
    dead_nodes: set[object] = set()

    for node in traced.graph.nodes:
        if node in dead_nodes:
            continue
        if node.op == "placeholder":
            value_id = f"arg{len(input_ids)}"
            node_ids[node] = value_id
            input_ids.append(value_id)
            values[value_id] = Value(value_id, _node_type(node))
        elif node.op == "get_attr":
            if node in empty_nodes:
                # Folding happens at the concatenation that consumes it; any
                # other use reaches ``_argument_ids`` and fails by name.
                continue
            raise NotImplementedError(
                _constant_tensor_message(traced, node)
            )
        elif node.op == "call_function":
            if node in node_ids:
                continue
            target = str(node.target)
            if node in empty_nodes and target in _EMPTY_COPY_TARGETS:
                # An identity copy of an empty constant carries no arithmetic;
                # it exists only to be concatenated away.
                continue
            if target == _MATRIX_TRANSPOSE_TARGET:
                input_node = node.args[0]
                input_type = _node_type(input_node)
                if len(input_type.shape) != 2:
                    raise ValueError(
                        "aten.t is only supported for rank-2 tensors, got "
                        f"shape {input_type.shape}"
                    )
                output_id = f"v{len(ops)}"
                ops.append(
                    Operation(
                        "transpose",
                        (_argument_ids(input_node, node_ids),),
                        (output_id,),
                        {"dim0": 1, "dim1": 0},
                    )
                )
                node_ids[node] = output_id
                values[output_id] = Value(output_id, _node_type(node))
                continue
            if target == _SELECT_TARGET:
                select_ops = _select_operations(node, node_ids, len(ops))
                for operation, value in select_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = select_ops[-1][0].outputs[0]
                continue
            if target == _INDEX_TARGET:
                inputs, attrs, output_type, permute_dims = (
                    _advanced_index_operation(node, node_ids)
                )
                if permute_dims is not None:
                    source_type = values[inputs[0]].type
                    permuted_shape = tuple(
                        source_type.shape[axis] for axis in permute_dims
                    )
                    permuted_id = f"v{len(ops)}"
                    ops.append(
                        Operation(
                            "permute",
                            (inputs[0],),
                            (permuted_id,),
                            {"dims": permute_dims},
                        )
                    )
                    values[permuted_id] = Value(
                        permuted_id,
                        TensorType(permuted_shape, source_type.dtype),
                    )
                    inputs = (permuted_id,) + inputs[1:]
                output_id = f"v{len(ops)}"
                ops.append(Operation("index", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
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
            canonical_target = _INPLACE_SCATTER_TARGETS.get(target, target)
            if canonical_target != target:
                _validate_inplace_legalization(node)
                target = canonical_target
            if target in _SCATTER_TARGETS or target in {
                _SCATTER_ADD_TARGET,
                _SCATTER_REDUCE_TARGET,
            }:
                inputs, attrs, output_type = _scatter_operation(node, node_ids)
                output_id = f"v{len(ops)}"
                ops.append(Operation("scatter", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target == _INDEX_ADD_TARGET:
                inputs, attrs, output_type = _index_add_operation(
                    node, node_ids
                )
                output_id = f"v{len(ops)}"
                ops.append(Operation("index_add", inputs, (output_id,), attrs))
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
                inputs, attrs, output_type, wanted = _layer_norm_operation(
                    node, node_ids
                )
                produced: dict[int, str] = {}
                if 0 in wanted:
                    output_id = f"v{len(ops)}"
                    ops.append(
                        Operation("layer_norm", inputs, (output_id,), attrs)
                    )
                    values[output_id] = Value(output_id, output_type)
                    produced[0] = output_id
                if 1 in wanted or 2 in wanted:
                    produced.update(
                        _layer_norm_statistics(
                            ops, values, inputs[0], attrs, wanted
                        )
                    )
                for user in node.users:
                    index = user.args[1]
                    if index in produced:
                        node_ids[user] = produced[index]
                    else:
                        dead_nodes.add(user)
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
            if target in _SORT_TARGETS:
                sort_ops, produced = _sort_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in sort_ops:
                    ops.append(operation)
                    values[value.id] = value
                for user in node.users:
                    index = user.args[1]
                    if index in produced:
                        node_ids[user] = produced[index]
                    else:
                        dead_nodes.add(user)
                continue
            if target in {_DIAGONAL_TARGET, _DIAGONAL_COPY_TARGET}:
                diagonal_ops, output_id = _diagonal_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in diagonal_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = output_id
                continue
            if target == _LOGSUMEXP_TARGET:
                logsumexp_ops, output_id = _logsumexp_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in logsumexp_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = output_id
                continue
            if target == _CONSTANT_PAD_TARGET and _constant_pad_crops(node):
                crop_ops, output_id = _constant_pad_crop_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in crop_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = output_id
                continue
            if target in _PAD_MODE_TARGETS:
                padding_ops, output_id = _padding_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in padding_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = output_id
                continue
            if target == _REPEAT_TARGET:
                repeat_ops, output_id = _repeat_operations(
                    node, node_ids, len(ops)
                )
                for operation, value in repeat_ops:
                    ops.append(operation)
                    values[value.id] = value
                node_ids[node] = output_id
                continue
            if target == _VECTOR_NORM_TARGET:
                inputs, attrs, output_type = _vector_norm_operation(
                    node, node_ids
                )
                output_id = f"v{len(ops)}"
                ops.append(Operation("vector_norm", inputs, (output_id,), attrs))
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target in _VARIANCE_TARGETS:
                inputs, attrs, output_type = _variance_operation(node, node_ids)
                output_id = f"v{len(ops)}"
                ops.append(
                    Operation(
                        _VARIANCE_TARGETS[target], inputs, (output_id,), attrs
                    )
                )
                node_ids[node] = output_id
                values[output_id] = Value(output_id, output_type)
                continue
            if target not in _SUPPORTED_OPS:
                raise ValueError(f"unsupported FX target: {target}")
            if target in _INPLACE_TARGETS:
                _validate_inplace_legalization(node)
            op_name = _SUPPORTED_OPS[target]
            call_args = node.args
            if target == "aten.masked_fill.Tensor":
                # ATen masks the first argument in place, which is exactly
                # where(mask, value, input) for a tensor fill value.
                masked, mask, fill = call_args
                op_name = "where"
                call_args = (mask, fill, masked)
            if target == "aten.cat.default" and empty_nodes:
                call_args = _fold_empty_cat_operands(
                    node, empty_nodes, node_ids
                )
                if call_args is None:
                    # A concatenation with one surviving operand is that
                    # operand, so the operation disappears entirely.
                    continue
            if (
                target == "aten.pow.Tensor_Scalar"
                and len(call_args) == 2
                and isinstance(call_args[1], int)
                and not isinstance(call_args[1], bool)
                and call_args[1] == 2
                and _node_type(node).dtype in {DType.I32, DType.I64}
            ):
                # make_fx lowers torch.square on integer tensors to pow(x, 2).
                op_name = "square"
                inputs = (_argument_ids(call_args[0], node_ids),)
                scalar_attrs = {}
            else:
                inputs, scalar_attrs = _operation_inputs(
                    op_name, call_args, node_ids, node.kwargs
                )
            attrs = _operation_attrs(op_name, node.kwargs)
            attrs.update(scalar_attrs)
            if op_name == "where" and len(inputs) == 3:
                inputs = (inputs[0],) + _promote_pointwise_operands(
                    "where", inputs[1:], values, ops, _node_type(node).dtype
                )
            elif op_name in _PROMOTABLE_POINTWISE_OPS or (
                op_name in _FLOAT_ONLY_POINTWISE_OPS
            ):
                inputs = _promote_pointwise_operands(
                    op_name, inputs, values, ops, _node_type(node).dtype
                )
            output_id = f"v{len(ops)}"
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


def _constant_value(traced: object, node: object) -> object:
    return getattr(traced, getattr(node, "target", ""), None)


def _empty_value_nodes(traced: object) -> set[object]:
    """Collect FX nodes whose value is an empty tensor.

    A model with ``use_cache=True`` starts its KV cache as zero-length tensors.
    ``make_fx`` lifts those into graph constants, and an empty tensor cannot be a
    plan input because the IR requires positive static dimensions. They are only
    ever concatenated, where a zero-length operand contributes nothing, so they
    are dropped at the concatenation instead of being given a type.
    """

    found: set[object] = set()
    for node in getattr(getattr(traced, "graph", None), "nodes", ()):
        tensor = getattr(node, "meta", {}).get("val")
        shape = getattr(tensor, "shape", None)
        if shape is not None and 0 in tuple(shape):
            found.add(node)
    return found


def _fold_empty_cat_operands(
    node: object,
    empty_nodes: set[object],
    node_ids: dict[object, str],
) -> tuple[object, ...] | None:
    """Drop zero-length operands from a ``cat``.

    Returns the arguments to lower, or ``None`` when a single operand survives
    and the concatenation is therefore the identity.
    """

    args = getattr(node, "args", ())
    if not args:
        return args
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        return args
    kept = tuple(operand for operand in operands if operand not in empty_nodes)
    if len(kept) == len(operands):
        return args
    if not kept:
        raise ValueError(
            "cat only concatenates zero-length constants and has no result"
        )
    if len(kept) == 1:
        node_ids[node] = _argument_ids(kept[0], node_ids)
        return None
    return (kept, *args[1:])


def _constant_tensor_message(traced: object, node: object) -> str:
    target = getattr(node, "target", "?")
    value = _constant_value(traced, node)
    shape = getattr(value, "shape", None)
    rendered = "?" if shape is None else tuple(shape)
    return (
        f"captured graph embeds constant tensor {target!r} of shape {rendered}; "
        "constant tensors are not plan inputs. A HuggingFace model traced with "
        "use_cache=True is the usual source: pass "
        "example_kwargs={'use_cache': False} to keep the cache out of the graph"
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


def _advanced_index_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType, tuple[int, ...] | None]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if kwargs:
        raise ValueError(
            "index received unsupported keyword arguments: "
            f"{sorted(kwargs)}"
        )
    if len(args) != 2:
        raise ValueError(f"index expects two arguments, got {len(args)}")

    source, indices = args
    if not isinstance(indices, (tuple, list)) or not indices:
        raise ValueError("index expects a non-empty sequence of indices")
    source_type = _node_type(source)
    if len(indices) > len(source_type.shape):
        raise ValueError(
            f"index has {len(indices)} indices for rank-{len(source_type.shape)} "
            "input"
        )

    index_nodes = [index for index in indices if index is not None]
    if not index_nodes:
        raise ValueError("index requires at least one tensor index")
    index_types = tuple(
        _node_type(index) for index in index_nodes
    )
    for index_type in index_types:
        if index_type.dtype not in {DType.I32, DType.I64}:
            raise ValueError(
                "index indices must be int32 or int64, "
                f"got {index_type.dtype.value}"
            )

    indexed_axes = tuple(
        axis for axis, index in enumerate(indices) if index is not None
    )
    first_axis = indexed_axes[0]
    last_axis = indexed_axes[-1]
    # PyTorch inserts the broadcast index dimensions where the indexed axes
    # were when they are adjacent, and at the front when they are separated by
    # a fixed axis. Moving the separated axes to the front turns the second
    # rule into the first, so the gather itself stays the same op.
    permute_dims: tuple[int, ...] | None = None
    if indexed_axes != tuple(range(first_axis, last_axis + 1)):
        permute_dims = indexed_axes + tuple(
            axis for axis in range(len(source_type.shape)) if axis not in indexed_axes
        )

    broadcast_shape = _broadcast_shapes(
        tuple(index_type.shape for index_type in index_types),
        "advanced index",
    )
    if permute_dims is None:
        expected_shape = (
            source_type.shape[:first_axis]
            + broadcast_shape
            + source_type.shape[last_axis + 1 :]
        )
        op_axes = indexed_axes
    else:
        remaining = tuple(
            source_type.shape[axis]
            for axis in permute_dims[len(indexed_axes) :]
        )
        expected_shape = broadcast_shape + remaining
        op_axes = tuple(range(len(indexed_axes)))
    expected_type = TensorType(expected_shape, source_type.dtype)
    output_type = _node_type(node)
    if output_type != expected_type:
        raise ValueError(
            f"index output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(expected_type.shape, expected_type.dtype)}"
        )

    inputs = (_argument_ids(source, node_ids),) + tuple(
        _argument_ids(index, node_ids) for index in index_nodes
    )
    return inputs, {"axes": op_axes}, output_type, permute_dims


def _broadcast_shapes(
    shapes: tuple[tuple[int, ...], ...], context: str
) -> tuple[int, ...]:
    rank = max(len(shape) for shape in shapes)
    result = [1] * rank
    for shape in shapes:
        offset = rank - len(shape)
        for axis, dim in enumerate(shape):
            current = result[offset + axis]
            if current != dim and current != 1 and dim != 1:
                raise ValueError(
                    f"{context} shapes cannot broadcast: {shapes}"
                )
            result[offset + axis] = max(current, dim)
    return tuple(result)


def _canonical_scatter_target(target: str) -> str:
    """Map an in-place scatter spelling onto its functional ATen target."""

    return _INPLACE_SCATTER_TARGETS.get(target, target)


def _validate_inplace_legalization(node: object) -> None:
    args = getattr(node, "args", ())
    if not args:
        raise ValueError("in-place operation has no input")
    source = args[0]
    users = set(getattr(source, "users", ()))
    if users != {node}:
        raise NotImplementedError(
            "in-place operation has aliasing observers and cannot be lowered "
            f"safely: {getattr(node, 'target', '<unknown>')}"
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
    target = _canonical_scatter_target(str(getattr(node, "target", "")))
    reduce_name: object = None
    if target == _SCATTER_REDUCE_TARGET:
        # `scatter_reduce` carries its reduction as a positional argument.
        if len(args) != 5:
            raise ValueError(
                f"scatter_reduce expects five arguments, got {len(args)}"
            )
        source, dim, indices, update, reduce_name = args
    else:
        if len(args) != 4:
            raise ValueError(f"scatter expects four arguments, got {len(args)}")
        source, dim, indices, update = args
    reduce, include_self = _scatter_reduce_attrs(target, reduce_name, kwargs)
    if reduce is None and kwargs:
        raise ValueError(
            f"scatter received unsupported keyword arguments: {sorted(kwargs)}"
        )

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
        if reduce is not None:
            attrs["reduce"] = reduce
            attrs["include_self"] = include_self

    output_type = _node_type(node)
    if output_type != source_type:
        raise ValueError(
            f"scatter output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )
    return inputs, attrs, output_type


def _scatter_reduce_attrs(
    target: str, reduce_name: object, kwargs: dict[str, object]
) -> tuple[str | None, bool]:
    """Read the reduction of ``scatter_reduce``/``scatter_add``.

    Returns ``(reduce, include_self)``, with ``reduce`` ``None`` for the plain
    overwrite forms.
    """

    if target == "aten.scatter_add.default":
        if kwargs:
            raise ValueError(
                "scatter_add received unsupported keyword arguments: "
                f"{sorted(kwargs)}"
            )
        return "add", True
    if target != _SCATTER_REDUCE_TARGET:
        return None, True
    unknown = sorted(set(kwargs) - {"reduce", "include_self"})
    if unknown:
        raise ValueError(
            "scatter_reduce received unsupported keyword arguments: "
            f"{unknown}"
        )
    if reduce_name is None:
        reduce_name = kwargs.get("reduce", "sum")
    if reduce_name not in _SCATTER_REDUCE_NAMES:
        raise ValueError(
            f"scatter_reduce.reduce {reduce_name!r} is not supported; expected "
            f"one of {sorted(_SCATTER_REDUCE_NAMES)}"
        )
    include_self = kwargs.get("include_self", True)
    if not isinstance(include_self, bool):
        raise ValueError(
            f"scatter_reduce.include_self must be a boolean, got {include_self!r}"
        )
    return _SCATTER_REDUCE_NAMES[str(reduce_name)], include_self


def _index_add_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    unknown = sorted(set(kwargs) - {"alpha"})
    if unknown:
        raise ValueError(
            f"index_add received unsupported keyword arguments: {unknown}"
        )
    if len(args) != 4:
        raise ValueError(f"index_add expects four arguments, got {len(args)}")

    source, dim, indices, update = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"index_add.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    update_type = _node_type(update)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("index_add requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"index_add.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != 1:
        raise NotImplementedError(
            "index_add requires a 1-D index tensor, got "
            f"{indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "index_add indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    if update_type.dtype != source_type.dtype:
        raise ValueError(
            "index_add source dtype must match the input dtype, got "
            f"{update_type.dtype.value} and {source_type.dtype.value}"
        )
    expected_update = (
        source_type.shape[:axis]
        + (indices_type.shape[0],)
        + source_type.shape[axis + 1 :]
    )
    if update_type.shape != expected_update:
        raise ValueError(
            f"index_add source shape {update_type.shape} does not match "
            f"{expected_update}"
        )
    alpha = kwargs.get("alpha", 1)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"index_add.alpha must be a scalar number, got {alpha!r}")
    if not isfinite(float(alpha)):
        raise ValueError(f"index_add.alpha must be finite, got {alpha!r}")

    output_type = _node_type(node)
    if output_type != source_type:
        raise ValueError(
            f"index_add output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )
    return (
        (
            _argument_ids(source, node_ids),
            _argument_ids(indices, node_ids),
            _argument_ids(update, node_ids),
        ),
        {"dim": axis, "alpha": float(alpha)},
        output_type,
    )


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
            raise ValueError(f"native_layer_norm output index must be 0, 1, or 2, got {index}")
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
    return tuple(inputs), attrs, output_type, frozenset(wanted)


def _diagonal_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.diagonal`` into permute/reshape/slice.

    ``A[i, i + offset]`` sits at flat index ``i * (m + 1) + offset`` once the
    two axes are trailing and flattened, so the diagonal is a strided slice
    with no copy.
    """

    input_node, offset, dim1, dim2 = _diagonal_arguments(node)
    input_type = _node_type(input_node)
    shape = input_type.shape
    rank = len(shape)
    rows, columns = shape[dim1], shape[dim2]
    start_row = max(0, -offset)
    start_column = max(0, offset)
    length = min(rows - start_row, columns - start_column)
    if length <= 0:
        raise NotImplementedError(
            f"diagonal offset {offset} leaves no elements on a {rows}x{columns} "
            "matrix, and the static IR has no zero-sized dimensions"
        )
    remaining = tuple(
        size
        for axis, size in enumerate(shape)
        if axis not in {dim1, dim2}
    )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    order = tuple(
        axis for axis in range(rank) if axis not in {dim1, dim2}
    ) + (dim1, dim2)
    if order != tuple(range(rank)):
        current = emit("permute", (current,), {"dims": order}, tuple(shape[axis] for axis in order))
    flattened_shape = remaining + (rows * columns,)
    current = emit("reshape", (current,), {"shape": flattened_shape}, flattened_shape)
    step = columns + 1
    start = start_row * columns + start_column
    sliced_shape = remaining + (length,)
    current = emit(
        "slice",
        (current,),
        {
            "dim": len(remaining),
            "start": start,
            "end": start + length * step,
            "step": step,
        },
        sliced_shape,
    )
    expected = _node_type(node).shape
    if sliced_shape != expected:
        raise ValueError(
            f"diagonal output shape {expected} does not match the strided "
            f"view {sliced_shape}"
        )
    return tuple(operations), current


def _diagonal_arguments(
    node: object,
) -> tuple[object, int, int, int]:
    """Normalize the ``diagonal`` offset and axis arguments."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("diagonal expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank < 2:
        raise NotImplementedError(
            f"diagonal requires rank 2 or more, got shape {input_type.shape}"
        )

    offset = args[1] if len(args) > 1 else kwargs.get("offset", 0)
    dim1 = args[2] if len(args) > 2 else kwargs.get("dim1", 0)
    dim2 = args[3] if len(args) > 3 else kwargs.get("dim2", 1)
    for name, value in (("offset", offset), ("dim1", dim1), ("dim2", dim2)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"diagonal.{name} must be an integer, got {value!r}")
    axis1, axis2 = dim1 % rank, dim2 % rank
    if axis1 == axis2:
        raise ValueError(
            f"diagonal.dim1 and dim2 must differ, both are {axis1}"
        )
    return input_node, offset, axis1, axis2


def _logsumexp_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.logsumexp`` into amax/masked_fill/exp/sum/log/add.

    ATen shifts by the reduced maximum for stability and replaces an infinite
    maximum with zero before the subtraction, so a +inf row sums to +inf and an
    all -inf row keeps -inf instead of producing ``inf - inf``.
    """

    input_node, dims, keepdim = _logsumexp_arguments(node)
    input_type = _node_type(input_node)
    shape = input_type.shape
    reduced_shape = tuple(
        1 if axis in dims else size for axis, size in enumerate(shape)
    )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
        dtype: DType | None = None,
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, dtype or input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    max_id = emit(
        "amax",
        (current,),
        {"dims": dims, "keepdim": True},
        reduced_shape,
    )
    infinite_id = emit(
        "isinf",
        (max_id,),
        {},
        reduced_shape,
        DType.BOOL,
    )
    shift_id = emit(
        "masked_fill",
        (max_id, infinite_id),
        {"value": 0.0},
        reduced_shape,
    )
    shifted_id = emit("sub", (current, shift_id), {}, shape)
    exp_id = emit("exp", (shifted_id,), {}, shape)
    sum_id = emit("sum", (exp_id,), {"dims": dims, "keepdim": True}, reduced_shape)
    log_id = emit("log", (sum_id,), {}, reduced_shape)
    total_id = emit("add", (log_id, shift_id), {}, reduced_shape)

    expected = _node_type(node).shape
    if not keepdim:
        total_id = emit("reshape", (total_id,), {"shape": expected}, expected)
    elif expected != reduced_shape:
        raise ValueError(
            f"logsumexp output shape {expected} does not match the reduced "
            f"{reduced_shape}"
        )
    return tuple(operations), total_id


def _logsumexp_arguments(
    node: object,
) -> tuple[object, tuple[int, ...], bool]:
    """Normalize the ``logsumexp`` dims and keepdim arguments."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("logsumexp expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError("logsumexp requires a ranked tensor")

    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    if not isinstance(keepdim, bool):
        raise ValueError(f"logsumexp.keepdim must be a boolean, got {keepdim!r}")
    if isinstance(dims, int) and not isinstance(dims, bool):
        normalized = (dims % rank,)
    elif isinstance(dims, (tuple, list)) and dims and all(
        isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
    ):
        normalized = tuple(dim % rank for dim in dims)
    elif isinstance(dims, (tuple, list)) and not dims:
        raise ValueError(
            "logsumexp needs at least one reduction axis; an empty dim tuple "
            "has no ATen meaning"
        )
    else:
        raise ValueError(f"logsumexp.dim must contain integers, got {dims!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"logsumexp.dim repeats an axis: {dims!r}")
    return input_node, normalized, keepdim


def _constant_pad_crops(node: object) -> bool:
    """Report whether a constant pad carries negative (cropping) entries."""

    args = getattr(node, "args", ())
    if len(args) < 2:
        return False
    pads = args[1]
    if not isinstance(pads, (tuple, list)):
        return False
    return any(
        isinstance(item, int) and not isinstance(item, bool) and item < 0
        for item in pads
    )


def _constant_pad_crop_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower a constant pad with negative entries as slice-then-pad.

    ATen derives the fill region from the distance past the source, so cropping
    the *padded* extent is the same as cropping the source first and padding
    what survives. The pad op only ever sees non-negative widths.
    """

    input_node, pads, value = _constant_pad_arguments(
        getattr(node, "args", ()), getattr(node, "kwargs", {})
    )
    input_type = _node_type(input_node)
    shape = input_type.shape
    rank = len(shape)
    padded_dims = len(pads) // 2
    if padded_dims > rank:
        raise ValueError(
            f"constant_pad_nd holds {padded_dims} padding pairs for a rank-"
            f"{rank} input"
        )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    current_shape = shape
    clamped: list[int] = []
    for position in range(padded_dims):
        axis = rank - 1 - position
        before, after = pads[2 * position], pads[2 * position + 1]
        clamped.extend((max(before, 0), max(after, 0)))
        if before >= 0 and after >= 0:
            continue
        start, end = max(-before, 0), shape[axis] - max(-after, 0)
        if start >= end:
            raise NotImplementedError(
                f"constant_pad_nd crop ({before}, {after}) empties axis {axis} "
                f"of size {shape[axis]}, which the static IR cannot represent"
            )
        current_shape = (
            current_shape[:axis] + (end - start,) + current_shape[axis + 1 :]
        )
        current = emit(
            "slice",
            (current,),
            {"dim": axis, "start": start, "end": end, "step": 1},
            current_shape,
        )

    expected = _node_type(node).shape
    if any(clamped):
        padded = list(current_shape)
        for position in range(padded_dims):
            axis = rank - 1 - position
            padded[axis] += clamped[2 * position] + clamped[2 * position + 1]
        current_shape = tuple(padded)
        current = emit(
            "pad",
            (current,),
            {"pads": tuple(clamped), "value": value},
            current_shape,
        )
    if current_shape != expected:
        raise ValueError(
            f"constant_pad_nd output shape {expected} does not match the "
            f"cropped and padded {current_shape}"
        )
    return tuple(operations), current


def _constant_pad_arguments(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[object, tuple[int, ...], float]:
    """Validate the constant pad arguments shared by both lowerings."""

    if len(args) not in {2, 3}:
        raise ValueError(
            f"constant_pad_nd expects two or three arguments, got {len(args)}"
        )
    input_node, pads = args[:2]
    value = args[2] if len(args) == 3 else kwargs.get("value", 0.0)
    mode = kwargs.get("mode")
    if mode is not None and mode != "constant":
        raise NotImplementedError(
            f"constant_pad_nd mode {mode!r} is not supported; only "
            "constant padding is lowered"
        )
    if not isinstance(pads, (tuple, list)) or not pads:
        raise ValueError(f"pad.pads must be a non-empty sequence, got {pads!r}")
    if len(pads) % 2 != 0:
        raise ValueError(
            "pad.pads must hold one (before, after) pair per padded "
            f"dimension, got {len(pads)} values"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in pads):
        raise ValueError(f"pad.pads must contain integers, got {pads!r}")
    if not isinstance(value, (bool, int, float)):
        raise ValueError(f"pad.value must be a scalar number, got {value!r}")
    if not isfinite(float(value)):
        raise ValueError(f"pad.value must be finite, got {value!r}")
    return input_node, tuple(int(item) for item in pads), float(value)


def _padding_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower reflect/replicate padding into slice, flip/expand and cat."""

    target = str(getattr(node, "target", ""))
    mode, padded_dims = _PAD_MODE_TARGETS[target]
    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"{target} expects two arguments, got {len(args)}")
    input_node, padding = args
    if not isinstance(padding, (tuple, list)) or len(padding) != 2 * padded_dims:
        raise ValueError(
            f"{target}.padding must hold {2 * padded_dims} entries, got {padding!r}"
        )
    if not all(
        isinstance(pad, int) and not isinstance(pad, bool) for pad in padding
    ):
        raise ValueError(f"{target}.padding must contain integers, got {padding!r}")

    input_type = _node_type(input_node)
    shape = input_type.shape
    rank = len(shape)
    if rank < padded_dims:
        raise ValueError(
            f"{target} needs at least rank {padded_dims}, got shape {shape}"
        )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    current_shape = shape
    for position in range(padded_dims):
        axis = rank - 1 - position
        before, after = padding[2 * position], padding[2 * position + 1]
        size = current_shape[axis]
        if before == 0 and after == 0:
            continue
        # Negative entries crop the padded extent the way ATen does: build the
        # full pad first and slice the window afterwards, which keeps a mirror
        # window anchored on the original extent.
        pad_before, pad_after = max(before, 0), max(after, 0)
        crop_before, crop_after = max(-before, 0), max(-after, 0)
        if mode == "reflect" and (pad_before >= size or pad_after >= size):
            raise ValueError(
                f"{target} reflection padding {pad_before}/{pad_after} must stay "
                f"below the {size} elements of axis {axis}"
            )

        pieces: list[str] = []
        if pad_before:
            single_shape = (
                current_shape[:axis] + (1,) + current_shape[axis + 1 :]
            )
            if mode == "reflect":
                window = (
                    current_shape[:axis] + (pad_before,) + current_shape[axis + 1 :]
                )
                piece = emit(
                    "slice",
                    (current,),
                    {"dim": axis, "start": 1, "end": pad_before + 1, "step": 1},
                    window,
                )
                piece = emit("flip", (piece,), {"dims": (axis,)}, window)
            else:
                piece = emit(
                    "slice",
                    (current,),
                    {"dim": axis, "start": 0, "end": 1, "step": 1},
                    single_shape,
                )
                window = (
                    current_shape[:axis] + (pad_before,) + current_shape[axis + 1 :]
                )
                piece = emit("expand", (piece,), {"shape": window}, window)
            pieces.append(piece)
        pieces.append(current)
        if pad_after:
            if mode == "reflect":
                window = (
                    current_shape[:axis] + (pad_after,) + current_shape[axis + 1 :]
                )
                piece = emit(
                    "slice",
                    (current,),
                    {
                        "dim": axis,
                        "start": size - pad_after - 1,
                        "end": size - 1,
                        "step": 1,
                    },
                    window,
                )
                piece = emit("flip", (piece,), {"dims": (axis,)}, window)
            else:
                single_shape = (
                    current_shape[:axis] + (1,) + current_shape[axis + 1 :]
                )
                piece = emit(
                    "slice",
                    (current,),
                    {"dim": axis, "start": size - 1, "end": size, "step": 1},
                    single_shape,
                )
                window = (
                    current_shape[:axis] + (pad_after,) + current_shape[axis + 1 :]
                )
                piece = emit("expand", (piece,), {"shape": window}, window)
            pieces.append(piece)

        padded_shape = (
            current_shape[:axis]
            + (size + pad_before + pad_after,)
            + current_shape[axis + 1 :]
        )
        current = (
            pieces[0]
            if len(pieces) == 1
            else emit("cat", tuple(pieces), {"dim": axis}, padded_shape)
        )
        current_shape = padded_shape

        if crop_before or crop_after:
            padded_size = padded_shape[axis]
            start, end = crop_before, padded_size - crop_after
            if start >= end:
                raise ValueError(
                    f"{target} crop ({before}, {after}) empties axis {axis} of "
                    f"size {padded_size}"
                )
            cropped_shape = (
                current_shape[:axis]
                + (end - start,)
                + current_shape[axis + 1 :]
            )
            current = emit(
                "slice",
                (current,),
                {"dim": axis, "start": start, "end": end, "step": 1},
                cropped_shape,
            )
            current_shape = cropped_shape

    expected = _node_type(node).shape
    if current_shape != expected:
        raise ValueError(
            f"{target} output shape {expected} does not match the padded "
            f"{current_shape}"
        )
    return tuple(operations), current


def _repeat_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.repeat`` into reshape/expand/reshape.

    PyTorch aligns repeats with the trailing axes and left-pads both the shape
    and the repeat list with ones, then reads ``out[i, j] = in[i % si, j % sj]``.
    Splitting every axis into ``(repeat, size)`` and broadcasting turns exactly
    that into a view chain.
    """

    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"repeat expects two arguments, got {len(args)}")
    input_node, repeats = args
    if not isinstance(repeats, (tuple, list)) or not repeats:
        raise ValueError(f"repeat.repeats must be a non-empty sequence, got {repeats!r}")
    if not all(
        isinstance(repeat, int) and not isinstance(repeat, bool) for repeat in repeats
    ):
        raise ValueError(f"repeat.repeats must contain integers, got {repeats!r}")
    if any(repeat < 0 for repeat in repeats):
        raise ValueError(f"repeat.repeats must not be negative, got {repeats!r}")
    if any(repeat == 0 for repeat in repeats):
        raise NotImplementedError(
            "repeat with a zero repeat produces an empty dimension, which the "
            "static IR does not model"
        )

    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    new_rank = max(rank, len(repeats))
    padded_shape = (1,) * (new_rank - rank) + input_type.shape
    padded_repeats = (1,) * (new_rank - len(repeats)) + tuple(repeats)
    output_shape = tuple(
        size * repeat for size, repeat in zip(padded_shape, padded_repeats)
    )
    if output_shape != _node_type(node).shape:
        raise ValueError(
            f"repeat output shape {_node_type(node).shape} does not match the "
            f"computed {output_shape}"
        )
    if padded_shape == output_shape:
        return (), _argument_ids(input_node, node_ids)

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str, inputs: tuple[str, ...], attrs: dict[str, object], shape: tuple[int, ...]
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(shape, input_type.dtype)),
            )
        )
        return value_id

    interleaved = tuple(
        item for size in padded_shape for item in (1, size)
    )
    expanded = tuple(
        item
        for size, repeat in zip(padded_shape, padded_repeats)
        for item in (repeat, size)
    )
    current = emit(
        "reshape",
        (_argument_ids(input_node, node_ids),),
        {"shape": interleaved},
        interleaved,
    )
    current = emit("expand", (current,), {"shape": expanded}, expanded)
    current = emit("reshape", (current,), {"shape": output_shape}, output_shape)
    return tuple(operations), current


def _vector_norm_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("linalg_vector_norm expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError("linalg_vector_norm requires a ranked tensor")

    ord_value = args[1] if len(args) > 1 else kwargs.get("ord", 2)
    if isinstance(ord_value, bool) or not isinstance(ord_value, (int, float)):
        raise NotImplementedError(
            f"linalg_vector_norm.ord must be a real number, got {ord_value!r}"
        )
    dims = args[2] if len(args) > 2 else kwargs.get("dim")
    keepdim = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
    if not isinstance(keepdim, bool):
        raise ValueError(
            f"linalg_vector_norm.keepdim must be a boolean, got {keepdim!r}"
        )

    if dims is None:
        normalized_dims = tuple(range(rank))
    elif isinstance(dims, int) and not isinstance(dims, bool):
        normalized_dims = (dims % rank,)
    elif isinstance(dims, (tuple, list)) and dims and all(
        isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
    ):
        normalized_dims = tuple(dim % rank for dim in dims)
    else:
        raise ValueError(
            f"linalg_vector_norm.dim must contain integers, got {dims!r}"
        )
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"linalg_vector_norm.dim repeats an axis: {dims!r}")

    return (
        (_argument_ids(input_node, node_ids),),
        {
            "ord": float(ord_value),
            "dims": tuple(sorted(normalized_dims)),
            "keepdim": keepdim,
        },
        _node_type(node),
    )


def _variance_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    """Normalize ``var``/``std`` spellings onto ``dims``/``keepdim``/``correction``."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{target} expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")

    if target.endswith(".correction"):
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        correction = kwargs.get("correction", 1)
        if correction is None:
            correction = 1
        keepdim = kwargs.get("keepdim", False)
    elif target.endswith(".dim"):
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        unbiased = args[2] if len(args) > 2 else kwargs.get("unbiased", True)
        correction = 1 if unbiased else 0
        keepdim = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
    else:
        unbiased = args[1] if len(args) > 1 else kwargs.get("unbiased", True)
        correction = 1 if unbiased else 0
        dims = None
        keepdim = False

    if not isinstance(keepdim, bool):
        raise ValueError(f"{target}.keepdim must be a boolean, got {keepdim!r}")
    if isinstance(correction, bool) or not isinstance(correction, (int, float)):
        raise NotImplementedError(
            f"{target}.correction must be a number, got {correction!r}"
        )
    if isinstance(correction, float) and not isfinite(correction):
        raise ValueError(
            f"{target}.correction must be finite, got {correction!r}"
        )
    if isinstance(correction, float) and correction.is_integer():
        correction = int(correction)

    if dims is None:
        normalized_dims = tuple(range(rank))
    elif isinstance(dims, int) and not isinstance(dims, bool):
        normalized_dims = (dims % rank,)
    elif isinstance(dims, (tuple, list)) and dims and all(
        isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
    ):
        normalized_dims = tuple(dim % rank for dim in dims)
    else:
        raise ValueError(f"{target}.dim must contain integers, got {dims!r}")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"{target}.dim repeats an axis: {dims!r}")

    return (
        (_argument_ids(input_node, node_ids),),
        {
            "dims": normalized_dims,
            "keepdim": keepdim,
            "correction": correction,
        },
        _node_type(node),
    )


def _sort_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], dict[int, str]]:
    """Lower ``sort``/``topk`` selections into single-output IR operations."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{target} expects a tensor input")

    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")

    if target == "aten.sort.default":
        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
        descending = args[2] if len(args) > 2 else kwargs.get("descending", False)
        stable = args[3] if len(args) > 3 else kwargs.get("stable", False)
        if not isinstance(descending, bool):
            raise ValueError(
                f"{target}.descending must be a boolean, got {descending!r}"
            )
        if not isinstance(stable, bool):
            raise ValueError(f"{target}.stable must be a boolean, got {stable!r}")
    else:
        if len(args) < 2:
            raise ValueError(f"{target} expects an input and k")
        dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
        largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
        ordered = args[4] if len(args) > 4 else kwargs.get("sorted", True)
        if not isinstance(largest, bool):
            raise ValueError(f"{target}.largest must be a boolean, got {largest!r}")
        if not isinstance(ordered, bool):
            raise ValueError(f"{target}.sorted must be a boolean, got {ordered!r}")
        k = args[1]
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"{target}.k must be a non-negative integer, got {k!r}")

    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
    if not -rank <= dim < rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")
    axis = dim % rank
    if target == "aten.topk.default" and k > input_type.shape[axis]:
        raise ValueError(
            f"{target}.k {k} exceeds the size {input_type.shape[axis]} of "
            f"dimension {axis}"
        )

    wanted: dict[int, object] = {}
    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(f"{target} outputs must be selected with getitem")
        user_args = getattr(user, "args", ())
        if len(user_args) != 2 or not isinstance(user_args[1], int):
            raise ValueError(f"malformed {target} getitem: {user!r}")
        index = user_args[1]
        if index not in {0, 1}:
            raise ValueError(f"{target} output index must be 0 or 1, got {index}")
        if user.users:
            wanted[index] = user

    operations: list[tuple[Operation, Value]] = []
    produced: dict[int, str] = {}

    def emit(
        name: str,
        attrs: dict[str, object],
        index: int,
        output_type: TensorType,
    ) -> None:
        output_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (Operation(name, (_argument_ids(input_node, node_ids),), (output_id,), attrs), Value(output_id, output_type))
        )
        produced[index] = output_id

    if target == "aten.sort.default":
        attrs: dict[str, object] = {"dim": axis, "descending": descending}
        if 0 in wanted:
            emit("sort", dict(attrs), 0, input_type)
        if 1 in wanted:
            emit(
                "argsort",
                dict(attrs),
                1,
                TensorType(input_type.shape, DType.I64),
            )
        return tuple(operations), produced

    k = int(args[1])
    output_shape = input_type.shape[:axis] + (k,) + input_type.shape[axis + 1 :]
    attrs = {"dim": axis, "k": k, "largest": largest}
    if 0 in wanted:
        emit("topk", dict(attrs), 0, TensorType(output_shape, input_type.dtype))
    if 1 in wanted:
        emit(
            "topk_indices",
            dict(attrs),
            1,
            TensorType(output_shape, DType.I64),
        )
    return tuple(operations), produced


def _layer_norm_statistics(
    ops: list[Operation],
    values: dict[str, Value],
    input_id: str,
    attrs: dict[str, object],
    wanted: frozenset[int],
) -> dict[int, str]:
    """Materialize the mean/rstd outputs PyTorch returns beside layer_norm.

    The native kernel computes both internally but only writes the normalized
    value, and the IR has single-output operations, so the statistics are
    rebuilt from a two-pass variance over the normalized axes. They only exist
    when the graph actually reads them.
    """

    input_type = values[input_id].type
    shape = input_type.shape
    normalized_shape = attrs["normalized_shape"]
    assert isinstance(normalized_shape, tuple)
    rank = len(shape)
    norm_rank = len(normalized_shape)
    if norm_rank > rank or shape[rank - norm_rank :] != normalized_shape:
        raise ValueError(
            f"layer_norm normalized_shape {normalized_shape} does not match "
            f"the trailing axes of {shape}"
        )
    dims = tuple(range(rank - norm_rank, rank))
    stats_shape = shape[: rank - norm_rank] + (1,) * norm_rank
    dtype = input_type.dtype

    def emit(
        op: str,
        inputs: tuple[str, ...],
        op_attrs: dict[str, object],
        output_shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{len(ops)}"
        ops.append(Operation(op, inputs, (value_id,), op_attrs))
        values[value_id] = Value(value_id, TensorType(output_shape, dtype))
        return value_id

    produced = {
        1: emit("mean", (input_id,), {"dims": dims, "keepdim": True}, stats_shape)
    }
    if 2 in wanted:
        centered = emit("sub", (input_id, produced[1]), {}, shape)
        squared = emit("mul", (centered, centered), {}, shape)
        variance = emit(
            "mean", (squared,), {"dims": dims, "keepdim": True}, stats_shape
        )
        shifted = emit(
            "add",
            (variance,),
            {"scalar": float(attrs["eps"]), "scalar_position": "rhs"},
            stats_shape,
        )
        produced[2] = emit("rsqrt", (shifted,), {}, stats_shape)
    return produced


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
    expected = "aten.mul.Tensor" if scale_op == "aten.rsqrt.default" else "aten.div.Tensor"
    return compute_input if target == expected else None


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
    if not all(
        isinstance(axis, int) and not isinstance(axis, bool) for axis in axes
    ):
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
    if normalized is None or _rms_norm_normalized_input(
        normalized, compute_input, scale, scale_op
    ) is None:
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
    if (
        weighted_output is not None
        and str(weighted_output.target) == "aten.mul.Tensor"
    ):
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


def _single_user(node: object) -> Any | None:
    users = tuple(getattr(node, "users", {}))
    return users[0] if len(users) == 1 else None


def _argument_ids(argument: object, node_ids: dict[object, str]) -> str:
    try:
        return node_ids[argument]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported FX argument: {argument!r}") from exc


def _promoted_pointwise_dtype(
    op_name: str,
    dtypes: set[DType],
    output_dtype: DType,
) -> DType | None:
    """Return the dtype PyTorch would apply operands of ``dtypes`` in."""

    if not dtypes <= _INTEGRAL_DTYPES | _FLOATING_DTYPES:
        # Boolean operands keep their own rules; nothing to promote here.
        return None
    if op_name == "div" and dtypes <= _INTEGRAL_DTYPES:
        # True division always produces a float, even for integer operands.
        return output_dtype if output_dtype in _FLOATING_DTYPES else None
    if op_name in _FLOAT_ONLY_POINTWISE_OPS and dtypes <= _INTEGRAL_DTYPES:
        # The kernel itself only takes floats, and PyTorch runs integer
        # operands through its floating-point opmath path.
        return output_dtype if output_dtype in _FLOATING_DTYPES else None
    if len(dtypes) == 1:
        return None
    if dtypes <= _INTEGRAL_DTYPES:
        return DType.I64 if DType.I64 in dtypes else DType.I32
    if dtypes & _INTEGRAL_DTYPES:
        floats = dtypes - _INTEGRAL_DTYPES
        return DType.F32 if len(floats) > 1 else next(iter(floats))
    # Two different float widths promote to float32, the width PyTorch uses
    # for the mixed pair, before the operator sees them.
    return DType.F32


def _promote_pointwise_operands(
    op_name: str,
    inputs: tuple[str, ...],
    values: dict[str, Value],
    ops: list[Operation],
    output_dtype: DType,
) -> tuple[str, ...]:
    """Insert casts so a pointwise op sees the operands PyTorch would."""

    dtypes = {values[input_id].type.dtype for input_id in inputs}
    target = _promoted_pointwise_dtype(op_name, dtypes, output_dtype)
    if target is None:
        return inputs
    promoted: list[str] = []
    for input_id in inputs:
        value = values[input_id]
        if value.type.dtype == target:
            promoted.append(input_id)
            continue
        cast_id = f"v{len(ops)}"
        values[cast_id] = Value(cast_id, TensorType(value.type.shape, target))
        ops.append(Operation("cast", (input_id,), (cast_id,), {}))
        promoted.append(cast_id)
    return tuple(promoted)


def _operation_inputs(
    op: str,
    args: tuple[object, ...],
    node_ids: dict[object, str],
    kwargs: dict[str, object] | None = None,
) -> tuple[tuple[str, ...], dict[str, object]]:
    kwargs = {} if kwargs is None else kwargs
    if op in {"argmax", "argmin"}:
        if not args:
            raise ValueError(f"{op} expects an input tensor")
        if len(args) > 3:
            raise ValueError(
                f"{op} expects one to three positional arguments, got {len(args)}"
            )
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if dim is not None and (
            not isinstance(dim, int) or isinstance(dim, bool)
        ):
            raise ValueError(f"{op}.dim must be an integer or None, got {dim!r}")
        if not isinstance(keepdim, bool):
            raise ValueError(f"{op}.keepdim must be a boolean, got {keepdim!r}")
        if dim is None and keepdim:
            raise ValueError(f"{op} cannot keep a dimension without a dim")
        return (
            (_argument_ids(args[0], node_ids),),
            {"dim": dim, "keepdim": keepdim},
        )
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

    if op == "arange":
        if len(args) not in {1, 2, 3}:
            raise ValueError(
                f"arange expects one to three arguments, got {len(args)}"
            )
        start = 0 if len(args) == 1 else args[0]
        end = args[0] if len(args) == 1 else args[1]
        step = 1 if len(args) < 3 else args[2]
        for name, value in (("start", start), ("end", end), ("step", step)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"arange.{name} must be a scalar number, got {value!r}"
                )
            if not isfinite(value):
                raise ValueError(
                    f"arange.{name} must be finite, got {value!r}"
                )
        if step == 0:
            raise ValueError("arange.step must be nonzero")
        return (), {"start": start, "end": end, "step": step}

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

    if op in {"softmax", "safe_softmax", "log_softmax"}:
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

    if op in {"cumsum", "cumprod"}:
        if len(args) not in {2, 3}:
            raise ValueError(
                f"{op} expects two or three arguments, got {len(args)}"
            )
        input_id, dim = args[:2]
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"{op}.dim must be an integer, got {dim!r}")
        if len(args) == 3 and args[2] is not None:
            dtype_name = str(args[2])
            if _TORCH_DTYPES.get(dtype_name) is None:
                raise ValueError(f"unsupported {op} dtype: {dtype_name}")
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

    if op == "pad":
        input_id, pads, value = _constant_pad_arguments(args, kwargs)
        if any(item < 0 for item in pads):
            raise NotImplementedError(
                "negative constant_pad_nd entries crop the padded extent and "
                f"must be lowered as a slice first, got {pads}"
            )
        return (_argument_ids(input_id, node_ids),), {
            "pads": pads,
            "value": value,
        }

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

    if op == "nan_to_num":
        if len(args) not in {1, 2, 3, 4}:
            raise ValueError(
                f"nan_to_num expects one to four arguments, got {len(args)}"
            )
        attrs: dict[str, object] = {}
        for position, name in ((1, "nan"), (2, "posinf"), (3, "neginf")):
            value = args[position] if len(args) > position else kwargs.get(name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"nan_to_num.{name} must be a scalar, got {value!r}"
                )
            attrs[name] = float(value)
        return ((_argument_ids(args[0], node_ids),), attrs)

    if op == "logit":
        if len(args) not in {1, 2}:
            raise ValueError(f"logit expects one or two arguments, got {len(args)}")
        attrs: dict[str, object] = {}
        eps = args[1] if len(args) == 2 else kwargs.get("eps")
        if eps is not None:
            if not isinstance(eps, (int, float)) or isinstance(eps, bool):
                raise ValueError(f"logit.eps must be a scalar, got {eps!r}")
            if not isfinite(eps) or eps < 0:
                raise ValueError(
                    f"logit.eps must be finite and non-negative, got {eps!r}"
                )
            attrs["eps"] = float(eps)
        return ((_argument_ids(args[0], node_ids),), attrs)

    if op == "lerp":
        if len(args) != 3:
            raise ValueError(f"lerp expects three arguments, got {len(args)}")
        start_id, end_id, weight = args
        inputs = (
            _argument_ids(start_id, node_ids),
            _argument_ids(end_id, node_ids),
        )
        try:
            weight_id = _argument_ids(weight, node_ids)
        except ValueError:
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ValueError(
                    f"unsupported lerp weight: {weight!r}"
                ) from None
            return inputs, {"weight": float(weight)}
        return (*inputs, weight_id), {}

    if op in {"addcmul", "addcdiv"}:
        # ATen's signature is (self, tensor1, tensor2, *, value=1).
        if len(args) not in {3, 4}:
            raise ValueError(
                f"{op} expects three or four positional arguments, "
                f"got {len(args)}"
            )
        inputs = tuple(
            _argument_ids(argument, node_ids) for argument in args[:3]
        )
        attrs: dict[str, object] = {}
        if len(args) == 4:
            if "value" in kwargs:
                raise ValueError(f"{op} received duplicate value arguments")
            value = args[3]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{op}.value must be a scalar, got {value!r}")
            attrs["value"] = float(value)
        return inputs, attrs

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
                attrs[name] = bound
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
            attrs[name] = bound
        else:
            attrs[f"{name}_input"] = len(inputs)
            inputs.append(bound_id)
        return tuple(inputs), attrs

    if op in {"leaky_relu", "elu"}:
        allowed = {2} if op == "leaky_relu" else {2, 3, 4}
        if len(args) not in allowed:
            raise ValueError(f"{op} expects {allowed} arguments, got {len(args)}")
        attrs: dict[str, object] = {}
        names = ["negative_slope"] if op == "leaky_relu" else [
            "alpha",
            "scale",
            "input_scale",
        ]
        for name, argument in zip(names, args[1:]):
            if not isinstance(argument, (int, float)) or isinstance(argument, bool):
                raise ValueError(f"{op}.{name} must be a number, got {argument!r}")
            attrs[name] = float(argument)
        return ((_argument_ids(args[0], node_ids),), attrs)

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
    scalars: list[tuple[int, bool | int | float]] = []
    for position, argument in enumerate(args):
        try:
            tensor_inputs.append(_argument_ids(argument, node_ids))
        except ValueError:
            if isinstance(argument, bool) and op in (
                _COMPARISON_OPS | _LOGICAL_BINARY_OPS | _BITWISE_BINARY_OPS
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
            scalars.append(
                (position, argument if isinstance(argument, int) else float(argument))
            )

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
        "arange",
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
    if op in {"argmax", "argmin"}:
        allowed.update({"dim", "keepdim"})
    if op in {"cumsum", "cumprod"}:
        allowed.update({"dim", "dtype"})
    if op == "flip":
        allowed.add("dims")
    if op == "pad":
        allowed.update({"value", "mode"})
    if op == "nan_to_num":
        allowed.update({"nan", "posinf", "neginf"})
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
        "arange",
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
        "argmax",
        "argmin",
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
