from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .ir import DType, Graph, Operation, TensorType, Value

_SUPPORTED_OPS = {
    "aten.mm.default": "matmul",
    "aten.addmm.default": "addmm",
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
            target = str(node.target)
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
