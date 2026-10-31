"""Elementwise kernels: validate the region, then write one thread per element.

The validators and the emitter are two halves of one thing. A pointwise region
is only expressible if every operator in it has an MSL spelling for its dtype,
so validation walks the region and the emitter then assumes it passed - which
is why they sit together instead of in separate files that would have to agree.

The zero-input kernels (``arange``, integer constants) are here too: they write
a buffer without reading anything, so they share the bounds arithmetic and
differ only in the expression, and ``arange`` additionally rebuilds the float
sequence with an fma remainder so the result rounds once.
"""

from __future__ import annotations

import mlx.core as mx
from math import ceil, prod
from typing import Any
from ..ir import DType
from ..ir import Graph
from ..ir import Operation
from ..ir import TensorType
from ..op_sets import CONSTANT_OPS as _CONSTANT_POINTWISE_OPS
from ..ir.ops import FLOATING_DTYPES as _FLOATING_DTYPES
from ..ir.ops import INTEGRAL_DTYPES as _INTEGRAL_DTYPES
from ..shapes import can_broadcast as _can_broadcast
from ..ir.ops import check_integer_constant
from ..ir.ops import check_operation
from ..ir.ops import integer_constant_value
from .dtypes import _MLX_DTYPES
from .expr import (
    _BITWISE_POINTWISE_OPS,
    _COMPARISON_POINTWISE_OPS,
    _ELEMENTS_PER_THREAD,
    _INTEGER_POINTWISE_OPS,
    _LOGICAL_POINTWISE_OPS,
    _NUMERIC_PREDICATE_POINTWISE_OPS,
    _emit_pointwise_expression,
    _emit_zero_input_expression,
    _msl_float_literal,
    _msl_integer_literal,
    _offset_expression,
    _pointwise_ctype,
    _split_double_to_floats,
)
from . import msl
from .msl import KernelArtifact, _MSL_HEADER


def _validate_pointwise_region(
    graph: Graph, body: tuple[Operation, ...], output_id: str
) -> None:
    for op in body:
        output_type = graph.value(op.outputs[0]).type
        input_types = tuple(graph.value(value_id).type for value_id in op.inputs)

        if op.op in _CONSTANT_POINTWISE_OPS:
            if op.inputs:
                raise ValueError(f"{op.op} must not have tensor inputs")
            value = op.attrs.get("value")
            if not isinstance(value, (bool, int, float)):
                raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
            if output_type.dtype in _INTEGRAL_DTYPES:
                integer_constant_value(op, output_type.dtype)
            continue

        if op.op in _COMPARISON_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL:
                raise ValueError(f"{op.op} must produce a boolean tensor")
            if not (
                len(input_types) == 2
                or (len(input_types) == 1 and "scalar" in op.attrs)
            ):
                raise ValueError(f"malformed comparison op: {op.op}")
            continue

        if op.op in _LOGICAL_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL:
                raise ValueError(f"{op.op} must produce a boolean tensor")
            if op.op == "logical_not":
                expected_inputs = len(input_types) == 1
            else:
                expected_inputs = len(input_types) == 2 or (
                    len(input_types) == 1 and isinstance(op.attrs.get("scalar"), bool)
                )
            if not expected_inputs:
                raise ValueError(f"malformed logical op: {op.op}")
            if any(input_type.dtype != DType.BOOL for input_type in input_types):
                raise ValueError(f"{op.op} requires boolean tensor inputs")
            continue

        if op.op in _BITWISE_POINTWISE_OPS:
            if output_type.dtype == DType.BOOL:
                if op.op == "bitwise_not":
                    expected_inputs = len(input_types) == 1
                else:
                    expected_inputs = len(input_types) == 2 or (
                        len(input_types) == 1
                        and isinstance(op.attrs.get("scalar"), bool)
                    )
                if not expected_inputs:
                    raise ValueError(f"malformed bitwise op: {op.op}")
                if any(input_type.dtype != DType.BOOL for input_type in input_types):
                    raise ValueError(f"boolean {op.op} requires boolean tensor inputs")
                continue
            _validate_integer_pointwise_op(op, output_type, input_types)
            continue

        if op.op in {"mul", "add"} and output_type.dtype == DType.BOOL:
            # PyTorch spells boolean conjunction and disjunction as ``*`` and
            # ``+``, which is how ATen's own composites such as ``isclose``
            # combine two masks.
            valid = len(input_types) == 2 or (
                len(input_types) == 1 and isinstance(op.attrs.get("scalar"), bool)
            )
            if not valid:
                raise ValueError(f"malformed boolean op: {op.op}")
            if any(input_type.dtype != DType.BOOL for input_type in input_types):
                raise ValueError(f"boolean {op.op} requires boolean tensor inputs")
            continue

        if op.op in _NUMERIC_PREDICATE_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL or len(input_types) != 1:
                raise ValueError(f"malformed predicate op: {op.op}")
            if input_types[0].dtype == DType.BOOL:
                raise ValueError(f"{op.op} requires a numeric tensor input")
            if input_types[0].dtype in _INTEGRAL_DTYPES and op.op != "signbit":
                raise ValueError(f"{op.op} does not accept integer tensor inputs")
            continue

        if op.op == "where":
            if len(input_types) != 3:
                raise ValueError("where requires condition, input and other tensors")
            if input_types[0].dtype != DType.BOOL:
                raise ValueError("where condition must be a boolean tensor")
            if output_type.dtype in _INTEGRAL_DTYPES:
                if any(
                    input_type.dtype != output_type.dtype
                    for input_type in input_types[1:]
                ):
                    raise ValueError(
                        "integer where requires matching integer branch dtypes"
                    )
            continue

        if op.op == "masked_fill":
            if len(input_types) != 2:
                raise ValueError("masked_fill requires input and mask tensors")
            if input_types[1].dtype != DType.BOOL:
                raise ValueError("masked_fill mask must be a boolean tensor")
            if output_type != input_types[0]:
                raise ValueError("masked_fill must preserve the input tensor type")
            value = op.attrs.get("value")
            if output_type.dtype == DType.BOOL:
                if not isinstance(value, bool):
                    raise ValueError("boolean masked_fill requires a boolean value")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("numeric masked_fill requires a numeric value")
            continue

        if op.op in {"logit", "erfc", "frac", "sinc"}:
            if len(input_types) != 1:
                raise ValueError(f"{op.op} requires one tensor input")
            if input_types[0].dtype not in _FLOATING_DTYPES:
                raise ValueError(f"{op.op} requires a floating-point tensor")
            if op.op == "logit" and "eps" in op.attrs:
                eps = op.attrs["eps"]
                if not isinstance(eps, (int, float)) or isinstance(eps, bool):
                    raise ValueError(f"logit.eps must be a scalar, got {eps!r}")
            continue

        if op.op in {"hypot", "copysign", "xlogy"}:
            valid = len(input_types) == 2 or (
                len(input_types) == 1 and "scalar" in op.attrs
            )
            if not valid:
                raise ValueError(f"malformed binary pointwise op: {op.op}")
            if any(
                input_type.dtype not in _FLOATING_DTYPES for input_type in input_types
            ):
                raise ValueError(f"{op.op} requires floating-point tensor operands")
            continue

        if op.op == "nan_to_num":
            if len(input_types) != 1:
                raise ValueError("nan_to_num requires one tensor input")
            if output_type != input_types[0]:
                raise ValueError("nan_to_num must preserve the input tensor type")
            for name in ("nan", "posinf", "neginf"):
                if name not in op.attrs:
                    continue
                value = op.attrs[name]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"nan_to_num.{name} must be a scalar, got {value!r}"
                    )
            continue

        if op.op == "lerp":
            floating_inputs = len(input_types) == 3 or (
                len(input_types) == 2 and "weight" in op.attrs
            )
            if not floating_inputs:
                raise ValueError("lerp requires one weight tensor or scalar")
            if any(
                input_type.dtype not in _FLOATING_DTYPES for input_type in input_types
            ):
                raise ValueError("lerp requires floating-point tensor operands")
            continue

        if op.op in {"addcmul", "addcdiv"}:
            if len(input_types) != 3:
                raise ValueError(f"{op.op} requires self, tensor1 and tensor2 tensors")
            if any(
                input_type.dtype not in _FLOATING_DTYPES for input_type in input_types
            ):
                raise ValueError(f"{op.op} requires floating-point tensor operands")
            value = op.attrs.get("value", 1.0)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{op.op}.value must be a scalar, got {value!r}")
            continue

        if output_type.dtype == DType.BOOL or any(
            input_type.dtype == DType.BOOL for input_type in input_types
        ):
            raise ValueError(f"{op.op} does not accept boolean tensor operands")
        if output_type.dtype in _INTEGRAL_DTYPES or any(
            input_type.dtype in _INTEGRAL_DTYPES for input_type in input_types
        ):
            _validate_integer_pointwise_op(op, output_type, input_types)
            continue

    if graph.value(output_id).type.dtype == DType.BOOL:
        producer = next(
            (op for op in reversed(body) if op.outputs[0] == output_id), None
        )
        if producer is None or producer.op not in (
            _COMPARISON_POINTWISE_OPS
            | _LOGICAL_POINTWISE_OPS
            | _BITWISE_POINTWISE_OPS
            | _NUMERIC_PREDICATE_POINTWISE_OPS
            | {"where", "masked_fill", "mul", "add"}
            | _CONSTANT_POINTWISE_OPS
        ):
            raise ValueError("boolean pointwise region has an unsupported root op")


def _validate_integer_pointwise_op(
    op: Operation,
    output_type: TensorType,
    input_types: tuple[TensorType, ...],
) -> None:
    if op.op not in _INTEGER_POINTWISE_OPS:
        raise ValueError(f"{op.op} does not accept integer tensor operands")
    if output_type.dtype not in _INTEGRAL_DTYPES:
        raise ValueError(f"{op.op} does not support integer tensor operands")
    if any(input_type.dtype not in _INTEGRAL_DTYPES for input_type in input_types):
        raise ValueError(f"{op.op} requires matching integer tensor operands")

    if op.op in {"neg", "abs", "sign", "square", "bitwise_not"}:
        if len(input_types) != 1 or input_types[0].dtype != output_type.dtype:
            raise ValueError(f"malformed integer unary op: {op.op}")
        return

    if op.op in {
        "add",
        "sub",
        "rsub",
        "mul",
        "pow",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "floor_divide",
        "trunc_divide",
        "remainder",
        "gcd",
        "lcm",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "heaviside",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    }:
        if len(input_types) == 2:
            if any(input_type.dtype != output_type.dtype for input_type in input_types):
                raise ValueError(f"integer {op.op} requires matching input dtypes")
        elif len(input_types) == 1:
            scalar = op.attrs.get("scalar")
            if not isinstance(scalar, int) or isinstance(scalar, bool):
                raise ValueError(
                    f"integer {op.op} scalar must be an integer, got {scalar!r}"
                )
            if op.op == "pow" and scalar < 0:
                raise ValueError(
                    "integer pow with a negative exponent raises in PyTorch, "
                    f"got {scalar}"
                )
        else:
            raise ValueError(f"malformed integer binary op: {op.op}")
        return

    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        if not input_types or input_types[0].dtype != output_type.dtype:
            raise ValueError(f"malformed integer {op.op} op")
        for input_type in input_types[1:]:
            if input_type.dtype != output_type.dtype:
                raise ValueError(f"integer {op.op} requires matching bound dtypes")
        for name in ("min", "max"):
            bound = op.attrs.get(name)
            if bound is not None and (
                not isinstance(bound, int) or isinstance(bound, bool)
            ):
                raise ValueError(
                    f"integer {op.op}.{name} must be an integer, got {bound!r}"
                )
        return

    raise AssertionError(f"unclassified integer pointwise op: {op.op}")


def _emit_integer_constant_source(op: Operation, output_type: TensorType) -> str:
    value = integer_constant_value(op, output_type.dtype)
    return _emit_zero_input_expression(
        _msl_integer_literal(value, output_type.dtype), output_type.shape
    )


def _emit_arange_source(op: Operation, output_type: TensorType) -> str:
    if output_type.dtype in _INTEGRAL_DTYPES:
        start = _msl_integer_literal(int(op.attrs["start"]), DType.I64)
        step = _msl_integer_literal(int(op.attrs["step"]), DType.I64)
        expression = f"(long)({start}) + (long)elem * (long)({step})"
        return _emit_zero_input_expression(expression, output_type.shape)
    # PyTorch forms each element as `start + index * step` and rounds once, so
    # `arange(0, 1, 0.1)` keeps 0.9 instead of the float32 product 0.90000004.
    # Metal has no double, so the constants are split at compile time and the
    # exact sum is rebuilt from the fma product residual and a two-sum, which
    # rounds like the double expression would.
    start_hi, start_lo = _split_double_to_floats(float(op.attrs["start"]))
    step_hi, step_lo = _split_double_to_floats(float(op.attrs["step"]))
    step_hi_text = _msl_float_literal(step_hi)
    start_hi_text = _msl_float_literal(start_hi)
    element_count = prod(output_type.shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
        "    float factor = (float)elem;",
        f"    float product = factor * {step_hi_text};",
        f"    float product_err = fma(factor, {step_hi_text}, -product);",
        f"    float sum = product + {start_hi_text};",
        "    float sum_base = sum - product;",
        f"    float sum_err = (product - (sum - sum_base)) + "
        f"({start_hi_text} - sum_base);",
        f"    float tail = sum_err + product_err + "
        f"factor * {_msl_float_literal(step_lo)} + "
        f"{_msl_float_literal(start_lo)};",
        "    out[elem] = (T)(sum + tail);",
        "}",
    ]
    return "\n".join(f"        {line}" for line in lines)


def _emit_pointwise_source(
    graph: Graph,
    input_ids: tuple[str, ...],
    body: tuple[Operation, ...],
    output_id: str,
    output_shape: tuple[int, ...],
) -> str:
    element_count = prod(output_shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
    ]
    env: dict[str, str] = {}
    common: dict[tuple[object, ...], str] = {}

    for position, value_id in enumerate(input_ids):
        variable = f"input_{position}"
        input_type = graph.value(value_id).type
        ctype = _pointwise_ctype(input_type.dtype)
        if input_type.shape:
            offset = _offset_expression(input_type.shape, output_shape, "elem")
            lines.append(f"    {ctype} {variable} = in_{position}[{offset}];")
        else:
            lines.append(f"    {ctype} {variable} = ({ctype})in_{position};")
        env[value_id] = variable

    for op_index, op in enumerate(body):
        arguments = [env[value_id] for value_id in op.inputs]
        key = (
            op.op,
            op.inputs,
            graph.value(op.outputs[0]).type.dtype.value,
            tuple((name, repr(value)) for name, value in sorted(op.attrs.items())),
        )
        existing = common.get(key)
        if existing is not None:
            env[op.outputs[0]] = existing
            continue
        output = f"temp_{op_index}"
        input_types = tuple(graph.value(value_id).type for value_id in op.inputs)
        output_type = graph.value(op.outputs[0]).type
        expression = _emit_pointwise_expression(
            op, arguments, input_types, output_type.dtype
        )
        ctype = _pointwise_ctype(output_type.dtype)
        lines.append(f"    {ctype} {output} = {expression};")
        env[op.outputs[0]] = output
        common[key] = output

    if output_id not in env:
        raise ValueError("pointwise region does not produce its declared output")
    lines.extend(
        [
            f"    out[elem] = (T)({env[output_id]});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


class PointwiseLowering:
    def _compile_elementwise(
        self,
        graph: Graph,
        input_ids: tuple[str, ...],
        body: tuple[Operation, ...],
        output_id: str,
        index: int,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        output_type = graph.value(output_id).type
        _validate_pointwise_region(graph, body, output_id)
        for value_id in input_ids:
            input_type = graph.value(value_id).type
            if not _can_broadcast(input_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast pointwise input {input_type.shape} "
                    f"to {output_type.shape}"
                )

        source = _emit_pointwise_source(
            graph, input_ids, body, output_id, output_type.shape
        )
        name = f"{kind}_{index}"
        kernel = msl._metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(input_ids))),
            ("out",),
            source,
            _MSL_HEADER,
        )

        element_count = prod(output_type.shape)
        grid = (ceil(element_count / _ELEMENTS_PER_THREAD), 1, 1)
        threadgroup = (min(256, grid[0]), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MSL_HEADER,
            input_ids=input_ids,
            output_ids=(output_id,),
            tensor_types=(output_type,),
            kind=kind,
        )
        return artifact, runtime

    def _compile_integer_constant(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        check_integer_constant(graph, op)
        output_type = graph.value(op.outputs[0]).type
        source = _emit_integer_constant_source(op, output_type)
        return self._compile_zero_input_kernel(op, index, source, output_type)

    def _compile_arange(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        check_operation(graph, op)
        output_type = graph.value(op.outputs[0]).type
        source = _emit_arange_source(op, output_type)
        return self._compile_zero_input_kernel(op, index, source, output_type)

    def _compile_zero_input_kernel(
        self,
        op: Operation,
        index: int,
        source: str,
        output_type: TensorType,
    ) -> tuple[KernelArtifact, Any]:
        name = f"{op.op}_{index}"
        kernel = msl._metal_kernel(name, (), ("out",), source, _MSL_HEADER)

        element_count = prod(output_type.shape)
        grid = (ceil(element_count / _ELEMENTS_PER_THREAD), 1, 1)
        threadgroup = (min(256, grid[0]), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime() -> mx.array:
            outputs = kernel(
                inputs=[],
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MSL_HEADER,
            input_ids=(),
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind=op.op,
        )
        return artifact, runtime
