"""Turning one IR expression into MSL, and the literals it needs to do it.

Everything here is a pure function over types and attribute values: given an
operator, its operand dtypes and its attributes, return the C expression that
computes one output element. The index arithmetic helpers live beside the
expression emitters because both are about addressing, and the pointwise
classification tables are here because the emitters are their only reader.

The integer path is separate rather than a flag because it is genuinely
different arithmetic: it wraps at the operand width, it has no NaN or infinity,
and it cannot be expressed with the float intrinsics at all.
"""

from __future__ import annotations

import struct
from math import isfinite, isinf, isnan, prod
from ..ir import DType
from ..ir import Operation
from ..ir import TensorType
from ..op_sets import CONSTANT_OPS as _CONSTANT_POINTWISE_OPS
from ..ir.ops import FLOATING_DTYPES as _FLOATING_DTYPES
from ..ir.ops import INTEGRAL_DTYPES as _INTEGRAL_DTYPES
from ..ir.ops import integer_constant_value

_ELEMENTS_PER_THREAD = 4


_DTYPE_ITEMSIZE = {
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.BOOL: 1,
    DType.I32: 4,
    DType.I64: 8,
}


_COMPARISON_POINTWISE_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}


_LOGICAL_POINTWISE_OPS = {
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
}


_BITWISE_POINTWISE_OPS = {
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
}


_NUMERIC_PREDICATE_POINTWISE_OPS = {"isnan", "isinf", "signbit"}


_FINITE_LIMITS = {
    DType.F16: 65504.0,
    DType.BF16: 3.3895313892515355e38,
    DType.F32: 3.4028234663852886e38,
}


_INTEGER_POINTWISE_OPS = {
    "add",
    "sub",
    "rsub",
    "mul",
    "pow",
    "neg",
    "abs",
    "sign",
    "square",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "floor_divide",
    "trunc_divide",
    "remainder",
    "heaviside",
    "gcd",
    "lcm",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "clamp",
    "clamp_min",
    "clamp_max",
    *_BITWISE_POINTWISE_OPS,
}


def _offset_expression(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    index_expression: str,
) -> str:
    if input_shape == output_shape:
        return index_expression

    offset = len(output_shape) - len(input_shape)
    terms: list[str] = []
    for output_axis, output_dim in enumerate(output_shape):
        input_axis = output_axis - offset
        if input_axis < 0 or input_shape[input_axis] == 1:
            continue
        output_stride = prod(output_shape[output_axis + 1 :])
        input_stride = prod(input_shape[input_axis + 1 :])
        coordinate = f"((({index_expression}) / {output_stride}) % {output_dim})"
        terms.append(f"{coordinate} * {input_stride}")
    return " + ".join(terms) if terms else "0"


def _split_double_to_floats(value: float) -> tuple[float, float]:
    """Split a double into a float head and the float tail of the remainder."""

    head = struct.unpack("<f", struct.pack("<f", value))[0]
    return head, value - head


def _emit_zero_input_expression(expression: str, output_shape: tuple[int, ...]) -> str:
    element_count = prod(output_shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
        f"    out[elem] = (T)({expression});",
        "}",
    ]
    return "\n".join(f"        {line}" for line in lines)


def _msl_integer_literal(value: int, dtype: DType) -> str:
    if dtype == DType.I64:
        if value == -(1 << 63):
            return "(-9223372036854775807LL - 1LL)"
        return f"{value}LL"
    if dtype == DType.I32:
        if value == -(1 << 31):
            return "(-2147483647 - 1)"
        return str(value)
    raise ValueError(f"{dtype.value} is not an integer dtype")


def _pointwise_ctype(dtype: DType) -> str:
    if dtype == DType.BOOL:
        return "bool"
    if dtype == DType.I32:
        return "int"
    if dtype == DType.I64:
        return "long"
    return "float"


def _wrap_integer_scalar(value: int, dtype: DType) -> int:
    bits = 32 if dtype == DType.I32 else 64
    modulus = 1 << bits
    wrapped = value % modulus
    if wrapped >= (1 << (bits - 1)):
        wrapped -= modulus
    return wrapped


def _checked_integer_scalar(value: int | float, dtype: DType) -> int:
    converted = int(value)
    bits = 32 if dtype == DType.I32 else 64
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if not lower <= converted <= upper:
        raise ValueError(f"integer scalar {value!r} overflows {dtype.value}")
    return converted


def _binary_operands(
    op: Operation,
    arguments: list[str],
    input_types: tuple[TensorType, ...],
) -> tuple[str, str]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar_value = op.attrs["scalar"]
    if isinstance(scalar_value, bool):
        scalar = "true" if scalar_value else "false"
    elif input_types[0].dtype in _INTEGRAL_DTYPES and isinstance(scalar_value, int):
        scalar = _msl_integer_literal(
            _wrap_integer_scalar(scalar_value, input_types[0].dtype),
            input_types[0].dtype,
        )
    else:
        scalar = _msl_float_literal(float(scalar_value))
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _msl_float_literal(value: float) -> str:
    if isnan(value):
        return "NAN"
    if isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)


def _emit_pointwise_expression(
    op: Operation,
    arguments: list[str],
    input_types: tuple[TensorType, ...],
    output_dtype: DType,
) -> str:
    if op.op in _CONSTANT_POINTWISE_OPS:
        value = op.attrs.get("value")
        if isinstance(value, bool):
            return "true" if value else "false"
        if output_dtype in _INTEGRAL_DTYPES:
            return _msl_integer_literal(
                integer_constant_value(op, output_dtype), output_dtype
            )
        if isinstance(value, (int, float)):
            return _msl_float_literal(float(value))
        raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
    if op.op == "nan_to_num" and output_dtype not in _FLOATING_DTYPES:
        # ATen only substitutes non-finite values, and an integer tensor has
        # none, so the lowering is the value itself.
        return arguments[0]
    if op.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }[op.op]
        return f"({left} {operator} {right})"
    if op.op == "logical_and":
        left, right = _binary_operands(op, arguments, input_types)
        return f"({left} && {right})"
    if op.op == "logical_or":
        left, right = _binary_operands(op, arguments, input_types)
        return f"({left} || {right})"
    if op.op == "logical_xor":
        left, right = _binary_operands(op, arguments, input_types)
        return f"({left} != {right})"
    if op.op == "logical_not":
        return f"(!{arguments[0]})"
    if op.op in _BITWISE_POINTWISE_OPS:
        if output_dtype == DType.BOOL:
            if op.op == "bitwise_not":
                return f"(!{arguments[0]})"
            left, right = _binary_operands(op, arguments, input_types)
            operator = {
                "bitwise_and": "&",
                "bitwise_or": "|",
                "bitwise_xor": "^",
            }[op.op]
            return f"({left} {operator} {right})"
        return _emit_integer_pointwise_expression(
            op, arguments, input_types, output_dtype
        )
    if op.op == "isnan":
        return f"metal::isnan({arguments[0]})"
    if op.op == "isinf":
        return f"metal::isinf({arguments[0]})"
    if op.op == "signbit":
        # A bit test rather than a comparison, so -0.0 reports true like
        # PyTorch. MSL has no integer overload for ``signbit``, and a signed
        # integer has no negative zero, so ``< 0`` is exact for those.
        if input_types[0].dtype in _INTEGRAL_DTYPES:
            return f"({arguments[0]} < 0)"
        return f"metal::signbit({arguments[0]})"
    if op.op == "where":
        return f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
    if op.op == "masked_fill":
        value = op.attrs.get("value")
        if isinstance(value, bool):
            fill = "true" if value else "false"
        elif output_dtype in _INTEGRAL_DTYPES:
            fill = _msl_integer_literal(
                _checked_integer_scalar(value, output_dtype), output_dtype
            )
        else:
            fill = _msl_float_literal(float(value))
        return f"({arguments[1]} ? {fill} : {arguments[0]})"
    if output_dtype in _INTEGRAL_DTYPES:
        return _emit_integer_pointwise_expression(
            op, arguments, input_types, output_dtype
        )
    if op.op == "silu":
        value = arguments[0]
        return f"{value} / (1.0f + metal::exp(-{value}))"
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            return (
                f"0.5f * {value} * (1.0f + metal::tanh("
                f"0.7978845608028654f * ({value} + "
                f"0.044715f * {value} * {value} * {value})))"
            )
        return f"0.5f * {value} * (1.0f + tsc_erf({value} * 0.7071067811865476f))"
    if op.op == "softplus":
        beta = _msl_float_literal(float(op.attrs["beta"]))
        threshold = _msl_float_literal(float(op.attrs["threshold"]))
        return f"tsc_softplus({arguments[0]}, {beta}, {threshold})"
    if op.op == "mish":
        return f"tsc_mish({arguments[0]})"
    if op.op == "logsigmoid":
        return f"tsc_logsigmoid({arguments[0]})"
    if op.op in {"mul", "div"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = "*" if op.op == "mul" else "/"
        return f"{left} {operator} {right}"
    if op.op == "pow":
        left, right = _binary_operands(op, arguments, input_types)
        return f"metal::pow({left}, {right})"
    if op.op in {"maximum", "minimum"}:
        left, right = _binary_operands(op, arguments, input_types)
        comparison = "<" if op.op == "maximum" else ">"
        selected = f"(({left}) {comparison} ({right}) ? ({right}) : ({left}))"
        return f"((metal::isnan({left}) || metal::isnan({right})) ? NAN : {selected})"
    if op.op in {"fmax", "fmin"}:
        # C99 fmax/fmin: a NaN operand is ignored instead of poisoning the
        # result, which is what ATen's kernels do and maximum/minimum do not.
        left, right = _binary_operands(op, arguments, input_types)
        function = "fmax" if op.op == "fmax" else "fmin"
        return f"tsc_{function}({left}, {right})"
    if op.op == "nextafter":
        left, right = _binary_operands(op, arguments, input_types)
        # The neighbour of a half value is a different float than the neighbour
        # of its float image, so the step is taken at the operand's width.
        function = {
            DType.F16: "tsc_nextafter_half",
            DType.BF16: "tsc_nextafter_bfloat",
        }.get(output_dtype, "tsc_nextafter")
        return f"{function}({left}, {right})"
    if op.op == "trunc_divide":
        left, right = _binary_operands(op, arguments, input_types)
        return f"metal::trunc({left} / {right})"
    if op.op == "logaddexp":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_logaddexp({left}, {right})"
    if op.op == "atan2":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_atan2({left}, {right})"
    if op.op == "floor_divide":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_floor_div_f32({left}, {right})"
    if op.op == "remainder":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_remainder_f32({left}, {right})"
    if op.op == "fmod":
        left, right = _binary_operands(op, arguments, input_types)
        return f"metal::fmod({left}, {right})"
    if op.op == "hypot":
        left, right = _binary_operands(op, arguments, input_types)
        # MSL declares hypot in the global namespace, like log1p.
        return f"hypot({left}, {right})"
    if op.op == "copysign":
        left, right = _binary_operands(op, arguments, input_types)
        return f"metal::copysign({left}, {right})"
    if op.op == "xlogy":
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_xlogy_f32({left}, {right})"
    if output_dtype == DType.BOOL and op.op in {"mul", "add"}:
        # ``&&`` and ``||`` short-circuit in MSL, which is harmless for two
        # pure predicate operands and keeps the boolean region a plain kernel.
        left, right = _binary_operands(op, arguments, input_types)
        operator_symbol = "&&" if op.op == "mul" else "||"
        return f"(({left}) {operator_symbol} ({right}))"
    if op.op == "heaviside":
        # ATen compares against zero first, so a NaN input takes the zero branch
        # and becomes 0 rather than the boundary value.
        value, boundary = _binary_operands(op, arguments, input_types)
        if output_dtype in _INTEGRAL_DTYPES:
            return f"(({value}) == 0 ? ({boundary}) : (({value}) > 0 ? 1 : 0))"
        return f"(({value}) == 0.0f ? ({boundary}) : (({value}) > 0.0f ? 1.0f : 0.0f))"
    if op.op == "logit":
        value = arguments[0]
        if "eps" in op.attrs:
            eps = float(op.attrs["eps"])
            lower = _msl_float_literal(eps)
            upper = _msl_float_literal(1.0 - eps)
            clamped = f"metal::min(metal::max({value}, {lower}), {upper})"
            return f"metal::log({clamped} / (1.0f - ({clamped})))"
        return f"metal::log(({value}) / (1.0f - ({value})))"
    if op.op == "erfc":
        return f"tsc_erfc({arguments[0]})"
    if op.op == "frac":
        value = arguments[0]
        return f"({value} - metal::trunc({value}))"
    if op.op == "sinc":
        return f"tsc_sinc({arguments[0]})"
    if op.op == "nan_to_num":
        value = arguments[0]
        if input_types[0].dtype not in _FLOATING_DTYPES:
            # ATen returns an integer tensor unchanged, and an elementwise
            # select over the same value is that copy.
            return value
        limit = _FINITE_LIMITS[input_types[0].dtype]
        nan_value = _msl_float_literal(float(op.attrs.get("nan", 0.0)))
        posinf_value = _msl_float_literal(float(op.attrs.get("posinf", limit)))
        neginf_value = _msl_float_literal(float(op.attrs.get("neginf", -limit)))
        return (
            f"(metal::isnan({value}) ? {nan_value} : "
            f"(metal::isinf({value}) ? "
            f"(({value}) > 0.0f ? {posinf_value} : {neginf_value}) : "
            f"({value})))"
        )
    if op.op == "lerp":
        if len(arguments) == 3:
            return f"tsc_lerp_f32({arguments[0]}, {arguments[1]}, {arguments[2]})"
        if len(arguments) != 2 or "weight" not in op.attrs:
            raise ValueError("malformed lerp op")
        weight = _msl_float_literal(float(op.attrs["weight"]))
        return f"tsc_lerp_f32({arguments[0]}, {arguments[1]}, {weight})"
    if op.op in {"addcmul", "addcdiv"}:
        if len(arguments) != 3:
            raise ValueError(f"{op.op} requires three tensor operands")
        value = _msl_float_literal(float(op.attrs.get("value", 1.0)))
        operator = "*" if op.op == "addcmul" else "/"
        return (
            f"{arguments[0]} + {value} * ({arguments[1]}) {operator} ({arguments[2]})"
        )
    if op.op in {"add", "sub", "rsub"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = "+" if op.op == "add" else "-"
        alpha = float(op.attrs.get("alpha", 1.0))
        return f"{left} {operator} {alpha!r} * {right}"
    if op.op == "neg":
        return f"-{arguments[0]}"
    if op.op == "abs":
        return f"metal::abs({arguments[0]})"
    if op.op == "sigmoid":
        return f"1.0f / (1.0f + metal::exp(-{arguments[0]}))"
    if op.op == "tanh":
        return f"metal::tanh({arguments[0]})"
    if op.op == "relu":
        return f"metal::max(0.0f, {arguments[0]})"
    if op.op == "leaky_relu":
        value = arguments[0]
        slope = _msl_float_literal(float(op.attrs.get("negative_slope", 0.01)))
        return f"(({value}) >= 0.0f ? ({value}) : {slope} * ({value}))"
    if op.op == "elu":
        value = arguments[0]
        alpha = _msl_float_literal(float(op.attrs.get("alpha", 1.0)))
        scale = float(op.attrs.get("scale", 1.0))
        input_scale = float(op.attrs.get("input_scale", 1.0))
        if scale == 1.0 and input_scale == 1.0:
            return (
                f"(({value}) >= 0.0f ? ({value}) : "
                f"{alpha} * (metal::exp({value}) - 1.0f))"
            )
        # ATen's signature is elu(x, alpha, scale, input_scale) and it scales
        # the negative branch by input_scale only, so the positive branch keeps
        # the raw value. selu is the scale=1.0507 spelling of this.
        scaled = (
            value
            if input_scale == 1.0
            else f"(({value}) * {_msl_float_literal(input_scale)})"
        )
        body = (
            f"metal::max(0.0f, {value}) + metal::min(0.0f, "
            f"{alpha} * (metal::exp({scaled}) - 1.0f))"
        )
        return f"(({body}) * {_msl_float_literal(scale)})" if scale != 1.0 else body
    if op.op == "hardswish":
        value = arguments[0]
        body = (
            f"({value}) * metal::min(metal::max(({value}) + 3.0f, 0.0f), 6.0f) / 6.0f"
        )
        return f"(metal::isnan({value}) ? NAN : {body})"
    if op.op == "hardsigmoid":
        value = arguments[0]
        body = f"metal::min(metal::max(({value}) + 3.0f, 0.0f), 6.0f) / 6.0f"
        return f"(metal::isnan({value}) ? NAN : {body})"
    if op.op == "threshold":
        value = arguments[0]
        threshold = _msl_float_literal(float(op.attrs["threshold"]))
        fill_value = _msl_float_literal(float(op.attrs["value"]))
        return (
            f"(metal::isnan({value}) ? {value} : "
            f"(({value}) > {threshold} ? ({value}) : {fill_value}))"
        )
    if op.op == "exp":
        return f"metal::exp({arguments[0]})"
    if op.op == "expm1":
        return f"tsc_expm1({arguments[0]})"
    if op.op == "exp2":
        return f"metal::exp2({arguments[0]})"
    if op.op == "log":
        return f"metal::log({arguments[0]})"
    if op.op == "log2":
        return f"metal::log2({arguments[0]})"
    if op.op == "log10":
        return f"metal::log10({arguments[0]})"
    if op.op == "log1p":
        return f"log1p({arguments[0]})"
    if op.op == "erf":
        return f"tsc_erf({arguments[0]})"
    if op.op == "lgamma":
        return f"tsc_lgamma({arguments[0]})"
    if op.op == "digamma":
        return f"tsc_digamma({arguments[0]})"
    if op.op == "i0":
        return f"tsc_i0({arguments[0]})"
    if op.op == "sin":
        return f"metal::sin({arguments[0]})"
    if op.op == "cos":
        return f"metal::cos({arguments[0]})"
    if op.op == "tan":
        return f"metal::tan({arguments[0]})"
    if op.op == "asin":
        return f"metal::asin({arguments[0]})"
    if op.op == "acos":
        return f"metal::acos({arguments[0]})"
    if op.op == "atan":
        return f"metal::atan({arguments[0]})"
    if op.op == "sinh":
        return f"metal::sinh({arguments[0]})"
    if op.op == "cosh":
        return f"metal::cosh({arguments[0]})"
    if op.op == "asinh":
        return f"metal::asinh({arguments[0]})"
    if op.op == "acosh":
        return f"metal::acosh({arguments[0]})"
    if op.op == "atanh":
        return f"metal::atanh({arguments[0]})"
    if op.op == "floor":
        return f"metal::floor({arguments[0]})"
    if op.op == "ceil":
        return f"metal::ceil({arguments[0]})"
    if op.op == "round":
        decimals = int(op.attrs.get("decimals", 0))
        if not decimals:
            return f"metal::rint({arguments[0]})"
        # ATen rescales, rounds to nearest-even on the scaled value, and
        # rescales back, which is the only way `decimals` can work without a
        # decimal floating-point type.
        scale = _msl_float_literal(10.0**decimals)
        value = arguments[0]
        return f"(metal::rint(({value}) * {scale}) / {scale})"
    if op.op == "trunc":
        return f"metal::trunc({arguments[0]})"
    if op.op == "sign":
        return f"metal::sign({arguments[0]})"
    if op.op == "square":
        value = arguments[0]
        return f"({value}) * ({value})"
    if op.op == "sqrt":
        return f"metal::sqrt({arguments[0]})"
    if op.op == "rsqrt":
        return f"metal::rsqrt({arguments[0]})"
    if op.op == "reciprocal":
        return f"1.0f / {arguments[0]}"
    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        lower = _clamp_bound_expression(op, arguments, "min", output_dtype)
        upper = _clamp_bound_expression(op, arguments, "max", output_dtype)
        if lower is not None and upper is not None:
            clipped = f"metal::min(metal::max({value}, {lower}), {upper})"
        elif lower is not None:
            clipped = f"metal::max({value}, {lower})"
        elif upper is not None:
            clipped = f"metal::min({value}, {upper})"
        else:
            return value
        invalid_inputs = [value]
        if lower is not None:
            invalid_inputs.append(lower)
        if upper is not None:
            invalid_inputs.append(upper)
        nan_check = " || ".join(
            f"metal::isnan({expression})" for expression in invalid_inputs
        )
        return f"(({nan_check}) ? NAN : {clipped})"
    raise NotImplementedError(f"cannot emit fused pointwise op: {op.op}")


def _emit_integer_pointwise_expression(
    op: Operation,
    arguments: list[str],
    input_types: tuple[TensorType, ...],
    output_dtype: DType,
) -> str:
    suffix = "i32" if output_dtype == DType.I32 else "i64"
    ctype = _pointwise_ctype(output_dtype)
    unsigned_ctype = "uint" if output_dtype == DType.I32 else "unsigned long"

    if op.op == "neg":
        return f"tsc_neg_{suffix}({arguments[0]})"
    if op.op == "abs":
        value = arguments[0]
        return f"(({value}) < 0 ? tsc_neg_{suffix}({value}) : ({value}))"
    if op.op == "sign":
        value = arguments[0]
        return f"(({value}) > 0 ? 1 : (({value}) < 0 ? -1 : 0))"
    if op.op == "square":
        value = arguments[0]
        return f"tsc_mul_{suffix}({value}, {value})"
    if op.op == "bitwise_not":
        value = arguments[0]
        return f"({ctype})(~({unsigned_ctype})({value}))"

    if op.op in {"add", "sub", "rsub", "mul", "pow"}:
        left, right = _binary_operands(op, arguments, input_types)
        if op.op == "mul":
            return f"tsc_mul_{suffix}({left}, {right})"
        if op.op == "pow":
            return f"tsc_pow_{suffix}({left}, {right})"
        alpha = op.attrs.get("alpha", 1)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(f"integer {op.op}.alpha must be an integer")
        if isinstance(alpha, float):
            if not isfinite(alpha) or not alpha.is_integer():
                raise ValueError(f"integer {op.op}.alpha must be an integer")
            alpha = int(alpha)
        if alpha != 1:
            alpha_literal = _msl_integer_literal(
                _wrap_integer_scalar(int(alpha), output_dtype), output_dtype
            )
            right = f"tsc_mul_{suffix}({right}, {alpha_literal})"
        function = "tsc_add" if op.op == "add" else "tsc_sub"
        return f"{function}_{suffix}({left}, {right})"

    if op.op in {"maximum", "minimum"}:
        left, right = _binary_operands(op, arguments, input_types)
        comparison = "<" if op.op == "maximum" else ">"
        return f"(({left}) {comparison} ({right}) ? ({right}) : ({left}))"

    if op.op in {"fmax", "fmin"}:
        # No NaN to ignore on the integer path, so these are max/min.
        left, right = _binary_operands(op, arguments, input_types)
        comparison = ">" if op.op == "fmax" else "<"
        return f"(({left}) {comparison} ({right}) ? ({left}) : ({right}))"

    if op.op in {"gcd", "lcm"}:
        left, right = _binary_operands(op, arguments, input_types)
        return f"tsc_{op.op}_{suffix}({left}, {right})"

    if op.op in {"bitwise_left_shift", "bitwise_right_shift"}:
        left, right = _binary_operands(op, arguments, input_types)
        function = "shl" if op.op == "bitwise_left_shift" else "shr"
        return f"tsc_{function}_{suffix}({left}, {right})"

    if op.op in {"floor_divide", "trunc_divide", "remainder"}:
        left, right = _binary_operands(op, arguments, input_types)
        function = {
            "floor_divide": "floor_div",
            "trunc_divide": "trunc_div",
            "remainder": "remainder",
        }[op.op]
        return f"tsc_{function}_{suffix}({left}, {right})"

    if op.op == "heaviside":
        # ATen compares against zero before the sign, so the boundary value only
        # applies to an exact zero.
        value, boundary = _binary_operands(op, arguments, input_types)
        return f"(({value}) == 0 ? ({boundary}) : (({value}) > 0 ? 1 : 0))"

    if op.op in {"bitwise_and", "bitwise_or", "bitwise_xor"}:
        left, right = _binary_operands(op, arguments, input_types)
        operator = {
            "bitwise_and": "&",
            "bitwise_or": "|",
            "bitwise_xor": "^",
        }[op.op]
        return (
            f"({ctype})(({unsigned_ctype})({left}) {operator} "
            f"({unsigned_ctype})({right}))"
        )

    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        lower = _clamp_bound_expression(op, arguments, "min", output_dtype)
        upper = _clamp_bound_expression(op, arguments, "max", output_dtype)
        if lower is not None and upper is not None:
            return (
                f"(({value}) > ({upper}) ? ({upper}) : "
                f"(({value}) < ({lower}) ? ({lower}) : ({value})))"
            )
        if lower is not None:
            return f"(({value}) < ({lower}) ? ({lower}) : ({value}))"
        if upper is not None:
            return f"(({value}) > ({upper}) ? ({upper}) : ({value}))"
        return value

    raise NotImplementedError(f"cannot emit integer pointwise op: {op.op}")


def _clamp_bound_expression(
    op: Operation,
    arguments: list[str],
    name: str,
    output_dtype: DType,
) -> str | None:
    input_position = op.attrs.get(f"{name}_input")
    if input_position is not None:
        if (
            not isinstance(input_position, int)
            or isinstance(input_position, bool)
            or input_position < 1
            or input_position >= len(arguments)
        ):
            raise ValueError(f"clamp.{name}_input is malformed")
        return arguments[input_position]
    bound = op.attrs.get(name)
    if bound is None:
        return None
    if not isinstance(bound, (int, float)) or isinstance(bound, bool):
        raise ValueError(f"clamp.{name} must be a scalar number")
    if output_dtype in _INTEGRAL_DTYPES:
        return _msl_integer_literal(
            _checked_integer_scalar(bound, output_dtype), output_dtype
        )
    value = float(bound)
    if isnan(value):
        return "NAN"
    if isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)
