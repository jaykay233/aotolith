"""The generic path: what an ATen call becomes when no fused pattern claims it.

This is the largest family by operator count and the smallest by code, because
the work is mechanical - map positionally, promote the operands to a common
dtype, attach the attributes ATen left in ``kwargs``. The promotion is here
rather than in codegen because PyTorch's type promotion is a property of the
operation, not of Metal."""

from __future__ import annotations

from math import isfinite
from typing import Any
from ..ir import DType, Operation, TensorType, Value
from .fx import _fold_empty_cat_operands, _node_type
from .state import Lowering
from .state import Handler
from .indexing import _validate_inplace_legalization
from .padding import _constant_pad_arguments
from .refs import _argument_ids
from .targets import (
    _BINARY_OPS,
    _BITWISE_BINARY_OPS,
    _COMPARISON_OPS,
    _COPY_TARGET,
    _DIVIDE_ROUNDING_TARGETS,
    _EYE_TARGETS,
    _FLOATING_DTYPES,
    _FLOAT_ONLY_POINTWISE_OPS,
    _INPLACE_TARGETS,
    _INTEGRAL_DTYPES,
    _INTEGRAL_ONLY_POINTWISE_OPS,
    _LDEXP_TARGET,
    _LOGICAL_BINARY_OPS,
    _NUMERIC_ATTRS,
    _PRELU_TARGET,
    _PROMOTABLE_POINTWISE_OPS,
    _REDUCTION_OPS,
    _SUPPORTED_OPS,
    _TORCH_DTYPES,
)


def _promoted_pointwise_dtype(
    op_name: str,
    dtypes: set[DType],
    output_dtype: DType,
    *,
    float_scalar: bool = False,
) -> DType | None:
    """Return the dtype PyTorch would apply operands of ``dtypes`` in."""

    if not dtypes <= _INTEGRAL_DTYPES | _FLOATING_DTYPES:
        # Boolean operands keep their own rules; nothing to promote here.
        return None
    if op_name == "div" and dtypes <= _INTEGRAL_DTYPES:
        # True division always produces a float, even for integer operands.
        return output_dtype if output_dtype in _FLOATING_DTYPES else None
    if float_scalar and dtypes <= _INTEGRAL_DTYPES:
        # A Python float is a higher promotion category than an integer
        # tensor, so ``int_tensor * 0.5`` is a float32 computation. The
        # tensor's own dtype wins when the categories match, which is why
        # ``half_tensor * 0.5`` stays half and needs no cast.
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
    input_count: int,
    *,
    float_scalar: bool = False,
) -> tuple[str, ...]:
    """Insert casts so a pointwise op sees the operands PyTorch would."""

    dtypes = {values[input_id].type.dtype for input_id in inputs}
    target = _promoted_pointwise_dtype(
        op_name, dtypes, output_dtype, float_scalar=float_scalar
    )
    if target is None:
        return inputs
    promoted: list[str] = []
    for input_id in inputs:
        value = values[input_id]
        if value.type.dtype == target:
            promoted.append(input_id)
            continue
        cast_id = f"v{len(values) - input_count}"
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
        if dim is not None and (not isinstance(dim, int) or isinstance(dim, bool)):
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
            raise ValueError(f"arange expects one to three arguments, got {len(args)}")
        start = 0 if len(args) == 1 else args[0]
        end = args[0] if len(args) == 1 else args[1]
        step = 1 if len(args) < 3 else args[2]
        for name, value in (("start", start), ("end", end), ("step", step)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"arange.{name} must be a scalar number, got {value!r}"
                )
            if not isfinite(value):
                raise ValueError(f"arange.{name} must be finite, got {value!r}")
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
        value = args[1] if op == "full_like" else (0.0 if op == "zeros_like" else 1.0)
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
        if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta <= 0:
            raise ValueError(f"softplus.beta must be positive, got {beta!r}")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
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
            raise ValueError(f"{op}.half_to_float must be a boolean, got {args[2]!r}")
        return ((_argument_ids(input_id, node_ids),), {"dim": dim})

    if op in {"cumsum", "cumprod", "logcumsumexp"}:
        arities = {2} if op == "logcumsumexp" else {2, 3}
        if len(args) not in arities:
            raise ValueError(f"{op} received unexpected arguments: {len(args)}")
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
        # An empty shape is the rank-0 reshape of a single element, which is
        # how ``einsum`` spells a full contraction; the contract then checks
        # that the input really has one element.
        if not isinstance(shape, (tuple, list)) or not all(
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
            isinstance(dim, int) and not isinstance(dim, bool) for dim in (dim0, dim1)
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
        if (
            not isinstance(dims, (tuple, list))
            or not dims
            or not all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
            )
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
        if (
            not isinstance(shape, (tuple, list))
            or not shape
            or not all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
            )
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
        if end is not None and (not isinstance(end, int) or isinstance(end, bool)):
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
        if (
            not isinstance(dims, (tuple, list))
            or not dims
            or not all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
            )
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
        elif (
            isinstance(shifts, (tuple, list))
            and shifts
            and all(
                isinstance(shift, int) and not isinstance(shift, bool)
                for shift in shifts
            )
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
        elif (
            isinstance(dims, (tuple, list))
            and dims
            and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims)
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

    if op == "round":
        if len(args) not in {1, 2}:
            raise ValueError(f"round expects one or two arguments, got {len(args)}")
        decimals = args[1] if len(args) == 2 else kwargs.get("decimals", 0)
        if not isinstance(decimals, int) or isinstance(decimals, bool):
            raise ValueError(f"round.decimals must be an integer, got {decimals!r}")
        attrs: dict[str, object] = {}
        if decimals:
            attrs["decimals"] = decimals
        return ((_argument_ids(args[0], node_ids),), attrs)

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
                raise ValueError(f"nan_to_num.{name} must be a scalar, got {value!r}")
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
                raise ValueError(f"unsupported lerp weight: {weight!r}") from None
            return inputs, {"weight": float(weight)}
        return (*inputs, weight_id), {}

    if op in {"addcmul", "addcdiv"}:
        # ATen's signature is (self, tensor1, tensor2, *, value=1).
        if len(args) not in {3, 4}:
            raise ValueError(
                f"{op} expects three or four positional arguments, got {len(args)}"
            )
        inputs = tuple(_argument_ids(argument, node_ids) for argument in args[:3])
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
                raise ValueError(f"unsupported {op} {name}: {bound!r}") from None
            attrs[name] = bound
        else:
            attrs[f"{name}_input"] = len(inputs)
            inputs.append(bound_id)
        return tuple(inputs), attrs

    if op in {"leaky_relu", "elu"}:
        names = (
            ["negative_slope"]
            if op == "leaky_relu"
            else [
                "alpha",
                "scale",
                "input_scale",
            ]
        )
        if len(args) > 1 + len(names):
            raise ValueError(
                f"{op} expects at most {1 + len(names)} arguments, got {len(args)}"
            )
        attrs: dict[str, object] = {}
        # The functional spellings pass these positionally, while the in-place
        # kernels keep the tuning constants as keywords, so both are read here.
        arguments = dict(zip(names, args[1:]))
        for name in names[len(arguments) :]:
            if name in kwargs:
                arguments[name] = kwargs[name]
        for name in names:
            if name not in arguments:
                continue
            argument = arguments[name]
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
        if dims is None:
            # ATen's decompositions spell a full reduction as ``dim=None``
            # rather than the one-argument ``sum.default``.
            normalized_dims = tuple(range(len(_node_type(input_id).shape)))
        elif isinstance(dims, int) and not isinstance(dims, bool):
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
            if not isinstance(argument, (int, float)) or isinstance(argument, bool):
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
                "scalar_position": ("lhs" if position == 0 or op == "rsub" else "rhs"),
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
    if op in {"cumsum", "cumprod", "logcumsumexp"}:
        allowed.update({"dim", "dtype"})
    if op == "flip":
        allowed.add("dims")
    if op == "pad":
        allowed.update({"value", "mode"})
    if op == "nan_to_num":
        allowed.update({"nan", "posinf", "neginf"})
    if op == "round":
        allowed.add("decimals")
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
                f"gelu.approximate must be 'none' or 'tanh', got {approximate!r}"
            )
        attrs["approximate"] = approximate
    return attrs


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _rounding_mode_operation(node: Any) -> str:
    """The IR op an ``aten.div`` spelling with a ``rounding_mode`` means.

    ATen folds the rounding mode into the target name rather than passing it as
    an attribute, so the same node can be three different IR operators: ``None``
    is true division, ``"floor"`` rounds towards negative infinity (the
    ``floor_divide`` operator, matching ``aten.floor_divide``), and ``"trunc"``
    rounds towards zero.
    """

    rounding_mode = node.kwargs.get("rounding_mode")
    if rounding_mode is None:
        return "div"
    if rounding_mode == "floor":
        return "floor_divide"
    if rounding_mode == "trunc":
        return "trunc_divide"
    raise ValueError(
        f"div.rounding_mode must be None, 'floor' or 'trunc', got {rounding_mode!r}"
    )


def lower_supported(lowering: Lowering, node: Any, target: str) -> bool:
    """Lower a node through the positional ATen -> core-op mapping.

    This is the path every operator takes unless a fused pattern claimed it
    first, so it is also where the three ATen spellings that do not map
    positionally are handled:

    * ``masked_fill`` masks its *first* argument in place, which is exactly
      ``where(mask, value, input)`` in that order;
    * a ``cat`` of zero-length constants loses those operands, and a ``cat``
      left with one operand is that operand, so the operation disappears;
    * ``make_fx`` lowers ``torch.square`` on integers to ``pow(x, 2)``, which
      is a multiplication the integer path cannot express as a power;
    * a ``div`` with a ``rounding_mode`` keyword has three different meanings,
      so the keyword picks between ``div``, ``floor_divide`` and
      ``trunc_divide`` instead of the target naming one.
    """

    if target not in _SUPPORTED_OPS and target not in _DIVIDE_ROUNDING_TARGETS:
        return False
    if target in _INPLACE_TARGETS or target in _DIVIDE_ROUNDING_TARGETS:
        _validate_inplace_legalization(node)
    op_name = _SUPPORTED_OPS.get(target, "div")
    call_kwargs = node.kwargs
    if target in _DIVIDE_ROUNDING_TARGETS:
        op_name = _rounding_mode_operation(node)
        call_kwargs = {
            name: value
            for name, value in node.kwargs.items()
            if name != "rounding_mode"
        }
    call_args = node.args
    if target == "aten.masked_fill.Tensor":
        masked, mask, fill = call_args
        op_name = "where"
        call_args = (mask, fill, masked)
    if target == "aten.cat.default" and lowering.empty_nodes:
        call_args = _fold_empty_cat_operands(
            node, lowering.empty_nodes, lowering.node_ids
        )
        if call_args is None:
            return True
    if (
        target == "aten.pow.Tensor_Scalar"
        and len(call_args) == 2
        and isinstance(call_args[1], int)
        and not isinstance(call_args[1], bool)
        and call_args[1] == 2
        and _node_type(node).dtype in {DType.I32, DType.I64}
    ):
        op_name = "square"
        inputs = (_argument_ids(call_args[0], lowering.node_ids),)
        scalar_attrs: dict[str, object] = {}
    else:
        inputs, scalar_attrs = _operation_inputs(
            op_name, call_args, lowering.node_ids, call_kwargs
        )
    attrs = _operation_attrs(op_name, call_kwargs)
    attrs.update(scalar_attrs)
    if op_name in _INTEGRAL_ONLY_POINTWISE_OPS:
        for input_id in inputs:
            if lowering.values[input_id].type.dtype not in _INTEGRAL_DTYPES:
                raise ValueError(f"{op_name} requires integral operands")
    # A Python float operand is a promotion category of its own: it lifts an
    # integer tensor to the default float dtype, which a table keyed only on the
    # tensor dtypes cannot see.
    float_scalar = isinstance(scalar_attrs.get("scalar"), float)
    if op_name == "where" and len(inputs) == 3:
        inputs = (inputs[0],) + _promote_pointwise_operands(
            "where",
            inputs[1:],
            lowering.values,
            lowering.ops,
            _node_type(node).dtype,
            len(lowering.input_ids),
        )
    elif op_name in _PROMOTABLE_POINTWISE_OPS or (op_name in _FLOAT_ONLY_POINTWISE_OPS):
        inputs = _promote_pointwise_operands(
            op_name,
            inputs,
            lowering.values,
            lowering.ops,
            _node_type(node).dtype,
            len(lowering.input_ids),
            float_scalar=float_scalar,
        )
    if op_name in _REDUCTION_OPS and "dims" not in attrs:
        input_type = lowering.values[inputs[0]].type
        attrs["dims"] = tuple(range(len(input_type.shape)))
        attrs["keepdim"] = False
    lowering.bind(node, lowering.emit(op_name, inputs, attrs, _node_type(node)))
    return True


def _lower_prelu(lowering: Lowering, node: Any, target: str) -> bool:
    """PReLU is a slope on the negative half: ``where(x > 0, x, x * weight)``.

    The weight arrives already reshaped by the caller's ``prelu`` (a scalar, or
    ``1``-padded around the channel axis), so it broadcasts like any operand.
    Written as three operators rather than one because that is what fuses: the
    selection joins the surrounding elementwise region.
    """

    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"{target} expects two arguments, got {len(args)}")
    value, weight = args
    output_type = _node_type(node)
    value_id = _argument_ids(value, lowering.node_ids)
    weight_id = _argument_ids(weight, lowering.node_ids)
    zero_id = lowering.emit(
        "scalar_tensor", (), {"value": 0}, TensorType((), output_type.dtype)
    )
    positive_id = lowering.emit(
        "gt", (value_id, zero_id), {}, TensorType(output_type.shape, DType.BOOL)
    )
    scaled_id = lowering.emit("mul", (value_id, weight_id), {}, output_type)
    result_id = lowering.emit(
        "where", (positive_id, value_id, scaled_id), {}, output_type
    )
    lowering.bind(node, result_id)
    return True


def _lower_copy(lowering: Lowering, node: Any, target: str) -> bool:
    """``copy_`` is the destination's value becoming the source's.

    Torch casts and broadcasts the source into the destination, so the lowering
    is those two operations and nothing else. Readers of the destination that
    precede the write already saw the old value, and the tracer rebinds the
    mutated name afterwards, so rebinding this node to the copied value is the
    whole write.
    """

    _validate_inplace_legalization(node)
    args = getattr(node, "args", ())
    if len(args) < 2:
        raise ValueError(f"{target} expects two arguments, got {len(args)}")
    destination_id = _argument_ids(args[0], lowering.node_ids)
    source_id = _argument_ids(args[1], lowering.node_ids)
    destination = lowering.values[destination_id].type
    source = lowering.values[source_id].type
    if source.dtype != destination.dtype:
        source_id = lowering.emit(
            "cast",
            (source_id,),
            {},
            TensorType(source.shape, destination.dtype),
        )
        source = TensorType(source.shape, destination.dtype)
    if source.shape != destination.shape:
        source_id = lowering.emit(
            "expand",
            (source_id,),
            {"shape": destination.shape},
            TensorType(destination.shape, destination.dtype),
        )
    lowering.bind(node, source_id)
    return True


def _lower_eye(lowering: Lowering, node: Any, target: str) -> bool:
    """``eye`` is the equality of two ranges, and nothing to launch.

    Torch's own reference implementation of ``eye`` is exactly that expression,
    typed with the narrowest integer that holds the larger side - ``uint8``
    below 256 - to keep the ranges small. This IR has no unsigned type, so the
    same expression is built over I32; the comparison is over integers either
    way and the cast to the declared dtype is what Torch does with it too.
    """

    args = getattr(node, "args", ())
    if not 1 <= len(args) <= 2:
        raise ValueError(f"{target} expects one or two arguments, got {len(args)}")
    rows = args[0]
    columns = args[1] if len(args) == 2 else rows
    for name, value in (("n", rows), ("m", columns)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{target}.{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(
                f"{target}.{name} must be positive, got {value!r}: the static IR "
                "has no zero-sized dimensions"
            )

    output_type = _node_type(node)
    row_id = lowering.emit(
        "arange",
        (),
        {"start": 0, "end": rows, "step": 1},
        TensorType((rows,), DType.I32),
    )
    column_id = lowering.emit(
        "arange",
        (),
        {"start": 0, "end": columns, "step": 1},
        TensorType((columns,), DType.I32),
    )
    rows_id = lowering.emit(
        "unsqueeze", (row_id,), {"dim": 1}, TensorType((rows, 1), DType.I32)
    )
    diagonal_id = lowering.emit(
        "eq", (rows_id, column_id), {}, TensorType((rows, columns), DType.BOOL)
    )
    lowering.bind(node, lowering.emit("cast", (diagonal_id,), {}, output_type))
    return True


def _lower_ldexp(lowering: Lowering, node: Any, target: str) -> bool:
    """``ldexp(x, e)`` is ``x * 2**e``, so the exponent becomes a scale.

    The exponent is cast to the result dtype before the power: ``2.0 ** k`` is
    exact for every integer ``k`` a float can represent, and the exponent is an
    integer tensor by definition, so the cast loses nothing and keeps the power
    a float one.
    """

    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"{target} expects two arguments, got {len(args)}")
    value_id = _argument_ids(args[0], lowering.node_ids)
    exponent_id = _argument_ids(args[1], lowering.node_ids)
    output_type = _node_type(node)
    exponent_type = lowering.values[exponent_id].type
    if exponent_type.dtype != output_type.dtype:
        exponent_id = lowering.emit(
            "cast",
            (exponent_id,),
            {},
            TensorType(exponent_type.shape, output_type.dtype),
        )
        exponent_type = TensorType(exponent_type.shape, output_type.dtype)
    two_id = lowering.emit(
        "scalar_tensor", (), {"value": 2.0}, TensorType((), output_type.dtype)
    )
    scale_id = lowering.emit("pow", (two_id, exponent_id), {}, exponent_type)
    lowering.bind(node, lowering.emit("mul", (value_id, scale_id), {}, output_type))
    return True


# The generic path has no target of its own: it is the fallback the loop calls
# when no family claimed the node.
HANDLERS: dict[str, tuple[Handler, ...]] = {
    _PRELU_TARGET: (_lower_prelu,),
    _COPY_TARGET: (_lower_copy,),
    **{target: (_lower_eye,) for target in _EYE_TARGETS},
    _LDEXP_TARGET: (_lower_ldexp,),
}
