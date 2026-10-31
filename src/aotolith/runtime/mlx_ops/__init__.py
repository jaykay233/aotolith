"""The MLX fallback: one dispatch table over the operators MLX already has.

Not every IR operation gets a Metal kernel. Views, gathers, and the long tail of
pointwise operators are exactly what MLX expresses natively, and compiling them
would only be slower than calling the framework. ``evaluate`` is the single
entry point: given an operation and its already-bound arguments, return the
array it produces.

The families are split by what the call looks like - a view changes metadata, a
pointwise call maps one op over an array, a scatter has to write through
``array.at`` - because that is also how they fail, and a failure should name the
family it happened in.
"""

from __future__ import annotations

import mlx.core as mx
from math import prod
from typing import Any
from ...ir import Graph
from ...ir import Operation
from ...codegen.dtypes import _MLX_DTYPES
from ...codegen.matmul import _matmul_epilogue_kind, _matmul_paired_weight_position
from ..bounds import (
    _check_advanced_index_indices,
    _check_take_indices,
    _mlx_take_checked,
)
from .attention import _evaluate_attention_mlx, _mlx_variance
from .layout import _MLX_LAYOUT_OPS, _evaluate_layout_mlx
from .matmul import _apply_mlx_matmul_epilogue, _apply_mlx_matmul_norm
from .pointwise import _evaluate_pointwise_mlx, _mlx_logaddexp
from .scatter import _mlx_scatter_reduce


def _mlx_cum_extremum(
    value: mx.array, axis: int, maximum: bool
) -> tuple[mx.array, mx.array]:
    """The running extremum of a scan and the position each step reports.

    MLX has ``cummax``/``cummin`` but returns only the values, so the position
    comes from the same loop rather than a second pass. ATen treats NaN as the
    largest value for ``cummax`` and the smallest for ``cummin``, which makes a
    NaN own the rest of the scan once it appears; the key folds that in while
    the reported value stays the element that was read. A repeated extremum
    reports the latest position, so the comparison is inclusive.
    """

    length = value.shape[axis]
    if length == 0:
        empty = mx.zeros_like(value)
        return empty, empty.astype(mx.int32)
    best = mx.take(value, 0, axis=axis)
    shape = best.shape
    best_index = mx.zeros(shape, dtype=mx.int32)
    floating = value.dtype in (mx.float16, mx.bfloat16, mx.float32, mx.float64)
    limit = float("inf") if maximum else float("-inf")
    best_key = mx.where(mx.isnan(best), limit, best) if floating else best
    values = [best]
    indices = [best_index]
    for position in range(1, length):
        current = mx.take(value, position, axis=axis)
        key = mx.where(mx.isnan(current), limit, current) if floating else current
        better = key >= best_key if maximum else key <= best_key
        best = mx.where(better, current, best)
        best_key = mx.where(better, key, best_key)
        best_index = mx.where(
            better, mx.full(shape, position, dtype=mx.int32), best_index
        )
        values.append(best)
        indices.append(best_index)
    return mx.stack(values, axis=axis), mx.stack(indices, axis=axis)


def _evaluate_mlx_op(
    graph: Graph, op: Operation, env: dict[str, mx.array]
) -> mx.array | tuple[mx.array, ...]:
    """Evaluate one IR operation with MLX fallback semantics.

    A single-output operation returns its array. A multi-output operation
    returns one array per declared output, in declaration order.
    """

    if op.op == "fused_elementwise":
        body = op.attrs.get("body")
        if not isinstance(body, tuple) or not all(
            isinstance(item, Operation) for item in body
        ):
            raise ValueError("fused_elementwise body must be a tuple of operations")
        body_env = dict(env)
        for item in body:
            body_env[item.outputs[0]] = _evaluate_mlx_op(graph, item, body_env)
        return body_env[op.outputs[0]]
    if op.op == "attention":
        return _evaluate_attention_mlx(op, env)
    if op.op == "rope":
        value, cosine, sine = (env[value_id] for value_id in op.inputs)
        half = value.shape[-1] // 2
        rotated = mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)
        return value * cosine + rotated * sine
    if op.op == "embedding":
        weight, indices = (env[value_id] for value_id in op.inputs)
        return _mlx_take_checked(weight, indices, axis=0)
    if op.op == "index_select":
        source, indices = (env[value_id] for value_id in op.inputs)
        return _mlx_take_checked(source, indices, axis=int(op.attrs["dim"]))
    if op.op == "index":
        source = env[op.inputs[0]]
        indices = tuple(env[value_id] for value_id in op.inputs[1:])
        axes = tuple(int(axis) for axis in op.attrs["axes"])
        for axis, index in zip(axes, indices, strict=True):
            _check_advanced_index_indices(index, source.shape[axis])
        indexing = (slice(None),) * axes[0] + indices
        return source[indexing]
    if op.op == "gather":
        source, indices = (env[value_id] for value_id in op.inputs)
        axis = int(op.attrs["dim"])
        _check_take_indices(indices, source.shape[axis])
        source_slices = [
            slice(0, indices.shape[dim]) if dim != axis else slice(None)
            for dim in range(len(source.shape))
        ]
        return mx.take_along_axis(source[tuple(source_slices)], indices, axis=axis)
    if op.op == "scatter":
        source, indices = (env[value_id] for value_id in op.inputs[:2])
        axis = int(op.attrs["dim"])
        _check_take_indices(indices, source.shape[axis])
        values = (
            mx.array(op.attrs["value"], dtype=source.dtype)
            if len(op.inputs) == 2
            else env[op.inputs[2]]
        )
        if len(op.inputs) == 3:
            values = values[tuple(slice(0, size) for size in indices.shape)]
        reduce = op.attrs.get("reduce")
        if reduce is not None:
            return _mlx_scatter_reduce(
                source,
                indices,
                values,
                axis,
                str(reduce),
                bool(op.attrs.get("include_self", True)),
            )
        source_slices = [
            slice(0, indices.shape[dim]) if dim != axis else slice(None)
            for dim in range(len(source.shape))
        ]
        slices = tuple(source_slices)
        updated = mx.put_along_axis(source[slices], indices, values, axis=axis)
        output = mx.array(source)
        output[slices] = updated
        return output
    if op.op == "index_add":
        source, indices, values = (env[value_id] for value_id in op.inputs)
        axis = int(op.attrs["dim"])
        alpha = float(op.attrs.get("alpha", 1.0))
        _check_take_indices(indices, source.shape[axis])
        selection: tuple[Any, ...] = (slice(None),) * axis + (indices,)
        contribution = values if alpha == 1.0 else values * alpha
        return source.at[selection].add(contribution)
    if op.op in {"prod", "amax", "amin"}:
        value = env[op.inputs[0]]
        dims = tuple(int(dim) % value.ndim for dim in op.attrs["dims"])
        keepdim = bool(op.attrs.get("keepdim", False))
        operation = {"prod": mx.prod, "amax": mx.max, "amin": mx.min}[op.op]
        reduced = operation(value, axis=dims, keepdims=keepdim)
        output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        return reduced.astype(output_dtype)
    if op.op in {"median", "median_indices"}:
        # MLX sorts along a single axis, so the reduced axes move to the back
        # and collapse into one, the same transformation ``vector_norm`` makes.
        value = env[op.inputs[0]]
        rank = value.ndim
        dims = tuple(int(dim) % rank for dim in op.attrs["dims"])
        keepdim = bool(op.attrs.get("keepdim", False))
        ignore_nan = bool(op.attrs["ignore_nan"])
        remaining = [axis for axis in range(rank) if axis not in dims]
        permuted = mx.transpose(value, remaining + list(dims))
        prefix = tuple(value.shape[axis] for axis in remaining)
        length = prod(value.shape[axis] for axis in dims)
        flat = permuted.reshape(prefix + (length,))
        # ATen returns the lower median: the element at (n - 1) / 2 of the
        # ascending order. ``nanmedian`` counts only the present elements and
        # sorts the NaNs past them, so the position is per row.
        nan_mask = mx.isnan(flat)
        present = ~nan_mask
        counted = mx.sum(present, axis=-1)
        ranked = mx.where(present, flat, float("inf")) if ignore_nan else flat
        order = mx.argsort(ranked, axis=-1)
        if ignore_nan:
            position = mx.maximum(counted - 1, 0) // 2
            selection = position[..., None]
        else:
            position = (length - 1) // 2
            selection = mx.full(prefix + (1,), position, dtype=mx.int32)
        index = mx.take_along_axis(order, selection, axis=-1)[..., 0]
        if op.op == "median_indices":
            picked = index
            if not ignore_nan:
                # ``median`` propagates NaN, and ATen names the first one as
                # the position of the median it reported.
                poisoned = mx.any(nan_mask, axis=-1)
                first_nan = mx.argmax(nan_mask.astype(mx.int32), axis=-1)
                picked = mx.where(poisoned, first_nan, picked)
        else:
            picked = mx.take_along_axis(ranked, index[..., None], axis=-1)[..., 0]
            if ignore_nan:
                # A row that is all NaN has no present element to report, and
                # ATen answers NaN rather than the +inf placeholder.
                chosen = mx.take_along_axis(present, index[..., None], axis=-1)[..., 0]
                picked = mx.where(chosen, picked, float("nan"))
            else:
                poisoned = mx.any(nan_mask, axis=-1)
                picked = mx.where(poisoned, float("nan"), picked)
        if keepdim:
            picked = picked.reshape(
                tuple(
                    1 if axis in dims else size for axis, size in enumerate(value.shape)
                )
            )
        output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        return picked.astype(output_dtype)
    if op.op == "log_softmax":
        value = env[op.inputs[0]]
        axis = int(op.attrs["dim"]) % value.ndim
        shifted = value - mx.logsumexp(value, axis=axis, keepdims=True)
        return shifted.astype(_MLX_DTYPES[graph.value(op.outputs[0]).type.dtype])
    if op.op == "vector_norm":
        value = env[op.inputs[0]]
        rank = value.ndim
        dims = tuple(int(dim) % rank for dim in op.attrs["dims"])
        keepdim = bool(op.attrs.get("keepdim", False))
        ord_value = float(op.attrs["ord"])
        output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        compute = (
            value
            if value.dtype in (mx.float32, mx.float64)
            else value.astype(mx.float32)
        )
        # MLX only takes a single norm axis, so the reduced axes move to the
        # back and collapse into one before the call.
        remaining = [axis for axis in range(rank) if axis not in dims]
        permuted = mx.transpose(compute, remaining + list(dims))
        prefix = tuple(compute.shape[axis] for axis in remaining)
        flattened = permuted.reshape(prefix + (-1,))
        reduced = mx.linalg.norm(flattened, ord=ord_value, axis=-1)
        if keepdim:
            reduced = reduced.reshape(
                tuple(
                    1 if axis in dims else size
                    for axis, size in enumerate(compute.shape)
                )
            )
        return reduced.astype(output_dtype)
    if op.op in {"var", "std"}:
        value = env[op.inputs[0]]
        dims = tuple(int(dim) % value.ndim for dim in op.attrs["dims"])
        keepdim = bool(op.attrs.get("keepdim", False))
        correction = op.attrs.get("correction", 1)
        output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        # PyTorch accumulates the variance in float32 for half inputs, so a
        # float32 pass is what keeps the two results within a few ulps.
        compute = (
            value
            if value.dtype in (mx.float32, mx.float64)
            else value.astype(mx.float32)
        )
        variance = _mlx_variance(compute, dims, keepdim, correction)
        result = variance if op.op == "var" else mx.sqrt(variance)
        return result.astype(output_dtype)
    if op.op in {"argmax", "argmin"}:
        value = env[op.inputs[0]]
        operation = mx.argmax if op.op == "argmax" else mx.argmin
        dim = op.attrs.get("dim")
        if dim is None:
            reduced = operation(value.reshape(-1))
        else:
            reduced = operation(
                value,
                axis=int(dim) % value.ndim,
                keepdims=bool(op.attrs.get("keepdim", False)),
            )
        return reduced.astype(_MLX_DTYPES[graph.value(op.outputs[0]).type.dtype])
    if op.op in {"sort", "argsort"}:
        value = env[op.inputs[0]]
        axis = int(op.attrs["dim"]) % value.ndim
        operation = mx.sort if op.op == "sort" else mx.argsort
        ordered = operation(value, axis=axis)
        if bool(op.attrs.get("descending", False)):
            ordered = mx.flip(ordered, axis=axis)
        return ordered.astype(_MLX_DTYPES[graph.value(op.outputs[0]).type.dtype])
    if op.op in {"topk", "topk_indices"}:
        value = env[op.inputs[0]]
        axis = int(op.attrs["dim"]) % value.ndim
        k = int(op.attrs["k"])
        largest = bool(op.attrs.get("largest", True))
        if op.op == "topk":
            if k == value.shape[axis]:
                # MLX's select-based topk does not order a full-length window,
                # so a full selection falls back to a sort.
                ordered = mx.argsort(value, axis=axis)
                if largest:
                    ordered = mx.flip(ordered, axis=axis)
                selected = mx.take_along_axis(value, ordered, axis=axis)
            else:
                # MLX returns the selected window in ascending order, so both
                # directions reverse it to match torch.topk's sorted output.
                raw = mx.topk(value if largest else -value, k, axis=axis)
                selected = mx.flip(raw, axis=axis)
                if not largest:
                    selected = -selected
        else:
            indices = mx.argsort(value, axis=axis)
            selection: list[slice] = [slice(None)] * value.ndim
            if largest:
                selection[axis] = slice(value.shape[axis] - k, value.shape[axis])
                indices = mx.flip(indices[tuple(selection)], axis=axis)
            else:
                selection[axis] = slice(0, k)
                indices = indices[tuple(selection)]
            selected = indices
        return selected.astype(_MLX_DTYPES[graph.value(op.outputs[0]).type.dtype])
    if op.op in {"cumsum", "cumprod"}:
        axis = int(op.attrs["dim"]) % len(env[op.inputs[0]].shape)
        scan = mx.cumsum if op.op == "cumsum" else mx.cumprod
        value = scan(env[op.inputs[0]], axis=axis)
        output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        return value.astype(output_dtype)
    if op.op in {"cummax", "cummin", "cummax_indices", "cummin_indices"}:
        value = env[op.inputs[0]]
        axis = int(op.attrs["dim"]) % len(value.shape)
        values, indices = _mlx_cum_extremum(value, axis, op.op.startswith("cummax"))
        picked = indices if op.op.endswith("_indices") else values
        return picked.astype(_MLX_DTYPES[graph.value(op.outputs[0]).type.dtype])
    if op.op == "logcumsumexp":
        # The scan in the log domain: each step is ``logaddexp`` of the running
        # total and the next element, which is stable by construction (the shift
        # keeps every exponent at or below zero) and returns -inf, +inf and NaN
        # the way ATen's kernels do instead of turning -inf - -inf into NaN.
        value = env[op.inputs[0]]
        axis = int(op.attrs["dim"]) % len(value.shape)
        compute = value if value.dtype == mx.float32 else value.astype(mx.float32)
        scanned = [mx.take(compute, 0, axis=axis)]
        for position in range(1, compute.shape[axis]):
            current = mx.take(compute, position, axis=axis)
            scanned.append(_mlx_logaddexp(scanned[-1], current))
        return mx.stack(scanned, axis=axis).astype(
            _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        )
    if op.op == "cast":
        return env[op.inputs[0]].astype(
            _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
        )
    if op.op in {"triu", "tril"}:
        operation = mx.triu if op.op == "triu" else mx.tril
        return operation(env[op.inputs[0]], int(op.attrs["diagonal"]))
    if op.op == "cat":
        rank = len(env[op.inputs[0]].shape)
        axis = int(op.attrs["dim"]) % rank
        return mx.concatenate([env[value_id] for value_id in op.inputs], axis=axis)
    if op.op in _MLX_LAYOUT_OPS:
        return _evaluate_layout_mlx(graph, op, env[op.inputs[0]])
    if op.op == "matmul":
        left, right = (env[value_id] for value_id in op.inputs[:2])
        left = _apply_mlx_matmul_norm(op, env, left)
        if op.attrs.get("transpose_a", False):
            left = mx.swapaxes(left, -1, -2)
        if op.attrs.get("transpose_b", False):
            right = mx.swapaxes(right, -1, -2)
        product = mx.matmul(left, right)
        if _matmul_epilogue_kind(op) == "silu_mul_paired":
            paired_position = _matmul_paired_weight_position(op, 1)
            assert paired_position is not None
            paired = env[op.inputs[paired_position]]
            if op.attrs.get("transpose_b", False):
                paired = mx.swapaxes(paired, -1, -2)
            return product * (mx.matmul(left, paired) / (1.0 + mx.exp(-product)))
        return _apply_mlx_matmul_epilogue(op, env, product)
    if op.op == "addmm":
        bias, left, right = (env[value_id] for value_id in op.inputs[:3])
        left = _apply_mlx_matmul_norm(op, env, left)
        if op.attrs.get("transpose_a", False):
            left = mx.swapaxes(left, -1, -2)
        if op.attrs.get("transpose_b", False):
            right = mx.swapaxes(right, -1, -2)
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        return _apply_mlx_matmul_epilogue(
            op, env, beta * bias + alpha * mx.matmul(left, right)
        )
    return _evaluate_pointwise_mlx(op, env, graph.value(op.outputs[0]).type)
