"""Attention and variance evaluation through MLX.

The reference path for both fused operators. It exists so a plan can run on a
machine that cannot compile the kernel, and so the kernel can be checked against
it.
"""

from __future__ import annotations

import mlx.core as mx
from ...ir import Operation
from ...codegen.attention import _zero_fully_masked_rows


def _mlx_variance(
    value: mx.array,
    dims: tuple[int, ...],
    keepdim: bool,
    correction: int | float,
) -> mx.array:
    """Return the corrected sample variance, integral or fractional correction.

    MLX's ``ddof`` takes an integer, so a fractional correction is applied the
    way ATen defines it: the biased variance times ``n / (n - correction)``.
    """

    if isinstance(correction, int):
        return mx.var(value, axis=dims, keepdims=keepdim, ddof=correction)
    count = 1
    for dim in dims:
        count *= value.shape[dim]
    scale = count / (count - correction)
    return mx.var(value, axis=dims, keepdims=keepdim) * scale


def _causal_mask(query: mx.array, key: mx.array) -> mx.array:
    """Torch's causal mask, aligned at the top left.

    MLX's own ``"causal"`` mask aligns the two sequences at the bottom right,
    which only agrees with Torch when the query and key lengths are equal. A
    shorter query attends to the first ``query_length`` keys under Torch's rule,
    so the mask is built explicitly as soon as the lengths differ.
    """

    rows = mx.arange(query.shape[-2])[:, None]
    columns = mx.arange(key.shape[-2])[None, :]
    additive = mx.array(0, dtype=query.dtype)
    blocked = mx.array(-float("inf"), dtype=query.dtype)
    return mx.where(rows >= columns, additive, blocked)


def _evaluate_attention_mlx(op: Operation, env: dict[str, mx.array]) -> mx.array:
    query, key, value = (env[value_id] for value_id in op.inputs[:3])
    explicit_mask = env[op.inputs[3]] if len(op.inputs) == 4 else None
    mask = explicit_mask
    if op.attrs.get("causal"):
        mask = (
            "causal" if query.shape[-2] == key.shape[-2] else _causal_mask(query, key)
        )
    scale = float(op.attrs["scale"])

    if query.ndim == 4:
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )
        if op.attrs.get("safe_softmax") and explicit_mask is not None:
            output = _zero_fully_masked_rows(explicit_mask, output)
        return output
    if query.ndim != 3:
        raise NotImplementedError(
            f"attention requires rank-3 or rank-4 inputs, got {query.shape}"
        )

    query = mx.expand_dims(query, axis=1)
    key = mx.expand_dims(key, axis=1)
    value = mx.expand_dims(value, axis=1)
    if mask is not None and mask.ndim == 3:
        mask = mx.expand_dims(mask, axis=1)
    output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask=mask
    )
    output = mx.squeeze(output, axis=1)
    if op.attrs.get("safe_softmax") and explicit_mask is not None:
        output = _zero_fully_masked_rows(mx.squeeze(explicit_mask, axis=1), output)
    return output
