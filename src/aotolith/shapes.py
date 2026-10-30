"""Operand-shape reasoning shared by the contracts and the backend codegen."""

from __future__ import annotations


def can_broadcast(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
    """Whether ``input_shape`` broadcasts to ``output_shape``."""

    if len(input_shape) > len(output_shape):
        return False
    offset = len(output_shape) - len(input_shape)
    return all(
        source == target or source == 1
        for source, target in zip(input_shape, output_shape[offset:])
    )


def broadcast_shapes(shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """The shape NumPy broadcasting produces for ``shapes``."""

    rank = max(len(shape) for shape in shapes)
    result = [1] * rank
    for shape in shapes:
        offset = rank - len(shape)
        for axis, dim in enumerate(shape):
            current = result[offset + axis]
            if current != dim and current != 1 and dim != 1:
                raise ValueError(f"index shapes cannot broadcast: {shapes}")
            result[offset + axis] = max(current, dim)
    return tuple(result)
