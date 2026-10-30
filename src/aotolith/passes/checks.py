"""Small predicates over operator attributes and static shapes.

These decide whether a match is even worth attempting, and they are deliberately
dumb: no graph walking, no shape inference, just a yes or no about values that
are already in hand.
"""

from __future__ import annotations


def _can_broadcast_to(shape: tuple[int, ...], output: tuple[int, ...]) -> bool:
    if len(shape) > len(output):
        return False
    offset = len(output) - len(shape)
    return all(dim in {1, output[offset + index]} for index, dim in enumerate(shape))


def _valid_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_ints(values: tuple[object, ...]) -> bool:
    return all(_valid_int(value) for value in values)


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
