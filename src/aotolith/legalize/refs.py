"""Resolving FX nodes to the IR values they are bound to.

The legalizer keeps a ``node_ids`` map from FX node to value id, and these are
the only three ways anything reads it: the operands of a node, whether a
consumer is the last one left, and the ids a graph output resolves to. They are
separate from both the builders and the loop because both need them.

A node with several results is the awkward case. ``node_ids`` holds the id of
one result - the one that is still used - and an operation's outputs list is the
authority on the others, which is why a multi-output builder reports its ids
explicitly instead of relying on the map.
"""

from __future__ import annotations

from typing import Any


def _argument_ids(argument: object, node_ids: dict[object, str]) -> str:
    try:
        return node_ids[argument]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported FX argument: {argument!r}") from exc


def _single_user(node: object) -> Any | None:
    users = tuple(getattr(node, "users", {}))
    return users[0] if len(users) == 1 else None


def _output_ids(output: object, node_ids: dict[object, str]) -> tuple[str, ...]:
    """Resolve the graph output, flattening any nesting it carries.

    A module that returns ``(output, (h, c))`` - ``nn.LSTM`` - traces to a
    nested tuple of FX nodes. The plan returns a flat tuple of arrays in the
    order those leaves appear, so a nested structure is walked in reading
    order rather than refused. ``None`` carries no data and is skipped:
    ``nn.MultiheadAttention(need_weights=False)`` returns ``(output, None)``,
    and the plan returns the output alone.
    """

    ids: list[str] = []

    def walk(argument: object) -> None:
        if isinstance(argument, (tuple, list)):
            for item in argument:
                walk(item)
            return
        if argument is None:
            return
        ids.append(_argument_ids(argument, node_ids))

    walk(output)
    if not ids:
        raise ValueError(f"unsupported FX output: {output!r}")
    return tuple(ids)
