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
    try:
        return (_argument_ids(output, node_ids),)
    except ValueError:
        pass
    if isinstance(output, (tuple, list)):
        return tuple(_argument_ids(item, node_ids) for item in output)
    raise ValueError(f"unsupported FX output: {output!r}")
