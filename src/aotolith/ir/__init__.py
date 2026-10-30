"""The typed SSA graph the compiler rewrites.

Three modules, in dependency order:

* ``types`` - ``DType``, ``TensorType``, ``Value``. The vocabulary only.
* ``op`` - ``Operation``, one operation with one or more outputs.
* ``graph`` - ``Graph``, which holds them and validates use-before-def.

Everything is re-exported here so callers keep writing ``from ..ir import
Graph, Operation``. That single import site is also what lets the internals
move again later without touching forty modules.
"""

from __future__ import annotations

from .graph import Graph
from .op import Operation
from .types import DType, TensorType, Value

__all__ = ["DType", "Graph", "Operation", "TensorType", "Value"]
