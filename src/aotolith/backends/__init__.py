"""The Apple GPU backend.

`AppleMetalBackend` is the entry point; the pieces it composes live in
`codegen` (Metal source), `runtime` (the compiled plan and the MLX fallback),
and `target` (device capability). Re-exported here so callers name one module.
"""

from __future__ import annotations

from ..codegen.msl import KernelArtifact
from ..runtime.plan import ArenaAssignment, CompiledPlan, ExecutionStats
from .apple import AppleMetalBackend

__all__ = [
    "AppleMetalBackend",
    "ArenaAssignment",
    "CompiledPlan",
    "ExecutionStats",
    "KernelArtifact",
]
