"""Backend implementations, and the interface they share.

A backend turns a lowered :class:`~aotolith.ir.Graph` into an executable
:class:`~aotolith.runtime.plan.CompiledPlan`. It is the last thing the compiler
asks for, and the only thing a target name selects, which is what keeps the
rest of the pipeline target-independent.

Two implementations exist today:

``AppleMetalBackend`` (:mod:`aotolith.backends.apple`)
    Generates Metal source per operation, selecting a kernel per node and
    falling back to MLX for what has no kernel. The performance target.

``ReferenceBackend`` (:mod:`aotolith.backends.reference`)
    Generates nothing and runs every node on the MLX interpreter. The
    correctness oracle and the portable target.

A backend only has to provide ``compile``; see :class:`Backend`. Everything
above this line - capture, legalization, rewriting, storage planning, the
artifact format - is shared, which is why a second backend is a file and not a
second pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..codegen.msl import KernelArtifact
from ..ir import Graph
from ..runtime.plan import ArenaAssignment, CompiledPlan, ExecutionStats
from .apple import AppleMetalBackend
from .reference import TARGET as REFERENCE_TARGET
from .reference import ReferenceBackend

#: Every target name this release can compile for. The compiler maps each name
#: to a backend instance; the CLI and the artifact loader validate against this
#: tuple so the three cannot disagree about what exists.
TARGETS: tuple[str, ...] = ("apple-metal", REFERENCE_TARGET)


@runtime_checkable
class Backend(Protocol):
    """What the compiler needs from a backend.

    Deliberately one method: a backend's whole job is "here is a graph, give me
    a plan that runs it". Anything a backend wants to know about the device, it
    asks the device - not the caller - so adding a target does not widen this.
    """

    def compile(self, graph: Graph, *, tune_inputs: Any = None) -> CompiledPlan:
        """Return an executable plan for ``graph``.

        ``tune_inputs`` are the example inputs autotuning may measure with; a
        backend that generates nothing ignores or refuses them.
        """


__all__ = [
    "REFERENCE_TARGET",
    "TARGETS",
    "AppleMetalBackend",
    "ArenaAssignment",
    "Backend",
    "CompiledPlan",
    "ExecutionStats",
    "KernelArtifact",
    "ReferenceBackend",
]
