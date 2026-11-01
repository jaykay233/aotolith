"""The MLX reference backend: a plan with no generated kernels at all.

Aotolith's first backend emits Metal source per operation and selects a kernel
per node, falling back to MLX for whatever has no kernel. This one never emits
anything: every node is executed by the MLX interpreter
(``aotolith.runtime.mlx_ops._evaluate_mlx_op``), so a plan compiled for this
target is a *portable* plan - it needs no metallib and no per-device kernel
compile - at the cost of the launch overhead the kernels were built to remove.

That makes it useful for three jobs:

* **A second opinion on behaviour.** Both plans run the same lowered graph, so
  a structural disagreement - an op that only one of them implements, a mask
  applied on one side and not the other, a cache the two write differently - is
  a bug in the plan, not a question about the model.
* **A host without Metal.** The artifact format has no device assumptions, so a
  reference plan loads and runs wherever MLX does.
* **The seam itself.** ``target="mlx-reference"`` is what proves the backend
  boundary is a boundary: the compiler, the artifact format and the runtime all
  take a plan without caring which producer made it.

It is *not* a bit-exact oracle, and the reason is worth stating plainly. MLX on
Apple GPUs does not evaluate a batched ``matmul`` in IEEE float32: on an M5 with
MLX 0.32 the same product differs from a float32 reference by about 1e-3
relative (the single-row path is exact, the batched one is not). Attention goes
through the same matrix engines. The generated kernels accumulate in float32
and are the numerically tighter of the two paths, which is the opposite of the
usual "hand-written kernel versus framework" assumption. So: compare structure,
shapes, dtypes, cache contents and decode decisions here; compare *precision*
against the generated kernel.

The ABI is deliberately identical. A reference plan serializes with the same
manifest, the same payloads and the same target field - only the kernel table
is empty and every selection says ``mlx``, which is exactly what a plan whose
kernels all fell back would record.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..ir import Graph
from ..planning import plan_storage
from ..runtime.plan import CompiledPlan
from ..selection import KernelSelection

#: The target name, as accepted by :func:`aotolith.compiler.compile_graph`.
TARGET = "mlx-reference"


class ReferenceBackend:
    """Build a plan that executes every operator on the MLX interpreter."""

    #: Recorded in selections and artifact manifests.
    name = TARGET

    def compile(
        self,
        graph: Graph,
        *,
        tune_inputs: Any = None,
    ) -> CompiledPlan:
        """Return a plan whose every node runs through MLX.

        ``tune_inputs`` exists only to satisfy the backend interface: there is
        nothing to tune when nothing is generated, so a caller that passes them
        has misunderstood what this target does and is told so rather than
        silently losing the measurements it expected.
        """

        if tune_inputs is not None:
            raise ValueError(
                "the MLX reference backend generates no kernels, so there is "
                "nothing to autotune"
            )
        graph.validate()
        device_name, architecture = _device_profile()
        selections = tuple(
            KernelSelection(
                op_index=index,
                op=op.op,
                output_id=op.outputs[0],
                implementation="mlx",
                kind=op.op,
                reason=(
                    "the reference backend generates no kernels: the operation "
                    "runs on the MLX interpreter"
                ),
                device_name=device_name,
                device_architecture=architecture,
            )
            for index, op in enumerate(graph.ops)
        )
        return CompiledPlan(graph, (), selections, {}, plan_storage(graph))


def _device_profile() -> tuple[str | None, str | None]:
    """The device a reference plan is being built on, for the artifact record.

    Recorded for diagnostics only. A reference plan does not depend on this
    device, which is why loading one never checks it.
    """

    try:
        info = mx.device_info()
    except (RuntimeError, ValueError):
        return None, None
    return (
        None if "device_name" not in info else str(info["device_name"]),
        None if "architecture" not in info else str(info["architecture"]),
    )
