"""One typed SSA operation.

An operation is the unit every later layer works on: a pass decides whether to
rewrite it, the contract table checks it, and the backend emits a kernel for it.
Keeping it in its own module means a consumer that only manipulates operations
does not also drag in graph validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Operation:
    """One typed SSA operation.

    An operation produces at least one value. Most ATen operators return a
    single tensor and therefore one output, but several return a tuple that a
    single kernel produces together - ``native_layer_norm`` writes the
    normalized value beside the statistics it already computes,
    ``native_batch_norm`` writes the running moments it updates, and the
    ``linalg`` decompositions return their factors from one factorization.
    Modelling those as one operation with several outputs keeps the tuple
    provenance explicit instead of re-deriving one output from the others.
    """

    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError(f"{self.op} must produce at least one value")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"{self.op} must produce distinct values: {self.outputs}")

    @property
    def output(self) -> str:
        """The only output of a single-output operation."""

        if len(self.outputs) != 1:
            raise ValueError(
                f"{self.op} produces {len(self.outputs)} values; select one "
                "instead of asking for a single output"
            )
        return self.outputs[0]
