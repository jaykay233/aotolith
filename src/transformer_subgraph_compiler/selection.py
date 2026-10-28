from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelSelection:
    op_index: int
    op: str
    output_id: str
    implementation: str
    kind: str
    reason: str
    artifact_name: str | None = None

    @property
    def is_native(self) -> bool:
        return self.implementation == "metal_kernel"
