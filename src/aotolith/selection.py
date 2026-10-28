from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelBenchmark:
    candidate: str
    artifact_name: str | None
    warmup_iterations: int
    iterations: int
    samples_seconds: tuple[float, ...]
    median_seconds: float | None
    min_seconds: float | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class KernelSelection:
    op_index: int
    op: str
    output_id: str
    implementation: str
    kind: str
    reason: str
    artifact_name: str | None = None
    benchmarks: tuple[KernelBenchmark, ...] = ()
    autotuned: bool = False
    device_name: str | None = None
    device_architecture: str | None = None

    @property
    def is_native(self) -> bool:
        return self.implementation == "metal_kernel"
