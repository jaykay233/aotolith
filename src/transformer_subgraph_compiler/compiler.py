from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .backends import AppleMetalBackend, CompiledPlan
from .frontend import capture_fx
from .ir import Graph
from .passes import fuse_pointwise


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: CompiledPlan
    input_graph: Graph
    lowered_graph: Graph


def compile_graph(graph: Graph, *, target: str = "apple-metal") -> CompilationResult:
    if target != "apple-metal":
        raise ValueError(f"unsupported target: {target}")

    lowered = fuse_pointwise(graph)
    plan = AppleMetalBackend().compile(lowered)
    return CompilationResult(plan=plan, input_graph=graph, lowered_graph=lowered)


def compile_callable(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    target: str = "apple-metal",
) -> CompilationResult:
    return compile_graph(capture_fx(fn, example_inputs), target=target)
