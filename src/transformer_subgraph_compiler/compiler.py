from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .backends import AppleMetalBackend, CompiledPlan
from .frontend import capture_fx
from .ir import Graph
from .passes import canonicalize_batched_matmul, fuse_attention, fuse_pointwise, simplify_layouts


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: CompiledPlan
    input_graph: Graph
    lowered_graph: Graph


def compile_graph(
    graph: Graph,
    *,
    target: str = "apple-metal",
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
) -> CompilationResult:
    if target != "apple-metal":
        raise ValueError(f"unsupported target: {target}")

    lowered = simplify_layouts(graph)
    lowered = canonicalize_batched_matmul(lowered)
    lowered = fuse_attention(lowered)
    lowered = fuse_pointwise(lowered)
    plan = AppleMetalBackend(
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
    ).compile(lowered)
    return CompilationResult(plan=plan, input_graph=graph, lowered_graph=lowered)


def compile_callable(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    target: str = "apple-metal",
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
) -> CompilationResult:
    return compile_graph(
        capture_fx(fn, example_inputs),
        target=target,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
    )
