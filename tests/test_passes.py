"""The rewrite pipeline is a sequence, and the sequence is the contract.

Every pass is individually covered by the pipeline tests: they compile a graph
and check the operators that come out. What those tests cannot see is the order,
because a wrong order shows up as a missed fusion rather than a failure. This
file pins the order, and pins that the driver runs exactly those passes.
"""

from __future__ import annotations

from typing import Any

import pytest

from aotolith import compiler, passes
from aotolith.ir import Graph


def test_the_pipeline_runs_every_pass_once_in_the_documented_order() -> None:
    expected = [
        "simplify_layouts",
        "canonicalize_batched_matmul",
        "fuse_attention",
        "fold_matmul_input_layouts",
        "fold_quantized_weights",
        "fuse_rope",
        "fuse_residual_epilogue",
        "fuse_pointwise",
        "fuse_gate_up_projection",
        "fuse_activation_epilogue",
        "eliminate_common_subexpressions",
        "fuse_rms_norm_epilogue",
    ]

    assert [name for name, _ in passes.PIPELINE] == expected


def test_every_step_is_a_pass_the_package_exports() -> None:
    exported = {step for _, step in passes.PIPELINE}

    for name, step in passes.PIPELINE:
        assert getattr(passes, name) is step, name
    assert len(exported) == len(passes.PIPELINE), "a pass is listed twice"


def test_the_compiler_uses_the_pipeline_rather_than_its_own_list() -> None:
    # The order used to be eleven assignments inside compile_graph, so this
    # asserts the compiler cannot drift back to keeping its own copy.
    calls: list[str] = []

    def record(name: str):
        def step(graph: Graph) -> Graph:
            calls.append(name)
            return graph

        return step

    pipeline = tuple((name, record(name)) for name, _ in passes.PIPELINE)
    graph = _empty_graph()

    original = passes.PIPELINE
    try:
        passes.driver.PIPELINE = pipeline
        result = passes.driver.rewrite(graph)
    finally:
        passes.driver.PIPELINE = original

    assert result is graph
    assert calls == [name for name, _ in pipeline]


def test_the_pipeline_leaves_the_graph_alone_when_there_is_nothing_to_do() -> None:
    # Every pass rebuilds the graph it was handed, so "does nothing" means the
    # result is equal, not identical.
    assert passes.rewrite(_empty_graph()) == _empty_graph()


def test_the_driver_is_what_the_compiler_calls() -> None:
    # A pass that raises is easier to attribute than a wrong graph, and this is
    # the one place the compiler's dependence on the driver is visible.
    graph = _empty_graph()

    def explode(_graph: Any) -> Graph:
        raise RuntimeError("pipeline reached")

    original = passes.driver.PIPELINE
    try:
        passes.driver.PIPELINE = (("explode", explode),)
        with pytest.raises(RuntimeError, match="pipeline reached"):
            compiler.compile_graph(graph)
    finally:
        passes.driver.PIPELINE = original


def _empty_graph() -> Graph:
    return Graph(inputs=(), values={}, ops=(), outputs=())
