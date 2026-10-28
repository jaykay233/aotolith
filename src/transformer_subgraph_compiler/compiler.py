from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .backends import AppleMetalBackend, CompiledPlan
from .frontend import capture_fx
from .ir import Graph
from .passes import (
    canonicalize_batched_matmul,
    eliminate_common_subexpressions,
    fuse_attention,
    fuse_pointwise,
    fold_matmul_input_layouts,
    simplify_layouts,
)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: CompiledPlan
    input_graph: Graph
    lowered_graph: Graph


@dataclass(frozen=True, slots=True)
class CompiledModulePlan:
    """An executable plan with module parameters captured as bound state."""

    compilation: CompilationResult
    input_count: int
    state_names: tuple[str, ...]
    _state_values: tuple[mx.array, ...]

    @property
    def plan(self) -> CompiledPlan:
        return self.compilation.plan

    @property
    def input_graph(self) -> Graph:
        return self.compilation.input_graph

    @property
    def lowered_graph(self) -> Graph:
        return self.compilation.lowered_graph

    def __call__(self, *inputs: Any) -> tuple[mx.array, ...]:
        if len(inputs) != self.input_count:
            raise ValueError(
                f"expected {self.input_count} inputs, received {len(inputs)}"
            )
        return self.plan(*inputs, *self._state_values)


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
    lowered = fold_matmul_input_layouts(lowered)
    lowered = fuse_pointwise(lowered)
    lowered = eliminate_common_subexpressions(lowered)
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


def compile_module(
    module: Any,
    example_inputs: Sequence[Any],
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    output_key: str | None = None,
    target: str = "apple-metal",
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
) -> CompiledModulePlan:
    """Compile an ``nn.Module`` and bind its state as immutable plan inputs.

    The returned plan does not retain the original module. Parameters and
    buffers are copied to MLX arrays during compilation and supplied to the
    lowered graph as private inputs.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for compile_module") from exc

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"expected torch.nn.Module, got {type(module).__name__}")
    if getattr(module, "training", False):
        raise ValueError("compile_module requires module.eval()")

    kwargs = dict(example_kwargs or {})
    tensor_kwargs = [
        name for name, value in kwargs.items() if isinstance(value, torch.Tensor)
    ]
    if tensor_kwargs:
        raise NotImplementedError(
            "tensor-valued example_kwargs are not graph inputs yet: "
            f"{tensor_kwargs}"
        )

    state = dict(module.named_parameters(remove_duplicate=False))
    state.update(module.named_buffers(remove_duplicate=False))
    state_names = tuple(state)
    state_values = tuple(state.values())
    input_count = len(example_inputs)

    def functional_forward(*args: Any) -> torch.Tensor:
        inputs = tuple(args[:input_count])
        bound_state = dict(zip(state_names, args[input_count:], strict=True))
        with torch.no_grad():
            output = torch.func.functional_call(
                module,
                bound_state,
                inputs,
                kwargs=kwargs,
                strict=True,
            )
        return _select_module_output(output, output_key)

    result = compile_graph(
        capture_fx(
            functional_forward,
            tuple(example_inputs) + state_values,
        ),
        target=target,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
    )
    bound_values = tuple(_torch_tensor_to_mlx(value) for value in state_values)
    return CompiledModulePlan(
        compilation=result,
        input_count=input_count,
        state_names=state_names,
        _state_values=bound_values,
    )


def _select_module_output(output: Any, output_key: str | None) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for compile_module") from exc

    selected = output
    if output_key is not None:
        if isinstance(output, Mapping):
            if output_key not in output:
                raise KeyError(f"module output has no key {output_key!r}")
            selected = output[output_key]
        else:
            try:
                selected = getattr(output, output_key)
            except AttributeError as exc:
                raise ValueError(
                    f"module output has no attribute {output_key!r}"
                ) from exc
    elif not isinstance(output, torch.Tensor):
        if hasattr(output, "logits"):
            selected = output.logits
        elif isinstance(output, Mapping) and len(output) == 1:
            selected = next(iter(output.values()))
        elif isinstance(output, (tuple, list)) and len(output) == 1:
            selected = output[0]

    if not isinstance(selected, torch.Tensor):
        raise ValueError(
            "module output must be a tensor or expose a tensor selected by "
            "output_key"
        )
    return selected


def _torch_tensor_to_mlx(value: Any) -> mx.array:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for compile_module") from exc

    cpu_value = value.detach().cpu()
    numpy_value = (
        cpu_value.float().numpy()
        if value.dtype == torch.bfloat16
        else cpu_value.numpy()
    )
    dtype = {
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.float32: mx.float32,
        torch.bool: mx.bool_,
        torch.int32: mx.int32,
        torch.int64: mx.int64,
    }.get(value.dtype)
    if dtype is None:
        raise TypeError(f"unsupported module state dtype: {value.dtype}")
    return mx.array(numpy_value).astype(dtype)
