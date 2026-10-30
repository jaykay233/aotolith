from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import mlx.core as mx

from .backends import AppleMetalBackend, CompiledPlan
from .frontend import capture_fx
from .ir import Graph
from .passes import (
    canonicalize_batched_matmul,
    eliminate_common_subexpressions,
    fuse_attention,
    fuse_activation_epilogue,
    fuse_gate_up_projection,
    fuse_pointwise,
    fuse_residual_epilogue,
    fuse_rms_norm_epilogue,
    fuse_rope,
    fold_matmul_input_layouts,
    simplify_layouts,
)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: CompiledPlan
    input_graph: Graph
    lowered_graph: Graph
    codegen_matmul: bool = False
    codegen_attention: bool = False
    autotune: bool = False
    autotune_warmup: int = 2
    autotune_iterations: int = 7

    def save(self, path: str | PathLike[str]) -> Path:
        from .artifact import save_compiled

        return save_compiled(self, path)


@dataclass(frozen=True, slots=True)
class CompiledModulePlan:
    """An executable plan with module parameters captured as bound state."""

    compilation: CompilationResult
    input_count: int
    state_names: tuple[str, ...]
    _state_values: tuple[mx.array, ...]
    # Tensor-valued ``example_kwargs``, promoted to graph inputs so callers can
    # pass them by name and change them per call. In ``graph.inputs`` they sit
    # between the positional inputs and the bound state.
    tensor_kwarg_names: tuple[str, ...] = ()

    @property
    def plan(self) -> CompiledPlan:
        return self.compilation.plan

    @property
    def input_graph(self) -> Graph:
        return self.compilation.input_graph

    @property
    def lowered_graph(self) -> Graph:
        return self.compilation.lowered_graph

    def save(self, path: str | PathLike[str]) -> Path:
        from .artifact import save_compiled

        return save_compiled(self, path)

    def __call__(
        self,
        *inputs: Any,
        verify_index_bounds: bool = True,
        **tensor_kwargs: Any,
    ) -> tuple[mx.array, ...]:
        if len(inputs) != self.input_count:
            raise ValueError(
                f"expected {self.input_count} inputs, received {len(inputs)}"
            )
        return self.plan(
            *inputs,
            *self._tensor_kwarg_values(tensor_kwargs),
            *self._state_values,
            verify_index_bounds=verify_index_bounds,
        )

    def _tensor_kwarg_values(
        self, supplied: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        """Return the promoted keyword inputs in graph order."""

        missing = [
            name for name in self.tensor_kwarg_names if name not in supplied
        ]
        if missing:
            raise ValueError(
                "missing tensor keyword inputs: " + ", ".join(missing)
            )
        unexpected = sorted(set(supplied) - set(self.tensor_kwarg_names))
        if unexpected:
            raise ValueError(
                "unexpected keyword inputs: " + ", ".join(unexpected)
            )
        return tuple(supplied[name] for name in self.tensor_kwarg_names)


def compile_graph(
    graph: Graph,
    *,
    target: str = "apple-metal",
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    autotune_warmup: int = 2,
    autotune_iterations: int = 7,
    autotune_inputs: Sequence[Any] | None = None,
) -> CompilationResult:
    if target != "apple-metal":
        raise ValueError(f"unsupported target: {target}")

    lowered = simplify_layouts(graph)
    lowered = canonicalize_batched_matmul(lowered)
    lowered = fuse_attention(lowered)
    lowered = fold_matmul_input_layouts(lowered)
    lowered = fuse_rope(lowered)
    lowered = fuse_residual_epilogue(lowered)
    lowered = fuse_pointwise(lowered)
    lowered = fuse_gate_up_projection(lowered)
    lowered = fuse_activation_epilogue(lowered)
    lowered = eliminate_common_subexpressions(lowered)
    lowered = fuse_rms_norm_epilogue(lowered)
    plan = AppleMetalBackend(
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        autotune_warmup=autotune_warmup,
        autotune_iterations=autotune_iterations,
    ).compile(lowered, tune_inputs=autotune_inputs)
    return CompilationResult(
        plan=plan,
        input_graph=graph,
        lowered_graph=lowered,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        autotune_warmup=autotune_warmup,
        autotune_iterations=autotune_iterations,
    )


def compile_callable(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    target: str = "apple-metal",
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    autotune_warmup: int = 2,
    autotune_iterations: int = 7,
) -> CompilationResult:
    return compile_graph(
        capture_fx(fn, example_inputs),
        target=target,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        autotune_warmup=autotune_warmup,
        autotune_iterations=autotune_iterations,
        autotune_inputs=example_inputs if autotune else None,
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
    autotune: bool = False,
    autotune_warmup: int = 2,
    autotune_iterations: int = 7,
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
    static_kwargs: dict[str, Any] = {}
    tensor_kwarg_names: list[str] = []
    tensor_kwarg_values: list[Any] = []
    nested: list[str] = []
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            tensor_kwarg_names.append(name)
            tensor_kwarg_values.append(value)
        elif _contains_tensor(value):
            nested.append(name)
        else:
            static_kwargs[name] = value
    if nested:
        raise NotImplementedError(
            "example_kwargs values that contain tensors are not graph inputs "
            f"yet: {nested}"
        )
    if "verify_index_bounds" in tensor_kwarg_names:
        raise ValueError(
            "a tensor keyword input cannot be named 'verify_index_bounds': "
            "the compiled plan uses that name for its own argument"
        )

    state = dict(module.named_parameters(remove_duplicate=False))
    state.update(module.named_buffers(remove_duplicate=False))
    state_names = tuple(state)
    state_values = tuple(state.values())
    input_count = len(example_inputs)
    forward_input_count = input_count + len(tensor_kwarg_names)
    bound_values = tuple(_torch_tensor_to_mlx(value) for value in state_values)

    def functional_forward(*args: Any) -> Any:
        inputs = tuple(args[:input_count])
        bound_state = dict(
            zip(state_names, args[forward_input_count:], strict=True)
        )
        call_kwargs = dict(static_kwargs)
        call_kwargs.update(
            zip(tensor_kwarg_names, args[input_count:forward_input_count])
        )
        with torch.no_grad():
            output = torch.func.functional_call(
                module,
                bound_state,
                inputs,
                kwargs=call_kwargs,
                strict=True,
            )
        return _select_module_output(output, output_key)

    traced_inputs = tuple(example_inputs) + tuple(tensor_kwarg_values)

    result = compile_graph(
        capture_fx(
            functional_forward,
            traced_inputs + state_values,
        ),
        target=target,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        autotune_warmup=autotune_warmup,
        autotune_iterations=autotune_iterations,
        autotune_inputs=(
            traced_inputs + state_values if autotune else None
        ),
    )
    return CompiledModulePlan(
        compilation=result,
        input_count=input_count,
        state_names=state_names,
        _state_values=bound_values,
        tensor_kwarg_names=tuple(tensor_kwarg_names),
    )


def _contains_tensor(value: Any) -> bool:
    """Return whether a keyword value holds a tensor below its top level."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for compile_module") from exc

    def contains(item: Any) -> bool:
        if isinstance(item, torch.Tensor):
            return True
        if isinstance(item, Mapping):
            return any(contains(child) for child in item.values())
        if isinstance(item, (list, tuple, set)):
            return any(contains(child) for child in item)
        return False

    return contains(value)



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
        elif isinstance(output, (tuple, list)):
            selected = tuple(output)

    if isinstance(selected, (tuple, list)):
        if not selected or not all(
            isinstance(value, torch.Tensor) for value in selected
        ):
            raise ValueError(
                "module sequence outputs must be a non-empty tuple or list "
                "of tensors"
            )
        return tuple(selected)
    if not isinstance(selected, torch.Tensor):
        raise ValueError(
            "module output must be a tensor, a tuple/list of tensors, or "
            "expose a tensor selected by output_key"
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
