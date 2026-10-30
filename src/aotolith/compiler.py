from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import mlx.core as mx

from .backends import AppleMetalBackend, CompiledPlan
from .capture import capture_fx
from .ir import Graph
from .passes import rewrite

# Container kinds recorded in a keyword template and in an artifact manifest.
_KWARG_LIST = "list"
_KWARG_TUPLE = "tuple"
_KWARG_DICT = "dict"


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
class _KwargLeaf:
    """A tensor keyword value that becomes one graph input."""


@dataclass(frozen=True, slots=True)
class _KwargStatic:
    """A keyword leaf with no tensor in it, kept as a tracing-time constant."""

    value: Any


@dataclass(frozen=True, slots=True)
class _KwargSequence:
    """A list/tuple keyword value rebuilt around whichever leaves are tensors."""

    kind: str
    items: tuple[Any, ...]
    # ``namedtuple`` subclasses are rebuilt through their constructor; a plain
    # list or tuple needs none. Only used while tracing, never serialized.
    factory: Any = None


@dataclass(frozen=True, slots=True)
class _KwargMapping:
    """A mapping keyword value rebuilt around whichever leaves are tensors."""

    items: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True, slots=True)
class _KwargTensor:
    """A promoted leaf: where it sits in the value and what it traced as."""

    path: tuple[tuple[str, Any], ...]
    value: Any


@dataclass(frozen=True, slots=True)
class CompiledModulePlan:
    """An executable plan with module parameters captured as bound state."""

    compilation: CompilationResult
    input_count: int
    # Parameter and buffer names, then one ``<constant N>`` entry per tensor that
    # tracing captured as a graph constant, in the order the graph reaches them.
    state_names: tuple[str, ...]
    _state_values: tuple[mx.array, ...]
    # Tensor-valued ``example_kwargs``, promoted to graph inputs so callers can
    # pass them by name and change them per call. In ``graph.inputs`` they sit
    # between the positional inputs and the bound state, in the order their
    # tensors appear inside each value. ``tensor_kwarg_templates`` records how
    # those leaves were nested, so a value such as ``(cos, sin)`` -- two graph
    # inputs behind one keyword -- is accepted in its original shape.
    tensor_kwarg_names: tuple[str, ...] = ()
    tensor_kwarg_templates: tuple[Any, ...] = ()

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

    def _tensor_kwarg_values(self, supplied: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the promoted keyword inputs in graph order."""

        missing = [name for name in self.tensor_kwarg_names if name not in supplied]
        if missing:
            raise ValueError("missing tensor keyword inputs: " + ", ".join(missing))
        unexpected = sorted(set(supplied) - set(self.tensor_kwarg_names))
        if unexpected:
            raise ValueError("unexpected keyword inputs: " + ", ".join(unexpected))
        values: list[Any] = []
        for name, template in zip(
            self.tensor_kwarg_names, self._kwarg_templates(), strict=True
        ):
            _collect_kwarg_tensors(
                template, supplied[name], values, f"keyword {name!r}"
            )
        return tuple(values)

    def _kwarg_templates(self) -> tuple[Any, ...]:
        """Templates for each promoted keyword, oldest artifacts included.

        Plans rebuilt from artifacts written before nested promotion recorded
        only a keyword name per graph input, which is exactly one tensor leaf
        at the top level.
        """

        if len(self.tensor_kwarg_templates) == len(self.tensor_kwarg_names):
            return self.tensor_kwarg_templates
        if self.tensor_kwarg_templates:
            raise ValueError(
                "tensor keyword templates do not match the tensor keyword names"
            )
        return tuple(_KwargLeaf() for _ in self.tensor_kwarg_names)


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

    lowered = rewrite(graph)
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
    tensor_kwarg_templates: list[Any] = []
    tensor_kwarg_values: list[Any] = []
    for name, value in kwargs.items():
        leaves: list[_KwargTensor] = []
        template = _kwarg_template(value, f"example_kwargs[{name!r}]", (), leaves)
        if leaves:
            tensor_kwarg_names.append(name)
            tensor_kwarg_templates.append(template)
            tensor_kwarg_values.extend(leaf.value for leaf in leaves)
        else:
            static_kwargs[name] = value
    if "verify_index_bounds" in tensor_kwarg_names:
        raise ValueError(
            "a tensor keyword input cannot be named 'verify_index_bounds': "
            "the compiled plan uses that name for its own argument"
        )

    with _tensor_attributes_as_buffers(module):
        return _compile_module_graph(
            module,
            example_inputs=example_inputs,
            static_kwargs=static_kwargs,
            tensor_kwarg_names=tensor_kwarg_names,
            tensor_kwarg_templates=tensor_kwarg_templates,
            tensor_kwarg_values=tensor_kwarg_values,
            output_key=output_key,
            target=target,
            codegen_matmul=codegen_matmul,
            codegen_attention=codegen_attention,
            autotune=autotune,
            autotune_warmup=autotune_warmup,
            autotune_iterations=autotune_iterations,
        )


@contextmanager
def _tensor_attributes_as_buffers(module: Any) -> Iterator[None]:
    """Temporarily register plain tensor attributes so tracing can bind them.

    A module sometimes keeps a constant tensor as a plain attribute
    (``self.scale = torch.tensor([...])``) instead of registering it. Tracing
    runs the forward pass under ``FakeTensorMode``, and a real tensor reaching an
    operator there is an error, so the module would fail before any graph
    exists. Registering the attribute as a buffer puts it in the state that
    ``functional_call`` binds, which keeps capture shape-only and turns the
    tensor into a bound plan input under the attribute's own name.

    The registration adds the name to ``_buffers`` and takes the same name out
    of the instance ``__dict__``: a plain attribute shadows the buffer, so
    ``functional_call``'s substitution would otherwise never reach ``forward``.
    The attribute holds the same tensor throughout, so the module behaves
    identically while this is in effect, and both changes are undone on the way
    out, including when capture raises.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - compile_module imports it first
        raise RuntimeError("PyTorch is required for compile_module") from exc

    registered: list[tuple[Any, str, Any]] = []
    try:
        for holder in module.modules():
            parameters = holder._parameters
            buffers = holder._buffers
            for name, value in list(vars(holder).items()):
                if name in parameters or name in buffers:
                    continue
                if not isinstance(value, torch.Tensor):
                    continue
                buffers[name] = value
                # A buffer is reached through ``__getattr__``; the plain
                # attribute lives in ``__dict__`` and would shadow it, keeping
                # ``functional_call``'s substitution invisible to ``forward``.
                del holder.__dict__[name]
                registered.append((holder, name, value))
        yield
    finally:
        for holder, name, value in reversed(registered):
            holder._buffers.pop(name, None)
            holder.__dict__[name] = value


def _compile_module_graph(
    module: Any,
    *,
    example_inputs: Sequence[Any],
    static_kwargs: Mapping[str, Any],
    tensor_kwarg_names: Sequence[str],
    tensor_kwarg_templates: Sequence[Any],
    tensor_kwarg_values: Sequence[Any],
    output_key: str | None,
    target: str,
    codegen_matmul: bool,
    codegen_attention: bool,
    autotune: bool,
    autotune_warmup: int,
    autotune_iterations: int,
) -> CompiledModulePlan:
    """Trace and lower the module once its tensor attributes are bound state."""

    import torch

    state = dict(module.named_parameters(remove_duplicate=False))
    state.update(module.named_buffers(remove_duplicate=False))
    state_names = tuple(state)
    state_values = tuple(state.values())
    input_count = len(example_inputs)
    forward_input_count = input_count + len(tensor_kwarg_values)

    def functional_forward(*args: Any) -> Any:
        inputs = tuple(args[:input_count])
        bound_state = dict(zip(state_names, args[forward_input_count:], strict=True))
        call_kwargs = dict(static_kwargs)
        promoted = iter(args[input_count:forward_input_count])
        for name, template in zip(
            tensor_kwarg_names, tensor_kwarg_templates, strict=True
        ):
            call_kwargs[name] = _build_kwarg_value(template, promoted)
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
    # Tensors that tracing captured as constants (a module attribute that is
    # not a parameter or buffer, or a tensor built inside ``forward``) become
    # private plan inputs in the order the graph reaches them, so the plan's
    # bound state is the module state followed by those constants.
    constants: list[Any] = []
    result = compile_graph(
        capture_fx(
            functional_forward,
            traced_inputs + state_values,
            promoted_constants=constants,
        ),
        target=target,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        autotune_warmup=autotune_warmup,
        autotune_iterations=autotune_iterations,
        autotune_inputs=(traced_inputs + state_values if autotune else None),
    )
    bound_values = _unique_mlx_state((*state_values, *constants))
    return CompiledModulePlan(
        compilation=result,
        input_count=input_count,
        state_names=tuple(state_names)
        + tuple(f"<constant {index}>" for index in range(len(constants))),
        _state_values=bound_values,
        tensor_kwarg_names=tuple(tensor_kwarg_names),
        tensor_kwarg_templates=tuple(tensor_kwarg_templates),
    )


def _kwarg_template(
    value: Any,
    where: str,
    path: tuple[tuple[str, Any], ...],
    leaves: list[_KwargTensor],
) -> Any:
    """Describe one keyword value, promoting every tensor inside it.

    Only tensors are promoted. A subtree with no tensors is kept whole as a
    tracing-time constant, so ``{"use_cache": False}`` still traces as a Python
    value and ``{"scale": 2.0, "mask": tensor}`` promotes just the mask.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for compile_module") from exc

    if isinstance(value, torch.Tensor):
        leaves.append(_KwargTensor(path, value))
        return _KwargLeaf()

    if isinstance(value, Mapping):
        found: list[_KwargTensor] = []
        items = tuple(
            (
                key,
                _kwarg_template(
                    child,
                    f"{where}[{key!r}]",
                    path + ((_KWARG_DICT, key),),
                    found,
                ),
            )
            for key, child in value.items()
        )
        if not found:
            return _KwargStatic(value)
        leaves.extend(found)
        return _KwargMapping(items)

    if isinstance(value, (list, tuple)):
        kind = _KWARG_LIST if isinstance(value, list) else _KWARG_TUPLE
        found = []
        items = tuple(
            _kwarg_template(
                child,
                f"{where}[{index}]",
                path + ((kind, index),),
                found,
            )
            for index, child in enumerate(value)
        )
        if not found:
            return _KwargStatic(value)
        leaves.extend(found)
        factory = (
            type(value)
            if kind == _KWARG_TUPLE
            and type(value) is not tuple
            and hasattr(type(value), "_fields")
            else None
        )
        return _KwargSequence(kind, items, factory)

    if _contains_tensor(value):
        raise NotImplementedError(
            f"{where} is a {type(value).__name__} that holds tensors; only "
            "lists, tuples and mappings promote their tensors to graph inputs"
        )
    return _KwargStatic(value)


def _build_kwarg_value(node: Any, values: Iterator[Any]) -> Any:
    """Rebuild a keyword value around the tensors bound to its leaves."""

    if isinstance(node, _KwargLeaf):
        return next(values)
    if isinstance(node, _KwargSequence):
        items = [_build_kwarg_value(child, values) for child in node.items]
        if node.factory is not None:
            return node.factory(*items)
        return items if node.kind == _KWARG_LIST else tuple(items)
    if isinstance(node, _KwargMapping):
        return {key: _build_kwarg_value(child, values) for key, child in node.items}
    if isinstance(node, _KwargStatic):
        return node.value
    raise TypeError(f"unknown keyword template {type(node).__name__}")


def _collect_kwarg_tensors(node: Any, value: Any, sink: list[Any], where: str) -> None:
    """Append the tensors inside one supplied keyword value, in graph order.

    The supplied value has to mirror the traced nesting exactly: same container
    shapes, same mapping keys. A caller that passes a different nesting would
    otherwise bind tensors to the wrong graph inputs.
    """

    if isinstance(node, _KwargLeaf):
        sink.append(value)
        return
    if isinstance(node, _KwargSequence):
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{where} expected "
                f"{'a list' if node.kind == _KWARG_LIST else 'a tuple'} of "
                f"{len(node.items)} values, got {type(value).__name__}"
            )
        if len(value) != len(node.items):
            raise ValueError(
                f"{where} expected {len(node.items)} values, got {len(value)}"
            )
        for index, (child, item) in enumerate(zip(node.items, value)):
            _collect_kwarg_tensors(child, item, sink, f"{where}[{index}]")
        return
    if isinstance(node, _KwargMapping):
        if not isinstance(value, Mapping):
            raise ValueError(
                f"{where} expected a mapping with keys "
                f"{[key for key, _ in node.items]}, got "
                f"{type(value).__name__}"
            )
        expected = [key for key, _ in node.items]
        if list(value) != expected:
            raise ValueError(
                f"{where} expected a mapping with keys {expected}, got {list(value)}"
            )
        for key, child in node.items:
            _collect_kwarg_tensors(child, value[key], sink, f"{where}[{key!r}]")
        return
    if isinstance(node, _KwargStatic):
        if type(value) is not type(node.value) or not _values_match(value, node.value):
            raise ValueError(
                f"{where} is not a tensor, so it was captured as a constant "
                f"while tracing ({node.value!r}); pass the traced value back "
                "or compile again with the value you need"
            )
        return
    raise TypeError(f"unknown keyword template {type(node).__name__}")


def _values_match(left: Any, right: Any) -> bool:
    """Compare two non-tensor leaves without letting exotic types raise."""

    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - an uncomparable leaf is a mismatch
        return False


def _kwarg_leaf_paths(
    node: Any,
) -> list[tuple[tuple[str, Any], ...]]:
    """Return the path of every tensor leaf, in graph-input order."""

    if isinstance(node, _KwargLeaf):
        return [()]
    if isinstance(node, _KwargSequence):
        return [
            ((node.kind, index),) + rest
            for index, child in enumerate(node.items)
            for rest in _kwarg_leaf_paths(child)
        ]
    if isinstance(node, _KwargMapping):
        return [
            ((_KWARG_DICT, key),) + rest
            for key, child in node.items
            for rest in _kwarg_leaf_paths(child)
        ]
    return []


def _kwarg_template_from_paths(
    paths: Sequence[Sequence[tuple[str, Any]]], leaf_count: int
) -> Any:
    """Rebuild a keyword template from the leaf paths stored in an artifact."""

    if leaf_count == 1 and all(len(path) == 0 for path in paths):
        return _KwargLeaf()

    def build(depth: int, members: Sequence[Sequence[tuple[str, Any]]]) -> Any:
        if any(len(path) == depth for path in members):
            if len(members) != 1:
                raise ValueError(
                    "artifact keyword path mixes a tensor leaf with nested tensors"
                )
            return _KwargLeaf()
        kind = members[0][depth][0]
        if kind == _KWARG_DICT:
            grouped: dict[Any, list[Sequence[tuple[str, Any]]]] = {}
            for path in members:
                step_kind, key = path[depth]
                if step_kind != _KWARG_DICT:
                    raise ValueError("artifact keyword path changes container kind")
                grouped.setdefault(key, []).append(path)
            return _KwargMapping(
                tuple(
                    (key, build(depth + 1, children))
                    for key, children in grouped.items()
                )
            )
        if kind not in (_KWARG_LIST, _KWARG_TUPLE):
            raise ValueError(f"artifact keyword path has unknown container {kind!r}")
        indexed: dict[int, list[Sequence[tuple[str, Any]]]] = {}
        for path in members:
            step_kind, key = path[depth]
            if step_kind != kind:
                raise ValueError("artifact keyword path changes container kind")
            if not isinstance(key, int) or isinstance(key, bool):
                raise ValueError("artifact keyword path has a non-integer index")
            indexed.setdefault(key, []).append(path)
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("artifact keyword path skips a container index")
        return _KwargSequence(
            kind,
            tuple(build(depth + 1, indexed[index]) for index in range(len(indexed))),
        )

    if not paths:
        raise ValueError("artifact keyword entry has no tensor leaves")
    return build(0, paths)


def _unique_mlx_state(values: Sequence[Any]) -> tuple[mx.array, ...]:
    """Convert each distinct state tensor once so aliases share one buffer.

    `named_parameters(remove_duplicate=False)` reports a tensor under every name
    it is reachable by, and a tied LM head is the common case. Converting each
    name separately would keep a second copy of the largest tensor in the model
    — 272 MB for a 0.5B model with a 151936x896 embedding — while the two graph
    inputs it feeds compute exactly the same numbers.

    The key includes strides because sharing a data pointer is not enough: a
    square tensor and its `.t()` view have the same pointer and shape but
    different contents. Identical pointer, dtype, shape and strides do mean
    identical elements.
    """

    converted: dict[tuple[Any, ...], mx.array] = {}
    unique: list[mx.array] = []
    for value in values:
        key = (
            value.data_ptr(),
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
        )
        array = converted.get(key)
        if array is None:
            array = _torch_tensor_to_mlx(value)
            converted[key] = array
        unique.append(array)
    return tuple(unique)


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
                "module sequence outputs must be a non-empty tuple or list of tensors"
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
