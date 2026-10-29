"""A ~200 line re-implementation of ``torch.fx.experimental.proxy_tensor.make_fx``.

Run with ``python examples/mini_make_fx.py``. It needs only PyTorch, no MLX and
no Aotolith backend, because it stops at the FX graph that
``aotolith.frontend.capture_fx`` later lowers into the SSA IR.

``make_fx`` is often mistaken for something that inspects bytecode or parses an
AST. It does neither. It runs the function on real tensors and records every
ATen operator that gets dispatched. Four layers, one per real component:

    Layer 0  Mini graph        <->  torch.fx.Graph / Node / Tracer
    Layer 1  Proxy + memo      <->  torch.fx.Proxy / set_proxy_slot / fetch_object_proxy
    Layer 2  Dispatch mode     <->  ProxyTorchDispatchMode.__torch_dispatch__
    Layer 3  Entry point       <->  torch.fx.experimental.proxy_tensor.make_fx

The whole trick is three lines in ``proxy_call``: build a graph node, let the
real kernel run, then bind the concrete output tensor to that node.
"""

from __future__ import annotations

import operator
import time
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# ---------------------------------------------------------------------------
# Layer 0: the graph is an ordered list of rows. No adjacency maps needed,
# because every node refers to earlier nodes by object identity and the
# execution order is the row order.
# ---------------------------------------------------------------------------


class Node:
    """One row: a placeholder input, a call, or the return value."""

    def __init__(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str,
    ) -> None:
        self.kind = kind
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __repr__(self) -> str:
        if self.kind == "placeholder":
            return f"%{self.name} : placeholder"
        if self.kind == "output":
            return f"return {_format(self.args[0])}"
        return f"%{self.name} = {_target_name(self.target)}({_format(self.args)})"


class Graph:
    def __init__(self) -> None:
        self.nodes: list[Node] = []

    def __repr__(self) -> str:
        body = "\n".join("    " + repr(node) for node in self.nodes)
        return "graph():\n" + body


def _target_name(target: Any) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    return getattr(target, "__name__", str(target))


def _format(value: Any) -> str:
    if isinstance(value, Node):
        return f"%{value.name}"
    if isinstance(value, tuple):
        inner = ", ".join(_format(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    return repr(value)


# ---------------------------------------------------------------------------
# Layer 1: Proxy is a handle to a not-yet-computed value. The tracer owns the
# graph plus a memo that answers "which node produced this tensor?".
# ---------------------------------------------------------------------------


class Proxy:
    """Handle for a value that only exists as a row in the graph so far."""

    def __init__(self, node: Node, tracer: "PythonKeyTracer") -> None:
        self.node = node
        self.tracer = tracer

    def __getitem__(self, index: int) -> "Proxy":
        # Indexing a multi-output op materialises a getitem row on demand. This
        # is where the %getitem nodes in real FX graphs come from: nothing
        # special-cases multi-output ops, they just get indexed like any value.
        return self.tracer.create_proxy(
            "call_function", operator.getitem, (self, index), {}
        )

    def __repr__(self) -> str:
        return f"%{self.node.name}"


class PythonKeyTracer:
    """Creates nodes and resolves tensors back to the nodes that produced them.

    The memo stands in for the ``proxy_slot`` that PyTorch attaches to a tensor.
    The real one is keyed by tracer and holds a weak reference; keying by
    ``id()`` and keeping the tensor alive is equivalent for a single tracer.
    """

    def __init__(self) -> None:
        self.graph = Graph()
        self._memo: dict[int, tuple[torch.Tensor, Proxy]] = {}
        self._counts: dict[str, int] = {}

    def _fresh_name(self, target: Any) -> str:
        base = _target_name(target).replace(".", "_")
        count = self._counts.get(base, 0)
        self._counts[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def create_node(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Node:
        node = Node(kind, target, args, kwargs, self._fresh_name(target))
        self.graph.nodes.append(node)
        return node

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Proxy:
        node = self.create_node(
            kind, target, self.create_arg(args), self.create_arg(kwargs)
        )
        return Proxy(node, self)

    def create_arg(self, value: Any) -> Any:
        """Replace proxies with their nodes, recursively through containers.

        Anything left over is a genuine constant: an int, a dtype, a list of
        sizes. Those get baked into the node as literal arguments.
        """

        if isinstance(value, Proxy):
            return value.node
        if isinstance(value, Node):
            return value
        if isinstance(value, tuple) and not isinstance(value, torch.Size):
            return tuple(self.create_arg(item) for item in value)
        if isinstance(value, list):
            return [self.create_arg(item) for item in value]
        if isinstance(value, dict):
            return {key: self.create_arg(item) for key, item in value.items()}
        return value

    def bind(self, tensor: torch.Tensor, proxy: Proxy) -> None:
        self._memo[id(tensor)] = (tensor, proxy)

    def fetch(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        entry = self._memo.get(id(value))
        if entry is None:
            raise RuntimeError(
                "tensor with shape "
                f"{tuple(value.shape)} was not produced by a traced input, so it "
                "has no node. make_fx requires every operand to be traceable."
            )
        return entry[1]


def tree_map(fn: Callable[[Any], Any], value: Any) -> Any:
    if isinstance(value, tuple) and not isinstance(value, torch.Size):
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return fn(value)


# ---------------------------------------------------------------------------
# Layer 2: the whole heart of make_fx. One dispatch callback, four steps.
# ---------------------------------------------------------------------------


class ProxyTorchDispatchMode(TorchDispatchMode):
    def __init__(self, tracer: PythonKeyTracer) -> None:
        self.tracer = tracer

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return proxy_call(self, func, args, kwargs or {})


def proxy_call(
    mode: ProxyTorchDispatchMode,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    tracer = mode.tracer

    # 1. Turn the live tensors into proxies. This is also the check that every
    #    operand is traceable; a tensor from outside the trace raises here.
    proxy_args = tree_map(tracer.fetch, args)
    proxy_kwargs = tree_map(tracer.fetch, kwargs)

    # 2. Record the node *before* running the op. If this happened afterwards,
    #    inner ops dispatched by this op would land in the graph first and the
    #    row order would no longer respect data dependencies.
    out_proxy = tracer.create_proxy("call_function", func, proxy_args, proxy_kwargs)

    # 3. Let the real kernel run. No manual "disable myself" is needed: the
    #    dispatcher pops this mode before invoking __torch_dispatch__, so a
    #    direct func(*args) cannot recurse. Real make_fx additionally wraps this
    #    in disable_proxy_modes_tracing(), which turns off the C++ PROXY infra
    #    mode rather than this Python-level one.
    out = func(*args, **kwargs)

    # 4. Tie the concrete outputs back to the node so later ops can find them.
    track_tensor_tree(out, out_proxy, tracer)

    return out


def track_tensor_tree(value: Any, proxy: Proxy, tracer: PythonKeyTracer) -> None:
    """Bind each output tensor to its proxy, projecting tuples one index at a time."""

    if isinstance(value, torch.Tensor):
        tracer.bind(value, proxy)
    elif isinstance(value, (tuple, list)) and not isinstance(value, torch.Size):
        for index, item in enumerate(value):
            track_tensor_tree(item, proxy[index], tracer)
    # SymInts and script objects are out of scope here.


# ---------------------------------------------------------------------------
# Layer 3: the entry point. Placeholders, run under the mode, emit the output.
# ---------------------------------------------------------------------------


def make_fx(fn: Callable[..., Any], *example_inputs: torch.Tensor) -> Graph:
    tracer = PythonKeyTracer()

    for index, tensor in enumerate(example_inputs):
        node = tracer.create_node("placeholder", f"x_{index}", (), {})
        tracer.bind(tensor, Proxy(node, tracer))

    # The function body has no idea it is being traced. It sees ordinary
    # tensors in and returns ordinary tensors out.
    with ProxyTorchDispatchMode(tracer):
        out = fn(*example_inputs)

    tracer.create_node("output", None, (tree_map(tracer.fetch, out),), {})
    return tracer.graph


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def _silu_matmul(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x @ w + b)


def _max_with_index(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.max(x, dim=-1)


def _rms_scale(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * x @ w


def _op_names(nodes: list[Any], field: str) -> list[str]:
    return [
        _target_name(node.target)
        for node in nodes
        if getattr(node, field) == "call_function"
    ]


def _demo_basic() -> None:
    print("=" * 66)
    print("1. A three-input function becomes four rows plus placeholders")
    print("=" * 66)
    inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
    print(make_fx(_silu_matmul, *inputs))


def _demo_multi_output() -> None:
    print()
    print("=" * 66)
    print("2. A multi-output op grows getitem rows because the tuple is indexed")
    print("=" * 66)
    print(make_fx(_max_with_index, torch.randn(3, 4)))


def _demo_fake_mode() -> None:
    print()
    print("=" * 66)
    print("3. FakeTensorMode makes capture shape-only instead of shape-and-data")
    print("=" * 66)
    size = 4096
    left, right = torch.randn(size, size), torch.randn(size, size)

    start = time.perf_counter()
    make_fx(lambda a, b: a @ b, left, right)
    real_seconds = time.perf_counter() - start

    from torch._subclasses import FakeTensorMode

    # `with FakeTensorMode()` with no device is the same as make_fx's
    # tracing_mode="fake", which is what aotolith.frontend.capture_fx selects.
    with FakeTensorMode():
        fake_left, fake_right = torch.randn(size, size), torch.randn(size, size)
        start = time.perf_counter()
        graph = make_fx(lambda a, b: a @ b, fake_left, fake_right)
        fake_seconds = time.perf_counter() - start

    print(graph)
    print()
    print(f"  {size}x{size} matmul, one trace each:")
    print(f"    real: {real_seconds * 1000:8.2f} ms   (computes the matmul)")
    print(f"    fake: {fake_seconds * 1000:8.2f} ms   (records shapes only)")
    print(f"    speedup: {real_seconds / fake_seconds:6.1f}x")


def _demo_against_torch() -> None:
    print()
    print("=" * 66)
    print("4. Same ops, same row count as torch's own make_fx")
    print("=" * 66)
    from torch.fx.experimental.proxy_tensor import make_fx as torch_make_fx

    inputs = (torch.randn(4, 8), torch.randn(8))
    mine = make_fx(_rms_scale, *inputs)
    reference = torch_make_fx(_rms_scale)(*inputs).graph

    mine_ops = _op_names(mine.nodes, "kind")
    reference_ops = _op_names(list(reference.nodes), "op")

    print("  mini :", mine_ops)
    print("  torch:", reference_ops)
    assert mine_ops == reference_ops, "operator sequence differs"
    assert len(mine.nodes) == len(reference.nodes), "row count differs"
    print(f"\n  {len(mine.nodes)} rows each, identical operator sequence")


def main() -> None:
    _demo_basic()
    _demo_multi_output()
    _demo_fake_mode()
    _demo_against_torch()
    print()
    print("=" * 66)
    print("Next: aotolith.frontend.capture_fx walks an FX graph like this one")
    print("and rewrites each row into an SSA Operation, keyed by node id.")
    print("=" * 66)


if __name__ == "__main__":
    main()
