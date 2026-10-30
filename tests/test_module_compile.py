from __future__ import annotations

import gc
import weakref
from pathlib import Path

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable, compile_module


def test_compile_module_binds_state_and_does_not_retain_module() -> None:
    class TinyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 3, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(value))

    torch.manual_seed(7)
    module = TinyModule().eval()
    value = torch.randn(2, 4)
    with torch.no_grad():
        expected = module(value)
    module_ref = weakref.ref(module)

    compiled = compile_module(module, (value,))
    assert compiled.state_names == ("linear.weight",)
    assert [op.op for op in compiled.input_graph.ops] == [
        "transpose",
        "matmul",
        "relu",
    ]

    del module
    gc.collect()
    assert module_ref() is None

    actual = compiled(value)[0]
    mx.eval(actual)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-3, rtol=1e-4)


def test_compile_module_binds_registered_buffer_and_does_not_retain_module() -> None:
    class BufferModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("offset", torch.arange(4, dtype=torch.float32))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.offset

    module = BufferModule().eval()
    value = torch.ones(2, 4)
    expected = module(value)
    module_ref = weakref.ref(module)

    compiled = compile_module(module, (value,))
    assert compiled.state_names == ("offset",)

    del module
    gc.collect()
    assert module_ref() is None

    actual = compiled(value)[0]
    mx.eval(actual)
    assert mx.allclose(actual, mx.array(expected.numpy()))


def test_compile_module_binds_a_tensor_literal_built_in_forward() -> None:
    """A ``torch.tensor(...)`` inside ``forward`` is a constant of the model.

    make_fx lifts such a literal into a graph attribute, which a bare
    ``capture_fx`` call rejects because a constant cannot be a plan input.
    ``compile_module`` binds it as private state instead, in the order the graph
    reaches it, so the compiled numbers match the eager module.
    """

    class LiteralModule(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            scale = torch.tensor([0.5, 0.25, 0.125])
            bias = torch.tensor([1.0, 2.0, 3.0])
            return (value * scale + bias) * 2.0

    torch.manual_seed(41)
    module = LiteralModule().eval()
    value = torch.randn(2, 3)
    expected = module(value)

    compiled = compile_module(module, (value,))
    assert compiled.state_names == ("<constant 0>", "<constant 1>")
    # Each literal arrives as a constant plus a fresh copy, which the frontend
    # keeps as an ``alias`` in the input graph and simplification erases.
    assert [op.op for op in compiled.input_graph.ops] == [
        "alias",
        "alias",
        "mul",
        "add",
        "mul",
    ]
    # The three pointwise steps fuse into one kernel over the bound constants.
    assert [op.op for op in compiled.lowered_graph.ops] == ["fused_elementwise"]

    actual = compiled(value)[0]
    mx.eval(actual)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-6, rtol=1e-6)


def test_compile_module_binds_a_plain_tensor_attribute_as_state() -> None:
    """A tensor kept on the module without ``register_buffer`` still compiles.

    Tracing runs the forward pass under fake tensors, which reject a real
    tensor, so ``self.scale = torch.tensor(...)`` used to fail before any graph
    existed. Binding it like a buffer makes it state under its own name, and the
    module is left exactly as the caller handed it over.
    """

    class AttributeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(2, 2, bias=False)
            self.block.mask = torch.tensor([1.0, 0.5])
            self.scale = torch.tensor([2.0, 3.0])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return (self.block(value) * self.block.mask) * self.scale

    torch.manual_seed(42)
    module = AttributeModule().eval()
    value = torch.randn(3, 2)
    expected = module(value)

    compiled = compile_module(module, (value,))
    # Parameters first, then buffers in module order: the root's own tensor
    # precedes the one living on the child module.
    assert compiled.state_names == ("block.weight", "scale", "block.mask")
    # The attributes are restored: still plain tensors, still in ``__dict__``.
    assert list(module._buffers) == []
    assert list(module.block._buffers) == []
    assert "scale" in module.__dict__
    assert "mask" in module.block.__dict__

    actual = compiled(value)[0]
    mx.eval(actual)
    # MLX's f32 GPU matmul rounds its inputs more coarsely than PyTorch's CPU
    # kernel, so the comparison stays at the tolerance the other matmul tests
    # use. A swapped binding would move a value by more than that.
    assert mx.allclose(
        actual, mx.array(expected.detach().numpy()), atol=2e-3, rtol=1e-3
    )


def test_compile_module_restores_tensor_attributes_after_a_failed_capture() -> None:
    class FailingModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.tensor([2.0, 3.0])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            offset = self.scale.sum()
            raise RuntimeError(f"forward refused to trace {offset.shape}")

    module = FailingModule().eval()
    with pytest.raises(RuntimeError, match="forward refused to trace"):
        compile_module(module, (torch.randn(2, 2),))

    assert list(module._buffers) == []
    assert "scale" in module.__dict__
    assert torch.equal(module.scale, torch.tensor([2.0, 3.0]))


def test_a_tensor_in_a_container_attribute_is_reported_actionably() -> None:
    """A tensor held inside a list attribute cannot be bound, and says so."""

    class ListModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scales = [torch.tensor([1.0, 2.0])]
            self.bias = torch.tensor([0.5, 0.5])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * self.scales[0] + self.bias

    module = ListModule().eval()
    with pytest.raises(NotImplementedError) as excinfo:
        compile_module(module, (torch.randn(2, 2),))

    message = str(excinfo.value)
    assert "example_inputs" in message
    assert "register_buffer" in message
    # The plain attribute that *can* be bound is restored on the way out.
    assert list(module._buffers) == []
    assert "bias" in module.__dict__


def test_bound_constant_survives_an_artifact_round_trip(tmp_path: Path) -> None:
    class LiteralModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 3, dtype=torch.bfloat16))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return (value @ self.weight) * torch.tensor([1.0, 2.0, 1.0])

    from aotolith import load_compiled_module

    torch.manual_seed(43)
    module = LiteralModule().eval()
    value = torch.randn(2, 3, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = module(value)

    compiled = compile_module(module, (value,))
    loaded = load_compiled_module(compiled.save(tmp_path / "literal.aoto"))
    assert loaded.state_names == ("weight", "<constant 0>")

    actual = loaded(_to_mlx(value))[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


def test_compile_module_supports_cumsum_and_multi_axis_indexing() -> None:
    class IndexingModule(torch.nn.Module):
        def forward(
            self,
            values: torch.Tensor,
            rows: torch.Tensor,
            columns: torch.Tensor,
        ) -> torch.Tensor:
            offsets = torch.cumsum(rows + 1, dim=-1)
            return values[rows, columns] + offsets.to(values.dtype)[:, None]

    module = IndexingModule().eval()
    values = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    rows = torch.tensor([[1], [3]], dtype=torch.int64)
    columns = torch.tensor([[0, 2, 5]], dtype=torch.int32)
    expected = module(values, rows, columns)

    compiled = compile_module(module, (values, rows, columns))
    op_names = [op.op for op in compiled.input_graph.ops]
    assert "cumsum" in op_names
    assert "index" in op_names

    actual = compiled(values, rows, columns)[0]
    mx.eval(actual)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-6, rtol=1e-6)


def test_inplace_add_is_legalized_when_unobserved() -> None:
    class AddModule(torch.nn.Module):
        def forward(self) -> torch.Tensor:
            values = torch.arange(5)
            values.add_(0)
            return torch.cumsum(values, dim=-1)

    module = AddModule().eval()
    compiled = compile_module(module, ())
    expected = module()
    assert "add" in [op.op for op in compiled.input_graph.ops]

    actual = compiled()[0]
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(expected.numpy()))


def test_inplace_add_rejects_aliasing_observer() -> None:
    def forward(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed = value + 1
        value.add_(1)
        return observed, value

    with pytest.raises(NotImplementedError, match="aliasing observers"):
        compile_callable(forward, (torch.arange(5, dtype=torch.float32),))


def _run_single_output(compiled, *tensors: torch.Tensor) -> mx.array:
    output = compiled.plan(*[mx.array(tensor.numpy()) for tensor in tensors])[0]
    mx.eval(output)
    return output


def test_inplace_index_add_is_legalized_when_unobserved() -> None:
    def forward(
        value: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        base = value.clone()
        base.index_add_(0, indices, update)
        return base * 2

    value = torch.zeros(4)
    indices = torch.tensor([0, 2])
    update = torch.randn(2)
    compiled = compile_callable(forward, (value, indices, update))
    assert "index_add" in [op.op for op in compiled.input_graph.ops]

    actual = _run_single_output(compiled, value, indices, update)
    expected = forward(value, indices, update)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-6, rtol=1e-6)


def test_inplace_scatter_add_accumulates_duplicate_indices() -> None:
    def forward(
        value: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        base = value.clone()
        base.scatter_add_(1, indices, update)
        return base

    value = torch.zeros(2, 3)
    indices = torch.tensor([[0, 2, 2]])
    update = torch.tensor([[1.0, 2.0, 4.0]])
    compiled = compile_callable(forward, (value, indices, update))
    scatter = [op for op in compiled.input_graph.ops if op.op == "scatter"]
    assert [op.attrs for op in scatter] == [
        {"dim": 1, "reduce": "add", "include_self": True}
    ]

    actual = _run_single_output(compiled, value, indices, update)
    # The duplicate index 2 must accumulate 2 + 4 instead of keeping the last
    # write, the same way the functional `scatter_add` does.
    assert mx.allclose(actual, mx.array(forward(value, indices, update).numpy()))


def test_inplace_scatter_value_is_legalized() -> None:
    def forward(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        base = value.clone()
        base.scatter_(1, indices, 5.0)
        return base

    value = torch.zeros(2, 3)
    indices = torch.tensor([[0, 2]])
    compiled = compile_callable(forward, (value, indices))
    actual = _run_single_output(compiled, value, indices)
    assert mx.array_equal(actual, mx.array(forward(value, indices).numpy()))


@pytest.mark.parametrize("reduce_name", ["amax", "prod"])
def test_inplace_scatter_reduce_is_legalized(reduce_name: str) -> None:
    def forward(
        value: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        base = value.clone()
        base.scatter_reduce_(1, indices, update, reduce_name)
        return base

    value = torch.zeros(2, 3)
    indices = torch.tensor([[0, 2, 2]])
    update = torch.tensor([[1.0, 2.0, 4.0]])
    compiled = compile_callable(forward, (value, indices, update))
    actual = _run_single_output(compiled, value, indices, update)
    assert mx.allclose(actual, mx.array(forward(value, indices, update).numpy()))


def test_inplace_scatter_rejects_aliasing_observer() -> None:
    def forward(
        value: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = value + 1
        mutated = value.index_add_(0, indices, update)
        return observed, mutated

    with pytest.raises(NotImplementedError, match="aliasing observers"):
        compile_callable(
            forward, (torch.zeros(3), torch.tensor([0, 1]), torch.randn(2))
        )


def test_inplace_scatter_returns_the_update_without_writing_back() -> None:
    def forward(
        value: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        value.index_add_(0, indices, update)
        return value

    value = torch.zeros(3)
    indices = torch.tensor([0, 2])
    update = torch.tensor([1.5, 2.5])
    compiled = compile_callable(forward, (value, indices, update))

    actual = _run_single_output(compiled, value, indices, update)
    assert mx.allclose(
        actual, mx.array(forward(value.clone(), indices, update).numpy())
    )
    # SSA legalization returns the updated value; the tensor the caller still
    # holds is untouched, which the documentation states explicitly.
    assert mx.array_equal(mx.array(value.numpy()), mx.zeros(3))


def test_tiny_huggingface_llama_logits_match_pytorch() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        use_cache=False,
    )
    torch.manual_seed(11)
    model = transformers.LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
    model_ref = weakref.ref(model)

    compiled = compile_module(
        model,
        (input_ids,),
        example_kwargs={"use_cache": False},
        codegen_matmul=True,
        codegen_attention=True,
    )
    assert {"matmul", "attention"} <= {kernel.kind for kernel in compiled.plan.kernels}
    assert {"cumsum", "index"} <= {op.op for op in compiled.input_graph.ops}

    del model
    gc.collect()
    assert model_ref() is None

    actual = compiled(input_ids)[0]
    mx.eval(actual)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)


def test_huggingface_llama_position_ids_are_a_graph_input() -> None:
    """`position_ids` is a tensor keyword a caller legitimately varies.

    Capturing it as a constant would compile a model that ignores the caller's
    positions, which is the kind of silent wrongness that still looks like
    success: the logits match for exactly the traced values.
    """

    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        use_cache=False,
    )
    torch.manual_seed(13)
    model = transformers.LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    traced_positions = torch.tensor([[3, 2, 1, 0]], dtype=torch.int64)

    compiled = compile_module(
        model,
        (input_ids,),
        example_kwargs={"use_cache": False, "position_ids": traced_positions},
        codegen_matmul=True,
        codegen_attention=True,
    )
    assert compiled.tensor_kwarg_names == ("position_ids",)

    def run(positions: torch.Tensor) -> tuple[torch.Tensor, mx.array]:
        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                position_ids=positions,
                use_cache=False,
            ).logits
        actual = compiled(
            mx.array(input_ids.numpy()),
            position_ids=mx.array(positions.numpy()),
        )[0]
        mx.eval(actual, mx.array(expected.numpy()))
        return expected, actual

    expected, actual = run(traced_positions)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)

    rotated = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    rotated_expected, rotated_actual = run(rotated)
    assert mx.allclose(
        rotated_actual,
        mx.array(rotated_expected.numpy()),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not mx.allclose(rotated_actual, actual, atol=1e-4, rtol=1e-4)


def test_huggingface_llama_compiles_with_the_default_cache_setting() -> None:
    """``use_cache=True`` must not make compilation fail.

    It is the default, so it is the first thing anyone tries. The empty cache
    tensors it puts in the graph are concatenated away rather than reported as
    an unsupported node kind, and the logits still match PyTorch.
    """

    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    torch.manual_seed(11)
    model = transformers.LlamaForCausalLM(config).eval()
    assert model.config.use_cache is True
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits

    compiled = compile_module(
        model,
        (input_ids,),
        codegen_matmul=True,
        codegen_attention=True,
    )

    actual = compiled(input_ids)[0]
    mx.eval(actual)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)


def test_compile_module_supports_multiple_tensor_outputs() -> None:
    class MultiOutputModule(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return value + 1, value * 2

    module = MultiOutputModule().eval()
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = module(value)

    compiled = compile_module(module, (value,))
    actual = compiled(value)
    mx.eval(*actual)

    assert len(actual) == 2
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert mx.allclose(
            actual_value,
            mx.array(expected_value.numpy()),
        )


class _MaskedScaleModule(torch.nn.Module):
    """Takes a tensor keyword input and a static keyword input.

    bfloat16 throughout: the assertions then hold for exactly the arithmetic
    the compiler emits, without the tolerance that f32 matmul needs because MLX
    rounds f32 operands for its tensor cores on this hardware.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8).to(torch.bfloat16))

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        output = value @ self.weight
        if mask is not None:
            output = output + mask
        return output * scale


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _to_mlx(value: torch.Tensor) -> mx.array:
    array = mx.array(value.detach().float().numpy())
    return array.astype(mx.bfloat16 if value.dtype == torch.bfloat16 else array.dtype)


def test_tensor_keyword_arguments_become_graph_inputs() -> None:
    torch.manual_seed(23)
    module = _MaskedScaleModule().eval()
    value = _bf16(2, 8)
    mask = _bf16(2, 8)

    compiled = compile_module(
        module, (value,), example_kwargs={"mask": mask, "scale": 2.0}
    )

    assert compiled.tensor_kwarg_names == ("mask",)
    # Public input, then the promoted keyword, then the bound parameter.
    assert len(compiled.plan.graph.inputs) == 3

    # The captured value is a placeholder, not a constant: a different mask at
    # call time has to change the result.
    other_mask = (_bf16(2, 8) * 4).to(torch.bfloat16)
    with torch.no_grad():
        expected = module(value, mask=other_mask, scale=2.0)
    actual = compiled(_to_mlx(value), mask=_to_mlx(other_mask))[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


def test_tensor_keyword_arguments_are_required_by_name() -> None:
    torch.manual_seed(24)
    module = _MaskedScaleModule().eval()
    compiled = compile_module(
        module,
        (_bf16(2, 8),),
        example_kwargs={"mask": _bf16(2, 8)},
    )

    with pytest.raises(ValueError, match="missing tensor keyword inputs: mask"):
        compiled(mx.zeros((2, 8), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="unexpected keyword inputs: scale"):
        compiled(
            mx.zeros((2, 8), dtype=mx.bfloat16),
            mask=mx.zeros((2, 8), dtype=mx.bfloat16),
            scale=mx.zeros((), dtype=mx.bfloat16),
        )


def test_tensor_keyword_arguments_survive_an_artifact_round_trip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(25)
    module = _MaskedScaleModule().eval()
    value = _bf16(2, 8)
    compiled = compile_module(
        module,
        (value,),
        example_kwargs={"mask": _bf16(2, 8), "scale": 1.5},
    )

    from aotolith import load_compiled_module

    artifact = compiled.save(tmp_path / "masked.aoto")
    loaded = load_compiled_module(artifact)
    assert loaded.tensor_kwarg_names == ("mask",)

    other_mask = _bf16(2, 8)
    with torch.no_grad():
        expected = module(value, mask=other_mask, scale=1.5)
    actual = loaded(_to_mlx(value), mask=_to_mlx(other_mask))[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


class _PairedKeywordModule(torch.nn.Module):
    """A module whose keyword argument is a pair of tensors.

    Newer ``transformers`` decoder layers receive ``position_embeddings=(cos,
    sin)`` exactly like this, and a caller has to be able to vary both halves.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8).to(torch.bfloat16))

    def forward(
        self,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        rotated = value @ self.weight
        output = rotated * cos + sin
        if bias is not None:
            output = output + bias
        return output


def test_nested_tensor_keywords_promote_every_leaf_in_order() -> None:
    torch.manual_seed(31)
    module = _PairedKeywordModule().eval()
    value = _bf16(2, 8)
    cos = _bf16(2, 8)
    sin = _bf16(2, 8)
    bias = _bf16(2, 8)

    compiled = compile_module(
        module,
        (value,),
        example_kwargs={
            "position_embeddings": (cos, sin),
            "bias": bias,
        },
    )

    # One public name covers both halves; the graph still receives two inputs.
    assert compiled.tensor_kwarg_names == ("position_embeddings", "bias")
    assert len(compiled.plan.graph.inputs) == 5  # value, cos, sin, bias, weight
    shapes = [
        compiled.plan.graph.value(graph_input).type.shape
        for graph_input in compiled.plan.graph.inputs
    ]
    assert shapes == [(2, 8), (2, 8), (2, 8), (2, 8), (8, 8)]

    other_cos = (_bf16(2, 8) * 2).to(torch.bfloat16)
    other_bias = _bf16(2, 8) * 0.5
    with torch.no_grad():
        expected = module(
            value,
            position_embeddings=(other_cos, sin),
            bias=other_bias,
        )
    actual = compiled(
        _to_mlx(value),
        position_embeddings=(_to_mlx(other_cos), _to_mlx(sin)),
        bias=_to_mlx(other_bias),
    )[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)

    # Both halves vary independently: swapping only `sin` changes the result.
    other_sin = (_bf16(2, 8) * 3).to(torch.bfloat16)
    with torch.no_grad():
        swapped = module(
            value,
            position_embeddings=(other_cos, other_sin),
            bias=other_bias,
        )
    swapped_actual = compiled(
        _to_mlx(value),
        position_embeddings=(_to_mlx(other_cos), _to_mlx(other_sin)),
        bias=_to_mlx(other_bias),
    )[0]
    mx.eval(swapped_actual)
    assert mx.allclose(swapped_actual, _to_mlx(swapped), atol=1e-2, rtol=1e-2)
    assert not mx.allclose(swapped_actual, actual, atol=1e-2, rtol=1e-2)


class _GatedModule(torch.nn.Module):
    """A mapping keyword with one tensor leaf and one constant leaf."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8).to(torch.bfloat16))

    def forward(
        self,
        value: torch.Tensor,
        gate: dict[str, object] | None = None,
    ) -> torch.Tensor:
        scaled = value @ self.weight
        return scaled * gate["mask"] * gate["factor"]


def test_mapping_keywords_promote_tensors_and_keep_other_leaves_static() -> None:
    torch.manual_seed(32)
    module = _GatedModule().eval()
    value = _bf16(2, 8)
    mask = _bf16(2, 8)

    compiled = compile_module(
        module,
        (value,),
        example_kwargs={"gate": {"mask": mask, "factor": 2.5}},
    )

    assert compiled.tensor_kwarg_names == ("gate",)
    assert len(compiled.plan.graph.inputs) == 3  # value, mask, weight

    other_mask = _bf16(2, 8)
    with torch.no_grad():
        expected = module(value, gate={"mask": other_mask, "factor": 2.5})
    actual = compiled(
        _to_mlx(value),
        gate={"mask": _to_mlx(other_mask), "factor": 2.5},
    )[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)

    # A non-tensor leaf is a tracing-time constant: accepting a different one
    # would silently compute with the traced value instead of the caller's.
    with pytest.raises(ValueError, match=r"\['factor'\] is not a tensor"):
        compiled(
            _to_mlx(value),
            gate={"mask": _to_mlx(other_mask), "factor": 4.0},
        )
    with pytest.raises(ValueError, match=r"expected a mapping with keys"):
        compiled(_to_mlx(value), gate={"mask": _to_mlx(other_mask)})


def test_nested_tensor_keywords_require_the_traced_nesting() -> None:
    torch.manual_seed(33)
    module = _PairedKeywordModule().eval()
    value = _bf16(2, 8)
    cos = _bf16(2, 8)
    sin = _bf16(2, 8)
    compiled = compile_module(
        module,
        (value,),
        example_kwargs={"position_embeddings": (cos, sin)},
    )

    with pytest.raises(ValueError, match="expected a tuple of 2 values"):
        compiled(_to_mlx(value), position_embeddings=_to_mlx(cos))
    with pytest.raises(ValueError, match="expected 2 values, got 3"):
        compiled(
            _to_mlx(value),
            position_embeddings=(
                _to_mlx(cos),
                _to_mlx(sin),
                _to_mlx(cos),
            ),
        )


def test_nested_tensor_keywords_survive_an_artifact_round_trip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(34)
    module = _PairedKeywordModule().eval()
    value = _bf16(2, 8)
    cos = _bf16(2, 8)
    sin = _bf16(2, 8)
    compiled = compile_module(
        module,
        (value,),
        example_kwargs={"position_embeddings": (cos, sin)},
    )

    from aotolith import load_compiled_module, read_artifact_manifest

    artifact = compiled.save(tmp_path / "paired.aoto")
    manifest = read_artifact_manifest(artifact)
    assert [
        (entry["name"], entry["path"]) for entry in manifest["entrypoint"]["kwargs"]
    ] == [
        ("position_embeddings", [["tuple", 0]]),
        ("position_embeddings", [["tuple", 1]]),
    ]

    loaded = load_compiled_module(artifact)
    assert loaded.tensor_kwarg_names == ("position_embeddings",)

    other_cos = _bf16(2, 8)
    with torch.no_grad():
        expected = module(value, position_embeddings=(other_cos, sin))
    actual = loaded(
        _to_mlx(value),
        position_embeddings=(_to_mlx(other_cos), _to_mlx(sin)),
    )[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


class _CachedAttentionModule(torch.nn.Module):
    """A list of ``(key, value)`` pairs, the usual past-key-value layout."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 8).to(torch.bfloat16))

    def forward(
        self,
        value: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        output = value @ self.weight
        for key, cached in past_key_values:
            output = output + cached @ key
        return output


def test_deeply_nested_tensor_keywords_promote_in_traversal_order() -> None:
    torch.manual_seed(35)
    module = _CachedAttentionModule().eval()
    value = _bf16(3, 4)
    key_one = _bf16(2, 8)
    cached_one = _bf16(3, 2)
    key_two = _bf16(5, 8)
    cached_two = _bf16(1, 5)

    compiled = compile_module(
        module,
        (value,),
        example_kwargs={
            "past_key_values": [
                (key_one, cached_one),
                (key_two, cached_two),
            ]
        },
    )

    assert compiled.tensor_kwarg_names == ("past_key_values",)
    assert [
        compiled.plan.graph.value(graph_input).type.shape
        for graph_input in compiled.plan.graph.inputs
    ] == [
        (3, 4),
        (2, 8),
        (3, 2),
        (5, 8),
        (1, 5),
        (4, 8),
    ]

    other_key = _bf16(5, 8)
    with torch.no_grad():
        expected = module(
            value,
            past_key_values=[
                (key_one, cached_one),
                (other_key, cached_two),
            ],
        )
    actual = compiled(
        _to_mlx(value),
        past_key_values=[
            (_to_mlx(key_one), _to_mlx(cached_one)),
            (_to_mlx(other_key), _to_mlx(cached_two)),
        ],
    )[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


def test_tensor_keywords_in_unsupported_containers_are_reported_by_name() -> None:
    torch.manual_seed(36)
    module = _PairedKeywordModule().eval()

    with pytest.raises(
        NotImplementedError, match=r"example_kwargs\['position_embeddings'\]"
    ):
        compile_module(
            module,
            (_bf16(2, 8),),
            example_kwargs={"position_embeddings": {_bf16(2, 8)}},
        )


def test_tied_state_shares_one_buffer() -> None:
    """Tied weights are one tensor under two names and should cost one buffer.

    `remove_duplicate=False` keeps both names because the graph binds them
    separately, but converting each name on its own would keep a second copy of
    the largest tensor in the model: 272 MB for a 0.5B model's embedding.
    """

    class TiedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.randn(8, 8).to(torch.bfloat16)
            self.left = torch.nn.Parameter(shared)
            self.right = torch.nn.Parameter(shared)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value @ self.left + value @ self.right

    torch.manual_seed(26)
    module = TiedModule().eval()
    value = _bf16(2, 8)
    with torch.no_grad():
        expected = module(value)

    compiled = compile_module(module, (value,))

    assert compiled.state_names == ("left", "right")
    left, right = compiled._state_values
    assert left is right

    actual = compiled(_to_mlx(value))[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


def test_a_transposed_view_is_not_treated_as_aliased_state() -> None:
    """Sharing a data pointer is not sharing contents.

    A square tensor and its `.t()` view have the same pointer, shape and dtype
    and different strides. Deduplicating on the pointer alone would make one of
    them silently compute with the other's numbers.
    """

    class TransposedView(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.randn(8, 8).to(torch.bfloat16)
            self.direct = torch.nn.Parameter(base)
            self.transposed = torch.nn.Parameter(base.t())

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value @ self.direct + value @ self.transposed

    torch.manual_seed(27)
    module = TransposedView().eval()
    assert module.direct.data_ptr() == module.transposed.data_ptr()
    assert module.direct.stride() != module.transposed.stride()

    value = _bf16(2, 8)
    with torch.no_grad():
        expected = module(value)
    compiled = compile_module(module, (value,))

    direct, transposed = compiled._state_values
    assert direct is not transposed

    actual = compiled(_to_mlx(value))[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


def test_nested_tensor_keyword_arguments_promote_the_inner_tensor() -> None:
    """A tensor inside a list is a graph input, not a captured constant.

    The old behaviour rejected any keyword value that merely contained a
    tensor, which left `past_key_values`, `position_embeddings` and other
    structured arguments uncompilable.
    """

    class ListMaskModule(torch.nn.Module):
        def forward(
            self, value: torch.Tensor, mask: list[torch.Tensor] | None = None
        ) -> torch.Tensor:
            return value + mask[0]

    torch.manual_seed(37)
    module = ListMaskModule().eval()
    value = _bf16(2, 8)
    compiled = compile_module(
        module,
        (value,),
        example_kwargs={"mask": [_bf16(2, 8)]},
    )

    other = _bf16(2, 8)
    with torch.no_grad():
        expected = module(value, mask=[other])
    actual = compiled(_to_mlx(value), mask=[_to_mlx(other)])[0]
    mx.eval(actual)
    assert mx.allclose(actual, _to_mlx(expected), atol=1e-2, rtol=1e-2)


def test_a_keyword_input_cannot_shadow_the_plan_argument() -> None:
    module = _MaskedScaleModule().eval()
    with pytest.raises(ValueError, match="verify_index_bounds"):
        compile_module(
            module,
            (_bf16(2, 8),),
            example_kwargs={"verify_index_bounds": _bf16(2, 8)},
        )
