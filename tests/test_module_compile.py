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
            self.register_buffer(
                "offset", torch.arange(4, dtype=torch.float32)
            )

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
    assert {"matmul", "attention"} <= {
        kernel.kind for kernel in compiled.plan.kernels
    }
    assert {"cumsum", "index"} <= {
        op.op for op in compiled.input_graph.ops
    }

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
        def forward(
            self, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.weight = torch.nn.Parameter(
            torch.randn(8, 8).to(torch.bfloat16)
        )

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


def test_nested_tensor_keyword_arguments_are_rejected_by_name() -> None:
    module = _MaskedScaleModule().eval()
    with pytest.raises(NotImplementedError, match=r"\['mask'\]"):
        compile_module(
            module,
            (_bf16(2, 8),),
            example_kwargs={"mask": [_bf16(2, 8)]},
        )


def test_a_keyword_input_cannot_shadow_the_plan_argument() -> None:
    module = _MaskedScaleModule().eval()
    with pytest.raises(ValueError, match="verify_index_bounds"):
        compile_module(
            module,
            (_bf16(2, 8),),
            example_kwargs={"verify_index_bounds": _bf16(2, 8)},
        )
