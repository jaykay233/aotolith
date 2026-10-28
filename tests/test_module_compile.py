from __future__ import annotations

import gc
import weakref

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
