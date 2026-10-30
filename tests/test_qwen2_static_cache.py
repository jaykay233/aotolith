from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import pytest
import torch

from aotolith.models.qwen2 import compile_qwen2_static_cache


def _reference_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, list[int]]:
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill_logits = output.logits
        past = output.past_key_values
        tokens = [int(prefill_logits[0, -1].argmax())]
        for _ in range(steps - 1):
            output = model(
                input_ids=torch.tensor([[tokens[-1]]], dtype=torch.int64),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            tokens.append(int(output.logits[0, -1].argmax()))
            past = output.past_key_values
    return prefill_logits, tokens


def test_tiny_qwen2_static_cache_matches_pytorch_decode() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        use_cache=True,
    )
    torch.manual_seed(23)
    model = transformers.Qwen2ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    expected_logits, expected_tokens = _reference_tokens(model, input_ids, 6)
    model_ref = weakref.ref(model)

    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=4,
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )
    del model
    gc.collect()
    assert model_ref() is None

    logits, compact_cache = runtime.prefill_ids(input_ids.tolist())
    mx.eval(logits, compact_cache)
    assert logits.shape == (1, 1, 64)
    assert compact_cache.shape == (2, 2, 1, 2, 4, 8)
    assert mx.allclose(
        logits,
        mx.array(expected_logits.numpy()),
        atol=1e-5,
        rtol=1e-5,
    )

    actual_tokens = runtime.greedy_generate(
        input_ids.tolist(),
        max_new_tokens=6,
        sync_each_step=True,
    )
    mx.eval(actual_tokens)
    assert actual_tokens.shape == (1, 6)
    assert actual_tokens.tolist() == [expected_tokens]

    # The gated MLP pairs its two projections into one launch per layer, so the
    # decode plan carries one fewer kernel per layer than it has projections.
    paired = [
        op
        for op in runtime.decode.plan.graph.ops
        if op.attrs.get("epilogue") == "silu_mul_paired"
    ]
    assert len(paired) == config.num_hidden_layers
    gated = [
        op
        for op in runtime.prefill.plan.graph.ops
        if op.attrs.get("epilogue") == "silu_mul_paired"
    ]
    assert gated == []


def test_decode_token_rejects_out_of_vocabulary_token() -> None:
    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)
    _, compact_cache = runtime.prefill_ids([[1, 2, 3, 4]])
    cache = runtime.pad_cache(compact_cache)
    mx.eval(cache)

    for invalid in (64, -1, 4096):
        with pytest.raises(IndexError, match="index out of range"):
            runtime.decode_token(
                mx.array([[invalid]], dtype=mx.int64), cache, 4
            )

    # The strict check must not change the valid path.
    logits, next_cache = runtime.decode_token(
        mx.array([[7]], dtype=mx.int64), cache, 4
    )
    mx.eval(logits, next_cache)
    assert logits.shape == (1, 1, 64)
    assert next_cache.shape == cache.shape


def test_decode_token_can_skip_the_token_bounds_check() -> None:
    """The opt-out path must produce the same values as the strict one."""

    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)
    _, compact_cache = runtime.prefill_ids([[1, 2, 3, 4]])
    cache = runtime.pad_cache(compact_cache)
    mx.eval(cache)
    token = mx.array([[7]], dtype=mx.int64)

    strict_logits, strict_cache = runtime.decode_token(token, cache, 4)
    lax_logits, lax_cache = runtime.decode_token(
        token, cache, 4, verify_token=False
    )
    mx.eval(strict_logits, strict_cache, lax_logits, lax_cache)
    assert mx.array_equal(strict_logits, lax_logits)
    assert mx.array_equal(strict_cache, lax_cache)


def _tiny_model(transformers: object, seed: int):
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        use_cache=True,
    )
    torch.manual_seed(seed)
    return transformers.Qwen2ForCausalLM(config).eval()


def _tiny_runtime(transformers: object):
    runtime = compile_qwen2_static_cache(
        _tiny_model(transformers, 31),
        prompt_length=4,
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )
    return runtime


def test_decode_step_uses_the_position_of_the_token_it_embeds() -> None:
    """The new token sits at ``valid_length - 1``, not ``valid_length``.

    The rotary lookup is the only thing that distinguishes the two, so compare
    the logits themselves rather than the greedy tokens: an off-by-one shifts
    every logit while often leaving ``argmax`` unchanged.
    """

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers, 47)
    prompt = [1, 2, 3, 4]
    input_ids = torch.tensor([prompt], dtype=torch.int64)
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        prefill_logits = output.logits
        first = int(output.logits[0, -1].argmax())
        expected = model(
            input_ids=torch.tensor([[first]], dtype=torch.int64),
            past_key_values=output.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        ).logits

    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(prompt),
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )
    del model

    logits, compact_cache = runtime.prefill_ids(prompt)
    mx.eval(logits, compact_cache)
    assert mx.allclose(
        logits, mx.array(prefill_logits.numpy()), atol=1e-5, rtol=1e-5
    )
    cache = runtime.pad_cache(compact_cache)
    token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
    mx.eval(token, cache)
    assert int(token.item()) == first

    step_logits, next_cache = runtime.decode_token(
        token, cache, len(prompt) + 1
    )
    mx.eval(step_logits, next_cache)
    assert mx.allclose(
        step_logits, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5
    )


def test_decode_step_writes_only_the_newest_slot() -> None:
    """One roll plus one pair: every older slot shifts by one and nothing else."""

    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)
    _, compact_cache = runtime.prefill_ids([[1, 2, 3, 4]])
    cache = runtime.pad_cache(compact_cache)
    mx.eval(cache)

    first_cache = runtime.decode_token(
        mx.array([[7]], dtype=mx.int64), cache, 5
    )[1]
    second_cache = runtime.decode_token(
        mx.array([[11]], dtype=mx.int64), cache, 5
    )[1]
    mx.eval(first_cache, second_cache)

    capacity = runtime.cache_capacity
    # Only the newest slot depends on the token that was embedded.
    assert not mx.allclose(
        first_cache[..., capacity - 1, :], second_cache[..., capacity - 1, :]
    )
    assert mx.array_equal(
        first_cache[..., : capacity - 1, :],
        second_cache[..., : capacity - 1, :],
    )
    # The prompt still occupies the window, shifted left by the roll.
    assert mx.array_equal(
        first_cache[..., : runtime.prompt_length - 1, :],
        cache[..., 1 : runtime.prompt_length, :],
    )

def test_autotune_compiles_a_model_whose_normalization_has_no_mlx_reference() -> None:
    """Autotune must skip the MLX comparison, not fail, when there is none.

    The prefill graph keeps standalone normalization because a 128-row operand
    cannot absorb the RMS scale into its projection prologue, and normalization
    only ever lowers to a generated kernel. Autotune used to evaluate the MLX
    fallback for every op in order to have a reference, which raised
    ``NotImplementedError`` before reaching the first projection.
    """

    transformers = pytest.importorskip("transformers")
    runtime = compile_qwen2_static_cache(
        _tiny_model(transformers, 37),
        prompt_length=4,
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
        autotune=True,
    )

    assert runtime.prefill.plan.selections
    normalized = [
        selection
        for selection in runtime.prefill.plan.selections
        if selection.op == "rms_norm"
    ]
    assert normalized
    for selection in normalized:
        # No MLX reference exists, so only the kernel is benchmarked.
        assert selection.is_native
        assert [item.candidate for item in selection.benchmarks] == [
            "metal_kernel"
        ]
        assert "only valid candidate" in selection.reason

    tokens = runtime.greedy_generate(
        [[1, 2, 3, 4]], max_new_tokens=6, sync_each_step=False
    )
    mx.eval(tokens)
    assert tokens.shape == (1, 6)

@pytest.mark.parametrize("hidden_act", ["silu", "gelu", "gelu_pytorch_tanh"])
def test_adapter_matches_pytorch_for_supported_activations(hidden_act) -> None:
    """The MLP calls the layer's own activation, so every lowerable one works.

    ``silu`` is the Qwen2 default, but ``Qwen2Config`` accepts others and the
    layer layout is identical, so the compiled logits have to follow the config
    rather than a hardcoded ``silu``.
    """

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        hidden_act=hidden_act,
    )
    torch.manual_seed(24)
    model = transformers.Qwen2ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    expected_logits, expected_tokens = _reference_tokens(model, input_ids, 5)

    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=4,
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )

    logits, _ = runtime.prefill_ids(input_ids.tolist())
    mx.eval(logits)
    assert mx.allclose(
        logits,
        mx.array(expected_logits.numpy()),
        atol=1e-5,
        rtol=1e-5,
    )
    tokens = runtime.greedy_generate(
        input_ids.tolist(), max_new_tokens=5, sync_each_step=True
    )
    mx.eval(tokens)
    assert tokens.tolist() == [expected_tokens]


def test_adapter_refuses_a_mlp_activation_it_does_not_implement() -> None:
    """A parameterised activation is not pointwise, so capture must not proceed.

    The adapter now calls the layer's own ``act_fn``, which covers every
    activation the compiler can lower. ``PReLU`` carries a parameter and is not
    one of them; the pre-flight capture turns that into a message naming the
    layer instead of a bare "unsupported FX target" from inside the model.
    """

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    for layer in model.model.layers:
        layer.mlp.act_fn = torch.nn.PReLU(config.intermediate_size)

    with pytest.raises(NotImplementedError, match="cannot lower the MLP activation"):
        compile_qwen2_static_cache(model, prompt_length=4, max_new_tokens=4)


def test_adapter_refuses_sliding_window_attention() -> None:
    """The adapter builds a plain causal mask, so a sliding window is a mismatch."""

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        layer_types=["sliding_attention", "sliding_attention"],
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    assert model.model.has_sliding_layers

    with pytest.raises(NotImplementedError, match="sliding-window"):
        compile_qwen2_static_cache(model, prompt_length=4, max_new_tokens=4)


def test_adapter_accepts_the_configurations_it_does_support() -> None:
    """Silu, tied or untied embeddings, and plain MHA all stay compilable."""

    transformers = pytest.importorskip("transformers")
    for overrides in (
        {"hidden_act": "silu"},
        {"tie_word_embeddings": False},
        {"num_key_value_heads": 4},
        {"hidden_act": "silu", "tie_word_embeddings": False},
    ):
        config = transformers.Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            **{"num_key_value_heads": 2, **overrides},
        )
        model = transformers.Qwen2ForCausalLM(config).eval()
        runtime = compile_qwen2_static_cache(
            model,
            prompt_length=4,
            max_new_tokens=4,
            codegen_matmul=True,
            codegen_attention=True,
        )
        tokens = runtime.greedy_generate(
            [[1, 2, 3, 4]], max_new_tokens=3, sync_each_step=False
        )
        mx.eval(tokens)
        assert tokens.shape == (1, 3), overrides
