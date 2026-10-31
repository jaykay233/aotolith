from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import pytest
import torch

from aotolith.models.qwen2 import (
    compile_qwen2_bucketed,
    compile_qwen2_static_cache,
    prefill_mask,
)


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
            runtime.decode_token(mx.array([[invalid]], dtype=mx.int64), cache, 4)

    # The strict check must not change the valid path.
    logits, next_cache = runtime.decode_token(mx.array([[7]], dtype=mx.int64), cache, 4)
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
    lax_logits, lax_cache = runtime.decode_token(token, cache, 4, verify_token=False)
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
    assert mx.allclose(logits, mx.array(prefill_logits.numpy()), atol=1e-5, rtol=1e-5)
    cache = runtime.pad_cache(compact_cache)
    token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
    mx.eval(token, cache)
    assert int(token.item()) == first

    step_logits, next_cache = runtime.decode_token(token, cache, len(prompt) + 1)
    mx.eval(step_logits, next_cache)
    assert mx.allclose(step_logits, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)


def test_decode_step_writes_only_the_newest_slot() -> None:
    """One roll plus one pair: every older slot shifts by one and nothing else."""

    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)
    _, compact_cache = runtime.prefill_ids([[1, 2, 3, 4]])
    cache = runtime.pad_cache(compact_cache)
    mx.eval(cache)

    first_cache = runtime.decode_token(mx.array([[7]], dtype=mx.int64), cache, 5)[1]
    second_cache = runtime.decode_token(mx.array([[11]], dtype=mx.int64), cache, 5)[1]
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
        assert [item.candidate for item in selection.benchmarks] == ["metal_kernel"]
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


def _bucketed_model(transformers: object, seed: int = 23):
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=40,
        tie_word_embeddings=True,
        use_cache=True,
    )
    torch.manual_seed(seed)
    return transformers.Qwen2ForCausalLM(config).eval()


def _bucketed_runtime(model: torch.nn.Module, widths: list[int]):
    return compile_qwen2_bucketed(
        model,
        prompt_lengths=widths,
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )


def test_prefill_mask_is_the_causal_mask_at_full_width() -> None:
    """A full-width prompt needs no padding, so the mask is plain causality."""

    import numpy as np

    mask = prefill_mask(4, 4)
    expected = np.tril(np.ones((4, 4), dtype=np.bool_))
    assert mask.shape == (1, 1, 4, 4)
    assert np.array_equal(mask[0, 0], expected)


def test_prefill_mask_hides_the_left_padding() -> None:
    """Every pad slot is excluded, and causality still holds for real tokens."""

    mask = prefill_mask(2, 4)[0, 0]
    # Two real tokens occupy the last two slots and keep positions 0 and 1.
    assert mask.tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]
    for query in range(4):
        for key in range(4):
            assert mask[query][key] == (key <= query and key >= 2)


@pytest.mark.parametrize("length", [1, 4, 8, 12, 16])
def test_bucketed_prefill_matches_pytorch_for_every_length_in_a_bucket(length) -> None:
    """Padding must change nothing the model can see.

    Each length is compared against the eager model on the real prompt, so a
    bucket plan that let a pad token through would show up as a logit
    difference even though its own exact-length plan would agree with it.
    """

    transformers = pytest.importorskip("transformers")
    model = _bucketed_model(transformers)
    prompt = list(range(1, length + 1))
    with torch.no_grad():
        expected = model(
            input_ids=torch.tensor([prompt], dtype=torch.int64),
            use_cache=True,
            logits_to_keep=1,
        ).logits

    runtime = _bucketed_runtime(model, [8, 16])
    logits, cache = runtime.prefill_ids([prompt])
    mx.eval(logits, cache)

    width = runtime.bucket_for(length).prompt_length
    assert width in (8, 16)
    assert logits.shape == (1, 1, 64)
    assert cache.shape == (2, 2, 1, 2, width + 8, 8)
    assert mx.allclose(logits, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)


def test_bucketed_runtime_picks_the_smallest_fitting_bucket() -> None:
    transformers = pytest.importorskip("transformers")
    runtime = _bucketed_runtime(_bucketed_model(transformers), [4, 8, 16])

    assert runtime.prompt_lengths == (4, 8, 16)
    assert runtime.max_prompt_length == 16
    assert [runtime.bucket_for(length).prompt_length for length in (1, 4, 5, 8, 9, 16)] == [
        4,
        4,
        8,
        8,
        16,
        16,
    ]


def test_bucketed_runtime_rejects_a_prompt_longer_than_every_bucket() -> None:
    transformers = pytest.importorskip("transformers")
    runtime = _bucketed_runtime(_bucketed_model(transformers), [4, 8])

    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        runtime.prefill_ids([[1] * 9])
    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        runtime.greedy_generate([[1] * 9], max_new_tokens=2)


def test_bucketed_runtime_rejects_unsorted_or_duplicate_widths() -> None:
    transformers = pytest.importorskip("transformers")
    model = _bucketed_model(transformers)

    with pytest.raises(ValueError, match="strictly increasing"):
        _bucketed_runtime(model, [8, 4])
    with pytest.raises(ValueError, match="strictly increasing"):
        _bucketed_runtime(model, [8, 8])
    with pytest.raises(ValueError, match="at least one bucket"):
        _bucketed_runtime(model, [])


def test_bucketed_generate_matches_a_per_length_compile() -> None:
    """The bucket must be invisible: same tokens as compiling that length alone."""

    transformers = pytest.importorskip("transformers")
    model = _bucketed_model(transformers)
    runtime = _bucketed_runtime(model, [8, 16])

    for length in (3, 8, 11, 16):
        prompt = [list(range(1, length + 1))]
        exact = compile_qwen2_static_cache(
            model,
            prompt_length=length,
            max_new_tokens=8,
            codegen_matmul=True,
            codegen_attention=True,
        )
        expected = exact.greedy_generate(prompt, max_new_tokens=6, sync_each_step=True)
        actual = runtime.greedy_generate(prompt, max_new_tokens=6, sync_each_step=True)
        mx.eval(expected, actual)
        assert actual.shape == (1, 6)
        assert actual.tolist() == expected.tolist(), length


def test_bucketed_runtime_keeps_one_copy_of_the_weights() -> None:
    """Buckets and the prefill/decode split must not multiply the model state.

    Every plan is traced from the same module, so the parameter arrays are
    identical by construction; only the rotary tables follow the cache
    capacity. Recomputation is not needed, and neither is a second copy.
    """

    transformers = pytest.importorskip("transformers")
    runtime = _bucketed_runtime(_bucketed_model(transformers), [8, 16])
    narrow, wide = runtime.buckets

    def values(plan):
        return dict(zip(plan.state_names, plan._state_values))

    first = values(narrow.prefill)
    decode = values(narrow.decode)
    # Prefill and decode are separate traces of the same weights.
    assert first == {name: value for name, value in decode.items()}

    for plan in (wide.prefill, wide.decode):
        shared = values(plan)
        for name, value in shared.items():
            reference = first[name]
            if value.shape != reference.shape:
                # The rotary tables scale with the cache capacity.
                assert name in {"cosine_table", "sine_table"}
                continue
            assert value is reference, name


@pytest.mark.parametrize("length", [3, 8, 11])
def test_bucketed_decode_matches_a_per_length_compile_step_by_step(length) -> None:
    """Compare every decode step's logits, not just the greedy tokens.

    Matching `argmax` chains is a weak test on a tiny random model: both sides
    can collapse to the same repeated token while the distributions behind it
    differ. Per-step logits, and the cache the step wrote, are the actual
    contract the padding has to preserve.
    """

    transformers = pytest.importorskip("transformers")
    model = _bucketed_model(transformers)
    prompt = [list(range(1, length + 1))]

    exact = compile_qwen2_static_cache(
        model,
        prompt_length=length,
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )
    bucketed = _bucketed_runtime(model, [8, 16])

    exact_logits, exact_compact = exact.prefill_ids(prompt)
    bucket_logits, bucket_cache = bucketed.prefill_ids(prompt)
    exact_cache = exact.pad_cache(exact_compact)
    mx.eval(exact_logits, exact_cache, bucket_logits, bucket_cache)

    assert mx.allclose(bucket_logits, exact_logits, atol=1e-5, rtol=1e-5)
    # The real tokens live in the trailing window of both caches; the bucket's
    # extra slots are the left padding the decode masks exclude.
    assert mx.allclose(
        bucket_cache[..., -length:, :],
        exact_cache[..., -length:, :],
        atol=1e-5,
        rtol=1e-5,
    )

    valid_length = length
    for _ in range(5):
        token = mx.argmax(exact_logits[:, -1, :], axis=-1, keepdims=True).astype(
            mx.int64
        )
        valid_length += 1
        exact_logits, exact_cache = exact.decode_token(token, exact_cache, valid_length)
        bucket = bucketed.bucket_for(length)
        bucket_logits, bucket_cache = bucket.decode_token(
            token, bucket_cache, valid_length
        )
        mx.eval(exact_logits, exact_cache, bucket_logits, bucket_cache)

        assert mx.allclose(bucket_logits, exact_logits, atol=1e-5, rtol=1e-5)
        assert mx.allclose(
            bucket_cache[..., -valid_length:, :],
            exact_cache[..., -valid_length:, :],
            atol=1e-5,
            rtol=1e-5,
        )


def test_bucketed_save_writes_one_loadable_archive_per_bucket(tmp_path) -> None:
    """Saving a bucket set keeps the ABI honest for every plan it holds."""

    from aotolith.artifact import load_compiled_module

    transformers = pytest.importorskip("transformers")
    runtime = _bucketed_runtime(_bucketed_model(transformers), [8, 16])

    paths = runtime.save(tmp_path / "qwen")
    names = sorted(path.name for path in paths)
    assert names == [
        "qwen.16.decode.aoto",
        "qwen.16.prefill.aoto",
        "qwen.8.decode.aoto",
        "qwen.8.prefill.aoto",
    ]
    for path in paths:
        loaded = load_compiled_module(path)
        assert len(loaded.lowered_graph.ops) > 0
        assert loaded.input_count == (3 if "prefill" in path.name else 4)
