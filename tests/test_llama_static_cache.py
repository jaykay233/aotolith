"""The second model family: Llama through the shared static-cache adapter.

Qwen2 and Llama are the same graph - ``model.model.layers``, the four attention
projections, the three MLP projections, two norms per layer, one final norm,
``lm_head`` - with one structural difference: Qwen2's attention projections
carry a bias and Llama's do not. These tests hold the adapter to the Llama
layout and, just as importantly, hold the family boundary: a Llama entry point
must refuse a Qwen2 checkpoint and the reverse, rather than compile either one
and produce plausible numbers.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

import aotolith as aoto
from aotolith.models import decoder
from aotolith.models.llama import (
    check_llama_model,
    compile_llama_bucketed,
    compile_llama_static_cache,
    prefill_mask,
)

PROMPT = [1, 2, 3, 4]


def _config(transformers, **overrides):
    settings = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
        "use_cache": True,
    }
    settings.update(overrides)
    return transformers.LlamaConfig(**settings)


def _tiny_model(transformers, seed: int = 23, **overrides):
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(_config(transformers, **overrides)).eval()


def _reference_logits(model: torch.nn.Module, prompt: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(
            input_ids=torch.tensor([prompt], dtype=torch.int64),
            use_cache=True,
            logits_to_keep=1,
        ).logits


def _reference_tokens(model: torch.nn.Module, prompt: list[int], steps: int) -> list[int]:
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([prompt], dtype=torch.int64),
            use_cache=True,
            logits_to_keep=1,
        )
        tokens = [int(output.logits[0, -1].argmax())]
        past = output.past_key_values
        for _ in range(steps - 1):
            output = model(
                input_ids=torch.tensor([[tokens[-1]]], dtype=torch.int64),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            tokens.append(int(output.logits[0, -1].argmax()))
    return tokens


def _tiny_runtime(transformers, seed: int = 23, **overrides):
    return compile_llama_static_cache(
        _tiny_model(transformers, seed, **overrides),
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )


def test_llama_prefill_and_decode_match_pytorch() -> None:
    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers)
    expected_logits = _reference_logits(model, PROMPT)
    expected_tokens = _reference_tokens(model, PROMPT, 6)

    runtime = compile_llama_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )

    logits, compact_cache = runtime.prefill_ids([PROMPT])
    mx.eval(logits, compact_cache)
    assert logits.shape == (1, 1, 64)
    assert compact_cache.shape == (2, 2, 1, 2, len(PROMPT), 8)
    assert mx.allclose(logits, mx.array(expected_logits.numpy()), atol=1e-5, rtol=1e-5)

    tokens = runtime.greedy_generate([PROMPT], max_new_tokens=6, sync_each_step=True)
    mx.eval(tokens)
    assert tokens.shape == (1, 6)
    assert tokens.tolist() == [expected_tokens]


def test_bias_free_attention_is_not_given_a_zero_bias() -> None:
    """The packed projection has no bias entry at all for Llama.

    Registering zeros would be numerically identical and structurally wrong: the
    traced graph would carry one add per layer that the checkpoint never asks
    for, and the "which family is this" story would stop being visible in the
    plan's state.
    """

    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)

    assert not any(name.startswith("qkv_bias_") for name in runtime.prefill.state_names)
    assert not any(name.startswith("qkv_bias_") for name in runtime.decode.state_names)
    # A zero bias would be one extra elementwise add per layer on the bias-free
    # path, so the plan's own operator count is the second witness.
    assert all(
        op.attrs.get("bias") is None
        for op in runtime.prefill.lowered_graph.ops
        if op.op == "matmul"
    )


def test_qwen2_attention_keeps_its_bias() -> None:
    """The other side of the same hook: a biased family still packs its bias."""

    transformers = pytest.importorskip("transformers")
    from aotolith.models.qwen2 import compile_qwen2_static_cache

    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(23)
    model = transformers.Qwen2ForCausalLM(config).eval()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_matmul=True,
    )
    assert any(name.startswith("qkv_bias_") for name in runtime.prefill.state_names)


@pytest.mark.parametrize("num_key_value_heads", [2, 4])
@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_llama_accepts_grouped_and_plain_attention(
    num_key_value_heads: int, tie_word_embeddings: bool
) -> None:
    """Grouped-query and plain multi-head attention are both the same graph."""

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(
        transformers,
        num_key_value_heads=num_key_value_heads,
        tie_word_embeddings=tie_word_embeddings,
    )
    expected = _reference_logits(model, PROMPT)
    runtime = compile_llama_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )
    logits, _ = runtime.prefill_ids([PROMPT])
    mx.eval(logits)
    assert mx.allclose(logits, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)


def test_llama3_rope_base_is_read_from_the_config() -> None:
    """``rope_theta`` is a config field, so a Llama 3 base must not be assumed.

    The table is compared against the checkpoint's own rotary embedding, which
    is the only place the base enters: a hardcoded 10000 would leave the logits
    close enough to look plausible on a short prompt and wrong everywhere else.
    """

    transformers = pytest.importorskip("transformers")
    for theta in (10000.0, 500000.0):
        model = _tiny_model(transformers, rope_theta=theta)
        expected = _reference_logits(model, PROMPT)
        runtime = compile_llama_static_cache(
            model,
            prompt_length=len(PROMPT),
            max_new_tokens=8,
            codegen_matmul=True,
            codegen_attention=True,
        )
        logits, _ = runtime.prefill_ids([PROMPT])
        mx.eval(logits)
        assert mx.allclose(
            logits, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5
        ), theta

        positions = torch.arange(runtime.cache_capacity).unsqueeze(0)
        table_cos, table_sin = model.model.rotary_emb(
            torch.zeros(1, 1, runtime.cache_capacity, runtime.head_dim), positions
        )
        state = dict(zip(runtime.prefill.state_names, runtime.prefill._state_values))
        assert mx.allclose(
            state["cosine_table"], mx.array(table_cos[0].numpy()), atol=1e-7
        )
        assert mx.allclose(
            state["sine_table"], mx.array(table_sin[0].numpy()), atol=1e-7
        )


def test_scaled_rope_is_refused_with_the_family_named() -> None:
    transformers = pytest.importorskip("transformers")
    model = _tiny_model(
        transformers,
        rope_scaling={"rope_type": "linear", "factor": 2.0},
    )
    with pytest.raises(NotImplementedError, match="Llama RoPE scaling"):
        compile_llama_static_cache(model, prompt_length=4, max_new_tokens=4)


def test_llama_entry_point_refuses_a_qwen2_checkpoint() -> None:
    """Compiling the wrong family against the same tree is the failure to avoid."""

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(23)
    model = transformers.Qwen2ForCausalLM(config).eval()

    with pytest.raises(ValueError, match="compiles llama"):
        check_llama_model(model)
    with pytest.raises(ValueError, match="compiles llama"):
        compile_llama_static_cache(model, prompt_length=4, max_new_tokens=4)
    with pytest.raises(ValueError, match="compiles llama"):
        compile_llama_bucketed(model, prompt_lengths=[4], max_new_tokens=4)


def test_qwen2_entry_point_refuses_a_llama_checkpoint() -> None:
    transformers = pytest.importorskip("transformers")
    from aotolith.models.qwen2 import check_qwen2_model

    model = _tiny_model(transformers)
    with pytest.raises(ValueError, match="compiles qwen2"):
        check_qwen2_model(model)


def test_the_shared_entry_point_accepts_either_family() -> None:
    """The pinned family is the frontend's promise, not the adapter's."""

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers)

    runtime = decoder.compile_decoder_static_cache(
        model, prompt_length=len(PROMPT), max_new_tokens=4
    )
    tokens = runtime.greedy_generate([PROMPT], max_new_tokens=2)
    mx.eval(tokens)
    assert tokens.shape == (1, 2)

    with pytest.raises(ValueError, match="compiles qwen2"):
        decoder.compile_decoder_static_cache(
            model,
            prompt_length=len(PROMPT),
            max_new_tokens=4,
            family="qwen2",
        )


def test_an_uncovered_model_type_lists_the_covered_families() -> None:
    """A third family that has not been checked must fail, not be guessed at."""

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers)
    model.config.model_type = "mystery-decoder"

    with pytest.raises(NotImplementedError, match="mystery-decoder") as caught:
        decoder.compile_decoder_static_cache(
            model, prompt_length=len(PROMPT), max_new_tokens=4
        )
    message = str(caught.value)
    assert "Qwen2/Qwen2.5" in message
    assert "Llama" in message


def test_a_refused_checkpoint_is_a_coded_error_a_program_can_branch_on() -> None:
    """A refusal is a class and a code, not just a sentence.

    "This checkpoint is unsupported" is a different decision from "this operator
    has no lowering": the first is answered by compiling another model, the
    second by waiting for a lowering. The hint has to say which, and the code
    has to be stable, because a serving layer logs the code and not the prose.
    """

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers)
    model.config.model_type = "mystery-decoder"

    with pytest.raises(aoto.UnsupportedModelError) as caught:
        decoder.compile_decoder_static_cache(
            model, prompt_length=len(PROMPT), max_new_tokens=4
        )
    assert caught.value.code == "AOTO-1004"
    assert caught.value.hint is not None
    # The class keeps the builtin the previous release raised, so a handler
    # written against it still catches this.
    assert isinstance(caught.value, NotImplementedError)
    assert isinstance(caught.value, aoto.CompilationError)


def test_the_runtime_names_the_family_it_was_compiled_from() -> None:
    """Which family a plan came from is part of the plan's identity.

    A host logs one line per loaded model, and "llama" or "qwen2" is what makes
    two artifacts in the same directory distinguishable without opening them.
    """

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers)

    single = compile_llama_static_cache(
        model, prompt_length=len(PROMPT), max_new_tokens=4
    )
    assert single.family == "llama"
    assert single.prefill.target == single.decode.target

    bucketed = compile_llama_bucketed(
        model, prompt_lengths=[4, 8], max_new_tokens=4
    )
    assert bucketed.family == "llama"
    assert {bucket.family for bucket in bucketed.buckets} == {"llama"}


def test_llama_bucketed_padding_is_invisible() -> None:
    """Every length in a Llama bucket must agree with the eager model."""

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers, max_position_embeddings=64)
    runtime = compile_llama_bucketed(
        model,
        prompt_lengths=[8, 16],
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )

    for length in (1, 5, 8, 13, 16):
        prompt = list(range(1, length + 1))
        expected = _reference_logits(model, prompt)
        logits, cache = runtime.prefill_ids([prompt])
        mx.eval(logits, cache)
        width = runtime.bucket_for(length).prompt_length
        assert cache.shape == (2, 2, 1, 2, width + 8, 8)
        assert mx.allclose(
            logits, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5
        ), length


def test_llama_bucketed_generation_matches_a_per_length_compile() -> None:
    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers)
    bucketed = compile_llama_bucketed(
        model,
        prompt_lengths=[8, 16],
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )

    for length in (3, 8, 12):
        prompt = [list(range(1, length + 1))]
        exact = compile_llama_static_cache(
            model,
            prompt_length=length,
            max_new_tokens=8,
            codegen_matmul=True,
            codegen_attention=True,
        )
        expected = exact.greedy_generate(prompt, max_new_tokens=5, sync_each_step=True)
        actual = bucketed.greedy_generate(prompt, max_new_tokens=5, sync_each_step=True)
        mx.eval(expected, actual)
        assert actual.tolist() == expected.tolist(), length


def test_llama_buckets_share_one_copy_of_the_weights() -> None:
    transformers = pytest.importorskip("transformers")
    runtime = compile_llama_bucketed(
        _tiny_model(transformers),
        prompt_lengths=[8, 16],
        max_new_tokens=8,
        codegen_matmul=True,
    )
    narrow, wide = runtime.buckets

    def values(plan):
        return dict(zip(plan.state_names, plan._state_values))

    reference = values(narrow.prefill)
    assert reference == values(narrow.decode)
    for plan in (wide.prefill, wide.decode):
        for name, value in values(plan).items():
            if value.shape != reference[name].shape:
                assert name in {"cosine_table", "sine_table"}
                continue
            assert value is reference[name], name


def test_llama_quantized_decode_matches_the_dense_plan() -> None:
    """int4 is a weight format, not a second graph: the hooks are the only change."""

    transformers = pytest.importorskip("transformers")
    model = _tiny_model(transformers, seed=31)
    dense = compile_llama_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_matmul=True,
        codegen_attention=True,
    )
    quantized = compile_llama_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_attention=True,
        quantize=16,
    )
    assert not any(
        name.startswith("qkv_bias_") for name in quantized.prefill.state_names
    )

    dense_logits, _ = dense.prefill_ids([PROMPT])
    quant_logits, _ = quantized.prefill_ids([PROMPT])
    mx.eval(dense_logits, quant_logits)
    # 4-bit weights against a 32-wide hidden state: the point is the same shape
    # and the same tokens, not bit equality with the float checkpoint.
    assert quant_logits.shape == dense_logits.shape
    assert mx.allclose(
        mx.argmax(quant_logits[:, -1, :], axis=-1),
        mx.argmax(dense_logits[:, -1, :], axis=-1),
    )


def test_llama_memory_budget_accounts_for_a_bias_free_model() -> None:
    from aotolith.errors import MemoryBudgetError

    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)
    footprint = runtime.memory_footprint()

    expected_cache = (
        runtime.num_layers
        * 2
        * runtime.num_key_value_heads
        * runtime.cache_capacity
        * runtime.head_dim
        * mx.array(0.0, dtype=runtime.decode._state_values[0].dtype).nbytes
    )
    assert footprint.cache_bytes == expected_cache
    assert runtime.check_memory_budget(footprint.step_peak_bytes) == footprint
    with pytest.raises(MemoryBudgetError, match="budget"):
        runtime.check_memory_budget(footprint.step_peak_bytes - 1)


def test_llama_plans_round_trip_through_the_artifact_format(tmp_path) -> None:
    """The artifact format carries no family field, and should not need one."""

    from aotolith.artifact import load_compiled_module

    transformers = pytest.importorskip("transformers")
    runtime = _tiny_runtime(transformers)
    logits, cache = runtime.prefill_ids([PROMPT])
    mx.eval(logits, cache)

    prefill_path, decode_path = runtime.save(tmp_path / "llama")
    assert prefill_path.name == "llama.prefill.aoto"
    assert decode_path.name == "llama.decode.aoto"
    for path, inputs in ((prefill_path, 3), (decode_path, 4)):
        loaded = load_compiled_module(path)
        assert loaded.input_count == inputs
        assert len(loaded.lowered_graph.ops) > 0


def test_llama_prefill_mask_matches_the_qwen2_one() -> None:
    """Padding is a property of the cache layout, not of the family."""

    mask = prefill_mask(2, 4)[0, 0]
    assert np.array_equal(mask, np.array([
        [False, False, False, False],
        [False, False, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]))
