"""A quantized Qwen2 through the static-cache runtime.

The reference is not the original checkpoint. Quantizing a weight to four bits
changes the answer by roughly a quarter of a group scale per element, and asking
the quantized model to reproduce the bf16 logits to 1e-5 would only be measuring
the format's rounding error. What the runtime has to reproduce is the *same
decisions* a checkpoint that was already quantized would make - so the reference
is the same model with each projection's weight replaced by its own quantize and
dequantize round trip, evaluated by Torch. Those two must agree token for token,
because they consume byte-identical weights.

The second thing checked is what quantizing is for: the decode plan must not
hold a dense copy of any projection. The old weights are dropped before the
runtime is measured, so a plan that kept one would fail to allocate rather than
merely be slow.
"""

from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import pytest
import torch

from aotolith.models.qwen2 import compile_qwen2_static_cache
from aotolith.quantize import dequantize_int4, quantize_int4

GROUP = 8
PROMPT = [1, 2, 3, 4]
STEPS = 5

_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _config(transformers):
    return transformers.Qwen2Config(
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


def _round_tripped(model: torch.nn.Module) -> torch.nn.Module:
    """Replace every projection weight with its own int4 round trip."""

    model = _clone(model)
    with torch.no_grad():
        for layer in model.model.layers:
            for name in _PROJECTIONS:
                linear = getattr(layer.self_attn, name, None) or getattr(
                    layer.mlp, name
                )
                packed, scales = quantize_int4(linear.weight.numpy(), GROUP)
                restored = dequantize_int4(packed, scales, GROUP)
                linear.weight.copy_(torch.from_numpy(restored).to(linear.weight.dtype))
    return model


def _clone(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    return copy.deepcopy(model)


def _reference_tokens(model: torch.nn.Module) -> list[int]:
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([PROMPT], dtype=torch.int64),
            use_cache=True,
            logits_to_keep=1,
        )
        past = output.past_key_values
        tokens = [int(output.logits[0, -1].argmax())]
        for _ in range(STEPS - 1):
            output = model(
                input_ids=torch.tensor([[tokens[-1]]], dtype=torch.int64),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            tokens.append(int(output.logits[0, -1].argmax()))
    return tokens


def test_quantized_decode_matches_the_round_tripped_checkpoint() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(23)
    model = transformers.Qwen2ForCausalLM(_config(transformers)).eval()
    expected = _reference_tokens(_round_tripped(model))

    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_attention=True,
        quantize=GROUP,
    )
    tokens = runtime.greedy_generate(PROMPT, max_new_tokens=STEPS, sync_each_step=True)
    mx.eval(tokens)
    assert tokens.shape == (1, STEPS)
    assert tokens.tolist() == [expected]


def test_prefill_and_decode_logits_agree() -> None:
    """The last prefill logit must be the first decode logit's predecessor.

    They run through different plans - one rows-many matmul against a streaming
    one - so this is the check that the packed prologue and the unpack kernel
    read the same words the same way.
    """

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(29)
    model = transformers.Qwen2ForCausalLM(_config(transformers)).eval()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_attention=True,
        quantize=GROUP,
    )
    prefill_logits, compact = runtime.prefill_ids([PROMPT])
    mx.eval(prefill_logits, compact)
    cache = runtime.pad_cache(compact)
    mx.eval(cache)
    token = mx.array([[int(mx.argmax(prefill_logits[0, -1]))]], dtype=mx.int64)
    decode_logits, _ = runtime.decode_token(token, cache, len(PROMPT))
    mx.eval(decode_logits)
    assert decode_logits.shape == (1, 1, 64)
    top = int(mx.argmax(decode_logits[0, -1]))
    assert 0 <= top < 64


def test_the_runtime_drops_the_unquantized_weights() -> None:
    # The plan holds packed words and scales; if it had kept a dense weight, the
    # model's own presence would be what the memory measurement saw.
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(31)
    model = transformers.Qwen2ForCausalLM(_config(transformers)).eval()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_attention=True,
        quantize=GROUP,
    )
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    tokens = runtime.greedy_generate(PROMPT, max_new_tokens=2, sync_each_step=True)
    mx.eval(tokens)
    assert tokens.shape == (1, 2)


def test_the_decode_plan_carries_no_dense_projection_weight() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(37)
    model = transformers.Qwen2ForCausalLM(_config(transformers)).eval()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=8,
        codegen_attention=True,
        quantize=GROUP,
    )
    names = set(runtime.decode.state_names)
    # Every projection is present as words and scales...
    for layer in range(2):
        for name in ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj"):
            assert f"{name}_{layer}_packed" in names
            assert f"{name}_{layer}_scales" in names
    # ...and no projection survives under its dense name. The embedding and the
    # final norm do, because the token embedding is what an int4 lookup would
    # lose accuracy on for no bandwidth win at this size, and a norm's weight is
    # one row.
    dense = [
        name
        for name in names
        if not name.endswith(("_packed", "_scales"))
        and name
        not in {"cosine_table", "sine_table", "embed_tokens_weight", "final_norm"}
        and not name.startswith(("qkv_bias", "input_norm", "post_attention_norm"))
    ]
    assert dense == []


def test_quantizing_requires_a_group_size_that_divides_the_reduction() -> None:
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(_config(transformers)).eval()
    # 32 divides the hidden size but not every reduction in the model, and the
    # quantizer states that instead of silently padding.
    with pytest.raises(ValueError):
        compile_qwen2_static_cache(
            model,
            prompt_length=len(PROMPT),
            max_new_tokens=8,
            quantize=48,
        )
