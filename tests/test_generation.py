from __future__ import annotations

import mlx.core as mx
import torch

from examples.tiny_transformer_generate import (
    CACHE_CAPACITY,
    HEAD_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    PROMPT_LENGTH,
    make_decode_mask,
    make_model_inputs,
    make_prefill_mask,
    make_rotary_tables,
    next_token,
    to_mlx,
    transformer_decode,
    transformer_forward,
    transformer_prefill,
)
from transformer_subgraph_compiler import compile_callable


def test_cached_prefill_decode_generates_tokens_on_metal() -> None:
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    prompt_positions = torch.arange(PROMPT_LENGTH, dtype=torch.int64)[None, :]
    cosine, sine = make_rotary_tables()
    embedding, final_norm, lm_head, *layer_weights = make_model_inputs()
    weights = (embedding, final_norm, lm_head, *layer_weights)
    prefill_mask = make_prefill_mask()

    forward_inputs = (
        prompt,
        prompt_positions,
        cosine,
        sine,
        *weights,
    )
    forward_result = compile_callable(
        transformer_forward,
        forward_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    reference_logits = transformer_forward(*forward_inputs).detach()
    forward_actual = forward_result.plan(*to_mlx(forward_inputs))[0]
    mx.eval(forward_actual)
    assert mx.allclose(
        forward_actual,
        mx.array(reference_logits.numpy()),
        atol=1e-4,
        rtol=1e-4,
    )

    prefill_inputs = (
        prompt,
        prompt_positions,
        cosine,
        sine,
        prefill_mask,
        *weights,
    )
    prefill_result = compile_callable(
        transformer_prefill,
        prefill_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    prefill_actual = prefill_result.plan(*to_mlx(prefill_inputs))
    mx.eval(*prefill_actual)
    assert len(prefill_actual) == 1 + 2 * NUM_LAYERS
    assert all(cache.shape == (1, 4, 8, 8) for cache in prefill_actual[1:])
    assert mx.allclose(
        prefill_actual[0],
        mx.array(reference_logits.numpy()),
        atol=1e-4,
        rtol=1e-4,
    )
    for plan in (forward_result, prefill_result):
        kinds = {kernel.kind for kernel in plan.plan.kernels}
        assert "attention" in kinds
        assert "matmul" in kinds

    empty_caches = tuple(
        torch.zeros(1, NUM_HEADS, CACHE_CAPACITY, HEAD_DIM)
        for _ in range(2 * NUM_LAYERS)
    )
    decode_inputs = (
        torch.tensor([[5]], dtype=torch.int32),
        torch.tensor([[PROMPT_LENGTH]], dtype=torch.int64),
        cosine,
        sine,
        make_decode_mask(PROMPT_LENGTH + 1),
        *empty_caches,
        *weights,
    )
    decode_result = compile_callable(
        transformer_decode,
        decode_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    assert "attention" in {kernel.kind for kernel in decode_result.plan.kernels}
    assert "matmul" in {kernel.kind for kernel in decode_result.plan.kernels}

    prefill_token = next_token(prefill_actual[0])
    reference_prefill_token = next_token(mx.array(reference_logits.numpy()))
    assert prefill_token == reference_prefill_token

    generated_tokens = [prefill_token]
    reference_generated_tokens = [reference_prefill_token]
    reference_sequence = prompt[0].tolist()
    caches = prefill_actual[1:]

    for step in range(3):
        fed_token = generated_tokens[-1]
        valid_length = PROMPT_LENGTH + step + 1
        runtime_inputs = (
            mx.array([[fed_token]], dtype=mx.int32),
            mx.array([[valid_length - 1]], dtype=mx.int64),
            *(mx.array(value.numpy()) for value in (cosine, sine)),
            mx.array(make_decode_mask(valid_length).numpy()),
            *caches,
            *(mx.array(value.numpy()) for value in weights),
        )
        decode_actual = decode_result.plan(*runtime_inputs)
        mx.eval(*decode_actual)

        reference_sequence.append(fed_token)
        reference_tokens = torch.tensor(
            [reference_sequence], dtype=torch.int32
        )
        reference_positions = torch.arange(
            len(reference_sequence), dtype=torch.int64
        )[None, :]
        reference_logits = transformer_forward(
            reference_tokens,
            reference_positions,
            cosine,
            sine,
            *weights,
        )[:, -1, :]
        actual_logits = decode_actual[0][:, -1, :]
        mx.eval(actual_logits)
        assert mx.allclose(
            actual_logits,
            mx.array(reference_logits.detach().numpy()),
            atol=1e-4,
            rtol=1e-4,
        )

        actual_next_token = next_token(decode_actual[0])
        reference_next_token = next_token(
            mx.array(reference_logits.detach().numpy())
        )
        assert actual_next_token == reference_next_token
        generated_tokens.append(actual_next_token)
        reference_generated_tokens.append(reference_next_token)
        caches = decode_actual[1:]

    assert generated_tokens == reference_generated_tokens
    assert len(caches) == 2 * NUM_LAYERS


def test_padded_batch_prefill_decode_tracks_valid_lengths() -> None:
    prompts = torch.tensor(
        [[1, 2, 3, 4], [0, 5, 6, 7]],
        dtype=torch.int32,
    )
    initial_valid_lengths = (4, 3)
    prompt_positions = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 2]],
        dtype=torch.int64,
    )
    cosine, sine = make_rotary_tables()
    embedding, final_norm, lm_head, *layer_weights = make_model_inputs()
    weights = (embedding, final_norm, lm_head, *layer_weights)

    prefill_inputs = (
        prompts,
        prompt_positions,
        cosine,
        sine,
        make_prefill_mask(
            prompts.shape[0], prompts.shape[1], initial_valid_lengths
        ),
        *weights,
    )
    prefill_result = compile_callable(
        transformer_prefill,
        prefill_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    prefill_actual = prefill_result.plan(*to_mlx(prefill_inputs))
    mx.eval(*prefill_actual)
    assert all(
        cache.shape == (2, NUM_HEADS, CACHE_CAPACITY, HEAD_DIM)
        for cache in prefill_actual[1:]
    )

    generated_tokens: list[list[int]] = [[], []]
    reference_generated_tokens: list[list[int]] = [[], []]
    reference_sequences = [
        prompts[row, -valid_length:].tolist()
        for row, valid_length in enumerate(initial_valid_lengths)
    ]
    for row, reference_sequence in enumerate(reference_sequences):
        reference_logits = transformer_forward(
            torch.tensor([reference_sequence], dtype=torch.int32),
            torch.arange(len(reference_sequence), dtype=torch.int64)[None, :],
            cosine,
            sine,
            *weights,
        )[:, -1, :]
        reference_token = next_token(
            mx.array(reference_logits.detach().numpy())
        )
        actual_token = next_token(prefill_actual[0][row : row + 1])
        assert actual_token == reference_token
        generated_tokens[row].append(actual_token)
        reference_generated_tokens[row].append(reference_token)

    empty_caches = tuple(
        torch.zeros(2, NUM_HEADS, CACHE_CAPACITY, HEAD_DIM)
        for _ in range(2 * NUM_LAYERS)
    )
    decode_inputs = (
        torch.zeros(2, 1, dtype=torch.int32),
        torch.tensor([[4], [3]], dtype=torch.int64),
        cosine,
        sine,
        make_decode_mask((5, 4)),
        *empty_caches,
        *weights,
    )
    decode_result = compile_callable(
        transformer_decode,
        decode_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    assert "attention" in {kernel.kind for kernel in decode_result.plan.kernels}
    assert "matmul" in {kernel.kind for kernel in decode_result.plan.kernels}

    caches = prefill_actual[1:]
    for step in range(3):
        current_valid_lengths = tuple(
            length + step for length in initial_valid_lengths
        )
        decode_actual = decode_result.plan(
            mx.array(
                [[tokens[-1]] for tokens in generated_tokens],
                dtype=mx.int32,
            ),
            mx.array(
                [[length] for length in current_valid_lengths],
                dtype=mx.int64,
            ),
            mx.array(cosine.numpy()),
            mx.array(sine.numpy()),
            mx.array(
                make_decode_mask(
                    tuple(length + 1 for length in current_valid_lengths)
                ).numpy()
            ),
            *caches,
            *(mx.array(weight.numpy()) for weight in weights),
        )
        mx.eval(*decode_actual)

        for row, reference_sequence in enumerate(reference_sequences):
            reference_sequence.append(generated_tokens[row][-1])
            reference_logits = transformer_forward(
                torch.tensor([reference_sequence], dtype=torch.int32),
                torch.arange(
                    len(reference_sequence), dtype=torch.int64
                )[None, :],
                cosine,
                sine,
                *weights,
            )[:, -1, :]
            actual_logits = decode_actual[0][row : row + 1, -1, :]
            assert mx.allclose(
                actual_logits,
                mx.array(reference_logits.detach().numpy()),
                atol=1e-4,
                rtol=1e-4,
            )
            actual_token = next_token(decode_actual[0][row : row + 1])
            reference_token = next_token(
                mx.array(reference_logits.detach().numpy())
            )
            assert actual_token == reference_token
            generated_tokens[row].append(actual_token)
            reference_generated_tokens[row].append(reference_token)

        caches = decode_actual[1:]

    assert generated_tokens == reference_generated_tokens
    assert all(len(token_ids) == 4 for token_ids in generated_tokens)
