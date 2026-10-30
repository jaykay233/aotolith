"""Compile static KV-cache prefill and decode graphs for batched generation.

The batch and cache shapes are fixed, while each request can have a different
valid prompt length. The cache is a correctness-oriented ABI demonstration:
decode shifts the full fixed-capacity cache and is not a paged cache.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import torch
import torch.nn.functional as F

import aotolith as aoto

PROMPT_LENGTH = 4
CACHE_CAPACITY = 8
HIDDEN_SIZE = 32
NUM_HEADS = 4
HEAD_DIM = 8
INTERMEDIATE_SIZE = 64
VOCABULARY_SIZE = 48
NUM_LAYERS = 2
PARAMETERS_PER_LAYER = 7


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(value, (HIDDEN_SIZE,), weight, eps=1e-6)


def apply_rope(
    value: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    cosine = F.embedding(positions, cosine_table)[:, None, :, :]
    sine = F.embedding(positions, sine_table)[:, None, :, :]
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cosine + rotated * sine


def project_qkv(
    value: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = rms_norm(value, norm_weight)
    query, key, attention_value = (normalized @ qkv_weight).split(HIDDEN_SIZE, dim=-1)
    batch_size, sequence_length = value.shape[:2]
    query = query.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_DIM).transpose(
        1, 2
    )
    key = key.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    attention_value = attention_value.reshape(
        batch_size, sequence_length, NUM_HEADS, HEAD_DIM
    ).transpose(1, 2)
    return (
        apply_rope(query, cosine_table, sine_table, positions),
        apply_rope(key, cosine_table, sine_table, positions),
        attention_value,
    )


def finish_layer(
    value: torch.Tensor,
    attention_output: torch.Tensor,
    output_weight: torch.Tensor,
    mlp_norm: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length = value.shape[:2]
    attention_output = attention_output.transpose(1, 2).reshape(
        batch_size, sequence_length, HIDDEN_SIZE
    )
    value = value + attention_output @ output_weight
    normalized = rms_norm(value, mlp_norm)
    gate = normalized @ gate_weight
    up = normalized @ up_weight
    return value + (F.silu(gate) * up) @ down_weight


def transformer_forward(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    embedding: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    *layer_weights: torch.Tensor,
) -> torch.Tensor:
    hidden = F.embedding(token_ids, embedding)
    for layer_index in range(NUM_LAYERS):
        start = layer_index * PARAMETERS_PER_LAYER
        (
            attention_norm,
            qkv_weight,
            output_weight,
            mlp_norm,
            gate_weight,
            up_weight,
            down_weight,
        ) = layer_weights[start : start + PARAMETERS_PER_LAYER]
        query, key, attention_value = project_qkv(
            hidden,
            attention_norm,
            qkv_weight,
            cosine_table,
            sine_table,
            positions,
        )
        attention_output = F.scaled_dot_product_attention(
            query, key, attention_value, is_causal=True
        )
        hidden = finish_layer(
            hidden,
            attention_output,
            output_weight,
            mlp_norm,
            gate_weight,
            up_weight,
            down_weight,
        )
    return rms_norm(hidden, final_norm) @ lm_head


def cached_layer(
    value: torch.Tensor,
    attention_norm: torch.Tensor,
    qkv_weight: torch.Tensor,
    output_weight: torch.Tensor,
    mlp_norm: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    positions: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_key: torch.Tensor | None,
    cache_value: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, attention_value = project_qkv(
        value,
        attention_norm,
        qkv_weight,
        cosine_table,
        sine_table,
        positions,
    )
    if cache_key is None:
        padding = CACHE_CAPACITY - key.shape[2]
        padding_shape = (key.shape[0], NUM_HEADS, padding, HEAD_DIM)
        zeros = torch.zeros(padding_shape, dtype=key.dtype)
        next_key = torch.cat((zeros, key), dim=2)
        next_value = torch.cat((zeros, attention_value), dim=2)
    else:
        next_key = torch.cat((cache_key[:, :, 1:, :], key), dim=2)
        next_value = torch.cat((cache_value[:, :, 1:, :], attention_value), dim=2)

    attention_output = F.scaled_dot_product_attention(
        query,
        next_key,
        next_value,
        attn_mask=attention_mask,
    )
    hidden = finish_layer(
        value,
        attention_output,
        output_weight,
        mlp_norm,
        gate_weight,
        up_weight,
        down_weight,
    )
    return hidden, next_key, next_value


def transformer_prefill(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    *layer_weights: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    hidden = F.embedding(token_ids, embedding)
    caches: list[torch.Tensor] = []
    for layer_index in range(NUM_LAYERS):
        start = layer_index * PARAMETERS_PER_LAYER
        hidden, key, value = cached_layer(
            hidden,
            *layer_weights[start : start + PARAMETERS_PER_LAYER],
            cosine_table,
            sine_table,
            positions,
            attention_mask,
            None,
            None,
        )
        caches.extend((key, value))
    return (rms_norm(hidden, final_norm) @ lm_head, *caches)


def transformer_decode(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_key_0: torch.Tensor,
    cache_value_0: torch.Tensor,
    cache_key_1: torch.Tensor,
    cache_value_1: torch.Tensor,
    embedding: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    *layer_weights: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    hidden = F.embedding(token_ids, embedding)
    caches = [cache_key_0, cache_value_0, cache_key_1, cache_value_1]
    next_caches: list[torch.Tensor] = []
    for layer_index in range(NUM_LAYERS):
        start = layer_index * PARAMETERS_PER_LAYER
        hidden, key, value = cached_layer(
            hidden,
            *layer_weights[start : start + PARAMETERS_PER_LAYER],
            cosine_table,
            sine_table,
            positions,
            attention_mask,
            caches[2 * layer_index],
            caches[2 * layer_index + 1],
        )
        next_caches.extend((key, value))
    return (rms_norm(hidden, final_norm) @ lm_head, *next_caches)


def make_model_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(2026)
    embedding = torch.randn(VOCABULARY_SIZE, HIDDEN_SIZE)
    final_norm = torch.randn(HIDDEN_SIZE)
    lm_head = torch.randn(HIDDEN_SIZE, VOCABULARY_SIZE)
    layer_weights: list[torch.Tensor] = []
    for _ in range(NUM_LAYERS):
        layer_weights.extend(
            (
                torch.randn(HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE, 3 * HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE, HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),
                torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),
                torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            )
        )
    return embedding, final_norm, lm_head, *layer_weights


def make_rotary_tables() -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = torch.arange(CACHE_CAPACITY, dtype=torch.float32)[
        :, None
    ] / torch.pow(
        torch.tensor(10000.0),
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)[None, :] / HEAD_DIM,
    )
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def make_prefill_mask(
    batch_size: int = 1,
    sequence_length: int = PROMPT_LENGTH,
    valid_lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    if valid_lengths is None:
        valid_lengths = (sequence_length,) * batch_size
    if len(valid_lengths) != batch_size:
        raise ValueError("valid_lengths must contain one entry per batch row")
    if any(length <= 0 or length > sequence_length for length in valid_lengths):
        raise ValueError("valid lengths must fit the padded prompt length")

    mask = torch.zeros(batch_size, 1, sequence_length, CACHE_CAPACITY, dtype=torch.bool)
    cache_start = CACHE_CAPACITY - sequence_length
    for batch_index, valid_length in enumerate(valid_lengths):
        first_query = sequence_length - valid_length
        first_valid = cache_start + first_query
        for query_position in range(first_query, sequence_length):
            mask[
                batch_index,
                0,
                query_position,
                first_valid : cache_start + query_position + 1,
            ] = True
    return mask


def make_decode_mask(valid_lengths: int | Sequence[int]) -> torch.Tensor:
    if isinstance(valid_lengths, int):
        valid_lengths = (valid_lengths,)
    mask = torch.zeros(len(valid_lengths), 1, 1, CACHE_CAPACITY, dtype=torch.bool)
    for batch_index, valid_length in enumerate(valid_lengths):
        mask[batch_index, 0, 0, -valid_length:] = True
    return mask


def to_mlx(values: tuple[torch.Tensor, ...]) -> tuple[mx.array, ...]:
    return tuple(mx.array(value.numpy()) for value in values)


def next_token(logits: mx.array) -> int:
    scores = logits[:, -1, :] if logits.ndim == 3 else logits
    return int(mx.argmax(scores, axis=-1).item())


def main() -> None:
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
        make_prefill_mask(prompts.shape[0], prompts.shape[1], initial_valid_lengths),
        *weights,
    )
    prefill_result = aoto.compile_callable(
        transformer_prefill,
        prefill_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    prefill_outputs = prefill_result.plan(*to_mlx(prefill_inputs))
    caches = prefill_outputs[1:]
    mx.eval(*prefill_outputs)

    empty_caches = tuple(
        torch.zeros(prompts.shape[0], NUM_HEADS, CACHE_CAPACITY, HEAD_DIM)
        for _ in range(2 * NUM_LAYERS)
    )
    decode_inputs = (
        torch.tensor([[5], [8]], dtype=torch.int32),
        torch.tensor([[4], [3]], dtype=torch.int64),
        cosine,
        sine,
        make_decode_mask((5, 4)),
        *empty_caches,
        *weights,
    )
    decode_result = aoto.compile_callable(
        transformer_decode,
        decode_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )

    generated = [
        [next_token(prefill_outputs[0][row : row + 1])]
        for row in range(prompts.shape[0])
    ]
    for step in range(3):
        valid_lengths = tuple(length + step for length in initial_valid_lengths)
        decode_outputs = decode_result.plan(
            mx.array([[tokens[-1]] for tokens in generated], dtype=mx.int32),
            mx.array([[length] for length in valid_lengths], dtype=mx.int64),
            mx.array(cosine.numpy()),
            mx.array(sine.numpy()),
            mx.array(
                make_decode_mask(tuple(length + 1 for length in valid_lengths)).numpy()
            ),
            *caches,
            *(mx.array(weight.numpy()) for weight in weights),
        )
        mx.eval(*decode_outputs)
        caches = decode_outputs[1:]
        for row, tokens in enumerate(generated):
            tokens.append(next_token(decode_outputs[0][row : row + 1]))

    print("prompt tokens:", prompts.tolist())
    print("valid prompt lengths:", initial_valid_lengths)
    print("generated tokens:", generated)
    print("cache shape:", tuple(caches[0].shape))
    print(
        "kernels:",
        [kernel.kind for kernel in decode_result.plan.kernels],
    )


if __name__ == "__main__":
    main()
