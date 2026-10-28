"""Compile static KV-cache prefill and decode graphs for greedy generation.

The cache is a correctness-oriented ABI demonstration. Decode shifts the full
fixed-capacity cache on every step; it is not a paged or streaming cache.
"""
from __future__ import annotations

import mlx.core as mx
import torch
import torch.nn.functional as F

from transformer_subgraph_compiler import compile_callable

BATCH_SIZE = 1
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
    query, key, attention_value = (normalized @ qkv_weight).split(
        HIDDEN_SIZE, dim=-1
    )
    sequence_length = value.shape[1]
    query = query.reshape(
        BATCH_SIZE, sequence_length, NUM_HEADS, HEAD_DIM
    ).transpose(1, 2)
    key = key.reshape(
        BATCH_SIZE, sequence_length, NUM_HEADS, HEAD_DIM
    ).transpose(1, 2)
    attention_value = attention_value.reshape(
        BATCH_SIZE, sequence_length, NUM_HEADS, HEAD_DIM
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
    sequence_length = value.shape[1]
    attention_output = attention_output.transpose(1, 2).reshape(
        BATCH_SIZE, sequence_length, HIDDEN_SIZE
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
        padding_shape = (BATCH_SIZE, NUM_HEADS, padding, HEAD_DIM)
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
    frequencies = torch.arange(
        CACHE_CAPACITY, dtype=torch.float32
    )[:, None] / torch.pow(
        torch.tensor(10000.0),
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)[None, :] / HEAD_DIM,
    )
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def make_prefill_mask() -> torch.Tensor:
    mask = torch.zeros(1, 1, PROMPT_LENGTH, CACHE_CAPACITY, dtype=torch.bool)
    start = CACHE_CAPACITY - PROMPT_LENGTH
    for query_position in range(PROMPT_LENGTH):
        mask[0, 0, query_position, start : start + query_position + 1] = True
    return mask


def make_decode_mask(valid_length: int) -> torch.Tensor:
    mask = torch.zeros(1, 1, 1, CACHE_CAPACITY, dtype=torch.bool)
    mask[0, 0, 0, -valid_length:] = True
    return mask


def to_mlx(values: tuple[torch.Tensor, ...]) -> tuple[mx.array, ...]:
    return tuple(mx.array(value.numpy()) for value in values)


def next_token(logits: mx.array) -> int:
    scores = logits[:, -1, :] if logits.ndim == 3 else logits
    return int(mx.argmax(scores, axis=-1).item())


def main() -> None:
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    prompt_positions = torch.arange(PROMPT_LENGTH, dtype=torch.int64)[None, :]
    cosine, sine = make_rotary_tables()
    embedding, final_norm, lm_head, *layer_weights = make_model_inputs()
    weights = (embedding, final_norm, lm_head, *layer_weights)

    prefill_inputs = (
        prompt,
        prompt_positions,
        cosine,
        sine,
        make_prefill_mask(),
        *weights,
    )
    prefill_result = compile_callable(
        transformer_prefill,
        prefill_inputs,
        codegen_attention=True,
        codegen_matmul=True,
    )
    prefill_outputs = prefill_result.plan(*to_mlx(prefill_inputs))
    caches = prefill_outputs[1:]
    mx.eval(*prefill_outputs)

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

    generated = [next_token(prefill_outputs[0])]
    for step in range(3):
        valid_length = PROMPT_LENGTH + step + 1
        decode_outputs = decode_result.plan(
            mx.array([[generated[-1]]], dtype=mx.int32),
            mx.array([[valid_length - 1]], dtype=mx.int64),
            mx.array(cosine.numpy()),
            mx.array(sine.numpy()),
            mx.array(make_decode_mask(valid_length).numpy()),
            *caches,
            *(mx.array(weight.numpy()) for weight in weights),
        )
        mx.eval(*decode_outputs)
        caches = decode_outputs[1:]
        generated.append(next_token(decode_outputs[0]))

    print("prompt tokens:", prompt.tolist()[0])
    print("generated tokens:", generated)
    print("cache shape:", tuple(caches[0].shape))
    print("kernels:", [kernel.kind for kernel in decode_result.plan.kernels])


if __name__ == "__main__":
    main()
