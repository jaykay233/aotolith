from __future__ import annotations

import mlx.core as mx
import torch
import torch.nn.functional as F

import aotolith as aoto

HIDDEN_SIZE = 32
NUM_HEADS = 4
HEAD_DIM = 8
INTERMEDIATE_SIZE = 64
VOCABULARY_SIZE = 48
BATCH_SIZE = 2
SEQUENCE_LENGTH = 5


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(value, (HIDDEN_SIZE,), weight, eps=1e-6)


def apply_rope(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cosine + rotated * sine


def decoder_layer(value: torch.Tensor, *parameters: torch.Tensor) -> torch.Tensor:
    (
        attention_norm,
        qkv_weight,
        output_weight,
        mlp_norm,
        gate_weight,
        up_weight,
        down_weight,
        cosine,
        sine,
    ) = parameters
    normalized = rms_norm(value, attention_norm)
    query, key, attention_value = (normalized @ qkv_weight).split(
        HIDDEN_SIZE, dim=-1
    )
    query = query.reshape(
        BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM
    ).transpose(1, 2)
    key = key.reshape(
        BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM
    ).transpose(1, 2)
    attention_value = attention_value.reshape(
        BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM
    ).transpose(1, 2)
    query = apply_rope(query, cosine, sine)
    key = apply_rope(key, cosine, sine)

    attention_output = F.scaled_dot_product_attention(
        query, key, attention_value, is_causal=True
    )
    attention_output = attention_output.transpose(1, 2).reshape(
        BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE
    )
    value = value + attention_output @ output_weight

    normalized = rms_norm(value, mlp_norm)
    gate = normalized @ gate_weight
    up = normalized @ up_weight
    mlp_output = (F.silu(gate) * up) @ down_weight
    return value + mlp_output


def tiny_transformer(
    token_ids: torch.Tensor,
    embedding: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    *layers: torch.Tensor,
) -> torch.Tensor:
    hidden = F.embedding(token_ids, embedding)
    parameters_per_layer = 9
    for layer_index in range(2):
        start = layer_index * parameters_per_layer
        hidden = decoder_layer(
            hidden,
            *layers[start : start + parameters_per_layer],
        )
    return rms_norm(hidden, final_norm) @ lm_head


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(123)
    token_ids = torch.randint(
        0, VOCABULARY_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.int32
    )
    inputs = [
        token_ids,
        torch.randn(VOCABULARY_SIZE, HIDDEN_SIZE),
        torch.randn(HIDDEN_SIZE),
        torch.randn(HIDDEN_SIZE, VOCABULARY_SIZE),
    ]

    frequencies = torch.arange(
        SEQUENCE_LENGTH, dtype=torch.float32
    )[:, None] / torch.pow(
        torch.tensor(10000.0),
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)[None, :] / HEAD_DIM,
    )
    frequencies = torch.cat((frequencies, frequencies), dim=-1)[None, None]
    cosine = frequencies.cos()
    sine = frequencies.sin()
    for _ in range(2):
        inputs.extend(
            (
                torch.randn(HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE, 3 * HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE, HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE),
                torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),
                torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),
                torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),
                cosine,
                sine,
            )
        )
    return tuple(inputs)


def main() -> None:
    inputs = make_inputs()
    result = aoto.compile_callable(
        tiny_transformer,
        inputs,
        codegen_matmul=True,
        codegen_attention=True,
    )

    print("lowered ops:", [op.op for op in result.lowered_graph.ops])
    print("kernels:", [kernel.kind for kernel in result.plan.kernels])

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(tiny_transformer(*inputs).detach().numpy())
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))
    print(f"logits shape: {actual.shape}")
    print(f"max abs error against PyTorch CPU: {max_error:.3e}")


if __name__ == "__main__":
    main()
