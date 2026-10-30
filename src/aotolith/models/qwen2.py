"""Static KV-cache compilation for HuggingFace Qwen2/Qwen2.5 models.

The compiler fixes every tensor shape at capture time. HuggingFace's default
``DynamicCache`` therefore cannot be passed through the graph boundary. This
adapter exposes an explicit cache ABI:

* prefill: ``input_ids`` + ``positions`` -> ``logits`` + compact cache
* decode: ``token`` + ``position`` + ``mask`` + full cache -> ``logits`` + cache

The host owns the greedy loop and the one-time left padding from prompt length
to cache capacity. Scheduling, paged allocation, batching, and serving remain
outside Aotolith.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from ..compiler import CompiledModulePlan, compile_module
from ..capture import capture_fx

# Probe points for the activation pre-flight check below. They span the range
# where activations differ, including a negative input, where SiLU is not
# monotonic.
_ACTIVATION_PROBES = (0.25, -0.75, 1.5, -2.5, 3.0, -6.0)


def _validate_activation(model: torch.nn.Module) -> None:
    """Reject an MLP activation that the compiler cannot lower.

    ``_finish_layer`` calls the layer's own ``act_fn``, so the traced graph is
    the model's computation for every activation the compiler can lower -
    ``silu``, both ``gelu`` spellings, ``relu`` and the rest of the pointwise
    set all work. What still needs rejecting is a function that is not a
    pointwise op at all: running it through the same capture the compiler uses
    turns that into a message naming the configured activation instead of a
    bare "unsupported FX target" from inside the model's capture.
    """

    probe = torch.tensor(_ACTIVATION_PROBES, dtype=torch.float32)
    for index, layer in enumerate(model.model.layers):
        activation = getattr(layer.mlp, "act_fn", None)
        if activation is None:
            raise NotImplementedError(
                f"layer {index} has no mlp.act_fn; the static-cache adapter "
                "requires a gated MLP"
            )
        try:
            capture_fx(activation, (probe,))
        except Exception as error:  # noqa: BLE001 - re-raised with context
            name = getattr(model.config, "hidden_act", "<unknown>")
            raise NotImplementedError(
                "the static-cache adapter cannot lower the MLP activation "
                f"{type(activation).__name__} (config hidden_act {name!r}) "
                f"used by layer {index}: {error}"
            ) from error


def _validate_no_sliding_attention(model: torch.nn.Module) -> None:
    """Reject a model that masks attention with anything other than causality.

    The adapter builds its own causal mask, so a config that makes layers use a
    sliding window would compile against the wrong mask instead of failing.
    Transformers exposes this on the decoder for current versions, and on the
    per-layer type list for the older ones.
    """

    if getattr(model.model, "has_sliding_layers", False):
        raise NotImplementedError(
            "sliding-window attention is not supported: the static-cache "
            "adapter builds a plain causal mask for every layer"
        )
    layer_types = getattr(model.config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)) and any(
        kind == "sliding_attention" for kind in layer_types
    ):
        raise NotImplementedError(
            "sliding-window attention is not supported: the static-cache "
            "adapter builds a plain causal mask for every layer"
        )


def _rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(value, (value.shape[-1],), weight, eps)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rotary(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return value * cosine + _rotate_half(value) * sine


class _Qwen2StaticBase(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        cache_capacity: int,
    ) -> None:
        super().__init__()
        config = model.config
        if getattr(config, "model_type", None) != "qwen2":
            raise ValueError(
                "Qwen2 static-cache adapter requires a qwen2 model, got "
                f"{getattr(config, 'model_type', None)!r}"
            )
        if getattr(config, "rope_scaling", None) is not None:
            raise NotImplementedError("Qwen2 RoPE scaling is not supported yet")
        _validate_activation(model)
        _validate_no_sliding_attention(model)

        self.model = model
        self.cache_capacity = cache_capacity
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        )
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))

        frequencies = torch.arange(cache_capacity, dtype=torch.float32)[
            :, None
        ] / torch.pow(
            torch.tensor(self.rope_theta, dtype=torch.float32),
            torch.arange(0, self.head_dim, 2, dtype=torch.float32)[None, :]
            / self.head_dim,
        )
        frequencies = torch.cat((frequencies, frequencies), dim=-1).to(
            next(model.parameters()).dtype
        )
        self.register_buffer("cosine_table", frequencies.cos(), persistent=False)
        self.register_buffer("sine_table", frequencies.sin(), persistent=False)
        # One projection per layer instead of three. Query, key and value share
        # the same normalized input, so a single matrix multiply over the packed
        # weights replaces three launches; the two key/value projections are far
        # too small to fill the GPU on their own.
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            self.register_buffer(
                f"qkv_weight_{index}",
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ),
                    dim=0,
                ).detach(),
                persistent=False,
            )
            self.register_buffer(
                f"qkv_bias_{index}",
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ),
                    dim=0,
                ).detach(),
                persistent=False,
            )

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.model.model.embed_tokens.weight)

    def _project_qkv(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.model.model.layers[layer_index]
        normalized = _rms_norm(hidden, layer.input_layernorm.weight, self.rms_norm_eps)
        projected = F.linear(
            normalized,
            self.get_buffer(f"qkv_weight_{layer_index}"),
            self.get_buffer(f"qkv_bias_{layer_index}"),
        )
        query, key, value = projected.split(
            (
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ),
            dim=-1,
        )
        return self._shape_qkv(query, key, value, hidden, positions)

    def _shape_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = hidden.shape[:2]
        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        cosine = F.embedding(positions, self.cosine_table)[:, None, :, :]
        sine = F.embedding(positions, self.sine_table)[:, None, :, :]
        return (
            _apply_rotary(query, cosine, sine),
            _apply_rotary(key, cosine, sine),
            value,
        )

    def _finish_layer(
        self,
        hidden: torch.Tensor,
        attention_output: torch.Tensor,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden.shape[:2]
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        hidden = hidden + F.linear(
            attention_output,
            layer.self_attn.o_proj.weight,
            None,
        )
        normalized = _rms_norm(
            hidden, layer.post_attention_layernorm.weight, self.rms_norm_eps
        )
        gate = F.linear(normalized, layer.mlp.gate_proj.weight, None)
        up = F.linear(normalized, layer.mlp.up_proj.weight, None)
        # The layer's own activation, so the traced graph is the model's
        # computation for every activation the compiler can lower; the
        # projection epilogue fold below recognizes ``silu`` only.
        down = F.linear(layer.mlp.act_fn(gate) * up, layer.mlp.down_proj.weight, None)
        return hidden + down

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = _rms_norm(hidden, self.model.model.norm.weight, self.rms_norm_eps)
        # The benchmark and serving path only need the next-token distribution.
        hidden = hidden[:, -1:, :]
        return F.linear(hidden, self.model.lm_head.weight, None)


class Qwen2PrefillModule(_Qwen2StaticBase):
    """Capture a fixed-length prompt and emit a compact per-layer KV cache."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._embed(input_ids)
        cache_values: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.model.model.layers):
            query, key, value = self._project_qkv(hidden, positions, layer_index)
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                enable_gqa=True,
            )
            hidden = self._finish_layer(hidden, attention_output, layer)
            cache_values.extend((key, value))

        cache = torch.stack(cache_values, dim=0).reshape(
            self.num_layers,
            2,
            input_ids.shape[0],
            self.num_key_value_heads,
            input_ids.shape[1],
            self.head_dim,
        )
        return self._logits(hidden), cache


class Qwen2DecodeModule(_Qwen2StaticBase):
    """Capture one decode token against a fixed-capacity packed KV cache."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._embed(input_ids)
        next_cache_values: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.model.model.layers):
            query, key, value = self._project_qkv(hidden, positions, layer_index)
            # One append moves key and value together: the cache already stores
            # both at the same slots, so a single copy replaces two.
            #
            # Writing straight to ``positions`` with a scatter reads better, but
            # MLX lowers it to an index-compare pass over the whole window while
            # a concatenate is a straight copy of the same bytes; measured on
            # decode it cost 0.18 ms per step, so the roll stays.
            next_key_value = torch.cat(
                (
                    cache[layer_index, :, :, :, 1:, :],
                    torch.stack((key, value), dim=0),
                ),
                dim=3,
            )
            attention_output = F.scaled_dot_product_attention(
                query,
                next_key_value[0],
                next_key_value[1],
                attn_mask=attention_mask,
                enable_gqa=True,
            )
            hidden = self._finish_layer(hidden, attention_output, layer)
            next_cache_values.append(next_key_value)

        next_cache = torch.stack(next_cache_values, dim=0).reshape(
            self.num_layers,
            2,
            input_ids.shape[0],
            self.num_key_value_heads,
            self.cache_capacity,
            self.head_dim,
        )
        return self._logits(hidden), next_cache


@dataclass(frozen=True, slots=True)
class Qwen2StaticCacheRuntime:
    """Executable Qwen2 prefill/decode plans plus host-side cache helpers."""

    prefill: CompiledModulePlan
    decode: CompiledModulePlan
    prompt_length: int
    max_new_tokens: int
    cache_capacity: int
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    _decode_masks: mx.array

    def prefill_ids(
        self,
        input_ids: mx.array | np.ndarray | list[int] | list[list[int]],
    ) -> tuple[mx.array, mx.array]:
        ids = _as_input_ids(input_ids)
        if ids.shape != (1, self.prompt_length):
            raise ValueError(
                f"expected prompt shape (1, {self.prompt_length}), got {ids.shape}"
            )
        positions = mx.arange(self.prompt_length, dtype=mx.int64)[None, :]
        logits, cache = self.prefill(ids, positions)
        return logits, cache

    def pad_cache(self, compact_cache: mx.array) -> mx.array:
        expected = (
            self.num_layers,
            2,
            1,
            self.num_key_value_heads,
            self.prompt_length,
            self.head_dim,
        )
        if compact_cache.shape != expected:
            raise ValueError(
                f"expected compact cache shape {expected}, got {compact_cache.shape}"
            )
        padding = mx.zeros(
            expected[:4] + (self.cache_capacity - self.prompt_length,) + expected[5:],
            dtype=compact_cache.dtype,
        )
        return mx.concatenate((padding, compact_cache), axis=4)

    def decode_token(
        self,
        token: mx.array,
        cache: mx.array,
        valid_length: int,
        *,
        verify_token: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Run one decode step and return ``(logits, cache)``.

        ``valid_length`` is the number of tokens the cache holds after this
        step, so the token being embedded sits at index ``valid_length - 1``.
        That index drives both the rotary table lookup and the attention mask.

        ``verify_token`` controls the embedding bounds checks. A device-resident
        index cannot be validated without a device sync, so the strict default
        costs one sync per step. Callers that already know the token is inside
        the vocabulary can pass ``False``; :meth:`greedy_generate` does that
        because every token it feeds back comes from an ``argmax`` over the
        logits. The rotary tables are indexed by ``valid_length - 1``, which
        this method keeps below ``cache_capacity``, so skipping the check cannot
        hide an out-of-range position lookup.
        """

        if valid_length < 1 or valid_length >= self.cache_capacity:
            raise ValueError(
                f"valid_length must be in [1, {self.cache_capacity}), "
                f"got {valid_length}"
            )
        logits, next_cache = self.decode(
            token,
            mx.array([[valid_length - 1]], dtype=mx.int64),
            self._decode_masks[valid_length - 1],
            cache,
            verify_index_bounds=verify_token,
        )
        # Decode is a chain of short bandwidth-bound kernels, so the host-side
        # graph build costs about a third of the GPU time per step. Handing the
        # step to the async executor lets the next step's build overlap instead
        # of serializing behind it; the arrays stay lazy and read correctly
        # through any later sync.
        mx.async_eval(logits, next_cache)
        return logits, next_cache

    def greedy_generate(
        self,
        input_ids: mx.array | np.ndarray | list[int] | list[list[int]],
        *,
        max_new_tokens: int | None = None,
        sync_each_step: bool = False,
    ) -> mx.array:
        steps = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if steps < 0 or steps > self.max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be in [0, {self.max_new_tokens}], got {steps}"
            )
        if steps == 0:
            return mx.zeros((1, 0), dtype=mx.int64)

        logits, compact_cache = self.prefill_ids(input_ids)
        cache = self.pad_cache(compact_cache)
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
        if sync_each_step:
            mx.eval(token, cache)
        generated = [token.reshape(-1)]
        valid_length = self.prompt_length
        for _ in range(steps - 1):
            valid_length += 1
            # Every token here is an ``argmax`` over the logits, so it always
            # indexes a real embedding row and the bounds check is redundant.
            logits, cache = self.decode_token(
                token, cache, valid_length, verify_token=False
            )
            token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
            if sync_each_step:
                mx.eval(token, cache)
            generated.append(token.reshape(-1))
        tokens = mx.stack(generated, axis=1)
        if not sync_each_step:
            mx.eval(tokens, cache)
        return tokens

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        path = Path(prefix)
        return (
            self.prefill.save(path.with_name(path.name + ".prefill.aoto")),
            self.decode.save(path.with_name(path.name + ".decode.aoto")),
        )


def compile_qwen2_static_cache(
    model: torch.nn.Module,
    *,
    prompt_length: int,
    max_new_tokens: int,
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
) -> Qwen2StaticCacheRuntime:
    """Compile a Qwen2/Qwen2.5 causal LM for greedy single-request generation.

    All shapes are static. The returned runtime releases the PyTorch module and
    retains only Aotolith plans and model state.
    """

    if prompt_length <= 0 or max_new_tokens < 0:
        raise ValueError(
            "prompt_length must be positive and max_new_tokens non-negative"
        )
    if getattr(model, "training", False):
        raise ValueError("compile_qwen2_static_cache requires model.eval()")
    cache_capacity = prompt_length + max_new_tokens
    if cache_capacity > int(model.config.max_position_embeddings):
        raise ValueError(
            "prompt_length + max_new_tokens exceeds the model's maximum "
            f"position embeddings ({model.config.max_position_embeddings})"
        )

    prefill_module = Qwen2PrefillModule(model, cache_capacity=cache_capacity).eval()
    decode_module = Qwen2DecodeModule(model, cache_capacity=cache_capacity).eval()
    example_ids = torch.zeros((1, prompt_length), dtype=torch.int64)
    example_positions = torch.arange(prompt_length, dtype=torch.int64)[None, :]
    prefill = compile_module(
        prefill_module,
        (example_ids, example_positions),
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
    )

    dtype = next(model.parameters()).dtype
    example_token = torch.zeros((1, 1), dtype=torch.int64)
    example_position = torch.zeros((1, 1), dtype=torch.int64)
    example_mask = torch.ones((1, 1, 1, cache_capacity), dtype=torch.bool)
    example_cache = torch.zeros(
        (
            prefill_module.num_layers,
            2,
            1,
            prefill_module.num_key_value_heads,
            cache_capacity,
            prefill_module.head_dim,
        ),
        dtype=dtype,
    )
    decode = compile_module(
        decode_module,
        (example_token, example_position, example_mask, example_cache),
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
    )

    masks = np.zeros((cache_capacity, 1, 1, 1, cache_capacity), dtype=np.bool_)
    for valid_length in range(1, cache_capacity + 1):
        masks[valid_length - 1, :, :, :, -valid_length:] = True
    runtime = Qwen2StaticCacheRuntime(
        prefill=prefill,
        decode=decode,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        cache_capacity=cache_capacity,
        num_layers=prefill_module.num_layers,
        num_key_value_heads=prefill_module.num_key_value_heads,
        head_dim=prefill_module.head_dim,
        _decode_masks=mx.array(masks),
    )
    # Do not leave the traced module (and a second reference to the state) in
    # the runtime through closures or exception frames.
    del prefill_module, decode_module
    return runtime


def _as_input_ids(
    input_ids: mx.array | np.ndarray | list[int] | list[list[int]],
) -> mx.array:
    if isinstance(input_ids, mx.array):
        ids = input_ids.astype(mx.int64)
    else:
        ids = mx.array(np.asarray(input_ids, dtype=np.int64))
    if ids.ndim == 1:
        ids = ids[None, :]
    return ids
