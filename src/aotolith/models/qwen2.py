"""Static KV-cache compilation for HuggingFace Qwen2/Qwen2.5 models.

The compiler fixes every tensor shape at capture time. HuggingFace's default
``DynamicCache`` therefore cannot be passed through the graph boundary. This
adapter exposes an explicit cache ABI:

* prefill: ``input_ids`` + ``positions`` + ``mask`` -> ``logits`` + compact cache
* decode: ``token`` + ``position`` + ``mask`` + full cache -> ``logits`` + cache

The host owns the greedy loop and the one-time left padding from prompt length
to cache capacity. Scheduling, paged allocation, batching, and serving remain
outside Aotolith.

A fixed prompt length means one pair of plans. :func:`compile_qwen2_bucketed`
compiles a pair per size bucket instead and left-pads a shorter prompt into the
smallest bucket that fits, so the host picks a bucket per request rather than
recompiling per prompt. Left padding is what keeps the cache layout: the decode
masks select the trailing window. It also means prefill needs an explicit mask,
because ``is_causal`` cannot tell a pad from a prompt token.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from ..compiler import CompiledModulePlan, compile_module
from ..capture import capture_fx
from .quantized import dequantize_i4, pack_weight

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


def prefill_mask(length: int, bucket: int) -> np.ndarray:
    """Build the rank-4 boolean mask for a left-padded prompt.

    Entry ``[query, key]`` is True when the key is a real token at or before
    the query, which is "causal and not padding" in one comparison. ``length``
    tokens are left-padded into a ``bucket``-wide plan, so the real tokens
    occupy the last ``length`` slots and the first ``length`` rotary positions.
    At ``length == bucket`` the result is the plain causal mask.
    """

    if bucket <= 0 or length <= 0 or length > bucket:
        raise ValueError(
            f"need 0 < length <= bucket, got length={length}, bucket={bucket}"
        )
    query = np.arange(bucket)[:, None]
    key = np.arange(bucket)[None, :]
    mask = (key <= query) & (key >= bucket - length)
    return mask[None, None]


class _Qwen2StaticBase(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        cache_capacity: int,
        group_size: int | None = None,
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

        self.cache_capacity = cache_capacity
        self.group_size = group_size
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
        self._install(model)

    # ------------------------------------------------------------------
    # Where the tensors come from.
    #
    # Every forward path reads a weight through one of the hooks below rather
    # than reaching into ``self.model``. That is what lets the quantized adapter
    # replace a projection's operand with an unpacked 4-bit weight - and what
    # lets it drop the float model entirely, so a compiled int4 plan holds no
    # bf16 projection weights at all.
    # ------------------------------------------------------------------

    def _install(self, model: torch.nn.Module) -> None:
        """Bind the module the dense adapter reads its weights from."""

        self.model = model
        self._activation = model.model.layers[0].mlp.act_fn
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

    #: Where each named projection lives under a decoder layer.
    _PROJECTION_OWNERS = {
        "o_proj": "self_attn",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "down_proj": "mlp",
    }

    def _weight(self, name: str, layer_index: int) -> torch.Tensor:
        """The right operand of one projection, as ``[out, in]``."""

        if name == "qkv":
            return self.get_buffer(f"qkv_weight_{layer_index}")
        if name == "lm_head":
            return self.model.lm_head.weight
        owner = getattr(self.model.model.layers[layer_index], self._PROJECTION_OWNERS[name])
        return getattr(owner, name).weight

    def _bias(self, name: str, layer_index: int) -> torch.Tensor | None:
        """The additive bias of one projection, or None for a bias-free one."""

        if name == "qkv":
            return self.get_buffer(f"qkv_bias_{layer_index}")
        return None

    def _normalization(self, name: str, layer_index: int) -> torch.Tensor:
        """One RMS-norm scale: ``input``, ``post_attention`` or ``final``."""

        if name == "final":
            return self.model.model.norm.weight
        layer = self.model.model.layers[layer_index]
        if name == "input":
            return layer.input_layernorm.weight
        return layer.post_attention_layernorm.weight

    def _embedding(self) -> torch.Tensor:
        return self.model.model.embed_tokens.weight

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self._embedding())

    def _project_qkv(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = _rms_norm(
            hidden, self._normalization("input", layer_index), self.rms_norm_eps
        )
        projected = F.linear(
            normalized,
            self._weight("qkv", layer_index),
            self._bias("qkv", layer_index),
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
        layer_index: int,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden.shape[:2]
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        hidden = hidden + F.linear(
            attention_output,
            self._weight("o_proj", layer_index),
            self._bias("o_proj", layer_index),
        )
        normalized = _rms_norm(
            hidden, self._normalization("post_attention", layer_index), self.rms_norm_eps
        )
        gate = F.linear(
            normalized,
            self._weight("gate_proj", layer_index),
            self._bias("gate_proj", layer_index),
        )
        up = F.linear(
            normalized,
            self._weight("up_proj", layer_index),
            self._bias("up_proj", layer_index),
        )
        # The layer's own activation, so the traced graph is the model's
        # computation for every activation the compiler can lower; the
        # projection epilogue fold below recognizes ``silu`` only.
        down = F.linear(
            self._activation(gate) * up,
            self._weight("down_proj", layer_index),
            self._bias("down_proj", layer_index),
        )
        return hidden + down

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = _rms_norm(hidden, self._normalization("final", 0), self.rms_norm_eps)
        # The benchmark and serving path only need the next-token distribution.
        hidden = hidden[:, -1:, :]
        return F.linear(hidden, self._weight("lm_head", 0), self._bias("lm_head", 0))


class Qwen2PrefillModule(_Qwen2StaticBase):
    """Capture a fixed-length prompt and emit a compact per-layer KV cache.

    The attention mask carries causality and padding in one tensor. A size
    bucket compiles one prompt length and left-pads anything shorter into it,
    so ``is_causal=True`` alone would let every real token attend to the pads
    sitting to its left. At full length :func:`prefill_mask` reduces to exactly
    the causal mask, so a single plan serves both cases.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._embed(input_ids)
        cache_values: list[torch.Tensor] = []
        for layer_index in range(self.num_layers):
            query, key, value = self._project_qkv(hidden, positions, layer_index)
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                enable_gqa=True,
            )
            hidden = self._finish_layer(hidden, attention_output, layer_index)
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
        for layer_index in range(self.num_layers):
            query, key, value = self._project_qkv(hidden, positions, layer_index)
            # One append moves key and value together: the cache already stores
            # both at the same slots, so a single copy replaces two.
            #
            # Writing straight to ``positions`` with a scatter reads better, but
            # MLX lowers it to an index-compare pass over the whole window while
            # a concatenate is a straight copy of the same bytes; measured on
            # decode it cost 0.18 ms per step, so the roll stays. This spelling
            # is only what the tracer records: ``fuse_kv_cache_update`` folds the
            # slice, stack, concatenate and both unpackings into one
            # ``kv_cache_append``, whose kernel is that single copy with the
            # index arithmetic inside it.
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
            hidden = self._finish_layer(hidden, attention_output, layer_index)
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


class _QuantizedProjections:
    """Weight-only int4 projections for the static-cache adapters.

    This is a mixin rather than a second adapter: the forward pass is identical,
    and only three hooks change - where a projection's operand comes from, where
    the norm scales come from, and the embedding. Everything the plan needs is
    registered as a buffer here, so the compiled plan's state is the packed
    weights and nothing else: the float model is *not* held, and no bf16
    projection weight survives into the plan.

    The weights are packed from the model at construction, which is the one
    place a checkpoint is read; after that the model can be released.
    """

    group_size: int

    #: The projections packed per decoder layer, plus the two whole-model ones.
    _PACKED_PROJECTIONS = ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")

    def _install(self, model: torch.nn.Module) -> None:
        if self.group_size is None:
            raise ValueError("a quantized adapter requires a group size")
        self._activation = model.model.layers[0].mlp.act_fn
        dtype = next(model.parameters()).dtype

        def register(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.detach(), persistent=False)

        def pack(name: str, weight: torch.Tensor) -> None:
            packed, scales = pack_weight(weight, self.group_size)
            self.register_buffer(f"{name}_packed", packed, persistent=False)
            self.register_buffer(
                f"{name}_scales", scales.to(dtype).contiguous(), persistent=False
            )

        register("embed_tokens_weight", model.model.embed_tokens.weight)
        register("final_norm", model.model.norm.weight)
        pack("lm_head_0", model.lm_head.weight)
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            pack(
                f"qkv_{index}",
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ),
                    dim=0,
                ),
            )
            register(
                f"qkv_bias_{index}",
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ),
                    dim=0,
                ),
            )
            register(f"input_norm_{index}", layer.input_layernorm.weight)
            register(
                f"post_attention_norm_{index}", layer.post_attention_layernorm.weight
            )
            for name in self._PACKED_PROJECTIONS[1:]:
                owner = getattr(layer, self._PROJECTION_OWNERS[name])
                pack(f"{name}_{index}", getattr(owner, name).weight)

    def _weight(self, name: str, layer_index: int) -> torch.Tensor:
        """The projection's weight, unpacked from the 4-bit words.

        This is where the compiler's fold happens: every call here is one
        ``dequantize_i4`` node feeding one matmul, and the pass rewrites the
        pair so the kernel streams the words instead of this matrix.
        """

        packed = self.get_buffer(f"{name}_{layer_index}_packed")
        scales = self.get_buffer(f"{name}_{layer_index}_scales")
        return dequantize_i4(packed, scales, self.group_size)

    def _bias(self, name: str, layer_index: int) -> torch.Tensor | None:
        if name == "qkv":
            return self.get_buffer(f"qkv_bias_{layer_index}")
        return None

    def _normalization(self, name: str, layer_index: int) -> torch.Tensor:
        if name == "final":
            return self.get_buffer("final_norm")
        if name == "input":
            return self.get_buffer(f"input_norm_{layer_index}")
        return self.get_buffer(f"post_attention_norm_{layer_index}")

    def _embedding(self) -> torch.Tensor:
        return self.get_buffer("embed_tokens_weight")


class Qwen2QuantizedPrefillModule(_QuantizedProjections, Qwen2PrefillModule):
    """Int4 prefill: the dense prompt pass with packed projection weights."""


class Qwen2QuantizedDecodeModule(_QuantizedProjections, Qwen2DecodeModule):
    """Int4 decode: the dense token step with packed projection weights."""


#: Which adapter classes a bucket compiles. The int4 pair is the same two
#: modules with the projection hooks replaced, so the compile path is shared.
_ADAPTER_TYPES = {
    False: (Qwen2PrefillModule, Qwen2DecodeModule),
    True: (Qwen2QuantizedPrefillModule, Qwen2QuantizedDecodeModule),
}


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
    _prefill_mask: mx.array

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
        logits, cache = self.prefill(ids, positions, self._prefill_mask)
        return logits, cache

    def prefill_padded(
        self,
        input_ids: mx.array | np.ndarray | list[int] | list[list[int]],
    ) -> tuple[mx.array, mx.array]:
        """Prefill a prompt no longer than the plan, left-padding the rest.

        The pads only ever occupy cache slots that the decode masks exclude, so
        the returned logits and the real part of the compact cache match a plan
        compiled for the exact length. That is what lets one plan serve a whole
        size bucket.
        """

        ids = _as_input_ids(input_ids)
        if ids.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {ids.shape[0]}")
        length = int(ids.shape[1])
        if length < 1 or length > self.prompt_length:
            raise ValueError(
                f"prompt length must be in [1, {self.prompt_length}], got {length}"
            )
        if length == self.prompt_length:
            return self.prefill_ids(ids)
        padding = self.prompt_length - length
        # Pad tokens stay below every real token's rotary position and are
        # masked out, so a zero id (and a zero position) is arbitrary but safe.
        padded_ids = mx.concatenate(
            (mx.zeros((1, padding), dtype=mx.int64), ids), axis=1
        )
        padded_positions = mx.concatenate(
            (
                mx.zeros((1, padding), dtype=mx.int64),
                mx.arange(length, dtype=mx.int64)[None, :],
            ),
            axis=1,
        )
        mask = mx.array(prefill_mask(length, self.prompt_length))
        logits, cache = self.prefill(padded_ids, padded_positions, mask)
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
        return _greedy_tokens(self, logits, cache, self.prompt_length, steps, sync_each_step)

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        path = Path(prefix)
        return (
            self.prefill.save(path.with_name(path.name + ".prefill.aoto")),
            self.decode.save(path.with_name(path.name + ".decode.aoto")),
        )


def _greedy_tokens(
    runtime: Qwen2StaticCacheRuntime,
    logits: mx.array,
    cache: mx.array,
    valid_length: int,
    steps: int,
    sync_each_step: bool,
) -> mx.array:
    """Run the host-side greedy loop over one bucket.

    ``valid_length`` is the real prompt length, not the bucket width: a
    left-padded bucket prefill leaves the pads outside the window the decode
    masks select, so the loop must start counting at the real length.
    """

    token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
    if sync_each_step:
        mx.eval(token, cache)
    generated = [token.reshape(-1)]
    for _ in range(steps - 1):
        valid_length += 1
        # Every token here is an ``argmax`` over the logits, so it always
        # indexes a real embedding row and the bounds check is redundant.
        logits, cache = runtime.decode_token(
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


def compile_qwen2_static_cache(
    model: torch.nn.Module,
    *,
    prompt_length: int,
    max_new_tokens: int,
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    quantize: int | None = None,
) -> Qwen2StaticCacheRuntime:
    """Compile a Qwen2/Qwen2.5 causal LM for greedy single-request generation.

    All shapes are static, and one prompt length means one pair of plans. Use
    :func:`compile_qwen2_bucketed` when the prompt length varies: it compiles a
    plan per size bucket and serves every shorter prompt by left-padding into
    the smallest bucket instead of recompiling.
    """

    return _compile_bucket(
        model,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        quantize=quantize,
    )


def _compile_bucket(
    model: torch.nn.Module,
    *,
    prompt_length: int,
    max_new_tokens: int,
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    quantize: int | None = None,
) -> Qwen2StaticCacheRuntime:
    """Compile the prefill/decode plan pair for one fixed prompt length."""

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

    prefill_type, decode_type = _ADAPTER_TYPES[quantize is not None]
    # A 4-bit weight is only worth its format if the kernel that reads it
    # streams the words; without the streaming kernel the model unpacks on MLX
    # and ends up slower than bf16. So asking for int4 turns the matmul codegen
    # on rather than leaving a trap for the caller.
    if quantize is not None:
        codegen_matmul = True
    prefill_module = prefill_type(
        model, cache_capacity=cache_capacity, group_size=quantize
    ).eval()
    decode_module = decode_type(
        model, cache_capacity=cache_capacity, group_size=quantize
    ).eval()
    example_ids = torch.zeros((1, prompt_length), dtype=torch.int64)
    example_positions = torch.arange(prompt_length, dtype=torch.int64)[None, :]
    example_prefill_mask = torch.from_numpy(
        prefill_mask(prompt_length, prompt_length)
    )
    prefill = compile_module(
        prefill_module,
        (example_ids, example_positions, example_prefill_mask),
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
        _prefill_mask=mx.array(prefill_mask(prompt_length, prompt_length)),
    )
    # Do not leave the traced module (and a second reference to the state) in
    # the runtime through closures or exception frames.
    del prefill_module, decode_module
    return runtime


def _share_plan_state(
    plans: Sequence[CompiledModulePlan],
) -> list[CompiledModulePlan]:
    """Rebind identical state entries to one array across plans.

    Every bucket traces the same module at a different sequence length, and
    prefill and decode are traced separately, so a naive bucket list holds one
    copy of the weights per plan. The weights are identical; only the rotary
    tables depend on the cache capacity. Rebinding by ``(name, shape, dtype)``
    leaves one copy of each weight for the whole bucket set.
    """

    if len(plans) < 2:
        return list(plans)
    canonical: dict[str, mx.array] = {}
    for name, value in zip(plans[0].state_names, plans[0]._state_values):
        canonical.setdefault(name, value)

    shared: list[CompiledModulePlan] = [plans[0]]
    for plan in plans[1:]:
        values = list(plan._state_values)
        changed = False
        for index, name in enumerate(plan.state_names):
            reference = canonical.get(name)
            if reference is None:
                continue
            value = values[index]
            if value.shape == reference.shape and value.dtype == reference.dtype:
                canonical.setdefault(name, value)
                if values[index] is not reference:
                    values[index] = reference
                    changed = True
        shared.append(replace(plan, _state_values=tuple(values)) if changed else plan)
    return shared


@dataclass(frozen=True, slots=True)
class Qwen2BucketedRuntime:
    """One compiled runtime per prompt-length bucket, dispatched by length.

    A request is served by the smallest bucket that can hold it, left-padded
    into that bucket's plan. Every prompt up to ``buckets[-1]`` therefore runs
    on already-compiled plans, and the only per-request host work is one
    padding copy and one mask build.
    """

    buckets: tuple[Qwen2StaticCacheRuntime, ...]
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("at least one bucket is required")
        widths = tuple(bucket.prompt_length for bucket in self.buckets)
        if list(widths) != sorted(set(widths)):
            raise ValueError(
                f"buckets must be strictly increasing, got {list(widths)}"
            )
        for bucket in self.buckets:
            if bucket.max_new_tokens != self.max_new_tokens:
                raise ValueError("every bucket must share one max_new_tokens")

    @property
    def prompt_lengths(self) -> tuple[int, ...]:
        """The bucket widths, ascending."""

        return tuple(bucket.prompt_length for bucket in self.buckets)

    @property
    def max_prompt_length(self) -> int:
        return self.buckets[-1].prompt_length

    def bucket_for(self, prompt_length: int) -> Qwen2StaticCacheRuntime:
        """Return the smallest bucket that holds ``prompt_length`` tokens."""

        for bucket in self.buckets:
            if prompt_length <= bucket.prompt_length:
                return bucket
        raise ValueError(
            f"prompt length {prompt_length} exceeds the largest bucket "
            f"({self.max_prompt_length}); compile another bucket or truncate"
        )

    def prefill_ids(
        self,
        input_ids: mx.array | np.ndarray | list[int] | list[list[int]],
    ) -> tuple[mx.array, mx.array]:
        """Prefill on the matching bucket and return ``(logits, cache)``.

        The cache is the full-capacity cache, already aligned so that the real
        tokens occupy the window the decode masks select.
        """

        ids = _as_input_ids(input_ids)
        if ids.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {ids.shape[0]}")
        bucket = self.bucket_for(int(ids.shape[1]))
        logits, compact_cache = bucket.prefill_padded(ids)
        return logits, bucket.pad_cache(compact_cache)

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

        ids = _as_input_ids(input_ids)
        if ids.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {ids.shape[0]}")
        prompt_length = int(ids.shape[1])
        bucket = self.bucket_for(prompt_length)
        logits, compact_cache = bucket.prefill_padded(ids)
        cache = bucket.pad_cache(compact_cache)
        return _greedy_tokens(bucket, logits, cache, prompt_length, steps, sync_each_step)

    def save(self, prefix: str | Path) -> tuple[Path, ...]:
        path = Path(prefix)
        saved: list[Path] = []
        for bucket in self.buckets:
            width = bucket.prompt_length
            saved.extend(
                bucket.save(path.with_name(f"{path.name}.{width}"))
            )
        return tuple(saved)


def compile_qwen2_bucketed(
    model: torch.nn.Module,
    *,
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    quantize: int | None = None,
) -> Qwen2BucketedRuntime:
    """Compile one plan pair per prompt-length bucket.

    Prompts shorter than a bucket are left-padded into it, so a serving loop
    picks a bucket once per request instead of compiling per length. The bucket
    widths are the only shape-varying knob: keep them on the boundaries the
    workload actually clusters on, since a wider bucket costs more prefill
    work for every prompt that lands in it.
    """

    widths = [int(width) for width in prompt_lengths]
    if not widths:
        raise ValueError("prompt_lengths must name at least one bucket")
    if any(width <= 0 for width in widths):
        raise ValueError(f"bucket widths must be positive, got {widths}")
    if widths != sorted(set(widths)):
        raise ValueError(f"bucket widths must be strictly increasing, got {widths}")

    runtimes = [
        _compile_bucket(
            model,
            prompt_length=width,
            max_new_tokens=max_new_tokens,
            codegen_matmul=codegen_matmul,
            codegen_attention=codegen_attention,
            autotune=autotune,
            quantize=quantize,
        )
        for width in widths
    ]
    # One copy of the weights for the whole bucket set. The plans never mutate
    # their state, so sharing the arrays is safe.
    shared = _share_plan_state(
        [plan for runtime in runtimes for plan in (runtime.prefill, runtime.decode)]
    )
    buckets = tuple(
        replace(runtime, prefill=shared[2 * index], decode=shared[2 * index + 1])
        for index, runtime in enumerate(runtimes)
    )
    return Qwen2BucketedRuntime(buckets=buckets, max_new_tokens=max_new_tokens)


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
