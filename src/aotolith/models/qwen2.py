"""Static KV-cache compilation for HuggingFace Qwen2/Qwen2.5 models.

This module is the Qwen2 frontend of :mod:`aotolith.models.decoder`. The
adapter itself is family-agnostic; what this module adds is a pinned family, so
an entry point that promises Qwen2 refuses a checkpoint of another family
instead of compiling it against the same module tree, and the public names keep
saying which family they mean.

What differs between families is read from the model, not assumed here: whether
the attention projections carry a bias (Qwen2's do, Llama's do not) and
``config.model_type``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .decoder import (
    DecoderBucketedRuntime,
    DecoderDecodeModule,
    DecoderMemoryFootprint,
    DecoderPrefillModule,
    DecoderStaticCacheRuntime,
    QuantizedDecoderDecodeModule,
    QuantizedDecoderPrefillModule,
    compile_decoder_bucketed,
    compile_decoder_static_cache,
    family_for,
    prefill_mask,
)

#: The family name this module pins.
FAMILY = "qwen2"

#: Qwen2 spellings of the shared adapter classes. They are aliases, not
#: subclasses: the pinned family is what the entry points below pass, and a
#: subclass that only changed a default would be a second thing to keep in sync.
Qwen2PrefillModule = DecoderPrefillModule
Qwen2DecodeModule = DecoderDecodeModule
Qwen2QuantizedPrefillModule = QuantizedDecoderPrefillModule
Qwen2QuantizedDecodeModule = QuantizedDecoderDecodeModule
Qwen2StaticCacheRuntime = DecoderStaticCacheRuntime
Qwen2BucketedRuntime = DecoderBucketedRuntime
Qwen2MemoryFootprint = DecoderMemoryFootprint

__all__ = [
    "FAMILY",
    "Qwen2BucketedRuntime",
    "Qwen2DecodeModule",
    "Qwen2MemoryFootprint",
    "Qwen2PrefillModule",
    "Qwen2QuantizedDecodeModule",
    "Qwen2QuantizedPrefillModule",
    "Qwen2StaticCacheRuntime",
    "check_qwen2_model",
    "compile_qwen2_bucketed",
    "compile_qwen2_static_cache",
    "prefill_mask",
]


def check_qwen2_model(model: torch.nn.Module) -> None:
    """Raise unless ``model`` is a Qwen2 checkpoint this adapter covers."""

    family_for(model, require=FAMILY)


def compile_qwen2_static_cache(
    model: torch.nn.Module,
    *,
    prompt_length: int,
    max_new_tokens: int,
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    quantize: int | None = None,
    target: str = "apple-metal",
) -> DecoderStaticCacheRuntime:
    """Compile a Qwen2/Qwen2.5 causal LM for greedy single-request generation."""

    check_qwen2_model(model)
    return compile_decoder_static_cache(
        model,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        quantize=quantize,
        family=FAMILY,
        target=target,
    )


def compile_qwen2_bucketed(
    model: torch.nn.Module,
    *,
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    codegen_matmul: bool = False,
    codegen_attention: bool = False,
    autotune: bool = False,
    quantize: int | None = None,
    target: str = "apple-metal",
) -> DecoderBucketedRuntime:
    """Compile one Qwen2 plan pair per prompt-length bucket."""

    check_qwen2_model(model)
    return compile_decoder_bucketed(
        model,
        prompt_lengths=prompt_lengths,
        max_new_tokens=max_new_tokens,
        codegen_matmul=codegen_matmul,
        codegen_attention=codegen_attention,
        autotune=autotune,
        quantize=quantize,
        family=FAMILY,
        target=target,
    )
