"""Static KV-cache compilation for HuggingFace Llama models.

This is the Llama frontend of :mod:`aotolith.models.decoder`, and the second
model family the static-cache adapter covers. It is the same graph, the same
cache ABI and the same bucket dispatch as Qwen2; what makes it a separate entry
point is that a Llama checkpoint's layout has been checked against the adapter
and that its attention is bias-free, which the shared adapter reads from the
modules.

Llama 2 and Llama 3 both use ``model_type == "llama"``. The adapter builds its
own rotary table from ``rope_theta``, so a Llama 3 checkpoint (base 500000)
works the same way as a Llama 2 one (10000) as long as ``rope_scaling`` is
unset - scaled or re-based RoPE variants (Llama 3.1's ``llama3`` scaling,
``linear``, ``dynamic``) are refused by name rather than compiled against the
wrong table.
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
FAMILY = "llama"

#: Llama spellings of the shared adapter classes.
LlamaPrefillModule = DecoderPrefillModule
LlamaDecodeModule = DecoderDecodeModule
LlamaQuantizedPrefillModule = QuantizedDecoderPrefillModule
LlamaQuantizedDecodeModule = QuantizedDecoderDecodeModule
LlamaStaticCacheRuntime = DecoderStaticCacheRuntime
LlamaBucketedRuntime = DecoderBucketedRuntime
LlamaMemoryFootprint = DecoderMemoryFootprint

__all__ = [
    "FAMILY",
    "LlamaBucketedRuntime",
    "LlamaDecodeModule",
    "LlamaMemoryFootprint",
    "LlamaPrefillModule",
    "LlamaQuantizedDecodeModule",
    "LlamaQuantizedPrefillModule",
    "LlamaStaticCacheRuntime",
    "check_llama_model",
    "compile_llama_bucketed",
    "compile_llama_static_cache",
    "prefill_mask",
]


def check_llama_model(model: torch.nn.Module) -> None:
    """Raise unless ``model`` is a Llama checkpoint this adapter covers."""

    family_for(model, require=FAMILY)


def compile_llama_static_cache(
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
    """Compile a Llama causal LM for greedy single-request generation."""

    check_llama_model(model)
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


def compile_llama_bucketed(
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
    """Compile one Llama plan pair per prompt-length bucket."""

    check_llama_model(model)
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
