#!/usr/bin/env python3
"""Compile a small Llama end to end and check it against PyTorch.

This is the shortest path through the whole compiler that exercises the second
model family: a real ``LlamaForCausalLM``, FX capture, lowering, plan
construction, and execution - and then a comparison with the same module's own
logits so "it ran" is not the standard being applied.

It downloads nothing. The config is small enough to build in memory, which is
also what makes it usable as a CI step: the shapes are checked, not the weights.

    python examples/tiny_llama_compile.py

Exits non-zero if the logits drift past ``ATOL``, if the plan disagrees with the
module's greedy continuation, or if the released module stops the plan from
running - the three ways this adapter can be quietly wrong.
"""

from __future__ import annotations

import gc

import mlx.core as mx
import numpy as np
import torch
import transformers

from aotolith.models.llama import compile_llama_static_cache

#: The tolerance the drift is held to, and the reason the generated kernels are
#: asked for below. With both matmul and attention compiled, the drift is ~1e-7
#: - fp32 round-off. Without them, the same graph runs on MLX's own kernels,
#: whose float32 matmul on this GPU lands at ~1e-3 relative (measured against a
#: float64 reference at 8e-4, flat in K, which is TF32-shaped rather than an
#: accumulation error). So the MLX fallback is a correctness-safe path but not a
#: bit-accurate one, and that difference is a property of the runtime, not of
#: this compiler.
ATOL = 1e-4

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
STEPS = 6


def _model() -> transformers.LlamaForCausalLM:
    torch.manual_seed(23)
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        use_cache=True,
    )
    return transformers.LlamaForCausalLM(config).eval()


def _reference(model: torch.nn.Module) -> tuple[np.ndarray, list[int]]:
    """The module's last-token logits and its own greedy continuation."""

    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([PROMPT], dtype=torch.int64),
            use_cache=True,
            logits_to_keep=1,
        )
        logits = output.logits[0, -1].numpy()
        tokens = [int(logits.argmax())]
        past = output.past_key_values
        for _ in range(STEPS - 1):
            output = model(
                input_ids=torch.tensor([[tokens[-1]]], dtype=torch.int64),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            tokens.append(int(output.logits[0, -1].argmax()))
    return logits, tokens


def main() -> int:
    model = _model()
    expected_logits, expected_tokens = _reference(model)

    runtime = compile_llama_static_cache(
        model,
        prompt_length=len(PROMPT),
        max_new_tokens=STEPS,
        codegen_matmul=True,
        codegen_attention=True,
    )

    print(f"family:        {runtime.family}")
    print(f"prefill ops:   {len(runtime.prefill.input_graph.ops)} captured, "
          f"{len(runtime.prefill.lowered_graph.ops)} lowered")
    print(f"decode ops:    {len(runtime.decode.input_graph.ops)} captured, "
          f"{len(runtime.decode.lowered_graph.ops)} lowered")
    print(f"kernels:       {len(runtime.prefill.plan.kernels)} prefill, "
          f"{len(runtime.decode.plan.kernels)} decode")

    logits, cache = runtime.prefill_ids([PROMPT])
    mx.eval(logits, cache)
    drift = float(np.abs(np.asarray(logits)[0, -1] - expected_logits).max())
    print(f"logit drift:   {drift:.3e} (atol {ATOL:.0e})")
    assert logits.shape == (1, 1, expected_logits.shape[-1]), logits.shape

    tokens = runtime.greedy_generate([PROMPT], max_new_tokens=STEPS, sync_each_step=True)
    mx.eval(tokens)
    got = tokens.tolist()[0]
    print(f"greedy tokens: {got}")
    assert got == expected_tokens, f"{got} != {expected_tokens}"

    # The plan holds its own copy of every parameter and buffer, so dropping the
    # module must not matter. This is the difference between a compiled plan and
    # a closure that happens to keep the module alive.
    del model
    gc.collect()
    again = runtime.greedy_generate([PROMPT], max_new_tokens=STEPS, sync_each_step=True)
    mx.eval(again)
    print(f"after release: {again.tolist()[0] == got and 'identical' or 'CHANGED'}")
    assert again.tolist()[0] == got

    if drift > ATOL:
        print(f"FAILED: logits drifted {drift:.3e} past {ATOL:.0e}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
