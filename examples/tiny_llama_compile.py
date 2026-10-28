from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import aotolith as aoto


def main() -> None:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        use_cache=False,
    )
    torch.manual_seed(11)
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
    model_ref = weakref.ref(model)

    compiled = aoto.compile_module(
        model,
        (input_ids,),
        example_kwargs={"use_cache": False},
        codegen_matmul=True,
        codegen_attention=True,
    )
    print(
        "graph:",
        f"{len(compiled.input_graph.ops)} -> {len(compiled.lowered_graph.ops)} ops",
    )
    print(
        "kernels:",
        ", ".join(
            sorted({kernel.kind for kernel in compiled.plan.kernels})
        ),
    )

    del model
    gc.collect()
    if model_ref() is not None:
        raise RuntimeError("compiled plan still retains the source module")

    actual = compiled(input_ids)[0]
    mx.eval(actual)
    max_error = float(mx.max(mx.abs(actual - mx.array(expected.numpy()))))
    print("logits shape:", tuple(actual.shape))
    print("max abs error:", f"{max_error:.3e}")
    if max_error > 1e-4:
        raise RuntimeError(f"logits mismatch: max abs error {max_error:.3e}")


if __name__ == "__main__":
    main()
