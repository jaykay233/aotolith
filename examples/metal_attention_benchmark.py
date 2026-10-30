from __future__ import annotations

import random
import sys
import time

import mlx.core as mx
import torch
import torch.nn.functional as F

import aotolith as aoto

_WARMUP_ELEMENTS = 2048

_ABLATION_CASES = ((1, 8, 512, 64), (1, 8, 2048, 128), (1, 64, 4096, 128))

# Each entry rewrites the emitted kernel so one stage of the pass disappears.
# The rewrites are textual on purpose: they remove work from the *generated*
# program, so what is timed below is the kernel the compiler actually ships,
# minus one stage. It also means a rewrite that stops matching raises instead of
# quietly timing the unmodified kernel.
_ABLATIONS = {
    "no QK matmul": ("tsc_qk.run(tsc_q, tsc_k, tsc_scores);", "// QK ablated"),
    "no PV matmul": ("tsc_pv.run(tsc_p, tsc_v, tsc_acc_", "// PV ablated"),
}


def _hot_operand(source: str, name: str, position: int) -> str:
    """Point one operand at head 0 so the loads stay but always hit cache.

    The instruction count and the access pattern are unchanged; only the
    addresses move. The gap between this and the real kernel is what the
    streaming operand loads cost.
    """

    pattern = f"const device T* {name} = in_{position} + "
    start = source.find(pattern)
    if start < 0:
        raise RuntimeError(f"the kernel no longer declares {name}")
    end = source.index(";", start)
    return source[:start] + f"const device T* {name} = in_{position}" + source[end:]


def _variant_source(source: str, name: str) -> str:
    if name in _ABLATIONS:
        needle, replacement = _ABLATIONS[name]
        if needle not in source:
            raise RuntimeError(f"ablation {name!r} no longer matches the kernel")
        return source.replace(needle, replacement)
    if name == "hot K":
        return _hot_operand(source, "key_head", 1)
    if name == "hot V":
        return _hot_operand(source, "value_head", 2)
    raise RuntimeError(f"unknown ablation {name!r}")


def _compiled_variant(
    attention, torch_inputs, name: str | None
) -> tuple[aoto.CompiledPlan, tuple[mx.array, ...]]:
    """Compile the pass, optionally rewriting the generated kernel source."""

    from aotolith.codegen import attention as attention_codegen

    original = attention_codegen._emit_mpp_attention_source
    if name is not None:

        def patched(**kwargs):
            return _variant_source(original(**kwargs), name)

        attention_codegen._emit_mpp_attention_source = patched
    try:
        native = aoto.compile_callable(attention, torch_inputs, codegen_attention=True)
    finally:
        attention_codegen._emit_mpp_attention_source = original
    kernel = native.plan.kernels[0]
    if "tensorcore" not in kernel.name:
        raise RuntimeError("this device did not select the tensor-core attention")
    inputs = tuple(
        mx.array(value.float().numpy(), dtype=mx.float16) for value in torch_inputs
    )
    return native.plan, inputs


def run_ablation(batch: int, heads: int, sequence: int, head_dim: int) -> None:
    """Split the pass cost across its matmuls and its device operand loads."""

    torch.manual_seed(0)
    torch_inputs = tuple(
        torch.randn(batch, heads, sequence, head_dim, dtype=torch.float16)
        for _ in range(3)
    )
    scale = head_dim**-0.5
    names = ["full kernel", "no QK matmul", "no PV matmul", "hot K", "hot V"]
    candidates = {}
    reference = None
    for name in names:
        plan, inputs = _compiled_variant(
            attention, torch_inputs, None if name == "full kernel" else name
        )
        candidates[name] = lambda plan=plan, inputs=inputs: plan(*inputs)[0]
        if reference is None:
            reference = candidates[name]
            mlx_inputs = inputs
    candidates["MLX SDPA"] = lambda: mx.fast.scaled_dot_product_attention(
        *mlx_inputs, scale=scale, mask="causal"
    )
    inner = 12 if sequence < 1024 else 4
    timings = benchmark(candidates, inner=inner)
    shape = f"{batch}x{heads}x{sequence}x{head_dim}"
    print(f"\nattention ablation {shape}")
    full = timings["full kernel"]
    for name in [*names, "MLX SDPA"]:
        saved = full - timings[name]
        share = (
            ""
            if name in {"full kernel", "MLX SDPA"}
            else f"  saves {saved / full:5.1%}"
        )
        print(f"  {name:<20} {timings[name] * 1e3:9.1f} us{share}")


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True)


def _spin_gpu(milliseconds: float = 350.0) -> None:
    """Raise the GPU clock before timing so a case is not penalised.

    Compiling a Metal kernel takes long enough for the clock to drop again, so
    this runs immediately before every measurement instead of once per process.
    """

    left = mx.random.normal((_WARMUP_ELEMENTS, _WARMUP_ELEMENTS)).astype(mx.float16)
    right = mx.random.normal((_WARMUP_ELEMENTS, _WARMUP_ELEMENTS)).astype(mx.float16)
    mx.eval(left, right)
    deadline = time.perf_counter() + milliseconds / 1000.0
    while time.perf_counter() < deadline:
        for _ in range(20):
            mx.eval(left @ right)
    mx.synchronize()


def _timed(fn, inner: int) -> float:
    start = time.perf_counter()
    outputs = [fn() for _ in range(inner)]
    mx.eval(*outputs)
    return (time.perf_counter() - start) / inner * 1000


def benchmark(candidates, inner: int, rounds: int = 7) -> dict[str, float]:
    """Paired timing: round-robin every candidate in shuffled order, keep the best.

    Interleaving matters because a single long sweep lets the GPU clock drift;
    paired rounds keep all candidates in the same thermal state.
    """

    _spin_gpu()
    for _ in range(3):
        for fn in candidates.values():
            _timed(fn, inner)
    best = {name: float("inf") for name in candidates}
    order = list(candidates)
    for _ in range(rounds):
        random.shuffle(order)
        for name in order:
            best[name] = min(best[name], _timed(candidates[name], inner))
    return best


def run_case(batch: int, heads: int, sequence: int, head_dim: int) -> None:
    torch.manual_seed(0)
    torch_inputs = tuple(
        torch.randn(
            batch,
            heads,
            sequence,
            head_dim,
            dtype=torch.float16,
        )
        for _ in range(3)
    )
    native = aoto.compile_callable(attention, torch_inputs, codegen_attention=True)
    kernel = native.plan.kernels[0]
    inputs = tuple(
        mx.array(value.float().numpy(), dtype=mx.float16) for value in torch_inputs
    )
    actual = native.plan(*inputs)[0]
    flops = 4 * batch * heads * sequence * (sequence // 2) * head_dim
    scale = head_dim**-0.5
    if flops < 5e10:
        expected = mx.array(attention(*torch_inputs).float().numpy(), dtype=mx.float16)
        reference = "PyTorch CPU"
    else:
        # A CPU reference this large would dominate the run and heat the machine
        # up right before timing, so large shapes check against MLX instead.
        expected = mx.fast.scaled_dot_product_attention(
            *inputs, scale=scale, mask="causal"
        )
        reference = "MLX SDPA"
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))

    inner = 12 if flops < 2e9 else 4
    timings = benchmark(
        {
            "custom": lambda: native.plan(*inputs)[0],
            "mlx": lambda: mx.fast.scaled_dot_product_attention(
                *inputs, scale=scale, mask="causal"
            ),
        },
        inner,
    )
    shape = f"{batch}x{heads}x{sequence}x{head_dim}"
    print(f"\nattention {shape}: {kernel.name}")
    print(f"max abs error against {reference}: {max_error:.3e}")
    print(
        f"custom kernel:         {timings['custom'] * 1e3:8.1f} us "
        f"({flops / timings['custom'] * 1e-9:.1f} TFLOPS)"
    )
    print(
        f"MLX SDPA:              {timings['mlx'] * 1e3:8.1f} us "
        f"({flops / timings['mlx'] * 1e-9:.1f} TFLOPS)"
    )
    ratio = timings["mlx"] / timings["custom"]
    verdict = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    print(f"custom vs MLX:         {verdict}")


def main() -> None:
    if "--ablate" in sys.argv[1:]:
        for batch, heads, sequence, head_dim in _ABLATION_CASES:
            run_ablation(batch, heads, sequence, head_dim)
        return
    run_case(1, 8, 512, 64)
    run_case(1, 8, 2048, 128)
    run_case(1, 64, 4096, 128)


if __name__ == "__main__":
    main()
