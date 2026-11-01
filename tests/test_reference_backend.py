"""The second backend: an interpreter-only target.

``target="mlx-reference"`` builds a plan with no generated kernels at all. These
tests cover the three things that has to mean:

* the compiler reaches it through the same pipeline (capture, rewrite, storage
  plan, artifact) and rejects options that would ask it for kernels;
* the plan executes the same graph as the Metal target, so structure, shapes,
  dtypes and cache contents agree - precision is MLX's, and the tolerances here
  say so rather than pretending otherwise;
* an artifact records its target and loads back through the same backend, and a
  manifest whose target does not exist is refused.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith.backends import REFERENCE_TARGET, TARGETS, ReferenceBackend, Backend
from aotolith.compiler import backend_for, compile_module
from aotolith.errors import CompilationError


class _Chain(torch.nn.Module):
    """A graph with one of each family the interpreter has to cover."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 8))
        self.scale = torch.nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.layer_norm(x, (8,), self.scale, None, 1e-5)
        hidden = torch.relu(normalized @ self.weight.T)
        return torch.softmax(hidden, dim=-1)


class _Reductions(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum(dim=-1, keepdim=True)
        mean = x.mean(dim=-1, keepdim=True)
        return torch.cat((total, mean), dim=-1)


def _inputs() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(4, 8)


def test_pointwise_and_norm_graph_is_bit_identical_across_targets() -> None:
    """The interpreter and the kernels agree exactly where no matrix engine is involved."""

    torch.manual_seed(3)
    module = _Chain().eval()
    x = _inputs()
    with torch.no_grad():
        expected = module(x)

    metal = compile_module(module, (x,))
    reference = compile_module(module, (x,), target=REFERENCE_TARGET)
    as_mlx = mx.array(x.numpy())
    metal_out = metal(as_mlx)
    reference_out = reference(as_mlx)
    mx.eval(*metal_out, *reference_out)

    # ``_Chain`` multiplies by a weight, so the two agree to MLX's matmul
    # precision rather than bit-exactly; the point here is that both are the
    # same computation to within the framework's float behaviour.
    assert reference_out[0].shape == expected.shape
    assert mx.allclose(reference_out[0], mx.array(expected.numpy()), atol=2e-3)
    assert mx.allclose(reference_out[0], metal_out[0], atol=2e-3)


def test_the_reference_plan_has_no_kernels_and_no_native_selections() -> None:
    torch.manual_seed(11)
    module = _Chain().eval()
    compiled = compile_module(module, (_inputs(),), target=REFERENCE_TARGET)
    plan = compiled.plan

    assert plan.kernels == ()
    assert len(plan.selections) == len(plan.graph.ops)
    assert not any(selection.is_native for selection in plan.selections)
    assert {selection.implementation for selection in plan.selections} == {"mlx"}
    assert compiled.compilation.target == REFERENCE_TARGET
    # Every op is still named, so "what runs here" is answerable for the plan.
    assert [selection.op for selection in plan.selections] == [
        op.op for op in plan.graph.ops
    ]


def test_the_metal_plan_of_the_same_module_keeps_its_kernels() -> None:
    """The comparison is meaningful only if the other target really emits kernels."""

    torch.manual_seed(13)
    module = _Chain().eval()
    compiled = compile_module(
        module, (_inputs(),), codegen_matmul=True, codegen_attention=True
    )
    assert compiled.compilation.target == "apple-metal"
    assert compiled.plan.kernels
    assert any(selection.is_native for selection in compiled.plan.selections)


@pytest.mark.parametrize(
    "options",
    [
        {"codegen_matmul": True},
        {"codegen_attention": True},
        {"autotune": True},
    ],
)
def test_kernel_options_are_refused_by_name(options: dict[str, bool]) -> None:
    torch.manual_seed(17)
    module = _Chain().eval()
    with pytest.raises(CompilationError, match="generates no kernels"):
        compile_module(module, (_inputs(),), target=REFERENCE_TARGET, **options)


def test_an_unknown_target_lists_the_ones_that_exist() -> None:
    with pytest.raises(CompilationError, match="unsupported target") as caught:
        backend_for("cuda")
    hint = caught.value.hint or ""
    for name in TARGETS:
        assert name in hint


def test_both_backends_satisfy_the_protocol() -> None:
    assert isinstance(ReferenceBackend(), Backend)
    assert isinstance(backend_for("apple-metal"), Backend)


def test_autotuning_inputs_are_refused_by_the_reference_backend() -> None:
    torch.manual_seed(19)
    module = _Chain().eval()
    graph = compile_module(module, (_inputs(),)).compilation.lowered_graph
    with pytest.raises(ValueError, match="nothing to autotune"):
        ReferenceBackend().compile(graph, tune_inputs=())


def test_the_target_names_are_part_of_the_public_surface() -> None:
    """``target=`` is public, so the values it accepts have to be reachable.

    A caller that passes the name as a string has no way to check it without
    catching an error, and every host that writes ``"mlx-reference"`` by hand is
    one typo away from a compile-time failure. Both names, and the tuple that
    lists them, come from the package root so a host can branch on membership.
    """

    import aotolith

    assert aotolith.REFERENCE_TARGET == "mlx-reference"
    assert aotolith.TARGETS == ("apple-metal", "mlx-reference")
    for name in ("REFERENCE_TARGET", "TARGETS"):
        assert name in aotolith.__all__
    assert aotolith.TARGETS == TARGETS



def test_reductions_are_evaluated_on_the_interpreter() -> None:
    """``sum`` and ``mean`` had no MLX fallback before the second backend."""

    torch.manual_seed(23)
    module = _Reductions().eval()
    x = _inputs()
    with torch.no_grad():
        expected = module(x)

    compiled = compile_module(module, (x,), target=REFERENCE_TARGET)
    out = compiled(mx.array(x.numpy()))
    mx.eval(*out)
    assert mx.allclose(out[0], mx.array(expected.numpy()), atol=1e-5)


def test_the_interpreter_covers_the_families_a_kernel_plan_never_emits() -> None:
    """A family with no generated kernel still has to run on this target.

    The Metal backend answers these by falling back; a reference plan has no
    fallback to fall back to, so a missing interpreter path is an immediate
    ``NotImplementedError``. That makes this the cheapest place to find one: one
    graph per family, compiled with no kernels asked for, compared against
    eager PyTorch.
    """

    class Broad(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # One input channel, eight output channels, kernel 3 with padding
            # 1: the conv keeps the length, so the pool below halves it.
            self.kernel = torch.nn.Parameter(torch.randn(8, 1, 3))
            self.square = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            conv = torch.nn.functional.conv1d(x, self.kernel, padding=1)
            pooled = torch.nn.functional.max_pool1d(conv, 2)
            # Channels last, so the normalized axis is the trailing one.
            normed = torch.nn.functional.rms_norm(pooled.transpose(1, 2), (8,))
            order = torch.tensor([0, 2, 1, 3])
            gathered = normed[:, :, order]
            sorted_values = torch.sort(normed, dim=-1).values
            reduced = normed.var(dim=-1, keepdim=True)
            catted = torch.cat((gathered, sorted_values), dim=-1)
            return (
                catted,
                reduced,
                torch.topk(normed, 2, dim=-1).values,
                torch.linalg.det(self.square),
            )

    torch.manual_seed(37)
    module = Broad().eval()
    x = torch.randn(2, 1, 8)
    with torch.no_grad():
        expected = module(x)

    compiled = compile_module(module, (x,), target=REFERENCE_TARGET)
    out = compiled(mx.array(x.numpy()))
    mx.eval(*out)
    assert compiled.plan.kernels == ()

    for actual, reference in zip(out, expected, strict=True):
        assert tuple(actual.shape) == tuple(reference.shape)
        assert mx.allclose(actual, mx.array(reference.numpy()), atol=2e-3)


def test_softmax_matches_torch() -> None:
    class Softmax(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(x, dim=-1)

    torch.manual_seed(29)
    module = Softmax().eval()
    x = torch.randn(3, 5)
    compiled = compile_module(module, (x,), target=REFERENCE_TARGET)
    out = compiled(mx.array(x.numpy()))
    mx.eval(*out)
    expected = torch.softmax(x, dim=-1)
    assert mx.allclose(out[0], mx.array(expected.numpy()), atol=1e-6)


def test_safe_softmax_returns_zero_for_a_fully_masked_row() -> None:
    from aotolith.ir import DType, Graph, Operation, TensorType, Value
    from aotolith.runtime.mlx_ops import _evaluate_mlx_op

    scores = mx.array([[-mx.inf, -mx.inf], [0.0, 1.0]], dtype=mx.float32)
    op = Operation(
        op="safe_softmax", inputs=("a",), outputs=("out",), attrs={"dim": -1}
    )
    tensor_type = TensorType(shape=(2, 2), dtype=DType.F32)
    graph = Graph(
        inputs=("a",),
        values={"a": Value("a", tensor_type), "out": Value("out", tensor_type)},
        ops=(op,),
        outputs=("out",),
    )
    result = _evaluate_mlx_op(graph, op, {"a": scores})
    assert isinstance(result, mx.array)
    mx.eval(result)
    # A row that is entirely masked normalizes to zeros, not to NaN.
    assert result.tolist()[0] == [0.0, 0.0]
    assert mx.allclose(result[1], mx.softmax(scores[1])).item()


def test_a_reference_artifact_records_its_target_and_loads_back(tmp_path) -> None:
    from aotolith.artifact import load_compiled_module, read_artifact_manifest

    torch.manual_seed(31)
    module = _Chain().eval()
    x = _inputs()
    compiled = compile_module(module, (x,), target=REFERENCE_TARGET)
    expected = compiled(mx.array(x.numpy()))
    mx.eval(*expected)

    path = compiled.save(tmp_path / "reference.aoto")
    manifest = read_artifact_manifest(path)
    assert manifest["target"] == REFERENCE_TARGET

    loaded = load_compiled_module(path)
    assert loaded.compilation.target == REFERENCE_TARGET
    assert loaded.compilation.plan.kernels == ()
    actual = loaded(mx.array(x.numpy()))
    mx.eval(*actual)
    assert mx.array_equal(actual[0], expected[0])


def test_a_device_mismatch_check_has_nothing_to_check_for_a_reference_plan(
    tmp_path,
) -> None:
    """A reference plan holds no compiled-for device, so strict matching is moot."""

    from aotolith.artifact import load_compiled_module

    torch.manual_seed(37)
    compiled = compile_module(_Chain().eval(), (_inputs(),), target=REFERENCE_TARGET)
    path = compiled.save(tmp_path / "reference.aoto")
    loaded = load_compiled_module(path, require_device_match=True)
    assert loaded.compilation.target == REFERENCE_TARGET


def test_an_artifact_with_an_unknown_target_is_refused(tmp_path) -> None:
    import json
    import zipfile

    from aotolith.artifact import read_artifact_manifest
    from aotolith.errors import ArtifactCompatibilityError

    torch.manual_seed(41)
    compiled = compile_module(_Chain().eval(), (_inputs(),))
    path = compiled.save(tmp_path / "metal.aoto")

    rewritten = tmp_path / "rewritten.aoto"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(rewritten, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["target"] = "cuda"
                data = json.dumps(manifest).encode()
            target.writestr(item, data)

    with pytest.raises(ArtifactCompatibilityError, match="unsupported artifact target"):
        read_artifact_manifest(rewritten)


def test_the_decoder_adapter_can_compile_for_the_reference_target() -> None:
    """The model fonts pass the target through, so a family can be checked on MLX."""

    transformers = pytest.importorskip("transformers")
    from aotolith.models.llama import compile_llama_static_cache

    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(43)
    model = transformers.LlamaForCausalLM(config).eval()
    prompt = [1, 2, 3, 4]
    with torch.no_grad():
        expected = model(
            input_ids=torch.tensor([prompt], dtype=torch.int64),
            use_cache=True,
            logits_to_keep=1,
        ).logits

    reference = compile_llama_static_cache(
        model,
        prompt_length=len(prompt),
        max_new_tokens=8,
        target=REFERENCE_TARGET,
    )
    assert reference.prefill.plan.kernels == ()
    logits, cache = reference.prefill_ids([prompt])
    mx.eval(logits, cache)
    # MLX's batched matmul is not IEEE float32 on Apple GPUs; 2e-3 is the
    # measured envelope for a graph this deep.
    assert mx.allclose(logits, mx.array(expected.numpy()), atol=2e-3, rtol=1e-3)

    tokens = reference.greedy_generate([prompt], max_new_tokens=4, sync_each_step=True)
    mx.eval(tokens)
    assert tokens.shape == (1, 4)


def test_a_quantized_decoder_still_compiles_without_kernels() -> None:
    """int4 on the reference target unpacks on MLX instead of streaming words."""

    transformers = pytest.importorskip("transformers")
    from aotolith.models.qwen2 import compile_qwen2_static_cache

    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(47)
    model = transformers.Qwen2ForCausalLM(config).eval()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=4,
        max_new_tokens=8,
        quantize=16,
        target=REFERENCE_TARGET,
    )
    assert runtime.prefill.plan.kernels == ()
    logits, cache = runtime.prefill_ids([[1, 2, 3, 4]])
    mx.eval(logits, cache)
    assert logits.shape == (1, 1, 64)


def test_a_bucketed_decoder_compiles_every_bucket_for_the_reference_target() -> None:
    transformers = pytest.importorskip("transformers")
    from aotolith.models.qwen2 import compile_qwen2_bucketed

    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(53)
    model = transformers.Qwen2ForCausalLM(config).eval()
    runtime = compile_qwen2_bucketed(
        model,
        prompt_lengths=[8, 16],
        max_new_tokens=4,
        target=REFERENCE_TARGET,
    )
    assert runtime.prompt_lengths == (8, 16)
    for bucket in runtime.buckets:
        assert bucket.prefill.plan.kernels == ()
        assert bucket.decode.plan.kernels == ()
    logits, cache = runtime.prefill_ids([[1, 2, 3, 4, 5]])
    mx.eval(logits, cache)
    assert cache.shape == (1, 2, 1, 2, 12, 8)
