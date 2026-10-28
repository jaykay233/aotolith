from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import mlx.core as mx
import torch

from aotolith import (
    ABI_VERSION,
    ArtifactCompatibilityError,
    compile_module,
)
from aotolith.backends.apple_metal import AppleMetalBackend


class TinyArtifactModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3))
        self.bias = torch.nn.Parameter(torch.randn(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value @ self.weight + self.bias)


class ParameterFreeArtifactModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value)


def test_module_artifact_round_trip_and_manifest(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = TinyArtifactModule().eval()
    value = torch.randn(2, 4)
    with torch.no_grad():
        expected = model(value)
    compiled = compile_module(model, (value,), codegen_matmul=True)
    artifact_path = compiled.save(tmp_path / "tiny.aoto")

    with zipfile.ZipFile(artifact_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert {
        "manifest.json",
        "input_graph.json",
        "lowered_graph.json",
        "kernels.json",
        "selections.json",
        "storage_plan.json",
        "state/00000.bin",
        "state/00001.bin",
    } <= names
    assert manifest["format"] == "aotolith"
    assert manifest["format_version"] == 4
    assert manifest["abi_version"] == ABI_VERSION
    assert manifest["entrypoint"]["kind"] == "module"
    assert [item["name"] for item in manifest["entrypoint"]["state"]] == [
        "weight",
        "bias",
    ]
    assert manifest["compile_options"] == {
        "codegen_matmul": True,
        "codegen_attention": False,
        "autotune": False,
        "autotune_warmup": 2,
        "autotune_iterations": 7,
    }
    assert manifest["profile"]["device_name"]

    loaded = __import__(
        "aotolith", fromlist=["load_compiled_module"]
    ).load_compiled_module(artifact_path)
    assert loaded.state_names == ("weight", "bias")
    actual = loaded(value)[0]
    mx.eval(actual)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-5, rtol=1e-5)


def test_parameter_free_module_keeps_module_entrypoint_kind(
    tmp_path: Path,
) -> None:
    model = ParameterFreeArtifactModule().eval()
    value = torch.randn(2, 3)
    artifact_path = compile_module(model, (value,)).save(
        tmp_path / "parameter-free.aoto"
    )

    with zipfile.ZipFile(artifact_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["entrypoint"]["kind"] == "module"
    assert manifest["entrypoint"]["state"] == []


def test_artifact_loads_and_executes_in_a_new_process(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = TinyArtifactModule().eval()
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        expected = model(value)[0].tolist()
    artifact_path = compile_module(model, (value,)).save(tmp_path / "process.aoto")

    script = """
import json
import sys
import mlx.core as mx
from aotolith import load_compiled_module
compiled = load_compiled_module(sys.argv[1])
output = compiled(mx.array([[1.0, 2.0, 3.0, 4.0]]))[0]
mx.eval(output)
print(json.dumps(output[0].tolist()))
"""
    environment = dict(os.environ)
    source_path = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_path, environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    actual = json.loads(completed.stdout.strip())
    assert actual == expected


def test_artifact_rejects_incompatible_abi(tmp_path: Path) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "abi.aoto")
    tampered_path = tmp_path / "tampered.aoto"

    with zipfile.ZipFile(artifact_path) as source, zipfile.ZipFile(
        tampered_path, "w"
    ) as destination:
        for name in source.namelist():
            payload = source.read(name)
            if name == "manifest.json":
                manifest = json.loads(payload)
                manifest["abi_version"] = "999.0"
                payload = json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ).encode()
            destination.writestr(name, payload)

    from aotolith import load_compiled_module

    try:
        load_compiled_module(tampered_path)
    except ArtifactCompatibilityError as exc:
        assert "ABI mismatch" in str(exc)
    else:
        raise AssertionError("incompatible artifact ABI was accepted")


def test_artifact_rejects_unsupported_target(tmp_path: Path) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "target.aoto")
    tampered_path = tmp_path / "target-tampered.aoto"
    with zipfile.ZipFile(artifact_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["target"] = "cuda"
    _rewrite_artifact(
        artifact_path,
        tampered_path,
        {
            "manifest.json": json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        },
    )

    from aotolith import load_compiled_module

    try:
        load_compiled_module(tampered_path)
    except ArtifactCompatibilityError as exc:
        assert "unsupported artifact target" in str(exc)
    else:
        raise AssertionError("unsupported artifact target was accepted")


def test_artifact_rejects_state_shape_mismatch(tmp_path: Path) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "shape.aoto")
    tampered_path = tmp_path / "shape-tampered.aoto"
    with zipfile.ZipFile(artifact_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["entrypoint"]["state"][1]["shape"] = [1, 3]
    _rewrite_artifact(
        artifact_path,
        tampered_path,
        {
            "manifest.json": json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        },
    )

    from aotolith import load_compiled_module

    try:
        load_compiled_module(tampered_path)
    except ArtifactCompatibilityError as exc:
        assert "has shape/dtype (1, 3)/float32" in str(exc)
    else:
        raise AssertionError("state shape mismatch was accepted")


def test_artifact_rejects_state_dtype_mismatch(tmp_path: Path) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "dtype.aoto")
    tampered_path = tmp_path / "dtype-tampered.aoto"
    with zipfile.ZipFile(artifact_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["entrypoint"]["state"][1]["dtype"] = "int32"
    _rewrite_artifact(
        artifact_path,
        tampered_path,
        {
            "manifest.json": json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        },
    )

    from aotolith import load_compiled_module

    try:
        load_compiled_module(tampered_path)
    except ArtifactCompatibilityError as exc:
        assert "has shape/dtype (3,)/int32" in str(exc)
    else:
        raise AssertionError("state dtype mismatch was accepted")


def test_artifact_persists_autotune_measurements(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = TinyArtifactModule().eval()
    value = torch.randn(2, 4)
    compiled = compile_module(
        model,
        (value,),
        autotune=True,
        autotune_warmup=1,
        autotune_iterations=2,
    )
    artifact_path = compiled.save(tmp_path / "autotuned.aoto")

    with zipfile.ZipFile(artifact_path) as archive:
        selections = json.loads(archive.read("selections.json"))
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["compile_options"]["autotune"] is True
    assert all(selection["autotuned"] for selection in selections)
    assert all(len(selection["benchmarks"]) >= 1 for selection in selections)
    assert any(
        benchmark["samples_seconds"]
        for selection in selections
        for benchmark in selection["benchmarks"]
    )

    from aotolith import load_compiled_module

    original_autotune = AppleMetalBackend._autotune

    def unexpected_autotune(*args, **kwargs):
        raise AssertionError("loading a tuned artifact must not rerun autotune")

    AppleMetalBackend._autotune = unexpected_autotune
    try:
        loaded = load_compiled_module(artifact_path)
    finally:
        AppleMetalBackend._autotune = original_autotune
    assert loaded.compilation.autotune
    assert loaded.plan.selections == compiled.plan.selections


def _rewrite_artifact(
    source_path: Path,
    destination_path: Path,
    replacements: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(
        destination_path, "w"
    ) as destination:
        for name in source.namelist():
            payload = replacements.get(name, source.read(name))
            destination.writestr(name, payload)


def test_artifact_rejects_tampered_kernel_table(tmp_path: Path) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "kernel.aoto")
    tampered_path = tmp_path / "kernel-tampered.aoto"
    with zipfile.ZipFile(artifact_path) as archive:
        kernels_payload = archive.read("kernels.json")
    _rewrite_artifact(
        artifact_path,
        tampered_path,
        {"kernels.json": kernels_payload + b"\n"},
    )

    from aotolith import load_compiled_module

    try:
        load_compiled_module(tampered_path)
    except ArtifactCompatibilityError as exc:
        assert "kernel table digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered kernel table was accepted")


def test_artifact_rejects_tampered_state_payload(tmp_path: Path) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "state.aoto")
    tampered_path = tmp_path / "state-tampered.aoto"
    with zipfile.ZipFile(artifact_path) as archive:
        state_payload = bytearray(archive.read("state/00000.bin"))
    state_payload[0] ^= 0xFF
    _rewrite_artifact(
        artifact_path,
        tampered_path,
        {"state/00000.bin": bytes(state_payload)},
    )

    from aotolith import load_compiled_module

    try:
        load_compiled_module(tampered_path)
    except ArtifactCompatibilityError as exc:
        assert "state payload 'state/00000.bin' digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered state payload was accepted")


def test_artifact_strict_device_match_rejects_other_profile(
    tmp_path: Path,
) -> None:
    model = TinyArtifactModule().eval()
    value = torch.randn(1, 4)
    artifact_path = compile_module(model, (value,)).save(tmp_path / "device.aoto")
    mismatched_path = tmp_path / "device-mismatched.aoto"
    with zipfile.ZipFile(artifact_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["profile"]["device_name"] = "Definitely not this device"
    _rewrite_artifact(
        artifact_path,
        mismatched_path,
        {
            "manifest.json": json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        },
    )

    from aotolith import load_compiled_module

    try:
        load_compiled_module(mismatched_path, require_device_match=True)
    except ArtifactCompatibilityError as exc:
        assert "artifact device mismatch" in str(exc)
    else:
        raise AssertionError("strict device matching accepted another profile")
