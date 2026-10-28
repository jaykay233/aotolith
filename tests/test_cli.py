from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from cli_fixture import TinyCliModule


_ROOT = Path(__file__).parents[1]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    python_paths = [_ROOT / "src", _ROOT / "tests"]
    existing = environment.get("PYTHONPATH")
    if existing:
        python_paths.extend(Path(item) for item in existing.split(os.pathsep))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    return subprocess.run(
        [sys.executable, "-m", "transformer_subgraph_compiler", *arguments],
        cwd=_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _input_spec() -> dict[str, object]:
    return {
        "shape": [2, 3],
        "dtype": "float32",
        "data": [[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]],
    }


def test_cli_compiles_inspects_verifies_and_runs_module(tmp_path: Path) -> None:
    artifact = tmp_path / "tiny.tsca"
    input_file = tmp_path / "inputs.json"
    input_file.write_text(json.dumps([_input_spec()]), encoding="utf-8")

    compile_result = _run_cli(
        "compile",
        "--module",
        "cli_fixture:TinyCliModule",
        "--input-file",
        str(input_file),
        "--output",
        str(artifact),
    )
    compile_summary = json.loads(compile_result.stdout)
    assert compile_summary["entrypoint"] == "module"
    assert compile_summary["input_count"] == 1
    assert artifact.exists()

    inspect_result = _run_cli("inspect", str(artifact))
    manifest = json.loads(inspect_result.stdout)
    assert manifest["format_version"] == 3
    assert manifest["entrypoint"]["input_count"] == 1
    assert [item["name"] for item in manifest["entrypoint"]["state"]] == [
        "weight",
        "bias",
    ]

    verify_result = _run_cli("verify", str(artifact), "--require-device-match")
    assert json.loads(verify_result.stdout)["verified"] is True

    run_result = _run_cli(
        "run",
        str(artifact),
        "--input",
        json.dumps(_input_spec()),
    )
    outputs = json.loads(run_result.stdout)
    assert len(outputs) == 1
    assert outputs[0]["shape"] == [2, 2]

    value = torch.tensor(_input_spec()["data"], dtype=torch.float32)
    expected = TinyCliModule()(value)
    assert outputs[0]["data"] == expected.reshape(-1).tolist()


def test_cli_compiles_callable_and_writes_run_output(tmp_path: Path) -> None:
    artifact = tmp_path / "callable.tsca"
    output_file = tmp_path / "output.json"
    spec = {
        "shape": [2, 2],
        "dtype": "float32",
        "data": [[-1.0, 2.0], [3.0, -4.0]],
    }

    compile_result = _run_cli(
        "compile",
        "--callable",
        "cli_fixture:scaled_relu",
        "--input",
        json.dumps(spec),
        "--input",
        '{"shape":[],"dtype":"float32","data":[2.0]}',
        "--output",
        str(artifact),
    )
    summary = json.loads(compile_result.stdout)
    assert summary["entrypoint"] == "callable"
    assert summary["input_count"] == 2

    _run_cli(
        "run",
        str(artifact),
        "--input",
        json.dumps(spec),
        "--input",
        '{"shape":[],"dtype":"float32","data":[2.0]}',
        "--output",
        str(output_file),
    )
    outputs = json.loads(output_file.read_text(encoding="utf-8"))
    assert outputs[0]["shape"] == [2, 2]
    assert outputs[0]["data"] == [-0.0, 4.0, 6.0, -0.0]
