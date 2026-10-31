from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import Any

import mlx.core as mx

from .artifact import (
    ArtifactError,
    load_compiled_module,
    read_artifact_manifest,
)
from .compiler import CompiledModulePlan, compile_callable, compile_module

_TORCH_DTYPES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "bool": "bool",
    "int32": "int32",
    "int64": "int64",
}
_MLX_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
    "bool": mx.bool_,
    "int32": mx.int32,
    "int64": mx.int64,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            return _compile(args)
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "run":
            return _run(args)
        if args.command == "bench":
            return _bench(args)
    except (ArtifactError, ImportError, OSError, TypeError, ValueError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aoto",
        description="Compile and run Aotolith artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="compile an importable callable, module, or HF model"
    )
    source = compile_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--callable",
        metavar="MODULE:ATTR",
        help="import path of a Python callable to trace",
    )
    source.add_argument(
        "--module",
        metavar="MODULE:ATTR",
        help="import path of an nn.Module, class, or module factory",
    )
    source.add_argument(
        "--hf-model",
        metavar="MODEL",
        help="HuggingFace model id or local directory",
    )
    compile_parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="JSON",
        help="positional tensor spec; may be repeated",
    )
    compile_parser.add_argument(
        "--input-file",
        type=Path,
        help="JSON file containing a list of positional tensor specs",
    )
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--target", default="apple-metal")
    compile_parser.add_argument("--codegen-matmul", action="store_true")
    compile_parser.add_argument("--codegen-attention", action="store_true")
    compile_parser.add_argument("--autotune", action="store_true")
    compile_parser.add_argument("--autotune-warmup", type=int, default=2)
    compile_parser.add_argument("--autotune-iterations", type=int, default=7)
    compile_parser.add_argument(
        "--model-kwargs",
        default="{}",
        metavar="JSON",
        help="keyword arguments for --module construction",
    )
    compile_parser.add_argument("--output-key")
    compile_parser.add_argument(
        "--forward-kwarg",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help=(
            "keyword argument for the traced forward pass; tensor specs are "
            "promoted to graph inputs, may be repeated"
        ),
    )
    compile_parser.add_argument("--trust-remote-code", action="store_true")
    compile_parser.add_argument("--revision")
    compile_parser.add_argument("--torch-dtype", default="float32")

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate and print an artifact manifest"
    )
    _add_artifact_path(inspect_parser)

    verify_parser = subparsers.add_parser(
        "verify", help="fully rebuild and verify an artifact"
    )
    _add_artifact_path(verify_parser)

    run_parser = subparsers.add_parser("run", help="execute an artifact")
    _add_artifact_path(run_parser)
    run_parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="JSON",
        help="positional MLX tensor spec; may be repeated",
    )
    run_parser.add_argument(
        "--input-file",
        type=Path,
        help="JSON file containing a list of positional tensor specs",
    )
    run_parser.add_argument(
        "--kwarg",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help=("tensor keyword input for the plan, in NAME=JSON form; may be repeated"),
    )
    run_parser.add_argument("--output", type=Path)
    _add_bench_parser(subparsers)
    return parser


def _add_bench_parser(subparsers: Any) -> None:
    bench_parser = subparsers.add_parser(
        "bench",
        help="measure fixed workloads and check them against a baseline",
    )
    bench_parser.add_argument("--repeats", type=int, default=11)
    bench_parser.add_argument("--warmups", type=int, default=3)
    bench_parser.add_argument(
        "--workload",
        action="append",
        default=[],
        metavar="NAME",
        help="measure only this workload; may be repeated",
    )
    bench_parser.add_argument(
        "--output",
        type=Path,
        help="write the report JSON to this path",
    )
    bench_parser.add_argument(
        "--baseline",
        type=Path,
        help="compare this run against a recorded report",
    )
    bench_parser.add_argument(
        "--update-baseline",
        type=Path,
        metavar="PATH",
        help="write this run as the baseline to compare future runs against",
    )
    bench_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help=(
            "fraction a timing may slow before it counts as a regression; "
            "only applied when the baseline names the same device"
        ),
    )


def _add_artifact_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--require-device-match",
        action="store_true",
        help="reject an artifact compiled for a different Apple GPU model",
    )


def _compile(args: argparse.Namespace) -> int:
    specs = _load_input_specs(args.input, args.input_file)
    torch_inputs = _torch_inputs(specs)
    common = {
        "target": args.target,
        "codegen_matmul": args.codegen_matmul,
        "codegen_attention": args.codegen_attention,
        "autotune": args.autotune,
        "autotune_warmup": args.autotune_warmup,
        "autotune_iterations": args.autotune_iterations,
    }

    if args.callable:
        if args.forward_kwarg:
            raise ValueError(
                "--forward-kwarg needs --module or --hf-model: a traced "
                "callable takes its arguments positionally"
            )
        function = _import_target(args.callable)
        compiled = compile_callable(function, torch_inputs, **common)
    elif args.module:
        module = _load_module(args.module, _load_json_object(args.model_kwargs))
        compiled = compile_module(
            module,
            torch_inputs,
            example_kwargs=_torch_forward_kwargs(args.forward_kwarg),
            output_key=args.output_key,
            **common,
        )
    else:
        module = _load_hf_model(
            args.hf_model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            torch_dtype=args.torch_dtype,
        )
        compiled = compile_module(
            module,
            torch_inputs,
            example_kwargs=_torch_forward_kwargs(args.forward_kwarg),
            output_key=args.output_key,
            **common,
        )

    artifact_path = compiled.save(args.output)
    summary = _artifact_summary(compiled)
    summary["artifact"] = str(artifact_path.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    manifest = read_artifact_manifest(
        args.artifact, require_device_match=args.require_device_match
    )
    summary = {
        "artifact": str(args.artifact.resolve()),
        "format_version": manifest["format_version"],
        "abi_version": manifest["abi_version"],
        "compiler_version": manifest.get("compiler_version"),
        "target": manifest["target"],
        "profile": manifest["profile"],
        "entrypoint": manifest["entrypoint"],
        "compile_options": manifest["compile_options"],
        "graph": manifest["graph"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _verify(args: argparse.Namespace) -> int:
    compiled = load_compiled_module(
        args.artifact, require_device_match=args.require_device_match
    )
    summary = _artifact_summary(compiled)
    summary["artifact"] = str(args.artifact.resolve())
    summary["verified"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _run(args: argparse.Namespace) -> int:
    specs = _load_input_specs(args.input, args.input_file)
    inputs = tuple(_mlx_input(spec) for spec in specs)
    kwarg_specs = _load_named_specs(args.kwarg)
    kwargs = {name: _mlx_input(spec) for name, spec in kwarg_specs.items()}
    compiled = load_compiled_module(
        args.artifact, require_device_match=args.require_device_match
    )
    outputs = compiled(*inputs, **kwargs)
    result = _encode_outputs(outputs)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


def _artifact_summary(compiled: Any) -> dict[str, Any]:
    plan = compiled.plan
    is_module = isinstance(compiled, CompiledModulePlan)
    return {
        "entrypoint": "module" if is_module else "callable",
        "input_count": (compiled.input_count if is_module else len(plan.graph.inputs)),
        "tensor_kwargs": list(compiled.tensor_kwarg_names) if is_module else [],
        "lowered_ops": len(plan.graph.ops),
        "kernels": len(plan.kernels),
        "native_selections": sum(
            selection.implementation == "metal_kernel" for selection in plan.selections
        ),
        "fallback_selections": sum(
            selection.implementation == "mlx" for selection in plan.selections
        ),
        "autotune": compiled.compilation.autotune
        if isinstance(compiled, CompiledModulePlan)
        else compiled.autotune,
        "arena_size_bytes": plan.storage_plan.arena_size_bytes,
        "peak_live_bytes": plan.storage_plan.peak_live_bytes,
    }


def _load_input_specs(
    literals: Sequence[str], input_file: Path | None
) -> list[dict[str, Any]]:
    specs: list[Any] = []
    if input_file is not None:
        payload = json.loads(input_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "inputs" in payload:
            payload = payload["inputs"]
        if not isinstance(payload, list):
            raise ValueError("--input-file must contain a JSON list")
        specs.extend(payload)
    specs.extend(json.loads(value) for value in literals)
    if not specs:
        raise ValueError("at least one --input or --input-file is required")
    normalized = [_normalize_tensor_spec(spec) for spec in specs]
    return normalized


def _load_named_specs(literals: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Parse repeated ``NAME=JSON`` tensor specs into normalized specs."""

    specs: dict[str, dict[str, Any]] = {}
    for literal in literals:
        name, separator, payload = literal.partition("=")
        if not separator or not name:
            raise ValueError(
                f"keyword tensor spec must use NAME=JSON syntax: {literal!r}"
            )
        if name in specs:
            raise ValueError(f"duplicate keyword tensor spec for {name!r}")
        specs[name] = _normalize_tensor_spec(json.loads(payload))
    return specs


def _torch_forward_kwargs(literals: Sequence[str]) -> dict[str, Any] | None:
    specs = _load_named_specs(literals)
    return {name: _torch_inputs([spec])[0] for name, spec in specs.items()} or None


def _normalize_tensor_spec(spec: object) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("each tensor spec must be a JSON object")
    dtype_name = str(spec.get("dtype", "float32")).lower()
    dtype = _TORCH_DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype {dtype_name!r}")
    shape = spec.get("shape")
    if not isinstance(shape, list):
        raise ValueError("tensor spec requires a shape list")
    normalized_shape = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in normalized_shape):
        raise ValueError(f"tensor dimensions must be positive: {normalized_shape}")
    data = spec.get("data")
    if not isinstance(data, list):
        raise ValueError("tensor spec requires a data list")
    flat_data = _flatten_data(data)
    if len(flat_data) != prod(normalized_shape):
        raise ValueError(
            f"tensor spec has {len(flat_data)} values for shape {normalized_shape}"
        )
    return {"dtype": dtype, "shape": normalized_shape, "data": flat_data}


def _flatten_data(value: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(_flatten_data(item))
        else:
            flattened.append(item)
    return flattened


def _torch_inputs(specs: Sequence[dict[str, Any]]) -> tuple[Any, ...]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to compile models") from exc
    tensors = []
    for spec in specs:
        dtype = getattr(torch, spec["dtype"])
        tensor = torch.tensor(spec["data"], dtype=dtype).reshape(spec["shape"])
        tensors.append(tensor)
    return tuple(tensors)


def _mlx_input(spec: dict[str, Any]) -> mx.array:
    return mx.array(spec["data"], dtype=_MLX_DTYPES[spec["dtype"]]).reshape(
        spec["shape"]
    )


def _encode_outputs(outputs: Sequence[mx.array]) -> list[dict[str, Any]]:
    encoded = []
    for output in outputs:
        mx.eval(output)
        value = output
        if output.dtype == mx.bfloat16:
            value = output.astype(mx.float32)
        encoded.append(
            {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "data": value.reshape((-1,)).tolist(),
            }
        )
    return encoded


def _import_target(spec: str) -> Any:
    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"import target must use MODULE:ATTR syntax: {spec!r}")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value


def _load_module(spec: str, kwargs: dict[str, Any]) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to compile modules") from exc
    target = _import_target(spec)
    module = target if isinstance(target, torch.nn.Module) else target(**kwargs)
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{spec!r} did not produce a torch.nn.Module")
    return module.eval()


def _load_hf_model(
    model_id: str,
    *,
    trust_remote_code: bool,
    revision: str | None,
    torch_dtype: str,
) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace compilation requires the 'hf' project extra"
        ) from exc
    dtype_name = _TORCH_DTYPES.get(torch_dtype.lower())
    if dtype_name is None:
        raise ValueError(f"unsupported --torch-dtype {torch_dtype!r}")
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
        torch_dtype=dtype,
    )
    return model.eval()


def _load_json_object(payload: str) -> dict[str, Any]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("expected a JSON object")
    return decoded


def _write_json(path: Path, payload: Any) -> None:
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _bench(args: argparse.Namespace) -> int:
    # Imported here so that ``aoto compile`` does not pay for the benchmark
    # module's numpy and MLX reference setup.
    from . import bench

    selected = bench.WORKLOADS
    if args.workload:
        known = {workload.name: workload for workload in bench.WORKLOADS}
        unknown = [name for name in args.workload if name not in known]
        if unknown:
            raise ValueError(
                "unknown workload(s): "
                + ", ".join(unknown)
                + "; known: "
                + ", ".join(known)
            )
        selected = tuple(known[name] for name in args.workload)

    report = bench.run(warmups=args.warmups, repeats=args.repeats, workloads=selected)
    print(bench.render_table(report))

    if args.output is not None:
        _write_json(args.output, report)
    if args.update_baseline is not None:
        _write_json(args.update_baseline, report)
        print(f"baseline written to {args.update_baseline}")
    if args.baseline is None:
        return 0

    baseline = json.loads(args.baseline.read_text())
    comparison = bench.compare(report, baseline, tolerance=args.tolerance)
    print(bench.summarize(comparison))
    return 0 if comparison.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
