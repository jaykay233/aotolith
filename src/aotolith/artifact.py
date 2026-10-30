from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from ._version import __version__
from .backends.apple_metal import AppleMetalBackend, CompiledPlan, KernelArtifact
from .ir import DType, Graph, Operation, TensorType, Value
from .planning import BufferSlot, StoragePlan, ValueLifetime
from .selection import KernelBenchmark, KernelSelection

if TYPE_CHECKING:
    from .compiler import CompilationResult, CompiledModulePlan

ARTIFACT_FORMAT = "aotolith"
# Format 5 serializes a kernel's outputs as parallel `output_ids` and
# `tensor_types` lists so a kernel may write several values; a multi-output IR
# operation is one kernel rather than a producer plus re-derivations of its
# siblings. Readers still accept format 4's singular `output_id`/`tensor_type`.
FORMAT_VERSION = 5
ABI_VERSION = "1.3"
# 1.3 added `entrypoint.kwargs`, the tensor-valued keyword inputs promoted to
# graph inputs. Artifacts without that field are the 1.2 layout, where every
# public input comes before the state, so they still load.
_SUPPORTED_ABI_VERSIONS = ("1.2", "1.3")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
    DType.BOOL: mx.bool_,
    DType.I32: mx.int32,
    DType.I64: mx.int64,
}


class ArtifactError(RuntimeError):
    """Base class for invalid or incompatible compiler artifacts."""


class ArtifactCompatibilityError(ArtifactError):
    """Raised when the current compiler cannot load an artifact ABI."""


def save_compiled(
    compiled: CompilationResult | CompiledModulePlan,
    path: str | os.PathLike[str],
) -> Path:
    """Serialize a compiled callable or module into a ``.aoto`` file.

    The archive layout and non-profile payload bytes are canonical. Autotune
    measurements are intentionally data-dependent and can differ between runs.
    """

    from .compiler import CompiledModulePlan, _kwarg_leaf_paths

    is_module = isinstance(compiled, CompiledModulePlan)
    if is_module:
        result = compiled.compilation
        input_count = compiled.input_count
        kwarg_paths = [
            (name, path)
            for name, template in zip(
                compiled.tensor_kwarg_names,
                compiled._kwarg_templates(),
                strict=True,
            )
            for path in _kwarg_leaf_paths(template)
        ]
        state_names = compiled.state_names
        state_values = compiled._state_values
    else:
        result = compiled
        input_count = len(result.plan.graph.inputs)
        kwarg_paths = []
        state_names = ()
        state_values = ()

    if len(state_names) != len(state_values):
        raise ValueError(
            f"artifact has {len(state_names)} state names but "
            f"{len(state_values)} state arrays"
        )
    if len(result.plan.graph.inputs) != (
        input_count + len(kwarg_paths) + len(state_values)
    ):
        raise ValueError(
            "graph inputs do not match public inputs, tensor keyword inputs "
            "and module state"
        )

    kwarg_entries: list[dict[str, object]] = []
    for index, (name, leaf_path) in enumerate(kwarg_paths):
        value_type = result.plan.graph.value(
            result.plan.graph.inputs[input_count + index]
        ).type
        kwarg_entries.append(
            {
                "name": name,
                "path": [[kind, key] for kind, key in leaf_path],
                "shape": list(value_type.shape),
                "dtype": value_type.dtype.value,
            }
        )

    lowered_graph_json = _encode_graph(result.lowered_graph)
    input_graph_json = _encode_graph(result.input_graph)
    lowered_graph_bytes = _canonical_json(lowered_graph_json)
    input_graph_bytes = _canonical_json(input_graph_json)
    kernels_bytes = _canonical_json(
        [_encode_kernel(kernel) for kernel in result.plan.kernels]
    )
    selections_bytes = _canonical_json(
        [_encode_selection(selection) for selection in result.plan.selections]
    )
    storage_bytes = _canonical_json(
        _encode_storage_plan(result.plan.storage_plan)
    )

    state_entries: list[dict[str, object]] = []
    state_payloads: list[tuple[str, bytes]] = []
    # Aliased state (a tied LM head is the usual case) shares one buffer at run
    # time, so it should share one payload here: two entries pointing at the
    # same file and digest describe the same bytes exactly, and the loader
    # recognises that without any new manifest field.
    shared_payloads: dict[int, tuple[str, str]] = {}
    for index, (name, array) in enumerate(
        zip(state_names, state_values, strict=True)
    ):
        mx.eval(array)
        shape = tuple(int(dim) for dim in array.shape)
        dtype = _dtype_from_mlx(array.dtype)
        payload = shared_payloads.get(id(array))
        if payload is None:
            raw = _array_bytes(array)
            payload = (f"state/{index:05d}.bin", _digest(raw))
            shared_payloads[id(array)] = payload
            state_payloads.append((payload[0], raw))
        state_entries.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype.value,
                "nbytes": array.nbytes,
                "file": payload[0],
                "digest": payload[1],
            }
        )

    try:
        device = mx.device_info()
    except (RuntimeError, ValueError):
        device = {}

    manifest = {
        "format": ARTIFACT_FORMAT,
        "format_version": FORMAT_VERSION,
        "abi_version": ABI_VERSION,
        "compiler_version": __version__,
        "target": "apple-metal",
        "entrypoint": {
            "kind": "module" if is_module else "callable",
            "input_count": input_count,
            "kwargs": kwarg_entries,
            "state": state_entries,
        },
        "compile_options": {
            "codegen_matmul": result.codegen_matmul,
            "codegen_attention": result.codegen_attention,
            "autotune": result.autotune,
            "autotune_warmup": result.autotune_warmup,
            "autotune_iterations": result.autotune_iterations,
        },
        "profile": {
            "device_name": device.get("device_name"),
            "architecture": device.get("architecture"),
        },
        "graph": {
            "input_digest": _digest(input_graph_bytes),
            "lowered_digest": _digest(lowered_graph_bytes),
            "inputs": list(result.plan.graph.inputs),
            "outputs": list(result.plan.graph.outputs),
        },
        "payload_digests": {
            "kernels": _digest(kernels_bytes),
            "selections": _digest(selections_bytes),
            "storage_plan": _digest(storage_bytes),
        },
        "payloads": {
            "input_graph": "input_graph.json",
            "lowered_graph": "lowered_graph.json",
            "kernels": "kernels.json",
            "selections": "selections.json",
            "storage_plan": "storage_plan.json",
        },
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            _write_zip_entry(
                archive, "manifest.json", _canonical_json(manifest)
            )
            _write_zip_entry(archive, "input_graph.json", input_graph_bytes)
            _write_zip_entry(
                archive, "lowered_graph.json", lowered_graph_bytes
            )
            _write_zip_entry(archive, "kernels.json", kernels_bytes)
            _write_zip_entry(
                archive,
                "selections.json",
                selections_bytes,
            )
            _write_zip_entry(
                archive,
                "storage_plan.json",
                storage_bytes,
            )
            for file_name, payload in state_payloads:
                _write_zip_entry(archive, file_name, payload)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_compiled_module(
    path: str | os.PathLike[str],
    *,
    require_device_match: bool = False,
) -> CompiledModulePlan:
    """Load and validate an artifact, rebuilding executable MLX kernels.

    ``require_device_match`` rejects artifacts compiled on a different Apple GPU
    model. The default keeps artifacts portable between Apple Silicon devices
    because Metal kernels are rebuilt from the validated graph at load time.
    """

    from .compiler import CompilationResult, CompiledModulePlan

    artifact_path = Path(path)
    with _open_artifact(artifact_path) as archive:
        manifest = _read_json(archive, "manifest.json")
        _validate_manifest(
            manifest, require_device_match=require_device_match
        )
        payloads = manifest["payloads"]
        graph_payload = _read_payload(
            archive,
            payloads["lowered_graph"],
            manifest["graph"]["lowered_digest"],
            "lowered graph",
        )
        input_graph_payload = _read_payload(
            archive,
            payloads["input_graph"],
            manifest["graph"]["input_digest"],
            "input graph",
        )
        kernels_payload = _read_payload(
            archive,
            payloads["kernels"],
            manifest["payload_digests"]["kernels"],
            "kernel table",
        )
        selections_payload = _read_payload(
            archive,
            payloads["selections"],
            manifest["payload_digests"]["selections"],
            "kernel selections",
        )
        storage_payload = _read_payload(
            archive,
            payloads["storage_plan"],
            manifest["payload_digests"]["storage_plan"],
            "storage plan",
        )
        lowered_graph = _decode_graph(_load_json_bytes(graph_payload))
        input_graph = _decode_graph(_load_json_bytes(input_graph_payload))
        lowered_graph.validate()
        input_graph.validate()
        if list(lowered_graph.inputs) != manifest["graph"]["inputs"]:
            raise ArtifactCompatibilityError("manifest graph inputs do not match graph")
        if list(lowered_graph.outputs) != manifest["graph"]["outputs"]:
            raise ArtifactCompatibilityError("manifest graph outputs do not match graph")

        state_entries = manifest["entrypoint"]["state"]
        loaded_state: dict[str, mx.array] = {}
        state_values: list[mx.array] = []
        for entry in state_entries:
            # Aliased entries point at the same payload file, and decoding it
            # once keeps them aliased in memory the way they were at run time.
            file_name = entry["file"] if isinstance(entry, Mapping) else None
            array = loaded_state.get(file_name)
            if array is None:
                array = _decode_state_array(archive, entry)
                if file_name is not None:
                    loaded_state[file_name] = array
            state_values.append(array)
        state_values = tuple(state_values)
        stored_kernels = tuple(
            _decode_kernel(item)
            for item in _load_json_list(kernels_payload)
        )
        stored_selections = tuple(
            _decode_selection(item)
            for item in _load_json_list(selections_payload)
        )
        stored_storage = _decode_storage_plan(
            _load_json_bytes(storage_payload)
        )

    compile_options = manifest["compile_options"]
    backend = AppleMetalBackend(
        codegen_matmul=bool(compile_options["codegen_matmul"]),
        codegen_attention=bool(compile_options["codegen_attention"]),
        autotune=False,
        autotune_warmup=int(compile_options.get("autotune_warmup", 2)),
        autotune_iterations=int(compile_options.get("autotune_iterations", 7)),
        selection_overrides={
            selection.output_id: selection.implementation
            for selection in stored_selections
        },
    )
    plan = backend.compile(lowered_graph)
    _validate_rebuilt_plan(
        plan,
        stored_kernels=stored_kernels,
        stored_selections=stored_selections,
        stored_storage=stored_storage,
    )
    plan.selections = stored_selections
    compilation = CompilationResult(
        plan=plan,
        input_graph=input_graph,
        lowered_graph=lowered_graph,
        codegen_matmul=bool(compile_options["codegen_matmul"]),
        codegen_attention=bool(compile_options["codegen_attention"]),
        autotune=bool(compile_options.get("autotune", False)),
        autotune_warmup=int(compile_options.get("autotune_warmup", 2)),
        autotune_iterations=int(compile_options.get("autotune_iterations", 7)),
    )
    state_names = tuple(str(entry["name"]) for entry in state_entries)
    kwarg_entries = manifest["entrypoint"].get("kwargs", [])
    input_count = int(manifest["entrypoint"]["input_count"])
    _validate_entrypoint_layout(
        lowered_graph,
        input_count=input_count,
        kwarg_entries=kwarg_entries,
        state_entries=state_entries,
        state_values=state_values,
    )
    kwarg_names, kwarg_templates = _kwarg_templates_from_entries(kwarg_entries)
    return CompiledModulePlan(
        compilation=compilation,
        input_count=input_count,
        state_names=state_names,
        _state_values=state_values,
        tensor_kwarg_names=kwarg_names,
        tensor_kwarg_templates=kwarg_templates,
    )


def read_artifact_manifest(
    path: str | os.PathLike[str],
    *,
    require_device_match: bool = False,
) -> dict[str, Any]:
    """Read and validate an artifact manifest without rebuilding kernels."""

    with _open_artifact(Path(path)) as archive:
        manifest = _read_json(archive, "manifest.json")
        _validate_manifest(
            manifest, require_device_match=require_device_match
        )
        return dict(manifest)


def _validate_manifest(
    manifest: Mapping[str, Any], *, require_device_match: bool = False
) -> None:
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ArtifactCompatibilityError(
            f"unsupported artifact format: {manifest.get('format')!r}"
        )
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ArtifactCompatibilityError(
            "artifact format version mismatch: "
            f"expected {FORMAT_VERSION}, got {manifest.get('format_version')!r}"
        )
    if manifest.get("abi_version") not in _SUPPORTED_ABI_VERSIONS:
        raise ArtifactCompatibilityError(
            "artifact ABI mismatch: expected one of "
            f"{list(_SUPPORTED_ABI_VERSIONS)}, "
            f"got {manifest.get('abi_version')!r}"
        )
    if manifest.get("target") != "apple-metal":
        raise ArtifactCompatibilityError(
            f"unsupported artifact target: {manifest.get('target')!r}"
        )
    required = {
        "entrypoint",
        "compile_options",
        "graph",
        "payload_digests",
        "profile",
        "payloads",
    }
    missing = required - set(manifest)
    if missing:
        raise ArtifactCompatibilityError(
            f"artifact manifest is missing fields: {sorted(missing)}"
        )
    if require_device_match:
        _validate_device_profile(manifest["profile"])


def _validate_device_profile(profile: object) -> None:
    if not isinstance(profile, Mapping):
        raise ArtifactCompatibilityError("artifact device profile is invalid")
    stored_name = profile.get("device_name")
    stored_architecture = profile.get("architecture")
    if not stored_name or not stored_architecture:
        raise ArtifactCompatibilityError(
            "artifact has no device profile for strict device matching"
        )
    try:
        current = mx.device_info()
    except (RuntimeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "cannot query the current Apple GPU for artifact validation"
        ) from exc
    current_name = current.get("device_name")
    current_architecture = current.get("architecture")
    if (
        current_name != stored_name
        or current_architecture != stored_architecture
    ):
        raise ArtifactCompatibilityError(
            "artifact device mismatch: "
            f"compiled for {stored_name!r} ({stored_architecture!r}), "
            f"running on {current_name!r} ({current_architecture!r})"
        )


def _validate_digest(payload: bytes, expected: object, label: str) -> None:
    actual = _digest(payload)
    if actual != expected:
        raise ArtifactCompatibilityError(
            f"{label} digest mismatch: expected {expected!r}, got {actual}"
        )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_payload(
    archive: zipfile.ZipFile,
    name: object,
    expected_digest: object,
    label: str,
) -> bytes:
    payload = archive.read(_safe_payload_name(name))
    _validate_digest(payload, expected_digest, label)
    return payload


def _validate_entrypoint_layout(
    graph: Graph,
    *,
    input_count: int,
    kwarg_entries: object,
    state_entries: object,
    state_values: tuple[mx.array, ...],
) -> None:
    if not isinstance(kwarg_entries, list):
        raise ArtifactCompatibilityError("artifact keyword entries are invalid")
    if not isinstance(state_entries, list):
        raise ArtifactCompatibilityError("artifact state entries are invalid")
    state_offset = input_count + len(kwarg_entries)
    if len(graph.inputs) != state_offset + len(state_values):
        raise ArtifactCompatibilityError(
            "artifact input count does not match graph inputs, keyword "
            "inputs and state"
        )
    for index, entry in enumerate(kwarg_entries):
        if not isinstance(entry, Mapping):
            raise ArtifactCompatibilityError(
                "artifact keyword entry is invalid"
            )
        _check_input_type(
            graph, graph.inputs[input_count + index], entry, "keyword"
        )
    _validate_kwarg_paths(kwarg_entries)
    for index, (entry, value) in enumerate(
        zip(state_entries, state_values, strict=True)
    ):
        if not isinstance(entry, Mapping):
            raise ArtifactCompatibilityError("artifact state entry is invalid")
        expected = graph.value(graph.inputs[state_offset + index]).type
        actual_shape = tuple(int(dim) for dim in value.shape)
        actual_dtype = _dtype_from_mlx(value.dtype)
        if actual_shape != expected.shape or actual_dtype != expected.dtype:
            raise ArtifactCompatibilityError(
                f"state {entry.get('name')!r} has shape/dtype "
                f"{actual_shape}/{actual_dtype.value}, expected "
                f"{expected.shape}/{expected.dtype.value}"
            )


def _validate_kwarg_paths(kwarg_entries: list[Any]) -> None:
    """Check that keyword paths describe exactly the stored tensor leaves."""

    from .compiler import _kwarg_leaf_paths, _kwarg_template_from_paths

    for name, group in _kwarg_entry_groups(kwarg_entries):
        try:
            template = _kwarg_template_from_paths(
                [_decode_kwarg_path(entry) for entry in group], len(group)
            )
        except ValueError as exc:
            raise ArtifactCompatibilityError(
                f"artifact keyword {name!r} path is invalid: {exc}"
            ) from exc
        if len(_kwarg_leaf_paths(template)) != len(group):
            raise ArtifactCompatibilityError(
                f"artifact keyword {name!r} path does not describe "
                f"{len(group)} tensor leaves"
            )


def _kwarg_templates_from_entries(
    kwarg_entries: list[Any],
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """Rebuild one template per promoted keyword from the manifest entries."""

    from .compiler import _kwarg_template_from_paths

    names: list[str] = []
    templates: list[Any] = []
    for name, group in _kwarg_entry_groups(kwarg_entries):
        names.append(name)
        templates.append(
            _kwarg_template_from_paths(
                [_decode_kwarg_path(entry) for entry in group], len(group)
            )
        )
    return tuple(names), tuple(templates)


def _kwarg_entry_groups(
    kwarg_entries: list[Any],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Group keyword entries by name, rejecting interleaved duplicates."""

    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    seen: set[str] = set()
    for entry in kwarg_entries:
        name = str(entry["name"])
        if groups and groups[-1][0] == name:
            groups[-1][1].append(entry)
            continue
        if name in seen:
            raise ArtifactCompatibilityError(
                f"artifact keyword {name!r} entries are not contiguous"
            )
        seen.add(name)
        groups.append((name, [entry]))
    return groups


def _decode_kwarg_path(
    entry: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    raw = entry.get("path")
    if raw is None:
        # Artifacts written before nested keyword promotion name one tensor
        # per keyword and have no path.
        return ()
    if not isinstance(raw, list):
        raise ArtifactCompatibilityError("artifact keyword path is invalid")
    from .compiler import _KWARG_DICT, _KWARG_LIST, _KWARG_TUPLE

    path: list[tuple[str, Any]] = []
    for step in raw:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            raise ArtifactCompatibilityError(
                "artifact keyword path step is invalid"
            )
        kind, key = step
        if kind not in (_KWARG_LIST, _KWARG_TUPLE, _KWARG_DICT):
            raise ArtifactCompatibilityError(
                f"artifact keyword path has unknown container {kind!r}"
            )
        if kind == _KWARG_DICT:
            if isinstance(key, (list, dict)) or key is None:
                raise ArtifactCompatibilityError(
                    "artifact keyword path has an invalid mapping key"
                )
        elif isinstance(key, bool) or not isinstance(key, int) or key < 0:
            raise ArtifactCompatibilityError(
                "artifact keyword path has an invalid sequence index"
            )
        path.append((str(kind), key))
    return tuple(path)


def _check_input_type(
    graph: Graph, input_id: str, entry: Mapping[str, Any], kind: str
) -> None:
    expected = graph.value(input_id).type
    shape = entry.get("shape")
    dtype = entry.get("dtype")
    if list(expected.shape) != shape or expected.dtype.value != dtype:
        raise ArtifactCompatibilityError(
            f"{kind} {entry.get('name')!r} has shape/dtype "
            f"{shape}/{dtype}, expected "
            f"{list(expected.shape)}/{expected.dtype.value}"
        )


def _validate_rebuilt_plan(
    plan: CompiledPlan,
    *,
    stored_kernels: tuple[KernelArtifact, ...],
    stored_selections: tuple[KernelSelection, ...],
    stored_storage: StoragePlan,
) -> None:
    if plan.kernels != stored_kernels:
        raise ArtifactCompatibilityError(
            "rebuilt kernels differ from the serialized compiler artifact"
        )
    rebuilt_selection_keys = tuple(
        (
            selection.op_index,
            selection.op,
            selection.output_id,
            selection.implementation,
            selection.kind,
            selection.artifact_name,
        )
        for selection in plan.selections
    )
    stored_selection_keys = tuple(
        (
            selection.op_index,
            selection.op,
            selection.output_id,
            selection.implementation,
            selection.kind,
            selection.artifact_name,
        )
        for selection in stored_selections
    )
    if rebuilt_selection_keys != stored_selection_keys:
        raise ArtifactCompatibilityError(
            "rebuilt backend selections differ from the serialized artifact"
        )
    if plan.storage_plan != stored_storage:
        raise ArtifactCompatibilityError(
            "rebuilt storage plan differs from the serialized artifact"
        )


def _encode_graph(graph: Graph) -> dict[str, object]:
    graph.validate()
    return {
        "inputs": list(graph.inputs),
        "values": {
            value_id: {
                "shape": list(value.type.shape),
                "dtype": value.type.dtype.value,
            }
            for value_id, value in sorted(graph.values.items())
        },
        "ops": [
            {
                "op": op.op,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": _encode_attr(op.attrs),
            }
            for op in graph.ops
        ],
        "outputs": list(graph.outputs),
    }


def _decode_graph(payload: Mapping[str, Any]) -> Graph:
    try:
        values = {
            value_id: Value(
                value_id,
                TensorType(
                    tuple(int(dim) for dim in value["shape"]),
                    DType(value["dtype"]),
                ),
            )
            for value_id, value in payload["values"].items()
        }
        ops = tuple(
            Operation(
                str(op["op"]),
                tuple(str(value) for value in op["inputs"]),
                tuple(str(value) for value in op["outputs"]),
                _decode_attr(op["attrs"]),
            )
            for op in payload["ops"]
        )
        graph = Graph(
            tuple(str(value) for value in payload["inputs"]),
            values,
            ops,
            tuple(str(value) for value in payload["outputs"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("invalid serialized graph") from exc
    graph.validate()
    return graph


def _encode_attr(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"$kind": "float", "value": "nan"}
        return {"$kind": "float", "value": "inf" if value > 0 else "-inf"}
    if isinstance(value, tuple):
        return {"$kind": "tuple", "items": [_encode_attr(item) for item in value]}
    if isinstance(value, list):
        return {"$kind": "list", "items": [_encode_attr(item) for item in value]}
    if isinstance(value, dict):
        return {
            "$kind": "dict",
            "items": [
                [_encode_attr(key), _encode_attr(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, Operation):
        return {
            "$kind": "operation",
            "op": value.op,
            "inputs": list(value.inputs),
            "outputs": list(value.outputs),
            "attrs": _encode_attr(value.attrs),
        }
    raise TypeError(f"cannot serialize IR attribute of type {type(value).__name__}")


def _decode_attr(value: object) -> object:
    if not isinstance(value, dict) or "$kind" not in value:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise ArtifactCompatibilityError(f"invalid serialized attribute: {value!r}")
    kind = value["$kind"]
    if kind == "float":
        return float(value["value"])
    if kind in {"tuple", "list"}:
        items = [_decode_attr(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {
            _decode_attr(key): _decode_attr(item) for key, item in value["items"]
        }
    if kind == "operation":
        return Operation(
            str(value["op"]),
            tuple(str(item) for item in value["inputs"]),
            tuple(str(item) for item in value["outputs"]),
            _decode_attr(value["attrs"]),
        )
    raise ArtifactCompatibilityError(f"unknown serialized attribute kind: {kind!r}")


def _encode_kernel(kernel: KernelArtifact) -> dict[str, object]:
    return {
        "name": kernel.name,
        "source": kernel.source,
        "header": kernel.header,
        "input_ids": list(kernel.input_ids),
        "output_ids": list(kernel.output_ids),
        "tensor_types": [
            {
                "shape": list(tensor_type.shape),
                "dtype": tensor_type.dtype.value,
            }
            for tensor_type in kernel.tensor_types
        ],
        "kind": kernel.kind,
    }


def _decode_kernel(payload: Mapping[str, Any]) -> KernelArtifact:
    try:
        # Format 4 kernels wrote one output under singular keys; the reader
        # still accepts them so an older archive loads instead of failing.
        if "output_ids" in payload:
            output_ids = tuple(str(value) for value in payload["output_ids"])
            tensor_types = tuple(
                TensorType(
                    tuple(int(dim) for dim in item["shape"]),
                    DType(item["dtype"]),
                )
                for item in payload["tensor_types"]
            )
        else:
            output_ids = (str(payload["output_id"]),)
            tensor_types = (
                TensorType(
                    tuple(int(dim) for dim in payload["tensor_type"]["shape"]),
                    DType(payload["tensor_type"]["dtype"]),
                ),
            )
        return KernelArtifact(
            name=str(payload["name"]),
            source=str(payload["source"]),
            header=str(payload["header"]),
            input_ids=tuple(str(value) for value in payload["input_ids"]),
            output_ids=output_ids,
            tensor_types=tensor_types,
            kind=str(payload["kind"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("invalid serialized kernel") from exc


def _encode_selection(selection: KernelSelection) -> dict[str, object]:
    return {
        "op_index": selection.op_index,
        "op": selection.op,
        "output_id": selection.output_id,
        "implementation": selection.implementation,
        "kind": selection.kind,
        "reason": selection.reason,
        "artifact_name": selection.artifact_name,
        "autotuned": selection.autotuned,
        "device_name": selection.device_name,
        "device_architecture": selection.device_architecture,
        "benchmarks": [
            {
                "candidate": benchmark.candidate,
                "artifact_name": benchmark.artifact_name,
                "warmup_iterations": benchmark.warmup_iterations,
                "iterations": benchmark.iterations,
                "samples_seconds": list(benchmark.samples_seconds),
                "median_seconds": benchmark.median_seconds,
                "min_seconds": benchmark.min_seconds,
                "error": benchmark.error,
            }
            for benchmark in selection.benchmarks
        ],
    }


def _decode_selection(payload: Mapping[str, Any]) -> KernelSelection:
    try:
        benchmarks = tuple(
            KernelBenchmark(
                candidate=str(item["candidate"]),
                artifact_name=(
                    None
                    if item["artifact_name"] is None
                    else str(item["artifact_name"])
                ),
                warmup_iterations=int(item["warmup_iterations"]),
                iterations=int(item["iterations"]),
                samples_seconds=tuple(
                    float(sample) for sample in item["samples_seconds"]
                ),
                median_seconds=(
                    None
                    if item["median_seconds"] is None
                    else float(item["median_seconds"])
                ),
                min_seconds=(
                    None
                    if item["min_seconds"] is None
                    else float(item["min_seconds"])
                ),
                error=None if item["error"] is None else str(item["error"]),
            )
            for item in payload.get("benchmarks", ())
        )
        return KernelSelection(
            op_index=int(payload["op_index"]),
            op=str(payload["op"]),
            output_id=str(payload["output_id"]),
            implementation=str(payload["implementation"]),
            kind=str(payload["kind"]),
            reason=str(payload["reason"]),
            artifact_name=(
                None
                if payload["artifact_name"] is None
                else str(payload["artifact_name"])
            ),
            benchmarks=benchmarks,
            autotuned=bool(payload.get("autotuned", False)),
            device_name=(
                None
                if payload.get("device_name") is None
                else str(payload["device_name"])
            ),
            device_architecture=(
                None
                if payload.get("device_architecture") is None
                else str(payload["device_architecture"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("invalid serialized kernel selection") from exc


def _encode_storage_plan(plan: StoragePlan) -> dict[str, object]:
    return {
        "lifetimes": [
            {
                "value_id": item.value_id,
                "shape": list(item.tensor_type.shape),
                "dtype": item.tensor_type.dtype.value,
                "start": item.start,
                "end": item.end,
                "size_bytes": item.size_bytes,
                "is_input": item.is_input,
                "is_output": item.is_output,
            }
            for item in plan.lifetimes
        ],
        "slots": [
            {
                "slot_id": slot.slot_id,
                "offset_bytes": slot.offset_bytes,
                "size_bytes": slot.size_bytes,
                "alignment_bytes": slot.alignment_bytes,
                "value_ids": list(slot.value_ids),
            }
            for slot in plan.slots
        ],
        "release_after_op": [list(values) for values in plan.release_after_op],
        "peak_live_bytes": plan.peak_live_bytes,
        "arena_size_bytes": plan.arena_size_bytes,
    }


def _decode_storage_plan(payload: Mapping[str, Any]) -> StoragePlan:
    try:
        return StoragePlan(
            lifetimes=tuple(
                ValueLifetime(
                    value_id=str(item["value_id"]),
                    tensor_type=TensorType(
                        tuple(int(dim) for dim in item["shape"]),
                        DType(item["dtype"]),
                    ),
                    start=int(item["start"]),
                    end=int(item["end"]),
                    size_bytes=int(item["size_bytes"]),
                    is_input=bool(item["is_input"]),
                    is_output=bool(item["is_output"]),
                )
                for item in payload["lifetimes"]
            ),
            slots=tuple(
                BufferSlot(
                    slot_id=int(item["slot_id"]),
                    offset_bytes=int(item["offset_bytes"]),
                    size_bytes=int(item["size_bytes"]),
                    alignment_bytes=int(item["alignment_bytes"]),
                    value_ids=tuple(str(value) for value in item["value_ids"]),
                )
                for item in payload["slots"]
            ),
            release_after_op=tuple(
                tuple(str(value) for value in values)
                for values in payload["release_after_op"]
            ),
            peak_live_bytes=int(payload["peak_live_bytes"]),
            arena_size_bytes=int(payload["arena_size_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("invalid serialized storage plan") from exc


def _array_bytes(array: mx.array) -> bytes:
    contiguous = array.reshape((-1,))
    mx.eval(contiguous)
    return bytes(memoryview(contiguous))


def _decode_state_array(
    archive: zipfile.ZipFile, entry: Mapping[str, Any]
) -> mx.array:
    try:
        file_name = _safe_payload_name(entry["file"])
        shape = tuple(int(dim) for dim in entry["shape"])
        dtype = DType(entry["dtype"])
        nbytes = int(entry["nbytes"])
        expected_digest = entry["digest"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("invalid artifact state metadata") from exc
    raw = archive.read(file_name)
    _validate_digest(raw, expected_digest, f"state payload {file_name!r}")
    if len(raw) != nbytes:
        raise ArtifactCompatibilityError(
            f"state payload {file_name!r} has {len(raw)} bytes, expected {nbytes}"
        )
    array = mx.array(list(raw), dtype=mx.uint8).view(_MLX_DTYPES[dtype])
    if len(shape) == 0:
        return array.reshape(())
    if math.prod(shape) != array.size:
        raise ArtifactCompatibilityError(
            f"state payload {file_name!r} cannot reshape to {shape}"
        )
    return array.reshape(shape)


def _dtype_from_mlx(dtype: mx.Dtype) -> DType:
    for candidate, mlx_dtype in _MLX_DTYPES.items():
        if dtype == mlx_dtype:
            return candidate
    raise TypeError(f"unsupported artifact state dtype: {dtype}")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_json_value(payload: bytes) -> object:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ArtifactCompatibilityError("invalid artifact JSON") from exc


def _load_json_bytes(payload: bytes) -> Mapping[str, Any]:
    decoded = _load_json_value(payload)
    if not isinstance(decoded, dict):
        raise ArtifactCompatibilityError("artifact JSON root must be an object")
    return decoded


def _load_json_list(payload: bytes) -> list[Any]:
    decoded = _load_json_value(payload)
    if not isinstance(decoded, list):
        raise ArtifactCompatibilityError("artifact JSON root must be a list")
    return decoded


def _read_json(archive: zipfile.ZipFile, name: object) -> Any:
    try:
        return json.loads(archive.read(_safe_payload_name(name)))
    except (json.JSONDecodeError, KeyError) as exc:
        raise ArtifactCompatibilityError(
            f"invalid or missing artifact JSON payload {name!r}"
        ) from exc


@contextmanager
def _open_artifact(path: Path):
    try:
        with zipfile.ZipFile(path, "r") as archive:
            yield archive
    except (FileNotFoundError, zipfile.BadZipFile) as exc:
        raise ArtifactCompatibilityError(
            f"cannot open compiler artifact {path}: {exc}"
        ) from exc


def _safe_payload_name(name: object) -> str:
    if not isinstance(name, str):
        raise ArtifactCompatibilityError(f"invalid artifact path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArtifactCompatibilityError(f"unsafe artifact path: {name!r}")
    return path.as_posix()


def _write_zip_entry(
    archive: zipfile.ZipFile, name: str, payload: bytes
) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)
