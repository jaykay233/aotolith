# Contributing

Thanks for looking at this. Aotolith is an experimental compiler for Transformer
models on Apple Silicon, and the parts that are worth contributing are the ones
that can be measured.

## What is in scope

The compiler owns graph capture, legalization, layout simplification, fusion,
CSE, tensor lifetimes, kernel selection, Metal code generation, and the versioned
artifact. It does **not** own request scheduling, continuous batching, paged KV
allocation, sampling, or serving — a change that adds one of those belongs in a
host runtime, not here. `docs/architecture.md` §16 lists the compiler-side
directions that are open.

## Setting up

```bash
git clone https://github.com/jaykay233/aotolith.git
cd aotolith
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The suite needs an Apple Silicon machine; `ruff` does not.

```bash
python -m pytest -q
ruff check src tests examples
python examples/tiny_llama_compile.py
```

## What a change needs

Five things, and the first one is the one that gets skipped:

1. **A stated motivation.** "It is cleaner" is not enough for a rewrite;
   "this drops 245 operators per decode step" is. If the change is about speed,
   the number that names the bottleneck belongs in the pull request.
2. **A test that fails before it.** Numerical changes get parity assertions
   against PyTorch; graph rewrites get op-count and bit-equality assertions;
   refusals get a test that the refusal happens with a message naming the
   operator.
3. **A measurement taken honestly**, if performance is the claim. Ablate the
   pass inside one session — off, on, off, on — so both arms share the machine's
   thermal state, and compare `min_ms` rather than the median, because
   contention only adds time. `docs/performance.md` records how the existing
   numbers were taken; match that standard.
4. **The right error class.** A new refusal is an `AotolithError`, not a bare
   `ValueError`: pick the class that matches the caller's decision (see
   [`docs/errors.md`](docs/errors.md)), and if none fits, add one with a new
   `code` and a row in both language versions of the catalog — `test_errors.py`
   fails if the catalog and the code disagree. The builtin bases are part of the
   contract: an existing handler must keep working.
5. **Both language versions of whatever documentation you touched**, plus a
   `CHANGELOG.md` entry under the version being prepared. The READMEs and
   `docs/` are paired: `README.md` / `README.zh-CN.md`, `docs/usage.md` /
   `docs/usage.zh-CN.md`, and so on. `docs/architecture.md` is Chinese-only.
   What counts as a change worth an entry is in
   [`docs/compatibility.md`](docs/compatibility.md).

## Adding an operator

The path is the same for every family:

1. A lowering in `src/aotolith/legalize/` that turns the ATen target into core
   IR ops. A target can have several handlers, tried in order; return `False`
   from one to let the next try (that is how `constant_pad_nd` separates the
   crop-only case).
2. A contract in `src/aotolith/ir/` — shape, dtype, and attribute validation, so
   an invalid graph fails at construction rather than in the kernel.
3. An implementation. Generated Metal goes in `src/aotolith/codegen/`; a
   composite that reuses existing ops needs no kernel at all; the MLX fallback
   in `src/aotolith/runtime/mlx_ops/` is required whenever a generated kernel is
   the only path, because autotuning and unsupported shapes need a second
   implementation to compare against.
4. A selection reason, so `plan.selections` says why this op took that path.
5. Tests: numerical parity with PyTorch, the refusal paths, and — for anything
   that claims to fuse — an op count.
6. A line in the operator table in `docs/architecture.md`, and in
   `docs/operators.md` if the execution strategy is worth explaining.

## Adding a model adapter

`src/aotolith/models/qwen2.py` is the reference. An adapter is a plain
`nn.Module` that re-expresses the model's forward in terms of IR-friendly
operations, so the compiler sees explicit shapes and an explicit cache instead of
a Python loop. It should validate what it depends on at construction time — the
activation function, the attention masking — and raise with a message that names
the layer, rather than producing plausible wrong numbers.

## Recording a baseline for a new chip

`benchmarks/baselines/` is a chip matrix: one file per chip, named by
`device_slug` of the device name the report records (`Apple M5 Pro` ->
`apple-m5-pro.json`). On a new machine:

```bash
aoto bench --rounds 3 --update-baseline benchmarks/baselines
aoto bench --check-baselines benchmarks/baselines
```

The first command writes this chip's file rather than replacing a neighbour's;
the second is what CI runs, and it catches the mistakes a hand edit makes — a
stale schema, a file renamed away from the chip it records, a chip recorded
twice. Commit the new file in its own change: a baseline recorded on a different
thermal state is not a code review, and mixing it into a feature change makes the
diff look like a regression. `--rounds 3` is not optional; a floor from fewer
than three repeats cannot be compared, so the gate would skip it.

Do not record a baseline for hardware you do not have, and do not edit a
`device.name` to make a run compare against another chip. `compare()` decides on
that field, so a wrong name produces a gate that passes for the wrong reason.

## Style

- `ruff check` runs with only `F` and `E9`: broken code, not formatting taste.
  Keep the line length at 88.
- Comments explain *why*. A comment that restates the code is deleted in review;
  a comment recording the measurement that ruled out the obvious alternative is
  the point of this repository.
- Type annotations are part of the public surface now that the wheel ships
  `py.typed`. Annotate new public functions and methods. `pyrightconfig.json`
  lists the modules CI type-checks; new public modules are added to it.
- Prefer a refusal with a message that names the operator and the reason over a
  silent fallback. A refusal a caller may want to branch on is an
  `AotolithError` from `src/aotolith/errors.py`, with a `hint` for the fix.
- A downgrade — a fallback, a refused capability, an autotune with no winner —
  belongs in the log at `WARNING` as well as in `plan.selections`, because the
  caller reading the log does not have the plan yet. Import the package logger
  (`from ._log import get_logger`), never `print`, and never configure handlers
  outside `_log.basic_config`.

## Releasing

Publishing is driven by a tag, and `.github/workflows/release.yml` checks that the
tag matches the version in `src/aotolith/_version.py` — the one place the version
lives; `pyproject.toml` reads it through `hatch`.

1. Write the `CHANGELOG.md` entry under a new version heading, and update the
   comparison links at the bottom of the file.
2. Bump `__version__` in `src/aotolith/_version.py`.
3. Commit, then tag: `git tag -a v0.64.0 -m "aotolith 0.63.0" && git push origin v0.64.0`.
4. The workflow builds the sdist and wheel, runs `twine check --strict`, asserts
   the wheel carries `py.typed`, and publishes with PyPI trusted publishing.

One-time setup, done in two web consoles rather than in this repository:

**PyPI** — at <https://pypi.org/manage/account/publishing/> add a *pending*
publisher (the project does not exist yet, so it lives under the account sidebar).
The five fields must match this repository exactly:

| Field | Value |
| --- | --- |
| PyPI Project Name | `aotolith` |
| Owner | `jaykay233` — the **GitHub** owner, not the PyPI login |
| Repository name | `aotolith` |
| Workflow name | `release.yml` — the bare file name |
| Environment name | `pypi` |

A pending publisher does not reserve the name until its first successful upload.

**GitHub** — the `pypi` environment needs two deployment branch policies, `tag:
v*` **and** `branch: main`. Both, because the workflow also has a
`workflow_dispatch` trigger for retrying a rejected upload: that run's ref is
`main`, and with only the tag policy the publish job would be blocked by the
protection rule exactly when it is needed. No secret or token is stored anywhere;
the workflow authenticates with OIDC.

Two failures worth recognising: `invalid-publisher` means one of the five PyPI
fields disagrees with the repository (usually `Workflow name` with a path prefix,
or the PyPI login typed into `Owner`); a "file already exists" rejection means the
version was published before, and PyPI does not allow replacing a file, so the
fix is a new version number rather than a retry.

## Reporting a bug

Include the model or the smallest callable that reproduces it, the Python and
MLX versions, the Apple GPU model, and — for a wrong number — the content of
`plan.selections`, which records what each operation actually ran as.
