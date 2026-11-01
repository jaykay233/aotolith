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

Four things, and the first one is the one that gets skipped:

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
4. **Both language versions of whatever documentation you touched.** The READMEs
   and `docs/` are paired: `README.md` / `README.zh-CN.md`,
   `docs/usage.md` / `docs/usage.zh-CN.md`, and so on. `docs/architecture.md` is
   Chinese-only.

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

## Style

- `ruff check` runs with only `F` and `E9`: broken code, not formatting taste.
  Keep the line length at 88.
- Comments explain *why*. A comment that restates the code is deleted in review;
  a comment recording the measurement that ruled out the obvious alternative is
  the point of this repository.
- Type annotations are part of the public surface now that the wheel ships
  `py.typed`. Annotate new public functions and methods.
- Prefer a refusal with a message that names the operator and the reason over a
  silent fallback.

## Releasing

Publishing is driven by a tag, and `.github/workflows/release.yml` checks that the
tag matches `[project].version` before it uploads.

1. Bump `version` in `pyproject.toml`.
2. Commit, then tag: `git tag v0.63.0 && git push origin v0.63.0`.
3. The workflow builds the sdist and wheel, runs `twine check --strict`, asserts
   the wheel carries `py.typed`, and publishes with PyPI trusted publishing.

One-time setup, done on PyPI rather than in this repository: add a trusted
publisher for project `aotolith` with workflow `release.yml`, environment
`pypi`. No API token is stored anywhere.

## Reporting a bug

Include the model or the smallest callable that reproduces it, the Python and
MLX versions, the Apple GPU model, and — for a wrong number — the content of
`plan.selections`, which records what each operation actually ran as.
