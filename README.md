# Transformer Subgraph Compiler

A compiler for Transformer subgraphs from PyTorch FX/ExportedProgram through
MLIR-level optimization to generated Triton kernels.

## Planned Pipeline

PyTorch FX
-> Torch/Linalg MLIR
-> fusion and tiling passes
-> Tile Schedule IR
-> Triton code generation
-> compile cache and runtime dispatcher

## Initial Target

- Llama/Qwen MLP block
- BF16
- NVIDIA Hopper
- correctness and performance comparison against PyTorch eager and Inductor

## Status

Early-stage project.
