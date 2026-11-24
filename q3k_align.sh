#!/bin/sh
# Does the megakernel's trajectory survive a real Q3_K file? The reference is
# llama.cpp on the same file, which is the alignment the objective asks for, so
# both sides read identical weights with identical greedy decoding.
#
#   q3k_align.sh <gguf> <ids.txt> <n>
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
G=${1:-/home/bianbu/models/MiniCPM5-2B-Q3KM.gguf}
P=${2:-/tmp/p512f.txt}
N=${3:-8}
echo "== megakernel -q3 on $(basename $G)"
./m3c.new -m "$G" -F "$P" -n "$N" -q3 -I -X 2>&1 |
    grep -E "weights: source|decode path|step 0|gemm  |attn  |decode: .*tok/s|gen:" || true
