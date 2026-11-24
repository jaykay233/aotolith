#!/bin/sh
# Same ids through both stacks, greedy, token by token.
#
#   cmp_prompt.sh ids.txt ngen
#
# mega runs first, then refgreedy (Q4_0 exact weights, libllama). Running them
# one after the other matters: two megakernels at once fight over the same TCM
# blocks and produce wrong tokens at a third of the speed.
set -e
P=${1:?ids file}
N=${2:-8}
M=${M:-/dev/shm/mc-q4o.gguf}
B=${B:-./m3c.new}
LLB=${LLB:-/home/bianbu/workspace/llama-cpp-31953bc/build-smt/bin}
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15

echo "== prompt $P, $N tokens"
echo "-- mega"
$B -m "$M" -F "$P" -n "$N" -X 2>&1 | grep -E "prefill: [0-9]+ tokens|step [0-9] ->"
echo "-- llama.cpp reference"
LD_LIBRARY_PATH=$LLB ./refgreedy "$M" "$P" "$N" 4096 2>&1 | grep -E "prompt|step [0-9] ->"
