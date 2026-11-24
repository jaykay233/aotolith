#!/bin/sh
# Narrow the length where megakernel and llama.cpp stop agreeing. One stack at a
# time: two megakernels share TCM and corrupt each other.
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=/dev/shm/mc-q4o.gguf
LLB=/home/bianbu/workspace/llama-cpp-31953bc/build-smt/bin
for N in "$@"; do
    awk -v n=$N '{for(i=1;i<=n;i++) print $i}' /home/bianbu/opt10k/p10k.txt > /tmp/px.txt
    a=$(./m3c.new -m $M -F /tmp/px.txt -n 3 -X 2>&1 | grep -E "^  step [0-9] ->" | sed 's/.*-> \([0-9-]*\).*/\1/' | tr '\n' ' ')
    b=$(LD_LIBRARY_PATH=$LLB ./refgreedy $M /tmp/px.txt 3 4096 2>&1 | grep -E "^  step [0-9] ->" | sed 's/.*-> \([0-9-]*\).*/\1/' | tr '\n' ' ')
    echo "n=$N  mega: $a | llama.cpp: $b"
done
