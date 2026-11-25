#!/bin/sh
# Canonical correctness record for the current default decode configuration:
# f16 KV, m2 accumulators, four positions batched, teacher-forced over 64 steps.
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/canon.res

MEGA_STEPLOG=/tmp/canon.logits ./m3c.w -m mc-q4o.gguf -F /tmp/p10k.txt -n 64 -X \
    -force refs/golden_10k_tokens.txt > /tmp/canon.raw 2>&1
echo "exit=$?" >> /tmp/canon.res
md5sum /tmp/canon.logits >> /tmp/canon.res
grep -E "attn split" /tmp/canon.raw >> /tmp/canon.res
grep -E "gemm  " /tmp/canon.raw >> /tmp/canon.res
grep -E "^decode: [0-9]+ tokens in" /tmp/canon.raw >> /tmp/canon.res
echo CANONDONE >> /tmp/canon.res
