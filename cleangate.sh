#!/bin/sh
# Final gate for the cleaned source: the QK refactor must be bit-identical to the
# committed behaviour and the exit path must return 0.
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/cl.res

MEGA_STEPLOG=/tmp/cl.logits ./m3c.clean -m mc-q4o.gguf -F /tmp/p10k.txt -n 32 -X \
    -force refs/golden_10k_tokens.txt > /tmp/cl.raw 2>&1
echo "exit=$?" >> /tmp/cl.res
md5sum /tmp/cl.logits >> /tmp/cl.res
grep -E "attn split" /tmp/cl.raw >> /tmp/cl.res
grep -E "gemm  " /tmp/cl.raw >> /tmp/cl.res
grep -E "^decode: [0-9]+ tokens in" /tmp/cl.raw >> /tmp/cl.res
echo CLDONE >> /tmp/cl.res
