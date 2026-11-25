#!/bin/sh
# Shutdown fix check: exit status must be 0, and the logits must be unchanged
# against the same binary's pre-fix output for the same arm (-P1 -> d532fda4).
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/fix.res

MEGA_STEPLOG=/tmp/fix1.logits ./m3c.w -m mc-q4o.gguf -F /tmp/p10k.txt -n 32 -X -P1 \
    -force refs/golden_10k_tokens.txt > /tmp/fix1.raw 2>&1
echo "exit_p1=$?" >> /tmp/fix.res
md5sum /tmp/fix1.logits >> /tmp/fix.res
grep -E "^decode: [0-9]+ tokens in" /tmp/fix1.raw >> /tmp/fix.res

MEGA_STEPLOG=/tmp/fix2.logits ./m3c.w -m mc-q4o.gguf -F /tmp/p10k.txt -n 32 -X \
    -force refs/golden_10k_tokens.txt > /tmp/fix2.raw 2>&1
echo "exit_default=$?" >> /tmp/fix.res
md5sum /tmp/fix2.logits >> /tmp/fix.res
grep -E "attn split" /tmp/fix2.raw >> /tmp/fix.res
grep -E "^decode: [0-9]+ tokens in" /tmp/fix2.raw >> /tmp/fix.res
echo FIXDONE >> /tmp/fix.res
