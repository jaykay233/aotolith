#!/bin/bash
# Sequential 10K measurement batch: int8 QK prefill A/B, decode path A/B, verify profiling.
cd "$WORK" || exit 1
./i8prefill.sh
./decode10k.sh
./specprof10k.sh
