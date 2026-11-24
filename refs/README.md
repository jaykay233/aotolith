# Golden references: the fixed results every change is scored against

A change is only an improvement if it is measured against something that is not
itself. Comparing two of our own variants (G=512 against G=1024, `-x100` on
against off) proves determinism and nothing else: the RoPE convention was wrong
in `mega3.cpp`, `ref_f32.py` and `layer0_ref.py` at the same time, so they agreed
with each other at 0.9979 cosine while sitting at 0.8688 against llama.cpp.

These files are generated **once** and then never regenerated. Every later
measurement is scored against them.

## What is here

| file | contents |
|---|---|
| `golden_10k_tokens.txt` | 64 greedy steps for `/tmp/p10k.txt` (9999 ids), llama.cpp `refgreedy` |
| `golden_10k_top32.txt` | per step, the reference's top 32 `id:logit`, for ranking regressions |
| `golden_10k_logits.bin` | sha256 `7b01b015889e51a5ce2067727c234803341b6ad10859d7473e2a80fe5865e07f`, 64 x 130560 f32, on the board at `refs/golden_10k_logits.bin` (33 MB, deliberately not committed) |

## How they were produced

    REF_TH=4 REF_TB=4 REF_STEPLOG=refs/golden_10k_logits.bin \
        ./refgreedy.new mc-q4o.gguf /tmp/p10k.txt 64 12288 > refs/golden_10k_tokens.txt

`REF_TH=4` is not a tuning choice. At 16 threads llama.cpp on this board emits
`[KPACK] publish roundtrip FAILED` and produces a degenerate chain
(`0 0 5 528 38 0 5 528 38 ...`); at 4 threads it reports no failures and a normal
chain. The golden has to come from a configuration that is actually correct.

## How a candidate is scored

Teacher forcing is what makes this a precision measurement instead of a
divergence hunt. A single near-tie splits two independent greedy chains and every
later step then compares different prefixes. `-force` walks the reference's own
chain, so step N of both sides is the same position of the same prefix:

    MEGA_STEPLOG=our.bin ./m3c.p -m mc-q4o.gguf -F /tmp/p10k.txt \
        -n 64 -G 1024 -force refs/golden_10k_tokens.txt
    python3 check_logits.py refs/golden_10k_logits.bin our.bin -n 64

`check_logits.py` reports per step the cosine, whether top-1 agrees, and the
reference's top-2 gap divided into how much of it survived, so a near-tie
resolved the other way is distinguishable from lost accuracy.

## Standing result

Base configuration, 10K, no `-x100`: **mean cosine 0.978702, top-1 60/64.**
Two of the four misses have a margin ratio of 0.0000, i.e. our argmax is exactly
the reference's rank 1.
