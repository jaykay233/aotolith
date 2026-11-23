# The end-to-end chain: prompt text in, tokens out

`mega_decode` used to start from a bare token id (`-p 692`) at position 0 and call
that a decode benchmark. That is not a chain: no tokenizer, no prompt, no context,
no weights provenance. This note records what the chain is now, what it measures,
and the one structural bug that the missing chain had been hiding.

## Shape

```
prompt text  --llama-tokenize-->  token ids  --prefill-->  KV cache  --dspark decode-->  token ids --detok.py--> text
```

* **Tokenizer.** The model is GPT-2 BPE, `tokenizer.ggml.model = gpt2`, 130560
  tokens and 129794 merges, byte-level. Prompt ids are produced by
  `llama-tokenize --ids`, so the chain starts from the *reference tokenizer's*
  ids. `detok.py` reverses the same table straight out of the GGUF, which is what
  turns ids back into text for comparison against `llama-simple`.
* **Weights.** `mc-q4o.gguf` is a Q4_0 quantization of the real checkpoint, not a
  toy: `general.name = MiniCPM5 2.6B`, 381 tensors and a 130560-entry vocabulary,
  identical in all three to `models/MiniCPM5-2B-F16.gguf` and
  `models/MiniCPM5-2B-Q8_0.gguf`. The float and Q8 originals are on the board at
  `/home/bianbu/models/`.
* **Prefill.** `target_forward(m, pos0, rowtok, want_logits)` is the same forward
  the decode round uses, driven with a whole prompt chunk at a time. Attention is
  causal per row (`g_attn_np = pos0 + r + 1`) and the KV is written at the row's
  own position, so a chunk needs no extra masking code. Logits are skipped
  (`want_logits = 0`) because a chunk would need `m * VOCAB` floats when only the
  last prompt position's distribution is ever read.
* **Hand-off.** Prefill consumes prompt tokens `0 .. n-2`. The last prompt token
  becomes the decode round's anchor, so round 0 already emits a generated token.

`-f ids.txt` is the prompt entry point; `MEGA_POLL`-style knobs that matter here are
`MEGA_ROWS` (row capacity of the row buffers, default 8) and `MEGA_PFCHUNK` (prefill
chunk, default `MEGA_ROWS`).

## The bug the chain exposed: RoPE pairing

`rope_fn` rotated interleaved pairs `(x[2i], x[2i+1])`, the GPT-NeoX convention.
This checkpoint is llama-architecture and rotates the halves of each head against
each other, `(x[i], x[i+64])`. Nothing had caught it because the only cross-check
in the tree was decode against *itself* (plain greedy against dspark), which is
invariant to a wrong-but-consistent rotation.

Evidence, prompt `The capital of France is` (ids `608 4894 304 6918 357`):

| | generated ids |
|---|---|
| reference `llama-simple` | `8181 33` |
| interleaved (old)      | `8181 350` |
| half-split (now)       | `8181 33` |

`MEGA_ROPE=inter` restores the old pairing. The fix is worth far more than the
alignment itself: the drafter is fed target features, and those features were
computed under the wrong rotation, so its accuracy had been capped by a
representation bug.

| dspark, 21-token prompt, s=3 | before | after |
|---|---|---|
| first-draft accuracy | 68.5% | **77.8%** |
| mean rank of target token in draft logits | 2644 | **47** |
| drafts accepted per round | 1.26 | **1.83** |

## Measured

Real weights, real prompt ids, greedy, 8 participants.

| stage | result |
|---|---|
| prefill, 4 tokens | 0.071 s |
| prefill, 20 tokens | 0.288 s (69.6 tok/s) |
| prefill, 512 tokens | 9.20 s (55.7 tok/s) |
| decode, 21-token context, plain | 17.57 tok/s (56.9 ms/round) |
| decode, 21-token context, dspark s=3 | 21.59 tok/s (131.3 ms/round) |
| decode, 512-token context, plain | 14.91 tok/s (67.1 ms/round) |

At 512 tokens the decode round splits 50.57 ms GEMM / 12.05 ms attention, and
prefill is GEMM-bound.

## Alignment status

Against `llama-simple` with the same Q4_0 file, greedy, same prompt ids:

* 5-token prompt: the first 9 generated tokens match, then `,` vs `.`.
* 21-token prompt: the first 21 generated tokens match, then `,` vs `.`.

Both divergences are a single near-tie at a comma. That is the expected shape of
remaining f32/f16 accumulation-order differences, not a structural error — but it
is not yet "exactly llama.cpp", and the objective asks for exact.

## What is standing between this and the objective

1. **Prefill GEMM is 230 GMAC/s, not 822.** `mega_decode` streams Q4_0 weights
   through `gemm_kernel_i8i4`, and `MEGA_PFSPEED` shows that rate is flat from 32
   to 512 rows (232.0 / 231.6 / 234.7 GMAC/s on `ffn_gate`). `mega3` loads a
   second, int8 weight copy (`-I`) and its `task_pf_gemm_i8` reaches 822. Prefill
   at 200 tok/s needs that representation; decode must keep Q4_0 because it
   streams every byte.
2. **Prefill attention is scalar.** At 10K the attention work is ~9 TMAC, which a
   scalar loop cannot do at all. `mega3`'s int8 tiled `pfa_*` attention does it in
   31.3 s, and that is the number the 174 tok/s prefill was built on.
3. Against those two, 10K prefill is 27.4 s GEMM + 31.3 s attention = 58.7 s =
   174 tok/s. 200 tok/s needs 51.2 s, so attention has to come down by ~7.5 s on
   top of adopting the int8 GEMM path.
4. Decode at 10K also needs a vectorised attention; 12.05 ms at 512 rows scales
   roughly linearly and would swamp the dspark gain.
