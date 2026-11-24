# dspark: what the draft model is, and what it will cost

> Index of conclusions, including every falsified attempt: `FINDINGS.md`.

The drafter for speculative decode is `MiniCPM5 2.6B DSpark` at
`/home/bianbu/models/dspark/dspark-bf16.gguf`, a 324M-parameter head over the target.
Architecture tag is `dflash`; the variant is DSpark, distinguished in llama.cpp by the
presence of `markov_w1.weight`.

Source of truth for the forward pass: `llama-mtmd/src/models/dflash.cpp` and
`src/models/dspark.cpp` on the board. Everything below is read off those, not inferred.

## Shape

| | |
|---|---|
| layers | 5 |
| embedding / ffn | 2048 / 6144 |
| heads / kv heads | 16 / 2 |
| head dim | 128 |
| rope base | 5e6 |
| rms eps | 1e-6 |
| block_size | 7 |
| target_layers | 2, 11, 21, 31, 40 |
| sample_from_anchor | true |
| has_confidence_head | true |
| markov rank | 256 |

The widths are identical to the target model, so every buffer, the IME2 GEMM, the
activation packer and the norm routines are reusable as-is. The draft needs its own
KV cache and its own set of weight blobs, nothing else.

## Forward pass

Drafting is three graph evaluations, in this order.

**1. Encoder.** Takes the target's hidden states at layers 2, 11, 21, 31, 40,
concatenates them into a 10240-wide vector, then

    g = rmsnorm(fc @ concat, output_norm_enc)          fc: 10240 -> 2048

`g` is the fused feature for that position. 21 MMAC per position, negligible.

**2. Inject.** For each of the 5 draft layers, from `g`:

    K = rope(rmsnorm_k(wk @ g))        wk: 2048 -> 256
    V = wv @ g                         wv: 2048 -> 256

and append to the draft's KV cache at that position. This is how the draft sees the
committed prefix: not through tokens, but through the fused target features. Note `V`
is *not* normalised here, because this model has a real `attn_v.weight` (the reference
only normalises V when `wv` is absent and V is shared with K). No Hadamard rotation:
that only applies to caches configured for it, and the base cache is not.

**3. Decoder.** One parallel pass over the block of 7 positions:

    x = tok_embd[block_tokens]                     the target's embedding table
    per layer:
        h  = rmsnorm(x, attn_norm)
        Q  = rmsnorm_q(wq @ h)     16 x 128
        K  = rmsnorm_k(wk @ h)      2 x 128
        V  = wv @ h                 2 x 128
        rope(Q at pos), rope(K at pos)
        a  = attn(Q, K, V over [0, n_committed) + this block), scale 1/sqrt(128)
        x  = x + wo @ a
        x  = x + down(silu(gate @ rmsnorm(x)) * (up @ rmsnorm(x)))
    x = rmsnorm(x, output_norm)
    logits_i = lm_head @ x_i                        the TARGET's lm_head

The block attention mask is **shared, not non-causal**: `self_attn_mask_swa.block_uniform
= true` makes every block query reuse the causal row of the block's FIRST query, so the
visible set is the prefix plus slot 0 (the anchor). The seven slots do not see each
other; they are independent denoising positions told apart only by rope position. That
is why slots 1..6 produce near-identical logits.

Block input is `[last_token, mask_token x (block_size - 1)]`: slot 0 is the embedding of
the committed token, every later slot is `llama_vocab_mask()`, which this vocabulary
records as `tokenizer.ggml.mask_token_id = 75982`. It is not the anchor repeated and it
is not a learned mask tensor -- this variant has no `mask_embd` (that tensor only exists
in the MLA `n_lora_kv > 0` branch of the reference).

**4. Markov head.** A rank-256 low-rank bigram bias, chained greedily over block
positions. With `sample_from_anchor = true` the chain starts from the anchor token:

    prev = anchor_token
    state = 0
    for i in 0 .. 6:
        w1p   = markov_w1[prev]                     gather one 256-row
        joint = joint_proj @ [state; w1p; x_i]      only for the rnn variant
        state = gate*state + (1-gate)*cand          (rnn variant only)
        markov = tanh(joint[2R:])                   or markov = w1p for the plain one
        bias  = markov_w2 @ markov                  256 -> 130560
        logits_i += bias
        draft_i = argmax(logits_i)
        prev = draft_i

This model has no `joint_proj`, so it is the plain variant: `markov = w1p` and there
is no running state. Confirm against `dspark_joint_proj` being absent from the tensor
list -- it is.

**5. Confidence head.** `conf_i = sigmoid(conf_proj @ [x_i; markov_i] + b)`, 2304 ->
1. Predicts per-position acceptance; useful later for trimming `K` to the number of
drafts likely to survive, which saves verification rows. Not needed to get working.

## Cost per round

At Q4_0 the draft's own weights are 174 MB (measured: llama-quantize takes the bf16
627 MB to 173.77 MiB at 4.5 bpw). Per round, with the block batched as 7 rows so each
weight is streamed once:

| stream | size | at 27.5 GB/s |
|---|---|---|
| draft weights (5 layers + fc + markov) | 174 MB | 6.3 ms |
| target lm_head, 7 positions batched | 267 MB | 9.7 ms |
| inject K/V (5 layers) | 5 MB | 0.2 ms |
| | | **~16 ms** |

The lm_head dominates and cannot be avoided: each draft position needs logits over the
full 130560 vocabulary. There is no `d2t` reduced-vocab mapping in this gguf, which
would have been the standard fix.

Against a measured 72.5 ms round that is 22% overhead. At two of three drafts accepted
that is 3 / 0.089 = 34 tok/s; at three of three, 45 tok/s. Both clear the 20 target.

## What is already done

Verification. `mega_decode -s K` runs a round as m = K+1 rows through one pass over
the weights, with the rows nested inside the GEMM's n-block walk, and a native
four-row IME2 kernel when m == 4. Exactness is proven against plain greedy with
`MEGA_REPLAY`, which replays true greedy output as the drafts: acceptance comes back
3/3 every round and the token streams diff identical. 55.2 tok/s at four rows.

So the drafter cannot make the output wrong. It can only lower the acceptance rate.

## What is left

1. Load `dspark-q4.gguf`: 62 tensors, all Q4_0 or f32, offsets dumped and sizes known.
   The linear layers repack into the same IME2 layout the target uses, which is why
   quantising the draft to Q4_0 first was worth doing.
2. Draft KV cache: 5 layers x MAXCTX x 2 heads x 128 dims, fp16, ~51 MB at 10K.
3. The encoder/inject step per committed position, so the draft cache stays in step.
4. The 7-position block pass and the Markov chain.
5. Wire `-s 3` to take drafts from the above instead of `MEGA_REPLAY`.

Confidence-gated `K` is a later optimisation, not a prerequisite.
