// Insert this block in spacemit_kernels::forward_flash_attn_ext_f16 (ime.cpp),
// immediately AFTER the REJECT block's `ggml_compute_forward_flash_attn_ext(params, dst); return;`
// and BEFORE the `// total rows in q` comment.
//
// Why: for single-token decode the one_chunk kernel parallelises over q head rows, so each
// thread re-reads the entire KV cache (at 10K ctx, 8 threads x 430 MB = 3.4 GB/token).
// The generic split-KV path chunks the keys across threads and reads KV exactly once.

        // Single-token, long-KV decode: fall through to the generic split-KV path.
        if (neq1 == 1 && neq3 == 1 && nek1 >= 512 &&
            q->type == GGML_TYPE_F32 &&
            k->type == GGML_TYPE_F16 && k->type == v->type) {
            if (getenv("FA_TRACE")) {
                static std::atomic<int> sk_on{0};
                if (sk_on.fetch_add(1) < 4) {
                    fprintf(stderr, "[FA_TRACE] SPLITKV neq1=%lld nek1=%lld DK=%lld DV=%lld nth=%d\n",
                            (long long) neq1, (long long) nek1, (long long) DK, (long long) DV,
                            (int) params->nth);
                }
            }
            ggml_compute_forward_flash_attn_ext(params, dst);
            return;
        }
