// Greedy reference driver: the same token ids in, the same ids out.
//
// `llama-simple` has no way to set n_ctx, so a 10K prompt is silently truncated
// to its 512-token default, and `llama-cli`'s non-interactive switches differ
// between builds. This links libllama directly: the ids come from a file (no
// tokeniser in the loop), all sampling is off, and it prints just the ids so the
// megakernel's chain can be diffed against them token by token.
//
// The prompt goes through in n_batch-sized slices because llama_decode asserts
// n_tokens <= cparams.n_batch; only the last slice asks for logits.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model ids.txt n_gen [n_ctx]\n", argv[0]); return 2; }
    const char * mpath = argv[1];
    const char * ipath = argv[2];
    const int ngen = atoi(argv[3]);
    const int nctx = argc > 4 ? atoi(argv[4]) : 12288;

    std::vector<llama_token> ids;
    if (FILE * f = fopen(ipath, "r")) {
        long v;
        while (fscanf(f, "%ld", &v) == 1) ids.push_back((llama_token) v);
        fclose(f);
    }
    if (ids.empty()) { fprintf(stderr, "no ids in %s\n", ipath); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) nctx;
    cp.n_batch = 2048;      // llama.cpp defaults; a single 10K decode asserts
    cp.n_ubatch = 512;
    cp.no_perf = true;
    // Embeddings are off by default so the graph's output stays logits. When
    // REF_EMB is set the final normed row is captured too, with no pooling so it
    // is the last token's vector rather than a sequence mean.
    cp.embeddings = getenv("REF_EMB") != nullptr;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const int np = (int) ids.size();
    for (int i0 = 0; i0 < np; i0 += 512) {
        const int n = (np - i0 < 512) ? (np - i0) : 512;
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = ids[i0 + i];
            b.pos[i] = i0 + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i0 + i == np - 1);
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", i0); return 1; }
        llama_batch_free(b);
    }

    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    // Optional hidden-state dump: with embeddings on, the graph's last node is the
    // final normed row, so this is the same vector the head multiplies, before any
    // quantisation of the head weights. Diffing it against the megakernel's own
    // pre-head row says whether a logit mismatch is upstream or in the head.
    const char * edump = getenv("REF_EMB");
    if (edump) {
        const float * e = llama_get_embeddings_ith(ctx, -1);
        if (e) {
            if (FILE * f = fopen(edump, "wb")) {
                fwrite(e, sizeof(float), (size_t) llama_model_n_embd(model), f);
                fclose(f);
                fprintf(stderr, "emb -> %s\n", edump);
            }
        } else fprintf(stderr, "no embeddings\n");
    }
    int pos = np - 1;
    printf("prompt %d\n", np);
    const char * ldump = getenv("REF_LOG");
    for (int s = 0; s < ngen; ++s) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        float bv = lg[0];
        for (int v = 1; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
        if (ldump && s == 0) {
            if (FILE * f = fopen(ldump, "wb")) {
                fwrite(lg, sizeof(float), (size_t) nv, f);
                fclose(f);
                fprintf(stderr, "logits -> %s\n", ldump);
            }
        }
        printf("  step %d -> %d (%.4f)\n", s, best, bv);
        fflush(stdout);
        if (s + 1 == ngen) break;
        llama_batch one = llama_batch_init(1, 0, 1);
        one.token[0] = best; one.pos[0] = ++pos; one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0; one.logits[0] = true; one.n_tokens = 1;
        if (llama_decode(ctx, one) != 0) { fprintf(stderr, "step decode failed\n"); return 1; }
        llama_batch_free(one);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
