// Greedy reference driver: the same token ids in, the same ids out.
//
// `llama-simple` has no way to set n_ctx, so a 10K prompt is silently truncated
// to its 512-token default, and `llama-cli`'s non-interactive switches differ
// between builds. This links libllama directly: tokenize nothing (the ids come
// from a file), disable all sampling but argmax, and print just the ids so the
// megakernel's own chain can be diffed against them token by token.
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
    cp.n_ctx = (uint32_t) (ids.size() + ngen + 32 > (size_t) nctx ? ids.size() + ngen + 32 : (size_t) nctx);
    cp.n_batch = (uint32_t) cp.n_ctx;
    cp.n_ubatch = (uint32_t) cp.n_ctx;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const int n_embd = llama_model_n_embd(model);
    (void) n_embd;
    const int np = (int) ids.size();
    llama_batch b = llama_batch_init(np, 0, 1);
    for (int i = 0; i < np; ++i) {
        b.token[i] = ids[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (i == np - 1);
    }
    b.n_tokens = np;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }

    llama_token cur = 0;
    int pos = np - 1;   // last prompt token sat at np-1; the next one goes at np
    printf("prompt %d\n", np);
    (void) cur;
    for (int s = 0; s < ngen; ++s) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        int best = 0; float bv = lg[0];
        for (int v = 1; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
        printf("  step %d -> %d (%.4f)\n", s, best, bv);
        fflush(stdout);
        if (s + 1 == ngen) break;
        llama_batch one = llama_batch_init(1, 0, 1);
        one.token[0] = best; one.pos[0] = pos + 1; one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0; one.logits[0] = true; one.n_tokens = 1;
        if (llama_decode(ctx, one) != 0) { fprintf(stderr, "step decode failed\n"); return 1; }
        llama_batch_free(one);
        cur = best; ++pos;
    }
    (void) cur;
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
