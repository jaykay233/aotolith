// Stubs for the ggml symbols the vendor objects reference.
//
// ime2.o and rvv.o are compiled from llama.cpp's SpacemiT backend. Most of what
// they contain is what we want -- the IME2 GEMM and the activation quantizers --
// but the objects also carry ggml-shaped entry points that pull in a handful of
// ggml helpers. Linking only these functions into a standalone probe means
// defining the symbols rather than dragging in llama.cpp, whose own build would
// need the whole ggml graph machinery.
//
// Nothing here is ever called: the GEMM and quantizer paths do not use them.
// ggml_abort is the exception -- it fires only if a contract check fails, so it
// aborts loudly instead of returning a value that would let a wrong result through.
#ifndef GGML_STUBS_H
#define GGML_STUBS_H

#include <cstddef>
#include <cstdlib>
#include <cstdio>

extern "C" {
 size_t   ggml_type_size(int) { return 0; }
 long long ggml_nelements(const void *) { return 0; }
 long long ggml_nrows(const void *) { return 0; }
 bool     ggml_can_repeat(const void *, const void *) { return false; }
 bool     ggml_are_same_shape(const void *, const void *) { return false; }
 bool     ggml_is_contiguous_rows(const void *) { return true; }
 bool     ggml_is_contiguous(const void *) { return true; }
 bool     ggml_is_transposed(const void *) { return false; }
 size_t   ggml_element_size(const void *) { return 1; }
 void ggml_abort(const char * f, int l, const char * m, ...) {
    fprintf(stderr, "ggml_abort %s:%d %s\n", f, l, m);
    abort();
}
}

#endif
