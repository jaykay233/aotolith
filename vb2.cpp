#include <cstdio>
#include <cstdint>
#include <riscv_vector.h>

static inline uint64_t rdtime(){ uint64_t t; __asm__ __volatile__("rdtime %0":"=r"(t)); return t; }
alignas(64) static _Float16 kbuf[8192];
alignas(64) static float    fbuf[8192];

#define REP4(a,s,v) a##0=__riscv_vfmacc_vf_f32m4(a##0,s,v,vl128); a##1=__riscv_vfmacc_vf_f32m4(a##1,s,v,vl128); \
                    a##2=__riscv_vfmacc_vf_f32m4(a##2,s,v,vl128); a##3=__riscv_vfmacc_vf_f32m4(a##3,s,v,vl128)
#define REP8(a,s,v) REP4(a,s,v); a##4=__riscv_vfmacc_vf_f32m4(a##4,s,v,vl128); a##5=__riscv_vfmacc_vf_f32m4(a##5,s,v,vl128); \
                    a##6=__riscv_vfmacc_vf_f32m4(a##6,s,v,vl128); a##7=__riscv_vfmacc_vf_f32m4(a##7,s,v,vl128)
#define SQ4(a) __riscv_vse32_v_f32m4(fbuf+0,a##0,vl128); __riscv_vse32_v_f32m4(fbuf+128,a##1,vl128); \
              __riscv_vse32_v_f32m4(fbuf+256,a##2,vl128); __riscv_vse32_v_f32m4(fbuf+384,a##3,vl128)
#define SQ8(a) SQ4(a); __riscv_vse32_v_f32m4(fbuf+512,a##4,vl128); __riscv_vse32_v_f32m4(fbuf+640,a##5,vl128); \
               __riscv_vse32_v_f32m4(fbuf+768,a##6,vl128); __riscv_vse32_v_f32m4(fbuf+896,a##7,vl128)
#define SQUARE8(a) SQ8(a);

int main(){
    for (int i=0;i<8192;i++){ kbuf[i]=(_Float16)((i%13)*0.01f); fbuf[i]=0.25f; }
    const int iters=300000;
    const float s=0.001f;
    #ifdef B_A
    {   // A: f32m4 accumulators, vl=128, 4 accs x 1 vfmacc each = 4 instr / 512 MAC
        const size_t vl128=128;
        const vfloat32m4_t v=__riscv_vle32_v_f32m4(fbuf,vl128);
        vfloat32m4_t a0=__riscv_vfmv_v_f_f32m4(0,vl128),a1=a0,a2=a0,a3=a0;
        uint64_t t0=rdtime();
        for(int it=0;it<iters;it++){ REP4(a,s,v); }
        uint64_t t1=rdtime();
        SQ4(a);
        double ns=(double)(t1-t0)*1000.0/24.0/iters/4.0;
        printf("A f32m4 vl128 x4 acc : %6.2f ns/instr  %6.1f GMAC/s/thread\n", ns, 128.0/ns);
    }
    #endif
#ifdef B_B
    {   // B: f32m4 accumulators, vl=128, 8 accs = 8 instr / 1024 MAC
        const size_t vl128=128;
        const vfloat32m4_t v=__riscv_vle32_v_f32m4(fbuf,vl128);
        vfloat32m4_t a0=__riscv_vfmv_v_f_f32m4(0,vl128),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        uint64_t t0=rdtime();
        for(int it=0;it<iters;it++){ REP8(a,s,v); }
        uint64_t t1=rdtime();
        SQUARE8(a);
        double ns=(double)(t1-t0)*1000.0/24.0/iters/8.0;
        printf("B f32m4 vl128 x8 acc : %6.2f ns/instr  %6.1f GMAC/s/thread\n", ns, 128.0/ns);
    }
    #endif
#ifdef B_C
    {   // C: f32m2 accumulators, vl=64, 8 accs = 8 instr / 512 MAC
        const size_t vl128=64;
        const vfloat32m2_t v=__riscv_vle32_v_f32m2(fbuf,vl128);
        vfloat32m2_t b0=__riscv_vfmv_v_f_f32m2(0,vl128),b1=b0,b2=b0,b3=b0,b4=b0,b5=b0,b6=b0,b7=b0;
        uint64_t t0=rdtime();
        for(int it=0;it<iters;it++){
            b0=__riscv_vfmacc_vf_f32m2(b0,s,v,vl128); b1=__riscv_vfmacc_vf_f32m2(b1,s,v,vl128);
            b2=__riscv_vfmacc_vf_f32m2(b2,s,v,vl128); b3=__riscv_vfmacc_vf_f32m2(b3,s,v,vl128);
            b4=__riscv_vfmacc_vf_f32m2(b4,s,v,vl128); b5=__riscv_vfmacc_vf_f32m2(b5,s,v,vl128);
            b6=__riscv_vfmacc_vf_f32m2(b6,s,v,vl128); b7=__riscv_vfmacc_vf_f32m2(b7,s,v,vl128);
        }
        uint64_t t1=rdtime();
        __riscv_vse32_v_f32m2(fbuf+0,b0,vl128); __riscv_vse32_v_f32m2(fbuf+64,b1,vl128);
        __riscv_vse32_v_f32m2(fbuf+128,b2,vl128); __riscv_vse32_v_f32m2(fbuf+192,b3,vl128);
        __riscv_vse32_v_f32m2(fbuf+256,b4,vl128); __riscv_vse32_v_f32m2(fbuf+320,b5,vl128);
        __riscv_vse32_v_f32m2(fbuf+384,b6,vl128); __riscv_vse32_v_f32m2(fbuf+448,b7,vl128);
        double ns=(double)(t1-t0)*1000.0/24.0/iters/8.0;
        printf("C f32m2 vl64  x8 acc : %6.2f ns/instr  %6.1f GMAC/s/thread\n", ns, 64.0/ns);
    }
    #endif
#ifdef B_D
    {   // D: widening vfwmacc f16m2->f32m4, vl=128 (the qk pattern)
        const size_t vl128=128;
        const vfloat16m2_t k=__riscv_vle16_v_f16m2(kbuf,vl128);
        vfloat32m4_t a0=__riscv_vfmv_v_f_f32m4(0,vl128),a1=a0,a2=a0,a3=a0;
        uint64_t t0=rdtime();
        for(int it=0;it<iters;it++){
            a0=__riscv_vfwmacc_vf_f32m4(a0,s,k,vl128); a1=__riscv_vfwmacc_vf_f32m4(a1,s,k,vl128);
            a2=__riscv_vfwmacc_vf_f32m4(a2,s,k,vl128); a3=__riscv_vfwmacc_vf_f32m4(a3,s,k,vl128);
        }
        uint64_t t1=rdtime();
        SQ4(a);
        double ns=(double)(t1-t0)*1000.0/24.0/iters/4.0;
        printf("D vfwmacc f32m4 x4   : %6.2f ns/instr  %6.1f GMAC/s/thread\n", ns, 128.0/ns);
    }
    printf("sink %f %f\n", fbuf[0], (double)kbuf[0]);
    return 0;
}
#endif
