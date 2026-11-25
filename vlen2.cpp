#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <riscv_vector.h>
int main() {
    for (int cpu = 0; cpu < 16; ++cpu) {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
        if (sched_setaffinity(0, sizeof s, &s) != 0) { printf("cpu %2d: cannot bind\n", cpu); continue; }
        cpu_set_t now; CPU_ZERO(&now);
        sched_getaffinity(0, sizeof now, &now);
        if (!CPU_ISSET(cpu, &now)) { printf("cpu %2d: bind lost\n", cpu); continue; }
        long vb = 0; asm volatile("csrr %0, vlenb" : "=r"(vb));
        const size_t v1 = __riscv_vsetvl_e32m1((size_t) 4096);
        const size_t v2 = __riscv_vsetvl_e32m2((size_t) 4096);
        printf("cpu %2d: vlenb=%3ld  e32m1=%2zu  e32m2=%2zu\n", cpu, vb, v1, v2);
    }
    return 0;
}
