/*
 * Compatibility translation unit for existing build commands.
 * Runtime functions are implemented in runtime_v5.c; defining wrappers here
 * would produce duplicate symbols when both files are linked.
 */
#include "runtime_v5.h"

/* benchmark timing — monotonic ms */
#include <time.h>
double mmc_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
