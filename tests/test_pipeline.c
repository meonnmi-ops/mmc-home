#include <stdio.h>
#include <stdlib.h>
int mmc_analyze_jsonl(char* path, int max_lines, int max_bytes, int* stats);
int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s file.jsonl max_lines max_bytes\n", argv[0]); return 1; }
    int stats[5] = {0};
    int rc = mmc_analyze_jsonl(argv[1], atoi(argv[2]), atoi(argv[3]), stats);
    printf("rc=%d\n", rc);
    printf("lines     = %d\n", stats[0]);
    printf("texts     = %d\n", stats[1]);
    printf("total_ch  = %d\n", stats[2]);
    printf("zg_reject = %d\n", stats[3]);
    printf("accepted  = %d\n", stats[4]);
    return 0;
}
