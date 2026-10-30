#include <stdio.h>
#include <stdlib.h>

static int loaded_size = 0;

void* mmc_load_binary(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }

    long size = ftell(f);
    if (size < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    void* data = malloc((size_t)size);
    if (!data) {
        fclose(f);
        return NULL;
    }

    if (fread(data, 1, (size_t)size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return NULL;
    }

    fclose(f);
    loaded_size = (int)size;
    return data;
}

int mmc_load_binary_size(void) {
    return loaded_size;
}
