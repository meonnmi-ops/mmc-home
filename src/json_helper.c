#include <string.h>

static int find_value(char* buf, int len, const char* key) {
    int klen = strlen(key);
    int i;
    for (i = 0; i < len - klen - 3; i++) {
        if (buf[i] == '"' &&
            strncmp(buf + i + 1, key, klen) == 0 &&
            buf[i + klen + 1] == '"' &&
            buf[i + klen + 2] == ':') {
            int p = i + klen + 3;
            while (p < len && (buf[p]==' ' || buf[p]=='\t')) p++;
            if (p < len && buf[p] == '"') return p + 1;
        }
    }
    return -1;
}

static int decode_str(char* buf, int len, int pos, char* out, int cap) {
    int o = 0;
    while (pos < len && buf[pos] != '"' && o < cap - 1) {
        if (buf[pos] == '\\' && pos+1 < len) {
            pos++;
            switch (buf[pos]) {
                case 'n': out[o++]='\n'; break;
                case 't': out[o++]='\t'; break;
                case 'r': out[o++]='\r'; break;
                case '"': out[o++]='"';  break;
                case '\\': out[o++]='\\'; break;
                case '/': out[o++]='/';  break;
                default: out[o++]=buf[pos];
            }
            pos++;
        } else {
            out[o++] = buf[pos++];
        }
    }
    out[o] = 0;
    return o;
}

int json_extract_text(char* buf, int len, char* out, int cap) {
    if (!buf || !out || cap <= 0 || len <= 7) return -1;

    /* 1. "text" */
    int pos = find_value(buf, len, "text");
    if (pos > 0) return decode_str(buf, len, pos, out, cap);

    /* 2. "instruction" + "\n" + "output" */
    int pi = find_value(buf, len, "instruction");
    int po = find_value(buf, len, "output");
    if (pi > 0 && po > 0 && pi < po) {
        int n = decode_str(buf, len, pi, out, cap);
        if (n < 0) return -1;
        if (n < cap - 1) out[n++] = '\n';
        int rest = decode_str(buf, len, po, out + n, cap - n);
        return (rest >= 0) ? (n + rest) : n;
    }

    /* 3. "output" only */
    if (po > 0) return decode_str(buf, len, po, out, cap);
    return -1;
}
