// bin2txt.cpp - print a .bin tensor file (input / filters / output) as readable text
//
// Usage: ./bin2txt file.bin              (prints to screen)
//        ./bin2txt file.bin out.txt      (writes to a text file)
//
// The last two dims are printed as a matrix; leading dims label each matrix.
//   input   {C, M, N}    -> C matrices of M x N
//   filters {K, C, W, W} -> K*C matrices of W x W
//   output  {K, M, N}    -> K matrices of M x N
#include "conv_io.h"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s file.bin [out.txt]\n", argv[0]);
        return 1;
    }
    Tensor t;
    if (!read_tensor(argv[1], t)) return 1;

    FILE* out = stdout;
    if (argc == 3 && !(out = fopen(argv[2], "w"))) {
        fprintf(stderr, "Cannot open %s\n", argv[2]);
        return 1;
    }

    int nd = (int)t.dims.size();
    fprintf(out, "dims:");
    for (int d : t.dims) fprintf(out, " %d", d);
    fprintf(out, "\n");

    int rows = nd >= 2 ? t.dims[nd - 2] : 1;
    int cols = t.dims[nd - 1];
    size_t per = (size_t)rows * cols;
    size_t nmat = t.count() / per;

    for (size_t m = 0; m < nmat; ++m) {
        // label like [k=1, c=0] or [c=2]
        if (nd > 2) {
            const char* names3[] = {"ch"};         // input: c, output: k
            const char* names4[] = {"k", "c"};     // filters {K,C,W,W}
            fprintf(out, "\n[");
            size_t rem = m;
            std::vector<size_t> idx(nd - 2);
            for (int d = nd - 3; d >= 0; --d) { idx[d] = rem % t.dims[d]; rem /= t.dims[d]; }
            for (int d = 0; d < nd - 2; ++d) {
                const char* nm = (nd == 4) ? names4[d] : (nd == 3 ? names3[0] : "i");
                fprintf(out, "%s%s=%zu", d ? ", " : "", nm, idx[d]);
            }
            fprintf(out, "]\n");
        }
        const float* p = t.data.data() + m * per;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) fprintf(out, "%9.5f ", p[(size_t)r * cols + c]);
            fprintf(out, "\n");
        }
    }
    if (out != stdout) fclose(out);
    return 0;
}
