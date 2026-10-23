// compare.cpp - check a test output against the CPU reference
//
// Usage: ./compare ref.bin test.bin [atol=1e-3] [rtol=1e-3]
// Element passes if |test - ref| <= atol + rtol * |ref|.
// Exit code 0 = PASS, 1 = FAIL.
#include "conv_io.h"
#include <cmath>
int main(int argc, char** argv) 
{
    if (argc < 3) 
    {
        fprintf(stderr, "Usage: %s ref.bin test.bin [atol] [rtol]\n", argv[0]);
        return 1;
    }
    double atol = argc > 3 ? atof(argv[3]) : 1e-3;
    double rtol = argc > 4 ? atof(argv[4]) : 1e-3;
    Tensor ref, tst;
    if (!read_tensor(argv[1], ref) || !read_tensor(argv[2], tst)) 
    {
        return 1;
    }
    if (ref.dims != tst.dims) 
    {
         printf("FAIL: shape mismatch\n"); 
         return 1; 
    }

    size_t bad = 0, first_bad = 0, worst = 0;
    double max_err = 0.0;
    for (size_t x = 0; x < ref.count(); ++x) 
    {
        double r = ref.data[x], t = tst.data[x];
        double err = std::fabs(t - r);
        if (std::isnan(t) || err > atol + rtol * std::fabs(r)) 
        {
            if (bad++ == 0) first_bad = x;
        }
        if (err > max_err || std::isnan(t)) 
        {
             max_err = err;
             worst = x; 
        }
    }
    if (bad) 
    {
        int M = ref.dims[1], N = ref.dims[2];
        size_t k = first_bad / ((size_t)M * N), rem = first_bad % ((size_t)M * N);
        printf("FAIL: %zu / %zu mismatches; first at (k=%zu,i=%zu,j=%zu) ref=%g got=%g; max_err=%g\n", bad, ref.count(), k, rem / N, rem % N,ref.data[first_bad], tst.data[first_bad], max_err);
        return 1;
    }
    printf("PASS (max_abs_err=%.3g at index %zu)\n", max_err, worst);
    return 0;
}
