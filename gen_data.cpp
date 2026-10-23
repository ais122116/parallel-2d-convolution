// gen_data.cpp - generate random input image and filter bank
//
// Usage: ./gen_data M N C K W seed input.bin filters.bin
//   input   values ~ U[0, 1]   (as per spec)
//   filters values ~ U[-1, 1]
#include "conv_io.h"
#include <random>
int main(int argc, char** argv) 
{
    if (argc != 9) 
    {
        fprintf(stderr, "Usage: %s M N C K W seed input.bin filters.bin\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]), N = atoi(argv[2]), C = atoi(argv[3]);
    int K = atoi(argv[4]), W = atoi(argv[5]);
    unsigned seed = (unsigned)strtoul(argv[6], nullptr, 10);
    if (M <= 0 || N <= 0 || C <= 0 || K <= 0 || W <= 0 || W % 2 == 0) 
    {
        fprintf(stderr, "Invalid sizes (W must be odd and positive)\n");
        return 1;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uin(0.0f, 1.0f), uf(-1.0f, 1.0f);
    Tensor in;
    in.dims = {C, M, N};
    in.data.resize(in.count());
    for (auto& x : in.data) 
    {
        x = uin(rng);
    }
    Tensor f;
    f.dims = {K, C, W, W};
    f.data.resize(f.count());
    for (auto& x : f.data)
    {
         x = uf(rng);
    }
    if (!write_tensor(argv[7], in) || !write_tensor(argv[8], f)) return 1;
    return 0;
}
