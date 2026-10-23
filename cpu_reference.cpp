#include "conv_io.h"
#include <chrono>
#include <cmath>
#include <cstring>
void conv2d_cpu(const float* in, const float* f, float* out,int M, int N, int C, int K, int W)
{
    const int R = W / 2;
    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j) 
            {
                double acc = 0.0;   
                for (int c = 0; c < C; ++c)
                {
                    const float* inc = in + (size_t)c * M * N;
                    const float* fkc = f + ((size_t)k * C + c) * W * W;
                    for (int u = 0; u < W; ++u) 
                    {
                        int ii = i + u - R;
                        if (ii < 0 || ii >= M)
                         continue;         
                        for (int v = 0; v < W; ++v)
                        {
                            int jj = j + v - R;
                            if (jj < 0 || jj >= N) continue;      // zero padding (cols)
                            acc += (double)inc[(size_t)ii * N + jj] * fkc[u * W + v];
                        }
                    }
                }
                
            out[((size_t)k * M + i) * N + j] = (float)acc;
            }
        }
    }
}
// Small hand-checkable cases.
static int selftest() 
{
    int fails = 0;    
    {
        int M = 3, N = 4, C = 1, K = 1, W = 1;
        std::vector<float> in(M * N), f(1, 1.0f), out(M * N);
        for (int x = 0; x < M * N; ++x) 
        {
            in[x] = 0.1f * x;
        }
        conv2d_cpu(in.data(), f.data(), out.data(), M, N, C, K, W);
        for (int x = 0; x < M * N; ++x)
        {
            if (out[x] != in[x]) 
            { 
                ++fails; 
                break; 
            }
        }
        printf("identity 1x1  %s\n", fails ? "FAIL" : "ok");
    }
    {
        int prev = fails;
        std::vector<float> in(9, 1.0f), f(9, 1.0f), out(9);
        conv2d_cpu(in.data(), f.data(), out.data(), 3, 3, 1, 1, 3);
        float exp[9] = {4, 6, 4, 6, 9, 6, 4, 6, 4};
        for (int x = 0; x < 9; ++x)
        {
            if (out[x] != exp[x])
             { 
                ++fails; 
                break;
             }
        }
        printf("zero padding 3x3  %s\n", fails > prev ? "FAIL" : "ok");
    }
    {
        int prev = fails;
        int M = 5, N = 5, C = 2, K = 2, W = 3;
        std::vector<float> in(C * M * N), f(K * C * W * W), out(K * M * N);
        for (int c = 0; c < C; ++c)
        {
            for (int x = 0; x < M * N; ++x) 
            {
                in[c * M * N + x] = (float)(c + 1);
            }
        }
        for (int k = 0; k < K; ++k)
        {
            for (int x = 0; x < C * W * W; ++x) 
            {
                f[k * C * W * W + x] = (float)(k + 1);
            }
        }
        conv2d_cpu(in.data(), f.data(), out.data(), M, N, C, K, W);
        for (int k = 0; k < K; ++k) 
        {
            float centre = out[((size_t)k * M + 2) * N + 2];
            float corner = out[((size_t)k * M + 0) * N + 0];
            if (centre != 27.0f * (k + 1) || corner != 12.0f * (k + 1))
            { 
                ++fails;
                 break; 
            }
        }
        printf("multi-channel  %s\n", fails > prev ? "FAIL" : "ok");
    }
    printf(fails ? "SELFTEST FAILED\n" : "SELFTEST PASSED\n");
    return fails ? 1 : 0;
}

int main(int argc, char** argv) 
{
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc != 4) 
    {
        fprintf(stderr, "Usage: %s input.bin filters.bin output.bin | --selftest\n", argv[0]);
        return 1;
    }
    Tensor in, f;
    if (!read_tensor(argv[1], in) || !read_tensor(argv[2], f))
    { return 1;
    }
    int M, N, C, K, W;
    if (!get_problem_dims(in, f, M, N, C, K, W)) 
    {
        return 1;
    }

    Tensor out;
    out.dims = {K, M, N};
    out.data.resize(out.count());

    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(in.data.data(), f.data.data(), out.data.data(), M, N, C, K, W);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!write_tensor(argv[3], out))
    { 
        return 1;
    }
    printf("M=%d N=%d C=%d K=%d W=%d\n", M, N, C, K, W);
    printf("time_ms: %.3f\n", ms);   // harness parses this line
    return 0;
}
