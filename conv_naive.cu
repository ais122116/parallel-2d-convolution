#include "conv_io.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_),  \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static const int REPS = 5;

__global__ void conv2d_naive(const float* in, const float* f, float* out,
                             int M, int N, int C, int K, int W)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // column
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // row
    const int k = blockIdx.z;                              // output channel
    if (i >= M || j >= N) return;

    const int R = W / 2;
    float acc = 0.f;
    for (int c = 0; c < C; ++c) {
        const float* inc = in + (size_t)c * M * N;              // input channel c
        const float* fkc = f + ((size_t)k * C + c) * W * W;     // filter slice (k, c)
        for (int u = 0; u < W; ++u) {
            const int ii = i + u - R;
            if (ii < 0 || ii >= M) continue;                    // zero padding (rows)
            for (int v = 0; v < W; ++v) {
                const int jj = j + v - R;
                if (jj < 0 || jj >= N) continue;                // zero padding (cols)
                acc += inc[(size_t)ii * N + jj] * fkc[u * W + v];
            }
        }
    }
    out[((size_t)k * M + i) * N + j] = acc;
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s input.bin filters.bin output.bin [block 8|16|32]\n", argv[0]);
        return 1;
    }
    const int B = argc == 5 ? atoi(argv[4]) : 16;
    if (B <= 0 || B * B > 1024) { fprintf(stderr, "block must be 1..32\n"); return 1; }

    Tensor in, f;
    if (!read_tensor(argv[1], in) || !read_tensor(argv[2], f)) return 1;
    int M, N, C, K, W;
    if (!get_problem_dims(in, f, M, N, C, K, W)) return 1;
    if (K > 65535) { fprintf(stderr, "K too large for grid.z\n"); return 1; }

    Tensor out;
    out.dims = {K, M, N};
    out.data.resize(out.count());

    const size_t in_bytes = in.count() * sizeof(float);
    const size_t f_bytes = f.count() * sizeof(float);
    const size_t out_bytes = out.count() * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaFree(0));   // start the GPU context here, so its one-time setup is not counted as malloc time

    float *d_in, *d_f, *d_out;
    auto tm0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_f, f_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    auto tm1 = std::chrono::high_resolution_clock::now();
    const double malloc_ms = std::chrono::duration<double, std::milli>(tm1 - tm0).count();

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // host -> device
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, in.data.data(), in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, f.data.data(), f_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));

    dim3 block(B, B);
    dim3 grid((N + B - 1) / B, (M + B - 1) / B, K);

    // warm-up (first launch includes one-time setup costs)
    conv2d_naive<<<grid, block>>>(d_in, d_f, d_out, M, N, C, K, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed runs
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < REPS; ++r)
        conv2d_naive<<<grid, block>>>(d_in, d_f, d_out, M, N, C, K, W);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(e1));
    float kern_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, e0, e1));
    kern_ms /= REPS;

    // device -> host
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, e0, e1));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    if (!write_tensor(argv[3], out)) return 1;

    const double gflops = 2.0 * K * M * N * C * W * W / (kern_ms * 1e6);
    printf("M=%d N=%d C=%d K=%d W=%d impl=gpu_naive block=%dx%d gpu=\"%s\"\n",
           M, N, C, K, W, B, B, prop.name);
    printf("copy_ms: h2d %.3f  d2h %.3f   kernel GFLOP/s: %.1f\n", h2d_ms, d2h_ms, gflops);
    printf("total_ms: %.3f  (malloc %.3f + h2d %.3f + kernel %.3f + d2h %.3f)\n",
           malloc_ms + h2d_ms + kern_ms + d2h_ms, malloc_ms, h2d_ms, kern_ms, d2h_ms);
    printf("time_ms: %.3f\n", kern_ms);   // kernel only - test scripts parse this line
    return 0;
}
