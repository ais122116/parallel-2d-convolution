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

// Block = TW x TH threads. Output tile = TW columns x (TH*PY) rows x KB filters.
template <int W, int KB, int TW, int TH, int PY>
__global__ void __launch_bounds__(TW * TH)
conv2d_tiled(const float* __restrict__ in, const float* __restrict__ f, float* __restrict__ out,
             int M, int N, int C, int K)
{
    constexpr int R = W / 2;
    constexpr int OH = TH * PY;                 // output tile height
    constexpr int SH = OH + W - 1;              // shared tile height incl. halo
    constexpr int SW = TW + W - 1;              // shared tile width  incl. halo
    constexpr int NT = TW * TH;                 // threads per block
    constexpr int WW = W * W;
    constexpr int NL = (SH * SW + NT - 1) / NT; // tile values loaded per thread
    constexpr int NWL = (KB * WW + NT - 1) / NT;// weights loaded per thread

    __shared__ float s_in[2][SH * SW];          // double-buffered input tile
    __shared__ float s_w[2][KB * WW];           // double-buffered weights (KB filters, 1 channel)

    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TW + tx;
    const int j0 = blockIdx.x * TW, i0 = blockIdx.y * OH, k0 = blockIdx.z * KB;

    float pin[NL], pw[NWL];                     // prefetch registers

    // global -> registers (coalesced: consecutive tid -> consecutive columns)
    auto fetch = [&](int c) {
        const float* inc = in + (size_t)c * M * N;
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const int idx = tid + l * NT;
            float x = 0.f;
            if (idx < SH * SW) {
                const int r = idx / SW, q = idx - r * SW;
                const int gi = i0 - R + r, gj = j0 - R + q;
                if (gi >= 0 && gi < M && gj >= 0 && gj < N) x = inc[(size_t)gi * N + gj];
            }
            pin[l] = x;                         // out of image -> 0 (zero padding)
        }
#pragma unroll
        for (int l = 0; l < NWL; ++l) {
            const int idx = tid + l * NT;
            float x = 0.f;
            if (idx < KB * WW) {
                const int kb = idx / WW, t = idx - kb * WW, k = k0 + kb;
                if (k < K) x = f[((size_t)k * C + c) * WW + t];
            }
            pw[l] = x;
        }
    };
    // registers -> shared buffer b
    auto store = [&](int b) {
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const int idx = tid + l * NT;
            if (idx < SH * SW) s_in[b][idx] = pin[l];
        }
#pragma unroll
        for (int l = 0; l < NWL; ++l) {
            const int idx = tid + l * NT;
            if (idx < KB * WW) s_w[b][idx] = pw[l];
        }
    };

    float acc[PY][KB];
#pragma unroll
    for (int p = 0; p < PY; ++p)
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) acc[p][kb] = 0.f;

    fetch(0);
    store(0);
    __syncthreads();

    for (int c = 0; c < C; ++c) {
        const int b = c & 1;
        const bool more = c + 1 < C;            // same for the whole block
        if (more) fetch(c + 1);                 // global loads in flight during compute

        const float* tile = s_in[b] + (ty * PY) * SW + tx;
#pragma unroll
        for (int v = 0; v < W; ++v) {
            float wr[KB][W];                    // filter column v for all KB filters
#pragma unroll
            for (int kb = 0; kb < KB; ++kb)
#pragma unroll
                for (int u = 0; u < W; ++u) wr[kb][u] = s_w[b][kb * WW + u * W + v];
            // walk down the PY+W-1 tile rows this thread needs; each value
            // contributes to output row p = r - u through filter row u
#pragma unroll
            for (int r = 0; r < PY + W - 1; ++r) {
                const float x = tile[r * SW + v];
#pragma unroll
                for (int u = 0; u < W; ++u) {
                    const int p = r - u;
                    if (p >= 0 && p < PY) {     // resolved at compile time
#pragma unroll
                        for (int kb = 0; kb < KB; ++kb) acc[p][kb] = fmaf(wr[kb][u], x, acc[p][kb]);
                    }
                }
            }
        }
        if (more) store(b ^ 1);                 // other buffer: nobody is reading it now
        __syncthreads();
    }

    // write results (coalesced along j)
    const int j = j0 + tx;
    if (j >= N) return;
#pragma unroll
    for (int p = 0; p < PY; ++p) {
        const int i = i0 + ty * PY + p;
        if (i >= M) break;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            const int k = k0 + kb;
            if (k < K) out[((size_t)k * M + i) * N + j] = acc[p][kb];
        }
    }
}

// ---------------------------------------------------------------------------
// Launch helpers: pick the template instance for runtime (W, KB)
// ---------------------------------------------------------------------------
static const int BW = 32;   // threads along a row
static const int BH = 8;    // threads along a column
static const int BPY = 4;   // output rows per thread -> output tile 32 x 32

template <int W, int KB>
static void launchT(const float* in, const float* f, float* out, int M, int N, int C, int K)
{
    dim3 block(BW, BH);
    dim3 grid((N + BW - 1) / BW, (M + BH * BPY - 1) / (BH * BPY), (K + KB - 1) / KB);
    conv2d_tiled<W, KB, BW, BH, BPY><<<grid, block>>>(in, f, out, M, N, C, K);
}

template <int W>
static bool byKB(int KB, const float* in, const float* f, float* out, int M, int N, int C, int K)
{
    switch (KB) {
    case 1: launchT<W, 1>(in, f, out, M, N, C, K); return true;
    case 2: launchT<W, 2>(in, f, out, M, N, C, K); return true;
    case 4: launchT<W, 4>(in, f, out, M, N, C, K); return true;
    case 8: launchT<W, 8>(in, f, out, M, N, C, K); return true;
    }
    return false;
}

static bool launch(int W, int KB, const float* in, const float* f, float* out, int M, int N, int C, int K)
{
    switch (W) {
    case 1: return byKB<1>(KB, in, f, out, M, N, C, K);
    case 3: return byKB<3>(KB, in, f, out, M, N, C, K);
    case 5: return byKB<5>(KB, in, f, out, M, N, C, K);
    case 7: return byKB<7>(KB, in, f, out, M, N, C, K);
    }
    return false;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s input.bin filters.bin output.bin [--kb 1|2|4|8] [--pinned]\n", argv[0]);
        return 1;
    }
    int KB = 0;
    bool pinned = false;
    for (int a = 4; a < argc; ++a) {
        if (!strcmp(argv[a], "--kb") && a + 1 < argc) KB = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--pinned")) pinned = true;
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
    }

    Tensor in, f;
    if (!read_tensor(argv[1], in) || !read_tensor(argv[2], f)) return 1;
    int M, N, C, K, W;
    if (!get_problem_dims(in, f, M, N, C, K, W)) return 1;
    if (W != 1 && W != 3 && W != 5 && W != 7) { fprintf(stderr, "conv_opt supports W = 1, 3, 5, 7\n"); return 1; }
    if (KB == 0) { KB = 1; while (KB < K && KB < 8) KB *= 2; }
    if (KB != 1 && KB != 2 && KB != 4 && KB != 8) { fprintf(stderr, "--kb must be 1, 2, 4 or 8\n"); return 1; }
    if ((K + KB - 1) / KB > 65535) { fprintf(stderr, "K too large for grid.z\n"); return 1; }

    Tensor out;
    out.dims = {K, M, N};
    out.data.resize(out.count());

    const size_t in_bytes = in.count() * sizeof(float);
    const size_t f_bytes = f.count() * sizeof(float);
    const size_t out_bytes = out.count() * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaFree(0));   // start the GPU context here, so its one-time setup is not counted as malloc time

    if (pinned) {   // page-lock the existing host buffers -> faster DMA copies
        CUDA_CHECK(cudaHostRegister(in.data.data(), in_bytes, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(out.data.data(), out_bytes, cudaHostRegisterDefault));
    }

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

    // warm-up
    launch(W, KB, d_in, d_f, d_out, M, N, C, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed runs
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < REPS; ++r) launch(W, KB, d_in, d_f, d_out, M, N, C, K);
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
    if (pinned) {
        CUDA_CHECK(cudaHostUnregister(in.data.data()));
        CUDA_CHECK(cudaHostUnregister(out.data.data()));
    }

    if (!write_tensor(argv[3], out)) return 1;

    const double gflops = 2.0 * K * M * N * C * W * W / (kern_ms * 1e6);
    printf("M=%d N=%d C=%d K=%d W=%d impl=gpu_opt block=%dx%d tile=%dx%d kb=%d pinned=%d gpu=\"%s\"\n",
           M, N, C, K, W, BW, BH, BW, BH * BPY, KB, pinned, prop.name);
    printf("copy_ms: h2d %.3f  d2h %.3f   kernel GFLOP/s: %.1f\n", h2d_ms, d2h_ms, gflops);
    printf("total_ms: %.3f  (malloc %.3f + h2d %.3f + kernel %.3f + d2h %.3f)\n",
           malloc_ms + h2d_ms + kern_ms + d2h_ms, malloc_ms, h2d_ms, kern_ms, d2h_ms);
    printf("time_ms: %.3f\n", kern_ms);   // kernel only - test scripts parse this line
    return 0;
}
