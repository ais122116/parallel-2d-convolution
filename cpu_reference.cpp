#include "conv_io.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
void conv2d_cpu(const float* in, const float* f, float* out, int M, int N, int C, int K, int W)
{
    const int R = W / 2;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int c = 0; c < C; ++c) {
                    const float* inc = in + (size_t)c * M * N;
                    const float* fkc = f + ((size_t)k * C + c) * W * W;
                    for (int u = 0; u < W; ++u) {
                        int ii = i + u - R;
                        if (ii < 0 || ii >= M) continue;          // zero padding (rows)
                        for (int v = 0; v < W; ++v) {
                            int jj = j + v - R;
                            if (jj < 0 || jj >= N) continue;      // zero padding (cols)
                            acc += (double)inc[(size_t)ii * N + jj] * fkc[u * W + v];
                        }
                    }
                }
                out[((size_t)k * M + i) * N + j] = (float)acc;
            }
}


// acc[j] += sum_{u,v} w[u][v] * rows[u][j + v - R]   for j in [0, N)
template <int W>
static inline void row_taps(float* __restrict acc, const float* const* rows,
                            const float* __restrict w, int N)
{
    constexpr int R = W / 2;
    const float* rw[W];
    for (int u = 0; u < W; ++u) rw[u] = rows[u];
    float wl[W * W];
    for (int t = 0; t < W * W; ++t) wl[t] = w[t];

    auto edge = [&](int j) {          
        float s = 0.f;
        for (int u = 0; u < W; ++u)
            for (int v = 0; v < W; ++v) {
                int jj = j + v - R;
                if (jj >= 0 && jj < N) s += wl[u * W + v] * rw[u][jj];
            }
        acc[j] += s;
    };
    const int jlo = std::min(R, N), jhi = std::max(N - R, jlo);
    for (int j = 0; j < jlo; ++j) edge(j);
#pragma omp simd
    for (int j = jlo; j < jhi; ++j) {   
        float s = 0.f;
#pragma GCC unroll 7
        for (int u = 0; u < W; ++u)
#pragma GCC unroll 7
            for (int v = 0; v < W; ++v)
                s += wl[u * W + v] * rw[u][j + v - R];
        acc[j] += s;
    }
    for (int j = jhi; j < N; ++j) edge(j);
}

// any W (fallback for widths other than 1/3/5/7)
static void row_taps_generic(float* acc, const float* const* rows, const float* w, int W, int N)
{
    const int R = W / 2;
    for (int u = 0; u < W; ++u)
        for (int v = 0; v < W; ++v) {
            float wt = w[u * W + v];
            if (wt == 0.f) continue;
            int off = v - R, j0 = std::max(0, -off), j1 = std::min(N, N - off);
            const float* src = rows[u] + off;
#pragma omp simd
            for (int j = j0; j < j1; ++j) acc[j] += wt * src[j];
        }
}

void conv2d_cpu_fast(const float* in, const float* f, float* out, int M, int N, int C, int K, int W)
{
    const int R = W / 2;

    // which (k, c) filter slices are all zero (e.g. per-channel filters)
    std::vector<char> nz((size_t)K * C);
    for (size_t s = 0; s < (size_t)K * C; ++s) {
        const float* p = f + s * W * W;
        nz[s] = std::any_of(p, p + W * W, [](float x) { return x != 0.f; });
    }
    const std::vector<float> zeros(N, 0.f);

#pragma omp parallel
    {
        std::vector<const float*> rows(W);
#pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float* acc = out + ((size_t)k * M + i) * N;
                std::fill(acc, acc + N, 0.f);
                for (int c = 0; c < C; ++c) {
                    if (!nz[(size_t)k * C + c]) continue;
                    const float* inc = in + (size_t)c * M * N;
                    const float* w = f + ((size_t)k * C + c) * W * W;
                    for (int u = 0; u < W; ++u) {
                        int ii = i + u - R;
                        rows[u] = (ii >= 0 && ii < M) ? inc + (size_t)ii * N : zeros.data();
                    }
                    switch (W) {
                    case 1: row_taps<1>(acc, rows.data(), w, N); break;
                    case 3: row_taps<3>(acc, rows.data(), w, N); break;
                    case 5: row_taps<5>(acc, rows.data(), w, N); break;
                    case 7: row_taps<7>(acc, rows.data(), w, N); break;
                    default: row_taps_generic(acc, rows.data(), w, W, N);
                    }
                }
            }
    }
}

typedef void (*ConvFn)(const float*, const float*, float*, int, int, int, int, int);


// Self tests

static int handchecked(ConvFn conv, const char* name)
{
    int fails = 0;
    {
        int M = 3, N = 4, C = 1, K = 1, W = 1;
        std::vector<float> in(M * N), f(1, 1.0f), out(M * N);
        for (int x = 0; x < M * N; ++x) in[x] = 0.1f * x;
        conv(in.data(), f.data(), out.data(), M, N, C, K, W);
        for (int x = 0; x < M * N; ++x)
            if (out[x] != in[x]) { ++fails; break; }
    }
    {
        std::vector<float> in(9, 1.0f), f(9, 1.0f), out(9);
        conv(in.data(), f.data(), out.data(), 3, 3, 1, 1, 3);
        float exp[9] = {4, 6, 4, 6, 9, 6, 4, 6, 4};
        for (int x = 0; x < 9; ++x)
            if (out[x] != exp[x]) { ++fails; break; }
    }
    {
        int M = 5, N = 5, C = 2, K = 2, W = 3;
        std::vector<float> in(C * M * N), f(K * C * W * W), out(K * M * N);
        for (int c = 0; c < C; ++c)
            for (int x = 0; x < M * N; ++x) in[c * M * N + x] = (float)(c + 1);
        for (int k = 0; k < K; ++k)
            for (int x = 0; x < C * W * W; ++x) f[k * C * W * W + x] = (float)(k + 1);
        conv(in.data(), f.data(), out.data(), M, N, C, K, W);
        for (int k = 0; k < K; ++k) {
            float centre = out[((size_t)k * M + 2) * N + 2], corner = out[((size_t)k * M + 0) * N + 0];
            if (centre != 27.0f * (k + 1) || corner != 12.0f * (k + 1)) { ++fails; break; }
        }
    }
    printf("%-6s hand-checked cases  %s\n", name, fails ? "FAIL" : "ok");
    return fails;
}

static int selftest()
{
    int fails = handchecked(conv2d_cpu, "naive") + handchecked(conv2d_cpu_fast, "fast");

    // fast vs reference on random problems, including odd and tiny sizes
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(0.f, 1.f), Wd(-1.f, 1.f);
    const int Ms[] = {1, 2, 7, 16, 33, 64}, Ns[] = {1, 3, 8, 31, 64, 100}, Ws[] = {1, 3, 5, 7, 9};
    int bad = 0, cases = 0;
    double worst = 0;
    for (int M : Ms)
        for (int N : Ns)
            for (int W : Ws) {
                int C = 1 + rng() % 5, K = 1 + rng() % 4;
                std::vector<float> in((size_t)C * M * N), f((size_t)K * C * W * W);
                std::vector<float> a((size_t)K * M * N), b(a.size());
                for (auto& x : in) x = U(rng);
                for (auto& x : f) x = Wd(rng);
                if (rng() % 3 == 0)  // some all-zero slices
                    std::fill(f.begin(), f.begin() + W * W, 0.f);
                conv2d_cpu(in.data(), f.data(), a.data(), M, N, C, K, W);
                conv2d_cpu_fast(in.data(), f.data(), b.data(), M, N, C, K, W);
                for (size_t x = 0; x < a.size(); ++x) {
                    double err = std::fabs((double)a[x] - b[x]) / std::max(1.0, std::fabs((double)a[x]));
                    worst = std::max(worst, err);
                    if (err > 1e-4) { ++bad; break; }
                }
                ++cases;
            }
    printf("fast vs naive, %d random cases: %s (worst rel. error %.2e)\n",
           cases, bad ? "FAIL" : "ok", worst);
    fails += bad;
    printf(fails ? "SELFTEST FAILED\n" : "SELFTEST PASSED\n");
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s input.bin filters.bin output.bin [fast|naive] | --selftest\n", argv[0]);
        return 1;
    }
    bool naive = argc == 5 && strcmp(argv[4], "naive") == 0;
    bool both = argc == 5 && strcmp(argv[4], "both") == 0;   // time naive AND fast in one run

    Tensor in, f;
    if (!read_tensor(argv[1], in) || !read_tensor(argv[2], f)) return 1;
    int M, N, C, K, W;
    if (!get_problem_dims(in, f, M, N, C, K, W)) return 1;

    Tensor out;
    out.dims = {K, M, N};
    out.data.resize(out.count());

    double naive_ms = 0.0;
    if (both) {   // plain serial version (conv2d_cpu), into its own output buffer
        std::vector<float> out_naive(out.count());
        auto n0 = std::chrono::high_resolution_clock::now();
        conv2d_cpu(in.data.data(), f.data.data(), out_naive.data(), M, N, C, K, W);
        auto n1 = std::chrono::high_resolution_clock::now();
        naive_ms = std::chrono::duration<double, std::milli>(n1 - n0).count();
    }

    ConvFn conv = naive ? conv2d_cpu : conv2d_cpu_fast;
    auto t0 = std::chrono::high_resolution_clock::now();
    conv(in.data.data(), f.data.data(), out.data.data(), M, N, C, K, W);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!write_tensor(argv[3], out)) return 1;
    int threads = 1;
#ifdef _OPENMP
    if (!naive) threads = omp_get_max_threads();
#endif
    printf("M=%d N=%d C=%d K=%d W=%d impl=%s threads=%d\n", M, N, C, K, W, naive ? "naive" : "fast", threads);
    if (both)
        printf("naive_ms: %.3f  fast_ms: %.3f  (fast is %.1fx faster)\n", naive_ms, ms, naive_ms / ms);
    printf("time_ms: %.3f\n", ms);   // harness parses this line
    return 0;
}
