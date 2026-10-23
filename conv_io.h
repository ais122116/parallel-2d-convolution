// conv_io.h - shared tensor file format + layout (usable from .cpp and .cu)
//
// FILE FORMAT (binary, little-endian):
//   int32  ndim
//   int32  dims[ndim]
//   float32 data[prod(dims)]      (row-major, last dim fastest)
//
// LAYOUTS:
//   input   : ndim=3, dims = {C, M, N}       in [(c*M + i)*N + j]
//   filters : ndim=4, dims = {K, C, W, W}    f  [((k*C + c)*W + u)*W + v]
//   output  : ndim=3, dims = {K, M, N}       out[(k*M + i)*N + j]
//
// Channel-major (CHW) is used so that, on the GPU, neighbouring threads
// (neighbouring j) read neighbouring addresses -> coalesced accesses.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

struct Tensor {
    std::vector<int32_t> dims;
    std::vector<float> data;
    size_t count() const {
        size_t n = 1;
        for (int d : dims) n *= (size_t)d;
        return n;
    }
};

inline bool write_tensor(const char* path, const Tensor& t) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s for writing\n", path); return false; }
    int32_t nd = (int32_t)t.dims.size();
    bool ok = fwrite(&nd, sizeof(int32_t), 1, fp) == 1 &&
              fwrite(t.dims.data(), sizeof(int32_t), nd, fp) == (size_t)nd &&
              fwrite(t.data.data(), sizeof(float), t.count(), fp) == t.count();
    fclose(fp);
    if (!ok) fprintf(stderr, "Write error on %s\n", path);
    return ok;
}

inline bool read_tensor(const char* path, Tensor& t) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    int32_t nd = 0;
    if (fread(&nd, sizeof(int32_t), 1, fp) != 1 || nd < 1 || nd > 8) {
        fprintf(stderr, "Bad header in %s\n", path); fclose(fp); return false;
    }
    t.dims.resize(nd);
    if (fread(t.dims.data(), sizeof(int32_t), nd, fp) != (size_t)nd) {
        fprintf(stderr, "Bad dims in %s\n", path); fclose(fp); return false;
    }
    t.data.resize(t.count());
    bool ok = fread(t.data.data(), sizeof(float), t.count(), fp) == t.count();
    fclose(fp);
    if (!ok) fprintf(stderr, "Truncated data in %s\n", path);
    return ok;
}

// Validates input/filter shapes; fills M, N, C, K, W. Returns false on mismatch.
inline bool get_problem_dims(const Tensor& in, const Tensor& f,
                             int& M, int& N, int& C, int& K, int& W) {
    if (in.dims.size() != 3 || f.dims.size() != 4) {
        fprintf(stderr, "Expected input ndim=3 and filters ndim=4\n"); return false;
    }
    C = in.dims[0]; M = in.dims[1]; N = in.dims[2];
    K = f.dims[0];  W = f.dims[2];
    if (f.dims[1] != C) { fprintf(stderr, "Channel mismatch: input C=%d, filter C=%d\n", C, f.dims[1]); return false; }
    if (f.dims[3] != W) { fprintf(stderr, "Filter must be square\n"); return false; }
    if (W % 2 == 0)     { fprintf(stderr, "Filter width must be odd (got %d)\n", W); return false; }
    return true;
}
