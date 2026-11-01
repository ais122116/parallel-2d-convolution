#include "conv_io.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_pnm(const char* path, int& C, int& M, int& N, std::vector<unsigned char>& px)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1 || (strcmp(magic, "P6") && strcmp(magic, "P5"))) {
        fprintf(stderr, "%s: only binary PPM (P6) / PGM (P5) supported\n", path);
        fclose(fp); return false;
    }
    C = magic[1] == '6' ? 3 : 1;
    int vals[3], n = 0;
    while (n < 3) {                      // width, height, maxval (skip comments)
        int ch = fgetc(fp);
        if (ch == '#') { while (ch != '\n' && ch != EOF) ch = fgetc(fp); continue; }
        if (ch == EOF) { fclose(fp); return false; }
        if (ch >= '0' && ch <= '9') { ungetc(ch, fp); if (fscanf(fp, "%d", &vals[n++]) != 1) { fclose(fp); return false; } }
    }
    fgetc(fp);                            // single whitespace before pixel data
    N = vals[0]; M = vals[1];
    if (vals[2] > 255) { fprintf(stderr, "16-bit images not supported\n"); fclose(fp); return false; }
    px.resize((size_t)M * N * C);
    bool ok = fread(px.data(), 1, px.size(), fp) == px.size();
    fclose(fp);
    if (!ok) fprintf(stderr, "%s: truncated\n", path);
    return ok;
}

static bool write_pnm(const char* path, int C, int M, int N, const std::vector<unsigned char>& px)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) return false;
    fprintf(fp, "P%c\n%d %d\n255\n", C == 3 ? '6' : '5', N, M);
    fwrite(px.data(), 1, px.size(), fp);
    fclose(fp);
    return true;
}

static unsigned char to_byte(float x) { return (unsigned char)std::lround(std::min(1.f, std::max(0.f, x)) * 255.f); }

// --------------------------------------------------------------------------
static int cmd_synth(int Wd, int Ht, const char* out)
{
    // gradient background + discs + rectangles + stripes + mild noise:
    // plenty of edges, flat areas and fine detail for the filters to act on
    std::vector<unsigned char> px((size_t)Wd * Ht * 3);
    unsigned rng = 12345;
    for (int y = 0; y < Ht; ++y)
        for (int x = 0; x < Wd; ++x) {
            float fx = (float)x / Wd, fy = (float)y / Ht;
            float r = 0.2f + 0.6f * fx, g = 0.2f + 0.6f * fy, b = 0.5f + 0.3f * std::sin(6.28f * (fx + fy));
            for (int d = 0; d < 6; ++d) {
                float cx = 0.15f + 0.14f * d, cy = 0.3f + 0.08f * (d % 3), rad = 0.05f + 0.01f * d;
                float dx = (fx - cx) * Wd / Ht, dy = fy - cy;
                if (dx * dx + dy * dy < rad * rad) { r = 0.9f - 0.1f * d; g = 0.1f * d; b = 0.3f; }
            }
            if (fy > 0.6f && fy < 0.85f && std::fmod(fx * 20.f, 2.f) < 1.f) { r *= 0.3f; g *= 0.3f; b *= 0.3f; }
            if (fy > 0.88f && ((x / 4 + y / 4) & 1)) { r = g = b = 0.95f; }   // fine checkerboard
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;   // xorshift32
            float noise = (rng & 255) / 255.f * 0.04f - 0.02f;
            size_t p = ((size_t)y * Wd + x) * 3;
            px[p] = to_byte(r + noise); px[p + 1] = to_byte(g + noise); px[p + 2] = to_byte(b + noise);
        }
    if (!write_pnm(out, 3, Ht, Wd, px)) return 1;
    printf("wrote %s (%dx%d RGB)\n", out, Wd, Ht);
    return 0;
}

static int cmd_img2tensor(const char* img, const char* out)
{
    int C, M, N;
    std::vector<unsigned char> px;
    if (!read_pnm(img, C, M, N, px)) return 1;
    Tensor t;
    t.dims = {C, M, N};
    t.data.resize(t.count());
    for (int c = 0; c < C; ++c)                  // interleaved HWC bytes -> planar CHW floats
        for (size_t p = 0; p < (size_t)M * N; ++p)
            t.data[(size_t)c * M * N + p] = px[p * C + c] / 255.f;
    if (!write_tensor(out, t)) return 1;
    printf("wrote %s  dims {C=%d, M=%d, N=%d}\n", out, C, M, N);
    return 0;
}

// --------------------------------------------------------------------------
struct Filt { const char* name; char mode; float w[9]; float scale; };

// classic 3x3 image-processing kernels
static const Filt EDGES8[8] = {
    {"identity",  'c', {0, 0, 0, 0, 1, 0, 0, 0, 0}, 1.f},
    {"box_blur",  'c', {1, 1, 1, 1, 1, 1, 1, 1, 1}, 1.f / 9},
    {"gaussian",  'c', {1, 2, 1, 2, 4, 2, 1, 2, 1}, 1.f / 16},
    {"sharpen",   'c', {0, -1, 0, -1, 5, -1, 0, -1, 0}, 1.f},
    {"sobel_x",   'a', {-1, 0, 1, -2, 0, 2, -1, 0, 1}, 1.f},
    {"sobel_y",   'a', {-1, -2, -1, 0, 0, 0, 1, 2, 1}, 1.f},
    {"laplacian", 'a', {0, 1, 0, 1, -4, 1, 0, 1, 0}, 1.f},
    {"emboss",    'c', {-2, -1, 0, -1, 1, 1, 0, 1, 2}, 1.f},
};

static int cmd_filters(const std::string& bank, int C, const char* out)
{
    Tensor t;
    if (bank == "edges8") {
        // each filter acts on the luminance: slice c = kernel * (ITU-R BT.601 weight of channel c)
        const float lum3[3] = {0.299f, 0.587f, 0.114f};
        int K = 8, W = 3;
        t.dims = {K, C, W, W};
        t.data.assign(t.count(), 0.f);
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < C; ++c) {
                float cw = C == 3 ? lum3[c] : 1.f / C;
                for (int x = 0; x < 9; ++x)
                    t.data[((size_t)k * C + c) * 9 + x] = EDGES8[k].w[x] * EDGES8[k].scale * cw;
            }
        if (!write_tensor(out, t)) return 1;
        printf("wrote %s  dims {K=8, C=%d, W=3, W=3}\nfilters:", out, C);
        std::string modes;
        for (auto& e : EDGES8) { printf(" %s", e.name); modes += e.mode; }
        printf("\nview with: ./imgtool tensor2img <output.bin> <prefix> %s\n", modes.c_str());
        return 0;
    }
    // per-channel colour filters: output channel k reads only input channel k
    std::vector<float> ker;
    int W;
    if (bank == "sharpen") { W = 3; ker = {0, -1, 0, -1, 5, -1, 0, -1, 0}; }
    else if (bank == "gauss3" || bank == "gauss5" || bank == "gauss7") {
        W = bank[5] - '0';
        std::vector<float> b(1, 1.f);            // binomial row: 1 2 1 / 1 4 6 4 1 / 1 6 15 20 15 6 1
        for (int n = 1; n < W; ++n) {
            std::vector<float> nb(n + 1, 1.f);
            for (int x = 1; x < n; ++x) nb[x] = b[x - 1] + b[x];
            b = nb;
        }
        float s = 0;
        for (float x : b) s += x;
        for (int u = 0; u < W; ++u)
            for (int v = 0; v < W; ++v) ker.push_back(b[u] * b[v] / (s * s));
    } else {
        fprintf(stderr, "unknown bank %s (edges8, gauss3, gauss5, gauss7, sharpen)\n", bank.c_str());
        return 1;
    }
    int K = C;
    t.dims = {K, C, W, W};
    t.data.assign(t.count(), 0.f);
    for (int k = 0; k < K; ++k)
        for (int x = 0; x < W * W; ++x) t.data[((size_t)k * C + k) * W * W + x] = ker[x];
    if (!write_tensor(out, t)) return 1;
    printf("wrote %s  dims {K=%d, C=%d, W=%d, W=%d}  (%s, per channel)\n", out, K, C, W, W, bank.c_str());
    printf("view with: ./imgtool %s\n", C == 3 ? "tensor2rgb <output.bin> <out.ppm>" : "tensor2img <output.bin> <prefix>");
    return 0;
}

// --------------------------------------------------------------------------
static int cmd_tensor2img(const char* in, const std::string& prefix, const std::string& modes)
{
    Tensor t;
    if (!read_tensor(in, t)) return 1;
    if (t.dims.size() != 3) { fprintf(stderr, "expected dims {K, M, N}\n"); return 1; }
    int K = t.dims[0], M = t.dims[1], N = t.dims[2];
    std::vector<unsigned char> px((size_t)M * N);
    for (int k = 0; k < K; ++k) {
        bool absmode = k < (int)modes.size() && modes[k] == 'a';
        const float* p = t.data.data() + (size_t)k * M * N;
        for (size_t x = 0; x < px.size(); ++x) px[x] = to_byte(absmode ? 4.f * std::fabs(p[x]) : p[x]);
        std::string name = prefix + "_" + std::to_string(k) + ".pgm";
        if (!write_pnm(name.c_str(), 1, M, N, px)) return 1;
        printf("wrote %s\n", name.c_str());
    }
    return 0;
}

static int cmd_tensor2rgb(const char* in, const char* out)
{
    Tensor t;
    if (!read_tensor(in, t)) return 1;
    if (t.dims.size() != 3 || t.dims[0] != 3) { fprintf(stderr, "expected dims {3, M, N}\n"); return 1; }
    int M = t.dims[1], N = t.dims[2];
    std::vector<unsigned char> px((size_t)M * N * 3);
    for (int c = 0; c < 3; ++c)
        for (size_t p = 0; p < (size_t)M * N; ++p) px[p * 3 + c] = to_byte(t.data[(size_t)c * M * N + p]);
    if (!write_pnm(out, 3, M, N, px)) return 1;
    printf("wrote %s\n", out);
    return 0;
}

static int cmd_compare(const char* a, const char* b)
{
    Tensor x, y;
    if (!read_tensor(a, x) || !read_tensor(b, y)) return 1;
    if (x.dims != y.dims) { printf("DIFFERENT SHAPES\n"); return 1; }
    double maxabs = 0, maxrel = 0;
    for (size_t i = 0; i < x.data.size(); ++i) {
        double d = std::fabs((double)x.data[i] - y.data[i]);
        maxabs = std::max(maxabs, d);
        maxrel = std::max(maxrel, d / std::max(1.0, std::fabs((double)x.data[i])));
    }
    bool ok = maxrel <= 1e-4;
    printf("max_abs_err %.3e  max_rel_err %.3e  %s\n", maxabs, maxrel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "synth" && argc == 5) return cmd_synth(atoi(argv[2]), atoi(argv[3]), argv[4]);
    if (cmd == "img2tensor" && argc == 4) return cmd_img2tensor(argv[2], argv[3]);
    if (cmd == "filters" && argc == 5) return cmd_filters(argv[2], atoi(argv[3]), argv[4]);
    if (cmd == "tensor2img" && (argc == 4 || argc == 5)) return cmd_tensor2img(argv[2], argv[3], argc == 5 ? argv[4] : "");
    if (cmd == "tensor2rgb" && argc == 4) return cmd_tensor2rgb(argv[2], argv[3]);
    if (cmd == "compare" && argc == 4) return cmd_compare(argv[2], argv[3]);
    fprintf(stderr,
            "usage:\n  imgtool synth <width> <height> <out.ppm>\n  imgtool img2tensor <in.ppm|in.pgm> <input.bin>\n"
            "  imgtool filters <edges8|gauss3|gauss5|gauss7|sharpen> <C> <filters.bin>\n"
            "  imgtool tensor2img <output.bin> <prefix> [modes]\n  imgtool tensor2rgb <output.bin> <out.ppm>\n"
            "  imgtool compare <a.bin> <b.bin>\n");
    return 1;
}
