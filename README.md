# Parallel 2D Convolution with CUDA

Multi-channel 2D convolution with zero padding, implemented on the CPU and on the GPU.

| Version | Source | Description |
|---|---|---|
| CPU reference | `cpu_reference.cpp` | `naive`: plain serial loops (used to verify everything); `fast`: OpenMP + SIMD |
| GPU V1 – naive | `conv_naive.cu` | one thread per output pixel, all reads from global memory |
| GPU V2 – optimized | `conv_opt1.cu` | shared-memory tiling with halo, register blocking, filter reuse, double buffering |

## Problem definition

- **Input:** image of `M × N` pixels with `C` channels, values in [0, 1]
- **Filters:** `K` filters, each of size `C × W × W` (W odd: 1, 3, 5, 7), values in [-1, 1]
- **Output:** `M × N` pixels with `K` channels

```
out[k][i][j] = Σ_c Σ_u Σ_v  in[c][i+u-R][j+v-R] · f[k][c][u][v],     R = W/2
```

Pixels outside the image are treated as 0 (zero padding), so the output is the same size as the input.

## Requirements

- `g++` with OpenMP support (C++17)
- CUDA toolkit (`nvcc`) and an NVIDIA GPU
- `python3` with `numpy` (only for `make check` and `make bench`)
- Nsight Systems / Nsight Compute (only for profiling)

## Files

| File | Purpose |
|---|---|
| `Makefile` | builds everything and runs all tests |
| `conv_io.h` | binary file format: read/write tensors, check dimensions |
| `cpu_reference.cpp` | CPU implementation (naive + fast) with `--selftest` |
| `conv_naive.cu` | GPU V1 (naive) |
| `conv_opt1.cu` | GPU V2 (optimized) |
| `gen_data.cpp` | generates a random input image and filter bank |
| `compare.cpp` | compares an output against the CPU reference (PASS / FAIL) |
| `imgtool.cpp` | converts images to and from tensors; builds classic filter banks |
| `bin2txt.cpp` | prints a `.bin` file as readable text (optional, for debugging) |
| `run_tests.sh` | correctness sweep over sizes, W, C, K → `results.csv` |
| `run_tests.py` | wider correctness check and benchmark sweep → `bench.csv` |
| `run_image_tests.sh` | applies real filters to a 4096×3072 image → `results/` |

## Quick start

```
make clean
make
```

Plain `make` runs the whole pipeline: it builds all programs, generates the test data, checks both GPU versions, runs the full correctness sweep, produces the filtered images, and times all versions on one large case (`timing.txt`). It takes a few minutes, mostly for the serial CPU timing.

## Make targets

| Command | What it does |
|---|---|
| `make` | **everything**: `build` + `data` + `verify` + `test` + `images` + `timing` |
| `make build` | only build the programs (`conv_opt` is built from `conv_opt1.cu`) |
| `make data` | CPU self-test, then generate 15 test cases in `testdata/` (`input.bin`, `filters.bin`, `ref.bin` each) |
| `make verify` | run `conv_naive` and `conv_opt` on every case in `testdata/` and compare with `ref.bin` |
| `make test` | run `run_tests.sh`: 140 configurations × 2 GPU versions → `results.csv` |
| `make images` | run `run_image_tests.sh`: filtered images and timings → `results/` |
| `make check` | `python3 run_tests.py check`: additional random correctness cases |
| `make bench` | `python3 run_tests.py bench`: timings and speedups of all versions → `bench.csv` |
| `make timing` | one 2048×2048 case (C = 16, K = 32, W = 7): serial CPU, CPU fast (1 and all threads), `conv_naive`, `conv_opt` → `timing.txt` |
| `make everything` | same as plain `make` |
| `make clean` | remove programs, CSV files, `timing.txt` and `results/` |
| `make distclean` | `clean` + remove `testdata/` and `timingdata/` |
| `make ARCH=sm_86` | build for a specific GPU architecture (default: `native`) |

To change the test cases, edit the `CASES` list in the Makefile (one `"M N C K W"` entry per case). To time a different size: `make timing TIME_CASE="1024 1024 16 32 7"`.

## Running a single case by hand

```
./gen_data 2048 2048 16 32 7 1 input.bin filters.bin
./cpu_reference input.bin filters.bin ref.bin
./conv_opt input.bin filters.bin out.bin
./compare ref.bin out.bin
```

`gen_data` arguments: `M N C K W seed input.bin filters.bin`. The same seed always produces the same data.

| Program | Usage |
|---|---|
| `cpu_reference` | `./cpu_reference input.bin filters.bin output.bin [fast\|naive\|both]`, or `--selftest` |
| `conv_naive` | `./conv_naive input.bin filters.bin output.bin [block 8\|16\|32]` |
| `conv_opt` | `./conv_opt input.bin filters.bin output.bin [--kb 1\|2\|4\|8] [--pinned]` |
| `compare` | `./compare ref.bin test.bin [atol=1e-3] [rtol=1e-3]` |
| `bin2txt` | `./bin2txt file.bin [out.txt]` |

Every program prints a `time_ms:` line with the compute time only (the kernel, for the GPU versions: average of 5 runs after one warm-up run). File reading and writing are never timed.

The GPU programs also print

```
total_ms: 113.308  (malloc 0.183 + h2d 17.489 + kernel 61.153 + d2h 34.483)
```

which is the full GPU cost of one convolution: `cudaMalloc`, copy to the GPU, one kernel run, copy back. The CUDA context is created before timing starts, so its one-time setup is not counted.

`cpu_reference ... both` runs the serial version and the fast (OpenMP) version on the same input and prints `naive_ms` and `fast_ms`. The serial version is slow on large inputs (about 77 s for the 2048×2048 timing case).

## File format

All inputs and outputs are binary files (little-endian):

```
int32 ndim | int32 dims[ndim] | float32 data[...]     (row-major)
```

| File | dims | Element index |
|---|---|---|
| input | `{C, M, N}` | `in[(c*M + i)*N + j]` |
| filters | `{K, C, W, W}` | `f[((k*C + c)*W + u)*W + v]` |
| output | `{K, M, N}` | `out[(k*M + i)*N + j]` |

We use a binary format because it is fast to read and write and much smaller than text. Use `bin2txt` to inspect small files.

## Correctness

Every GPU result is compared with the CPU reference. A value passes if

```
|got − ref| ≤ 1e-3 + 1e-3 · |ref|
```

GPU results are not bit-identical to the CPU because floating-point additions happen in a different order. The largest error observed is 6.29e-05 (0.0000629), well below the 1e-03 limit; most errors are around 1e-07. All 310 checks pass (30 in `make verify`, 280 in `make test`).

Test coverage includes:
- square and rectangular images;
- power-of-two, non-power-of-two even and odd sizes, from 1×1 up to 2048×2048 and 4096×3072;
- W = 1, 3, 5, 7;
- every C and K value from {3, 4, 8, 16, 32, 64, 127}.

## Implementation notes

### GPU V1: `conv_naive.cu`
- Grid `(N/B, M/B, K)` with B×B threads (default 16×16). Each thread computes one output pixel of one output channel.
- Every input value and weight is read from global memory, and boundary checks implement the zero padding.
- Works for any odd W.

### GPU V2: `conv_opt1.cu`
- **Block shape:** 32×8 threads, and each thread computes 4 output rows (PY = 4), giving a 32×32 output tile per block.
- **Shared-memory tiling with halo:** for each input channel, the block loads its tile plus the `R`-pixel halo into shared memory once. Out-of-image pixels are stored as 0, so the compute loop needs no bounds checks.
- **Filter reuse (KB):** each block computes up to 8 output channels from the same input tile, so the tile is loaded once instead of KB times.
- **Register blocking:** each thread keeps PY × KB accumulators in registers. One value read from shared memory feeds up to W·KB multiply-adds, and the weights of the current filter column are kept in registers.
- **Coalesced access:** `threadIdx.x` runs along an image row, so a warp reads or writes 32 consecutive floats.
- **Double buffering:** the next channel's tile is fetched into registers while the current channel is being computed, so only one `__syncthreads()` is needed per channel.
- **Compile-time specialisation:** W and KB are template parameters, so all inner loops are fully unrolled. Supported widths: W = 1, 3, 5, 7 (the widths required by the project).

## Results (NVIDIA RTX A2000 12 GB, 22-core CPU)

From `make timing`: M = N = 2048, C = 16, K = 32, W = 7 (210 billion floating-point operations).

| Version | Kernel time (compute only) | GFLOP/s | Total (kernel + malloc + copies) | Speedup vs serial CPU |
|---|---|---|---|---|
| CPU reference (serial) | 76,531 ms | 2.7 | — | 1× |
| GPU V1 `conv_naive` | 442 ms | 476 | 495 ms | 173× |
| GPU V2 `conv_opt` | 61 ms | 3441 | 113 ms | 1251× |

`conv_opt` is 1251× faster than the serial CPU (675× including malloc and copies) and 7.2× faster than `conv_naive`. About 52 ms of its total is copying data over PCIe.

Extra comparison with the optimized CPU code on the same case: CPU fast takes 3,722 ms on 1 thread and 316 ms on 22 threads, so the `conv_opt` kernel is 61× and 5.2× faster respectively.

Image filters on a 4096×3072 RGB image (`make images`):

| Filter | CPU serial | CPU fast (best) | GPU naive | GPU optimized |
|---|---|---|---|---|
| edges8 (8 filters, W = 3) | 2165 ms | 23.5 ms | 20.0 ms | 3.0 ms |
| gauss7 (W = 7) | 3814 ms | 11.5 ms | 24.5 ms | 3.5 ms |

Run `make bench` for the full sweep across image sizes, C, K and W.

## Profiling

Nsight Systems (timeline: kernels, memory copies, API calls):

```
./gen_data 2048 2048 16 32 7 1 input.bin filters.bin
nsys profile -o prof_naive --stats=true --force-overwrite true ./conv_naive input.bin filters.bin out.bin
nsys profile -o prof_opt   --stats=true --force-overwrite true ./conv_opt   input.bin filters.bin out.bin
```

Open the `.nsys-rep` files in the Nsight Systems GUI (`nsys-ui`) to view the timeline.
