# Parallel 2D Convolution — Week 1

## Build & run
```
make            # builds cpu_reference, gen_data, compare
make test       # CPU selftest + full sweep -> results.csv
make gpu        # (later) builds conv_naive.cu, conv_opt.cu
```

## Tools
| Program | Usage |
|---|---|
| `gen_data` | `./gen_data M N C K W seed input.bin filters.bin` |
| `cpu_reference` | `./cpu_reference input.bin filters.bin output.bin` or `--selftest` |
| `compare` | `./compare ref.bin test.bin [atol=1e-3] [rtol=1e-3]` |
| `run_tests.sh` | runs every config, checks GPU versions vs CPU, writes CSV |

GPU programs must use the same CLI as `cpu_reference` and print `time_ms: <value>`.

## File format (binary, little-endian)
```
int32 ndim | int32 dims[ndim] | float32 data[...]   (row-major)
```
- input:   dims `{C, M, N}`  → `in[(c*M + i)*N + j]`, values in [0, 1]
- filters: dims `{K, C, W, W}` → `f[((k*C + c)*W + u)*W + v]`, values in [-1, 1]
- output:  dims `{K, M, N}`  → `out[(k*M + i)*N + j]`

## Definition
`out[k][i][j] = Σ_c Σ_u Σ_v in[c][i+u-R][j+v-R] · f[k][c][u][v]`, with `R = W/2`.
Out-of-bounds input is treated as 0 (zero padding); output has the same M×N size.
W must be odd (1, 3, 5, 7).
