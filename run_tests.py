#!/usr/bin/env python3
"""
Test harness for the multi-channel 2D convolution project.

  python3 run_tests.py check            # correctness of every version vs the CPU reference
  python3 run_tests.py check --quick    # small subset, a few seconds
  python3 run_tests.py bench            # timings + speedups -> results.csv

Every program is run as
    <exe> input.bin filters.bin output.bin [extra args]
and must print a line "time_ms: <number>" (time of the convolution only).
Versions whose executable does not exist yet are skipped automatically.

File format (little-endian, must match conv_io.h):
    int32 ndims, int32 dims[ndims], float32 data[prod(dims)]   (row-major)
    input   dims = [C, M, N]      values in [0, 1]
    filters dims = [K, C, W, W]   values in [-1, 1]
    output  dims = [K, M, N]
"""
import argparse, csv, os, re, shutil, subprocess, sys, tempfile
import numpy as np

# (name, executable, extra args) -- edit paths here
IMPLS = [
    ("cpu_naive", "./cpu_reference", ["naive"]),
    ("cpu_fast",  "./cpu_reference", ["fast"]),
    ("gpu_naive", "./conv_naive",    []),
    ("gpu_opt",   "./conv_opt",      []),
]

WIDTHS = [1, 3, 5, 7]
# (C, K) pairs covering every value of the set {3, 4, 8, 16, 32, 64, 127}
CK_PAIRS = [(3, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 127),
            (127, 3), (3, 127), (64, 64), (127, 127)]

# square / rectangular, power-of-two, even, odd, tiny
CHECK_SIZES = [(1, 1), (1, 7), (5, 3), (16, 16), (17, 17), (31, 64), (64, 64),
               (100, 100), (127, 255), (128, 128), (256, 192), (333, 517), (512, 512)]
QUICK_SIZES = [(1, 1), (5, 3), (16, 16), (31, 64), (100, 100)]

BENCH_SIZES = [(256, 256), (512, 512), (1000, 1000), (1024, 1024),
               (1920, 1080), (2048, 2048), (4096, 3072)]
BENCH_CK = [(3, 8), (16, 16), (64, 64)]

NAIVE_LIMIT = 2e9    # max multiply-adds for which the serial naive CPU version is run
CHECK_LIMIT = 2e10   # max multiply-adds per correctness case


# ---------------------------------------------------------------- file I/O
def write_tensor(path, arr):
    with open(path, "wb") as f:
        np.array([arr.ndim], dtype="<i4").tofile(f)
        np.array(arr.shape, dtype="<i4").tofile(f)
        np.ascontiguousarray(arr, dtype="<f4").tofile(f)


def read_tensor(path):
    with open(path, "rb") as f:
        ndim = int(np.fromfile(f, dtype="<i4", count=1)[0])
        shape = tuple(np.fromfile(f, dtype="<i4", count=ndim))
        data = np.fromfile(f, dtype="<f4")
    return data.reshape(shape)


# ---------------------------------------------------------------- helpers
def work(M, N, C, K, W):
    return K * M * N * C * W * W          # multiply-adds


def generate(d, M, N, C, K, W, rng):
    write_tensor(os.path.join(d, "input.bin"), rng.random((C, M, N), dtype=np.float32))
    write_tensor(os.path.join(d, "filters.bin"),
                 rng.uniform(-1, 1, (K, C, W, W)).astype(np.float32))


def run(impl, d, timeout):
    name, exe, extra = impl
    out = os.path.join(d, f"out_{name}.bin")
    cmd = [exe, os.path.join(d, "input.bin"), os.path.join(d, "filters.bin"), out] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{name} exited with {p.returncode}: {p.stderr.strip()[:200]}")
    m = re.search(r"time_ms:\s*([-+0-9.eE]+)", p.stdout)
    return out, float(m.group(1)) if m else float("nan")


def rel_err(ref, x):
    """max |ref - x| / max(1, |ref|)  (same metric as the C++ self-test)"""
    if ref.shape != x.shape or not np.all(np.isfinite(x)):
        return float("inf")
    r = ref.astype(np.float64)
    return float(np.max(np.abs(r - x) / np.maximum(1.0, np.abs(r))))


def available():
    impls = [i for i in IMPLS if os.path.isfile(i[1]) and os.access(i[1], os.X_OK)]
    if not any(i[0].startswith("cpu") for i in impls):
        sys.exit("cpu_reference not found -- build it first")
    print("versions found:", ", ".join(i[0] for i in impls))
    return impls


def by_name(impls, name):
    return next((i for i in impls if i[0] == name), None)


# ---------------------------------------------------------------- check
def check(args):
    impls = available()
    rng = np.random.default_rng(args.seed)
    sizes = QUICK_SIZES if args.quick else CHECK_SIZES
    fails = total = 0
    d = tempfile.mkdtemp(prefix="conv_")
    try:
        for s, (M, N) in enumerate(sizes):
            for w, W in enumerate(WIDTHS):
                # pick a (C, K) pair, rotating through the set, within the work limit
                start = (s * len(WIDTHS) + w) % len(CK_PAIRS)
                C, K = next(p for p in CK_PAIRS[start:] + CK_PAIRS[:start]
                            if work(M, N, *p, W) <= CHECK_LIMIT)
                ops = work(M, N, C, K, W)
                generate(d, M, N, C, K, W, rng)

                ref_impl = by_name(impls, "cpu_naive") if ops <= NAIVE_LIMIT else None
                ref_impl = ref_impl or by_name(impls, "cpu_fast")
                ref = read_tensor(run(ref_impl, d, args.timeout)[0])

                line = f"M={M:4d} N={N:4d} C={C:3d} K={K:3d} W={W}  ref={ref_impl[0]:9s}"
                for impl in impls:
                    if impl is ref_impl or impl[0] == "cpu_naive":
                        continue
                    total += 1
                    try:
                        err = rel_err(ref, read_tensor(run(impl, d, args.timeout)[0]))
                        ok = err <= args.tol
                        line += f" | {impl[0]} {'ok  ' if ok else 'FAIL'} {err:.1e}"
                    except Exception as e:
                        ok = False
                        line += f" | {impl[0]} ERROR ({e})"
                    fails += not ok
                print(line, flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{total - fails}/{total} comparisons passed (tolerance {args.tol:g})")
    print("ALL PASSED" if fails == 0 else "SOME FAILED")
    return 1 if fails else 0


# ---------------------------------------------------------------- bench
def bench(args):
    impls = available()
    rng = np.random.default_rng(args.seed)
    rows, fails = [], 0
    d = tempfile.mkdtemp(prefix="conv_")
    try:
        for (M, N) in BENCH_SIZES:
            for (C, K) in BENCH_CK:
                for W in WIDTHS:
                    ops = work(M, N, C, K, W)
                    if ops > args.max_work:
                        print(f"skip M={M} N={N} C={C} K={K} W={W} (work {ops:.1e} > --max-work)")
                        continue
                    generate(d, M, N, C, K, W, rng)
                    times, outs = {}, {}
                    for impl in impls:
                        if impl[0] == "cpu_naive" and ops > NAIVE_LIMIT:
                            continue
                        ts = []
                        for _ in range(args.repeats):
                            out, t = run(impl, d, args.timeout)
                            ts.append(t)
                        times[impl[0]], outs[impl[0]] = min(ts), out

                    ref = read_tensor(outs["cpu_fast"]) if "cpu_fast" in outs else None
                    msg = f"M={M:4d} N={N:4d} C={C:3d} K={K:3d} W={W}"
                    for name, t in times.items():
                        err = rel_err(ref, read_tensor(outs[name])) if ref is not None else float("nan")
                        bad = err > args.tol
                        fails += bad
                        rows.append({
                            "M": M, "N": N, "C": C, "K": K, "W": W, "impl": name,
                            "time_ms": f"{t:.4f}",
                            "gflops": f"{2 * ops / (t * 1e6):.2f}" if t > 0 else "",
                            "speedup_vs_cpu_fast": f"{times['cpu_fast'] / t:.2f}" if "cpu_fast" in times and t > 0 else "",
                            "speedup_vs_cpu_naive": f"{times['cpu_naive'] / t:.2f}" if "cpu_naive" in times and t > 0 else "",
                            "max_rel_err": f"{err:.1e}",
                        })
                        msg += f" | {name} {t:9.3f} ms{' WRONG' if bad else ''}"
                    print(msg, flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if rows:
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nresults written to {args.out}")
    if fails:
        print(f"WARNING: {fails} results disagreed with cpu_fast")
    return 1 if fails else 0


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "bench"])
    ap.add_argument("--quick", action="store_true", help="check: small subset only")
    ap.add_argument("--tol", type=float, default=1e-3, help="max relative error (default 1e-3)")
    ap.add_argument("--repeats", type=int, default=3, help="bench: runs per case, min is kept")
    ap.add_argument("--max-work", type=float, default=2e11, help="bench: skip cases above this many multiply-adds")
    ap.add_argument("--out", default="results.csv", help="bench: CSV output file")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(check(args) if args.mode == "check" else bench(args))
