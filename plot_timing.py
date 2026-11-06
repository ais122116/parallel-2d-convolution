#!/usr/bin/env python3
"""Plot the 2048x2048, C=16, K=32, W=7 timing case (output of `make timing`).

Usage:  python3 plot_timing.py            # reads timing.txt if present
Writes: charts/timing_2048.png
"""
import os, re
import matplotlib
matplotlib.use("Agg")                     # no display needed (works over ssh)
import matplotlib.pyplot as plt

# values from the last `make timing` run (used if timing.txt is missing)
times = {
    "CPU naive\n(1 thread)":   77366.095,
    "CPU fast\n(1 thread)":     3820.646,
    "CPU fast\n(22 threads)":    305.430,
    "GPU naive":                 441.679,
    "GPU opt":                    61.193,
}

# if timing.txt exists, take the numbers from it instead
if os.path.exists("timing.txt"):
    t = open("timing.txt").read()
    num = lambda pat: float(re.findall(pat, t)[-1])
    try:
        times["CPU naive\n(1 thread)"]   = num(r"naive_ms:\s*([\d.]+)")
        times["CPU fast\n(22 threads)"]  = num(r"fast_ms:\s*([\d.]+)")
        tm = re.findall(r"time_ms:\s*([\d.]+)", t)   # order: fast22, fast1, gpu_naive, gpu_opt
        times["CPU fast\n(1 thread)"] = float(tm[-3])
        times["GPU naive"]            = float(tm[-2])
        times["GPU opt"]              = float(tm[-1])
    except (IndexError, ValueError):
        print("could not parse timing.txt, using built-in values")

names  = list(times)
ms     = list(times.values())
serial = times["CPU naive\n(1 thread)"]
speed  = [serial / v for v in ms]
colors = ["#9e9e9e", "#bdbdbd", "#757575", "#64b5f6", "#1565c0"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("2048 × 2048, C = 16, K = 32, W = 7  ·  RTX A2000 12 GB, 22-thread CPU")

b = a1.bar(names, ms, color=colors)
a1.set_yscale("log")
a1.set_ylabel("time (ms, log scale)")
a1.set_title("Compute time")
for r, v in zip(b, ms):
    a1.text(r.get_x() + r.get_width() / 2, v * 1.15,
            f"{v/1000:.1f} s" if v >= 1000 else f"{v:.0f} ms", ha="center")

b = a2.bar(names, speed, color=colors)
a2.set_yscale("log")
a2.set_ylabel("speedup vs serial CPU (log scale)")
a2.set_title("Speedup")
for r, v in zip(b, speed):
    a2.text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:.0f}×", ha="center")

for a in (a1, a2):
    a.set_ylim(top=max(a.get_ylim()[1], 1) * 3)
    a.grid(axis="y", alpha=0.3)

plt.tight_layout()
os.makedirs("charts", exist_ok=True)
plt.savefig("charts/timing_2048.png", dpi=150)
print("wrote charts/timing_2048.png")
