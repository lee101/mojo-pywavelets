"""Benchmark Mojo kernels against PyWavelets on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pywt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopywavelets as mpw  # noqa: E402


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cases():
    rng = np.random.default_rng(0)

    x1m = np.ascontiguousarray(rng.normal(size=1_000_000))
    yield "dwt db4 symmetric, 1M", lambda: mpw.dwt(x1m, "db4"), lambda: pywt.dwt(x1m, "db4")

    yield (
        "dwt db20 symmetric, 1M",
        lambda: mpw.dwt(x1m, "db20"),
        lambda: pywt.dwt(x1m, "db20"),
    )

    cA, cD = mpw.dwt(x1m, "db4")
    uA, uD = pywt.dwt(x1m, "db4")
    yield (
        "idwt db4 symmetric, 500k bands",
        lambda: mpw.idwt(cA, cD, "db4"),
        lambda: pywt.idwt(uA, uD, "db4"),
    )

    yield (
        "wavedec db4 level 6, 1M",
        lambda: mpw.wavedec(x1m, "db4", level=6),
        lambda: pywt.wavedec(x1m, "db4", level=6),
    )

    image = np.ascontiguousarray(rng.normal(size=(2048, 2048)))
    yield (
        "dwt2 db4, 2048 x 2048",
        lambda: mpw.dwt2(image, "db4"),
        lambda: pywt.dwt2(image, "db4"),
    )

    image_small = image[:1024, :1024]
    yield (
        "wavedec2 db4 level 4, 1024 x 1024",
        lambda: mpw.wavedec2(image_small, "db4", level=4),
        lambda: pywt.wavedec2(image_small, "db4", level=4),
    )


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| Case | mojo-pywavelets | PyWavelets | PyWavelets / Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_time = best_time(ours)
        upstream_time = best_time(upstream)
        ratio = upstream_time / mojo_time
        result = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
