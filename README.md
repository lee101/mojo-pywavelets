# mojo-pywavelets

`mojo-pywavelets` is a standalone Mojo implementation of the compute-heavy
filter-bank kernels behind PyWavelets' discrete wavelet transforms. Its Python
package, `mojopywavelets`, follows PyWavelets' public names and signatures for
the covered subset, so existing code can commonly use:

```python
import mojopywavelets as pywt
```

The implementation does not import PyWavelets at runtime. The discrete filter
banks exposed by the pinned PyWavelets development dependency are vendored in
the package; PyWavelets itself is used only for parity tests and benchmarks.

## Coverage

Implemented:

- `Wavelet`, `Modes`, `families`, and `wavelist` for discrete wavelets
- `dwt`, `idwt`, `downcoef`, `dwt_coeff_len`, and `dwt_max_level`
- `wavedec` and `waverec`
- `dwt2`, `idwt2`, `wavedec2`, and `waverec2`
- `dwtn`, `idwtn`, `wavedecn`, `waverecn`, and `dwtn_max_level`
- every PyWavelets extension mode: zero, constant, symmetric, periodic, smooth,
  periodization, reflect, antisymmetric, and antireflect
- arbitrary transform axes, real and complex input, PyWavelets-compatible
  output dtypes, non-contiguous NumPy input, and custom filter banks

Not implemented:

- stationary transforms (`swt`, `swt2`, `swtn`)
- continuous transforms, continuous wavelets, and `cwt`
- wavelet packets
- `upcoef`, fully separable transforms, and coefficient packing helpers
- object arrays

This is intentionally a discrete-transform port rather than a complete copy of
every PyWavelets feature.

## Install

Install the pinned Mojo toolchain and Python dependencies, then build the shared
library:

```bash
pixi install
pixi run build
```

Run the parity suite with:

```bash
pixi run test
```

## Usage

```python
import numpy as np
import mojopywavelets as pywt

signal = np.arange(1024, dtype=np.float64)
coeffs = pywt.wavedec(signal, "db4", mode="symmetric", level=4)
restored = pywt.waverec(coeffs, "db4", mode="symmetric")

assert np.allclose(restored, signal)
print([band.shape for band in coeffs])
```

The same code is checked in at `examples/quickstart.py`; run it with:

```bash
pixi run python examples/quickstart.py
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic. Times are the best of five warmed runs. The ratio is
PyWavelets time divided by mojo-pywavelets time, so values above 1 mean Mojo is
faster.

| Case | mojo-pywavelets | PyWavelets | PyWavelets / Mojo | Result |
|---|---:|---:|---:|---|
| dwt db4 symmetric, 1M | 13.88 ms | 5.75 ms | 0.41x | slower |
| dwt db20 symmetric, 1M | 15.27 ms | 32.08 ms | 2.10x | faster |
| idwt db4 symmetric, 500k bands | 3.55 ms | 3.44 ms | 0.97x | slower |
| wavedec db4 level 6, 1M | 17.64 ms | 12.94 ms | 0.73x | slower |
| dwt2 db4, 2048 x 2048 | 150.68 ms | 171.49 ms | 1.14x | faster |
| wavedec2 db4 level 4, 1024 x 1024 | 40.76 ms | 32.64 ms | 0.80x | slower |

The long-filter analysis and large single-level 2D cases benefit from Mojo SIMD
and thresholded parallelism. PyWavelets is faster in the other measured cases.
These results are reported as measured, including the slower cases.

No GPU path is included.

## How it works

NumPy owns every input, filter, and output allocation. The Python layer moves
the selected transform axis to the final dimension and makes one contiguous,
row-major view. It passes raw buffer addresses and validated extents through
`ctypes`; the C-ABI Mojo exports reject inconsistent arguments before
reconstructing typed pointers and process all rows in one call. The contiguous
NumPy owners remain live until each call returns, and no Mojo allocation crosses
the FFI boundary.

Analysis applies the low- and high-pass filters together while downsampling.
Interior filter windows use native-width SIMD loads and reductions, while
boundaries and incomplete vectors use scalar paths. Synthesis packs even and
odd filter phases and reconstructs output pairs with SIMD dot products.
Periodization has its own phase and sizing path to match PyWavelets exactly.
Large independent workloads are partitioned across physical CPU cores;
smaller transforms stay serial to avoid thread overhead. Built-in wavelet
metadata is cached across multilevel calls. Float32 and complex64 outputs
preserve PyWavelets' public dtype behavior, while the current kernel computes
through float64 internally.

The implementation is tested directly against PyWavelets 1.8.0 across every
vendored discrete filter bank, all extension modes, multiple dtypes, arbitrary
axes, multilevel transforms, and 2D/nD reconstruction.
