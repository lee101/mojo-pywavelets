import numpy as np
import mojopywavelets as pywt


signal = np.arange(1024, dtype=np.float64)
coeffs = pywt.wavedec(signal, "db4", mode="symmetric", level=4)
restored = pywt.waverec(coeffs, "db4", mode="symmetric")

assert np.allclose(restored, signal)
print([band.shape for band in coeffs])
