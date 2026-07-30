"""ctypes bindings for the Mojo filter-bank kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYWAVELETS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pywavelets.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mpw_dwt_f64": ([I] * 10, I),
    "mpw_idwt_f64": ([I] * 10, I),
}

_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first "
                "or set MOJOPYWAVELETS_LIB"
            )
        _lib = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    """Return a non-null address for an array kept alive by the caller."""
    address = int(array.ctypes.data)
    if address == 0:
        raise RuntimeError("NumPy returned a null buffer address")
    return address


def check_status(operation: str, status: int) -> None:
    if status != 0:
        raise RuntimeError(
            f"Mojo {operation} kernel rejected its validated buffers "
            f"(status {status})"
        )
