"""Single-level discrete wavelet transforms backed by Mojo."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, check_status, lib
from ._wavelet import Modes, _as_wavelet

_MODE_CODES = {
    "zero": 0,
    "constant": 1,
    "symmetric": 2,
    "periodic": 3,
    "smooth": 4,
    "reflect": 5,
    "antisymmetric": 6,
    "antireflect": 7,
    "periodization": 8,
    "per": 8,
}


def _mode_code(mode) -> int:
    if isinstance(mode, (int, np.integer)):
        try:
            mode = Modes.modes[int(mode)]
        except (IndexError, ValueError):
            raise ValueError(f"Unknown mode name '{mode}'") from None
    try:
        return _MODE_CODES[str(mode)]
    except KeyError:
        raise ValueError(f"Unknown mode name '{mode}'") from None


def _axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return axis


def _result_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype == np.float16 or dtype == np.float32:
        return np.dtype(np.float32)
    if dtype == np.complex64:
        return np.dtype(np.complex64)
    if dtype == np.complex128:
        return dtype
    return np.dtype(np.float64)


def dwt_coeff_len(data_len, filter_len, mode) -> int:
    data_len = int(data_len)
    filter_len = int(getattr(filter_len, "dec_len", filter_len))
    if data_len < 1:
        raise ValueError("Value of data_len must be greater than zero.")
    if filter_len < 1:
        raise ValueError("Value of filter_len must be greater than zero.")
    if _mode_code(mode) == 8:
        return (data_len + 1) // 2
    return (data_len + filter_len - 1) // 2


def dwt_max_level(data_len, filter_len) -> int:
    data_len = int(data_len)
    filter_len = int(getattr(filter_len, "dec_len", filter_len))
    if data_len < 0:
        raise ValueError("Input data length must be non-negative.")
    if filter_len < 2:
        raise ValueError("invalid wavelet filter length")
    if data_len < filter_len - 1:
        return 0
    return int(math.log2(data_len // (filter_len - 1)))


def _dwt_real(data: np.ndarray, wavelet, mode_code: int, axis: int):
    moved = np.moveaxis(data, axis, -1)
    shape = moved.shape
    n = shape[-1]
    coeff_len = (
        (n + 1) // 2
        if mode_code == 8
        else (n + wavelet.dec_len - 1) // 2
    )
    src = np.ascontiguousarray(moved, dtype=np.float64)
    dec_lo = np.ascontiguousarray(wavelet.dec_lo[::-1], dtype=np.float64)
    dec_hi = np.ascontiguousarray(wavelet.dec_hi[::-1], dtype=np.float64)
    approx = np.empty(shape[:-1] + (coeff_len,), dtype=np.float64)
    detail = np.empty_like(approx)
    rows = src.size // n
    status = lib().mpw_dwt_f64(
        addr(src),
        rows,
        n,
        addr(dec_lo),
        addr(dec_hi),
        wavelet.dec_len,
        addr(approx),
        addr(detail),
        coeff_len,
        mode_code,
    )
    check_status("dwt", status)
    return np.moveaxis(approx, -1, axis), np.moveaxis(detail, -1, axis)


def dwt(data, wavelet, mode="symmetric", axis=-1):
    array = np.asarray(data)
    if array.ndim < 1:
        raise ValueError("Input data must be at least 1D")
    if array.dtype == object:
        raise TypeError("Input must be a numeric array-like")
    axis = _axis(axis, array.ndim)
    if array.shape[axis] < 1:
        raise ValueError("Input data length must be greater than zero.")
    wav = _as_wavelet(wavelet)
    mode_code = _mode_code(mode)
    if array.shape[axis] == 1 and mode_code in {5, 7}:
        raise ValueError(
            "Input data length must be greater than 1 for [anti]reflect mode."
        )
    dtype = _result_dtype(array.dtype)
    if np.iscomplexobj(array):
        ar, dr = _dwt_real(array.real, wav, mode_code, axis)
        ai, di = _dwt_real(array.imag, wav, mode_code, axis)
        return (ar + 1j * ai).astype(dtype), (dr + 1j * di).astype(dtype)
    approx, detail = _dwt_real(array, wav, mode_code, axis)
    return approx.astype(dtype, copy=False), detail.astype(dtype, copy=False)


def _idwt_real(approx, detail, wavelet, mode_code: int, axis: int):
    a_moved = np.moveaxis(approx, axis, -1)
    d_moved = np.moveaxis(detail, axis, -1)
    if a_moved.shape != d_moved.shape:
        raise ValueError("Coefficients arrays must have the same size.")
    shape = a_moved.shape
    coeff_len = shape[-1]
    result_len = (
        2 * coeff_len
        if mode_code == 8
        else 2 * coeff_len - wavelet.rec_len + 2
    )
    if result_len < 1:
        raise ValueError("Invalid coefficient array length.")
    a = np.ascontiguousarray(a_moved, dtype=np.float64)
    d = np.ascontiguousarray(d_moved, dtype=np.float64)
    if mode_code == 8:
        rec_lo = np.ascontiguousarray(wavelet.rec_lo, dtype=np.float64)
        rec_hi = np.ascontiguousarray(wavelet.rec_hi, dtype=np.float64)
    else:
        rec_lo = np.ascontiguousarray(
            [*wavelet.rec_lo[-2::-2], *wavelet.rec_lo[-1::-2]],
            dtype=np.float64,
        )
        rec_hi = np.ascontiguousarray(
            [*wavelet.rec_hi[-2::-2], *wavelet.rec_hi[-1::-2]],
            dtype=np.float64,
        )
    result = np.empty(shape[:-1] + (result_len,), dtype=np.float64)
    rows = a.size // coeff_len
    status = lib().mpw_idwt_f64(
        addr(a),
        addr(d),
        rows,
        coeff_len,
        addr(rec_lo),
        addr(rec_hi),
        wavelet.rec_len,
        addr(result),
        result_len,
        int(mode_code == 8),
    )
    check_status("idwt", status)
    return np.moveaxis(result, -1, axis)


def idwt(cA, cD, wavelet, mode="symmetric", axis=-1):
    if cA is None and cD is None:
        raise ValueError("At least one coefficient parameter must be specified.")
    reference = np.asarray(cD if cA is None else cA)
    if reference.ndim < 1:
        raise ValueError("Input array must be at least 1D")
    axis = _axis(axis, reference.ndim)
    a = np.zeros_like(reference) if cA is None else np.asarray(cA)
    d = np.zeros_like(reference) if cD is None else np.asarray(cD)
    wav = _as_wavelet(wavelet)
    mode_code = _mode_code(mode)
    dtype = _result_dtype(np.result_type(a.dtype, d.dtype))
    if np.iscomplexobj(a) or np.iscomplexobj(d):
        real = _idwt_real(a.real, d.real, wav, mode_code, axis)
        imag = _idwt_real(a.imag, d.imag, wav, mode_code, axis)
        return (real + 1j * imag).astype(dtype)
    return _idwt_real(a, d, wav, mode_code, axis).astype(dtype, copy=False)


def downcoef(part, data, wavelet, mode="symmetric", level=1):
    if part not in {"a", "d"}:
        raise ValueError("Argument 1 must be 'a' or 'd'.")
    if level < 1:
        raise ValueError("Value of level must be greater than 0.")
    approx = np.asarray(data)
    for current in range(level):
        approx, detail = dwt(approx, wavelet, mode)
    return approx if part == "a" else detail
