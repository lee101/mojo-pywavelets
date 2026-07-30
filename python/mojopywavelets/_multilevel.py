"""Multilevel 1D, 2D, and nD wavelet decompositions."""

from __future__ import annotations

import warnings

import numpy as np

from ._dwt import dwt, dwt_max_level, idwt
from ._multidim import dwt2, dwtn, idwt2, idwtn, _axes, _per_axis
from ._wavelet import _as_wavelet


def _level(sizes, wavelets, level):
    maximum = min(
        dwt_max_level(size, _as_wavelet(wav).dec_len)
        for size, wav in zip(sizes, wavelets)
    )
    if level is None:
        return maximum
    level = int(level)
    if level < 0:
        raise ValueError("Level value is too low. Minimum level is 0.")
    if level > maximum:
        warnings.warn(
            f"Level value of {level} is too high: all coefficients will "
            "experience boundary effects.",
            UserWarning,
            stacklevel=3,
        )
    return level


def wavedec(data, wavelet, mode="symmetric", level=None, axis=-1):
    array = np.asarray(data)
    wav = _as_wavelet(wavelet)
    axis = axis + array.ndim if axis < 0 else axis
    level = _level([array.shape[axis]], [wav], level)
    details = []
    approx = array
    for _ in range(level):
        approx, detail = dwt(approx, wav, mode, axis)
        details.append(detail)
    return [approx, *reversed(details)]


def waverec(coeffs, wavelet, mode="symmetric", axis=-1):
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        raise ValueError("Expected a non-empty sequence of coefficient arrays.")
    if len(coeffs) == 1:
        return coeffs[0]
    approx = coeffs[0]
    for detail in coeffs[1:]:
        if approx is not None and detail is not None:
            axis_norm = axis + np.ndim(approx) if axis < 0 else axis
            if approx.shape[axis_norm] == detail.shape[axis_norm] + 1:
                slices = [slice(None)] * np.ndim(approx)
                slices[axis_norm] = slice(0, detail.shape[axis_norm])
                approx = approx[tuple(slices)]
            elif approx.shape[axis_norm] != detail.shape[axis_norm]:
                raise ValueError("coefficient shape mismatch")
        approx = idwt(approx, detail, wavelet, mode, axis)
    return approx


def wavedec2(data, wavelet, mode="symmetric", level=None, axes=(-2, -1)):
    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("Expected input data to have at least 2 dimensions.")
    axes = _axes(tuple(axes), array.ndim)
    if len(axes) != 2 or len(set(axes)) != 2:
        raise ValueError("Expected 2 unique axes")
    wavelets = _per_axis(wavelet, axes)
    level = _level([array.shape[axis] for axis in axes], wavelets, level)
    details = []
    approx = array
    for _ in range(level):
        approx, detail = dwt2(approx, wavelet, mode, axes)
        details.append(detail)
    return [approx, *reversed(details)]


def waverec2(coeffs, wavelet, mode="symmetric", axes=(-2, -1)):
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        raise ValueError("Expected a non-empty sequence of coefficient arrays.")
    if len(coeffs) == 1:
        return coeffs[0]
    approx = np.asarray(coeffs[0])
    for details in coeffs[1:]:
        if not isinstance(details, (list, tuple)) or len(details) != 3:
            raise ValueError("Detail coefficients must be a 3-tuple.")
        detail_arrays = [np.asarray(item) for item in details if item is not None]
        if detail_arrays:
            shape = detail_arrays[0].shape
            slices = tuple(
                slice(None, -1 if a_len == d_len + 1 else None)
                for a_len, d_len in zip(approx.shape, shape)
            )
            approx = approx[slices]
        approx = idwt2((approx, details), wavelet, mode, axes)
    return approx


def dwtn_max_level(shape, wavelet, axes=None):
    shape = tuple(shape)
    axes = _axes(axes, len(shape))
    wavelets = _per_axis(wavelet, axes)
    return min(
        dwt_max_level(shape[axis], _as_wavelet(wav).dec_len)
        for axis, wav in zip(axes, wavelets)
    )


def wavedecn(data, wavelet, mode="symmetric", level=None, axes=None):
    array = np.asarray(data)
    axes = _axes(axes, array.ndim)
    wavelets = _per_axis(wavelet, axes)
    level = _level([array.shape[axis] for axis in axes], wavelets, level)
    details = []
    approx = array
    approx_key = "a" * len(axes)
    for _ in range(level):
        bands = dwtn(approx, wavelet, mode, axes)
        approx = bands.pop(approx_key)
        details.append(bands)
    return [approx, *reversed(details)]


def waverecn(coeffs, wavelet, mode="symmetric", axes=None):
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        raise ValueError("Expected a non-empty sequence of coefficients.")
    if len(coeffs) == 1:
        return coeffs[0]
    approx = np.asarray(coeffs[0])
    key_len = len(next(iter(coeffs[1])))
    axes = _axes(tuple(range(key_len)) if axes is None else axes, approx.ndim)
    approx_key = "a" * key_len
    for details in coeffs[1:]:
        sample = next((np.asarray(v) for v in details.values() if v is not None), None)
        if sample is not None:
            slices = tuple(
                slice(0, sample.shape[i]) if approx.shape[i] == sample.shape[i] + 1 else slice(None)
                for i in range(approx.ndim)
            )
            approx = approx[slices]
        bands = dict(details)
        bands[approx_key] = approx
        approx = idwtn(bands, wavelet, mode, axes)
    return approx
