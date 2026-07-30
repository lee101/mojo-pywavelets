"""Two- and n-dimensional transforms composed from the batched Mojo kernel."""

from __future__ import annotations

import numpy as np

from ._dwt import dwt, idwt


def _per_axis(value, axes):
    if isinstance(value, (tuple, list)):
        if len(value) != len(axes):
            raise ValueError("The number of wavelets/modes must match the axes")
        return list(value)
    return [value] * len(axes)


def _axes(axes, ndim):
    if axes is None:
        axes = tuple(range(ndim))
    result = tuple(axis + ndim if axis < 0 else axis for axis in axes)
    if any(axis < 0 or axis >= ndim for axis in result):
        raise np.exceptions.AxisError("Axis greater than data dimensions")
    return result


def dwtn(data, wavelet, mode="symmetric", axes=None):
    array = np.asarray(data)
    if array.ndim < 1:
        raise ValueError("Input data must be at least 1D")
    axes = _axes(axes, array.ndim)
    wavelets = _per_axis(wavelet, axes)
    modes = _per_axis(mode, axes)
    coeffs = {"": array}
    for axis, wav, extension in zip(axes, wavelets, modes):
        next_coeffs = {}
        for key, values in coeffs.items():
            approx, detail = dwt(values, wav, extension, axis)
            next_coeffs[key + "a"] = approx
            next_coeffs[key + "d"] = detail
        coeffs = next_coeffs
    return coeffs


def idwtn(coeffs, wavelet, mode="symmetric", axes=None):
    coeffs = {key: np.asarray(value) for key, value in coeffs.items() if value is not None}
    if not coeffs:
        raise ValueError("`coeffs` must contain at least one non-null wavelet band")
    key_len = max(map(len, coeffs))
    if any(len(key) != key_len or not set(key) <= {"a", "d"} for key in coeffs):
        raise ValueError("Invalid coefficient dictionary keys")
    sample = next(iter(coeffs.values()))
    axes = _axes(tuple(range(key_len)) if axes is None else axes, sample.ndim)
    if len(axes) != key_len:
        raise ValueError("The number of axes must match the coefficient keys")
    wavelets = _per_axis(wavelet, axes)
    modes = _per_axis(mode, axes)
    current = coeffs
    for _, (axis, wav, extension) in reversed(
        list(enumerate(zip(axes, wavelets, modes)))
    ):
        merged = {}
        prefixes = {key[:-1] for key in current}
        for prefix in prefixes:
            merged[prefix] = idwt(
                current.get(prefix + "a"),
                current.get(prefix + "d"),
                wav,
                extension,
                axis,
            )
        current = merged
    return current[""]


def dwt2(data, wavelet, mode="symmetric", axes=(-2, -1)):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")
    coeffs = dwtn(data, wavelet, mode, axes)
    return coeffs["aa"], (coeffs["da"], coeffs["ad"], coeffs["dd"])


def idwt2(coeffs, wavelet, mode="symmetric", axes=(-2, -1)):
    if len(tuple(axes)) != 2:
        raise ValueError("Expected 2 axes")
    approx, (horizontal, vertical, diagonal) = coeffs
    return idwtn(
        {"aa": approx, "da": horizontal, "ad": vertical, "dd": diagonal},
        wavelet,
        mode,
        axes,
    )
