"""Numerical and behavioral parity with PyWavelets for the covered API."""

from __future__ import annotations

import numpy as np
import pytest
import pywt

import mojopywavelets as mpw

RNG = np.random.default_rng(2026)
FAMILY_WAVELETS = [
    "haar",
    "db2",
    "db8",
    "sym3",
    "sym12",
    "coif1",
    "coif8",
    "bior1.3",
    "bior3.5",
    "rbio2.8",
    "dmey",
]


def assert_coeffs_close(ours, upstream, **kwargs):
    if isinstance(ours, dict):
        assert ours.keys() == upstream.keys()
        for key in ours:
            assert_coeffs_close(ours[key], upstream[key], **kwargs)
    elif isinstance(ours, (tuple, list)):
        assert len(ours) == len(upstream)
        for left, right in zip(ours, upstream):
            assert_coeffs_close(left, right, **kwargs)
    else:
        assert ours.shape == upstream.shape
        assert ours.dtype == upstream.dtype
        assert np.allclose(ours, upstream, **kwargs)


def test_catalog_matches_upstream():
    assert mpw.wavelist(kind="discrete") == pywt.wavelist(kind="discrete")
    assert mpw.families() == pywt.families()[:7]
    assert mpw.families(short=False) == pywt.families(short=False)[:7]
    assert mpw.Modes.modes == pywt.Modes.modes


@pytest.mark.parametrize("name", pywt.wavelist(kind="discrete"))
def test_every_vendored_filter_bank(name):
    ours = mpw.Wavelet(name)
    upstream = pywt.Wavelet(name)
    assert ours.dec_len == upstream.dec_len
    assert ours.rec_len == upstream.rec_len
    assert ours.family_name == upstream.family_name
    assert ours.short_family_name == upstream.short_family_name
    assert ours.symmetry == upstream.symmetry
    assert ours.orthogonal == upstream.orthogonal
    assert ours.biorthogonal == upstream.biorthogonal
    for left, right in zip(ours.filter_bank, upstream.filter_bank):
        assert np.array_equal(left, right)
    data = np.linspace(-1.0, 1.0, 129)
    assert_coeffs_close(
        mpw.dwt(data, ours),
        pywt.dwt(data, upstream),
        atol=5e-14,
    )


@pytest.mark.parametrize("mode", pywt.Modes.modes)
@pytest.mark.parametrize("length", [1, 2, 7, 32, 65])
def test_dwt_all_extension_modes(mode, length):
    data = RNG.normal(size=length)
    if length == 1 and mode in {"reflect", "antireflect"}:
        with pytest.raises(ValueError):
            mpw.dwt(data, "db4", mode)
        with pytest.raises(ValueError):
            pywt.dwt(data, "db4", mode)
        return
    assert_coeffs_close(
        mpw.dwt(data, "db4", mode),
        pywt.dwt(data, "db4", mode),
        atol=2e-14,
    )


@pytest.mark.parametrize("wavelet", FAMILY_WAVELETS)
def test_dwt_wavelet_families(wavelet):
    data = RNG.normal(size=127)
    assert_coeffs_close(
        mpw.dwt(data, wavelet),
        pywt.dwt(data, wavelet),
        atol=3e-14,
    )


@pytest.mark.parametrize("mode", pywt.Modes.modes)
def test_idwt_all_extension_modes(mode):
    data = RNG.normal(size=63)
    ours_coeffs = mpw.dwt(data, "sym6", mode)
    upstream_coeffs = pywt.dwt(data, "sym6", mode)
    assert_coeffs_close(
        mpw.idwt(*ours_coeffs, "sym6", mode),
        pywt.idwt(*upstream_coeffs, "sym6", mode),
        atol=3e-14,
    )


@pytest.mark.parametrize(
    "dtype",
    [np.int16, np.float16, np.float32, np.float64, np.complex64, np.complex128],
)
def test_dtype_and_complex_parity(dtype):
    data = np.arange(48, dtype=np.float64).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        data += (1j * np.arange(48)[::-1]).astype(dtype)
    ours = mpw.dwt(data, "bior2.4")
    upstream = pywt.dwt(data, "bior2.4")
    assert_coeffs_close(
        ours,
        upstream,
        rtol=2e-6,
        atol=2e-6,
    )
    assert_coeffs_close(
        mpw.idwt(*ours, "bior2.4"),
        pywt.idwt(*upstream, "bior2.4"),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_arbitrary_axis(axis):
    data = RNG.normal(size=(9, 10, 11))
    assert_coeffs_close(
        mpw.dwt(data, "db3", axis=axis),
        pywt.dwt(data, "db3", axis=axis),
        atol=2e-14,
    )


def test_noncontiguous_input():
    data = RNG.normal(size=(40, 30))[::2, ::3]
    assert not data.flags.c_contiguous
    assert_coeffs_close(mpw.dwt(data, "db3", axis=0), pywt.dwt(data, "db3", axis=0))


def test_empty_batch_dimension_does_not_dereference_buffers():
    data = np.empty((0, 9), dtype=np.float64)
    assert_coeffs_close(
        mpw.dwt(data, "db3", axis=1),
        pywt.dwt(data, "db3", axis=1),
    )


def test_simd_tail_filter():
    data = RNG.normal(size=257)
    assert_coeffs_close(
        mpw.dwt(data, "db3"),
        pywt.dwt(data, "db3"),
        atol=2e-14,
    )


@pytest.mark.parametrize("length", [999_973, 1_000_003])
def test_parallel_threshold(length):
    data = RNG.normal(size=length)
    ours = mpw.dwt(data, "db4")
    upstream = pywt.dwt(data, "db4")
    assert_coeffs_close(ours, upstream, atol=2e-14)
    assert_coeffs_close(
        mpw.idwt(*ours, "db4"),
        pywt.idwt(*upstream, "db4"),
        atol=3e-14,
    )


def test_parallel_synthesis():
    approx = RNG.normal(size=1_000_003)
    detail = RNG.normal(size=1_000_003)
    assert_coeffs_close(
        mpw.idwt(approx, detail, "db4"),
        pywt.idwt(approx, detail, "db4"),
        atol=3e-14,
    )


@pytest.mark.parametrize("mode", ["symmetric", "periodization"])
def test_idwt_missing_band(mode):
    data = RNG.normal(size=64)
    cA, cD = mpw.dwt(data, "db5", mode)
    uA, uD = pywt.dwt(data, "db5", mode)
    assert_coeffs_close(mpw.idwt(cA, None, "db5", mode), pywt.idwt(uA, None, "db5", mode))
    assert_coeffs_close(mpw.idwt(None, cD, "db5", mode), pywt.idwt(None, uD, "db5", mode))


def test_custom_wavelet():
    upstream = pywt.Wavelet("db2")
    ours = mpw.Wavelet("custom", filter_bank=upstream.filter_bank)
    data = RNG.normal(size=33)
    assert_coeffs_close(mpw.dwt(data, ours), pywt.dwt(data, upstream))


def test_odd_length_custom_wavelet_is_padded_like_upstream():
    bank = [[1.0, 2.0, 3.0]] * 4
    ours = mpw.Wavelet("custom", filter_bank=bank)
    upstream = pywt.Wavelet("custom", filter_bank=bank)
    assert ours.filter_bank == [list(filt) for filt in upstream.filter_bank]
    data = RNG.normal(size=17)
    assert_coeffs_close(mpw.dwt(data, ours), pywt.dwt(data, upstream))


@pytest.mark.parametrize("length", [1, 2, 3, 16, 31, 100])
@pytest.mark.parametrize("filter_len", [2, 4, 10])
@pytest.mark.parametrize("mode", ["symmetric", "periodization"])
def test_length_helpers(length, filter_len, mode):
    assert mpw.dwt_coeff_len(length, filter_len, mode) == pywt.dwt_coeff_len(
        length, filter_len, mode
    )
    assert mpw.dwt_max_level(length, filter_len) == pywt.dwt_max_level(
        length, filter_len
    )


@pytest.mark.parametrize("mode", ["symmetric", "periodization", "antireflect"])
@pytest.mark.parametrize("level", [0, 1, 3, None])
def test_wavedec_and_waverec(mode, level):
    data = RNG.normal(size=(3, 129))
    ours = mpw.wavedec(data, "db4", mode, level, axis=1)
    upstream = pywt.wavedec(data, "db4", mode, level, axis=1)
    assert_coeffs_close(ours, upstream, atol=3e-14)
    assert_coeffs_close(
        mpw.waverec(ours, "db4", mode, axis=1),
        pywt.waverec(upstream, "db4", mode, axis=1),
        atol=3e-14,
    )


def test_downcoef():
    data = RNG.normal(size=128)
    for part in "ad":
        assert_coeffs_close(
            mpw.downcoef(part, data, "sym5", level=3),
            pywt.downcoef(part, data, "sym5", level=3),
            atol=3e-14,
        )


@pytest.mark.parametrize("mode", ["symmetric", "periodization", "reflect"])
def test_dwt2_and_idwt2(mode):
    data = RNG.normal(size=(3, 17, 22))
    wavelet = ("db2", "sym4")
    axes = (1, 2)
    ours = mpw.dwt2(data, wavelet, mode, axes)
    upstream = pywt.dwt2(data, wavelet, mode, axes)
    assert_coeffs_close(ours, upstream, atol=3e-14)
    assert_coeffs_close(
        mpw.idwt2(ours, wavelet, mode, axes),
        pywt.idwt2(upstream, wavelet, mode, axes),
        atol=3e-14,
    )


def test_dwtn_and_idwtn():
    data = RNG.normal(size=(8, 9, 10))
    ours = mpw.dwtn(data, ("haar", "db2", "sym3"), axes=(0, 1, 2))
    upstream = pywt.dwtn(data, ("haar", "db2", "sym3"), axes=(0, 1, 2))
    assert_coeffs_close(ours, upstream, atol=3e-14)
    assert_coeffs_close(
        mpw.idwtn(ours, ("haar", "db2", "sym3"), axes=(0, 1, 2)),
        pywt.idwtn(upstream, ("haar", "db2", "sym3"), axes=(0, 1, 2)),
        atol=3e-14,
    )


def test_idwtn_sparse_coefficients():
    data = RNG.normal(size=(8, 9))
    ours = mpw.dwtn(data, "db2")
    upstream = pywt.dwtn(data, "db2")
    sparse_ours = {"aa": ours["aa"]}
    sparse_upstream = {"aa": upstream["aa"]}
    assert_coeffs_close(
        mpw.idwtn(sparse_ours, "db2"),
        pywt.idwtn(sparse_upstream, "db2"),
        atol=3e-14,
    )


@pytest.mark.parametrize("mode", ["symmetric", "periodization"])
def test_wavedec2_and_waverec2(mode):
    data = RNG.normal(size=(3, 33, 42))
    ours = mpw.wavedec2(data, "bior2.2", mode, level=2, axes=(1, 2))
    upstream = pywt.wavedec2(data, "bior2.2", mode, level=2, axes=(1, 2))
    assert_coeffs_close(ours, upstream, atol=3e-14)
    assert_coeffs_close(
        mpw.waverec2(ours, "bior2.2", mode, axes=(1, 2)),
        pywt.waverec2(upstream, "bior2.2", mode, axes=(1, 2)),
        atol=3e-14,
    )


def test_wavedecn_and_waverecn():
    data = RNG.normal(size=(17, 20, 7))
    ours = mpw.wavedecn(data, "db2", level=2, axes=(0, 1))
    upstream = pywt.wavedecn(data, "db2", level=2, axes=(0, 1))
    assert_coeffs_close(ours, upstream, atol=3e-14)
    assert mpw.dwtn_max_level(data.shape, "db2", axes=(0, 1)) == pywt.dwtn_max_level(
        data.shape, "db2", axes=(0, 1)
    )
    assert_coeffs_close(
        mpw.waverecn(ours, "db2", axes=(0, 1)),
        pywt.waverecn(upstream, "db2", axes=(0, 1)),
        atol=3e-14,
    )


def test_per_alias_matches_periodization():
    data = RNG.normal(size=51)
    assert_coeffs_close(
        mpw.dwt(data, "db3", "per"),
        pywt.dwt(data, "db3", "periodization"),
    )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        mpw.dwt([], "db2")
    with pytest.raises(ValueError):
        mpw.idwt(None, None, "db2")
    with pytest.raises(ValueError):
        mpw.dwt([1, 2], "not-a-wavelet")
    with pytest.raises(ValueError):
        mpw.dwt([1, 2], "haar", "not-a-mode")
