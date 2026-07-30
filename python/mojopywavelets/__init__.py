"""Discrete wavelet transforms accelerated by Mojo."""

from ._dwt import (
    downcoef,
    dwt,
    dwt_coeff_len,
    dwt_max_level,
    idwt,
)
from ._multidim import dwt2, dwtn, idwt2, idwtn
from ._multilevel import (
    dwtn_max_level,
    wavedec,
    wavedec2,
    wavedecn,
    waverec,
    waverec2,
    waverecn,
)
from ._wavelet import Modes, Wavelet, families, wavelist

__version__ = "0.1.0"

__all__ = [
    "Modes",
    "Wavelet",
    "downcoef",
    "dwt",
    "dwt2",
    "dwt_coeff_len",
    "dwt_max_level",
    "dwtn",
    "dwtn_max_level",
    "families",
    "idwt",
    "idwt2",
    "idwtn",
    "wavedec",
    "wavedec2",
    "wavedecn",
    "wavelist",
    "waverec",
    "waverec2",
    "waverecn",
]
