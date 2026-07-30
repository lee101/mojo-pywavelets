"""Discrete wavelet metadata and the vendored filter-bank catalogue."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import re

from ._filters import FILTER_BANKS


class Wavelet:
    def __init__(self, name: str, filter_bank=None):
        self.name = str(name)
        if filter_bank is None:
            try:
                info = FILTER_BANKS[self.name]
            except KeyError:
                raise ValueError(f"Unknown wavelet name '{self.name}'") from None
            bank = info["filter_bank"]
            for key, value in info.items():
                if key != "filter_bank":
                    setattr(self, key, value)
            self._builtin = True
        else:
            bank = getattr(filter_bank, "filter_bank", filter_bank)
            if not isinstance(bank, Sequence) or len(bank) != 4:
                raise ValueError("filter_bank must contain four filter sequences")
            lengths = {len(f) for f in bank}
            if len(lengths) != 1 or next(iter(lengths)) < 2:
                raise ValueError("all filters must have the same length of at least 2")
            self.family_name = ""
            self.short_family_name = ""
            self.symmetry = "unknown"
            self.orthogonal = False
            self.biorthogonal = False
            self.vanishing_moments_phi = 0
            self.vanishing_moments_psi = 0
            self._builtin = False
        converted = [[float(value) for value in filt] for filt in bank]
        if len(converted[0]) % 2:
            for filt in converted:
                filt.append(0.0)
        self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi = converted

    @property
    def dec_len(self) -> int:
        return len(self.dec_lo)

    @property
    def rec_len(self) -> int:
        return len(self.rec_lo)

    @property
    def filter_bank(self) -> list[list[float]]:
        return [self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi]

    @property
    def inverse_filter_bank(self) -> list[list[float]]:
        return [
            self.rec_lo[::-1],
            self.rec_hi[::-1],
            self.dec_lo[::-1],
            self.dec_hi[::-1],
        ]

    def __repr__(self) -> str:
        return f"Wavelet {self.name!r}"


class _Modes:
    modes = [
        "zero",
        "constant",
        "symmetric",
        "periodic",
        "smooth",
        "periodization",
        "reflect",
        "antisymmetric",
        "antireflect",
    ]

    zero = 0
    constant = 1
    symmetric = 2
    periodic = 3
    smooth = 4
    periodization = 5
    reflect = 6
    antisymmetric = 7
    antireflect = 8


Modes = _Modes()


@lru_cache(maxsize=None)
def _builtin_wavelet(name: str) -> Wavelet:
    return Wavelet(name)


def _as_wavelet(wavelet) -> Wavelet:
    return (
        wavelet
        if isinstance(wavelet, Wavelet)
        else _builtin_wavelet(str(wavelet))
    )


def families(short: bool = True) -> list[str]:
    key = "short_family_name" if short else "family_name"
    order = ["haar", "db", "sym", "coif", "bior", "rbio", "dmey"]
    by_short = {
        info["short_family_name"]: info[key] for info in FILTER_BANKS.values()
    }
    return [by_short[name] for name in order]


def wavelist(family=None, kind: str = "all") -> list[str]:
    if kind not in {"all", "discrete", "continuous"}:
        raise ValueError(f"Unrecognized kind: {kind}")
    if kind == "continuous":
        return []
    def natural_key(name):
        return [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", name)
        ]

    names = sorted(FILTER_BANKS, key=natural_key)
    if family is None:
        return names
    return [
        name
        for name in names
        if FILTER_BANKS[name]["short_family_name"] == family
        or FILTER_BANKS[name]["family_name"] == family
    ]
