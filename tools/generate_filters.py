"""Regenerate the vendored discrete filter-bank table from PyWavelets."""

from pathlib import Path
from pprint import pformat

import pywt


def main() -> None:
    banks = {}
    for name in pywt.wavelist(kind="discrete"):
        wavelet = pywt.Wavelet(name)
        banks[name] = {
            "filter_bank": tuple(tuple(float(x) for x in f) for f in wavelet.filter_bank),
            "family_name": wavelet.family_name,
            "short_family_name": wavelet.short_family_name,
            "symmetry": wavelet.symmetry,
            "orthogonal": bool(wavelet.orthogonal),
            "biorthogonal": bool(wavelet.biorthogonal),
            "vanishing_moments_phi": wavelet.vanishing_moments_phi,
            "vanishing_moments_psi": wavelet.vanishing_moments_psi,
        }
    target = Path(__file__).parents[1] / "python" / "mojopywavelets" / "_filters.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        '"""Vendored PyWavelets-compatible discrete filter banks."""\n\n'
        + "FILTER_BANKS = "
        + pformat(banks, width=1_000_000, sort_dicts=True)
        + "\n"
    )


if __name__ == "__main__":
    main()
