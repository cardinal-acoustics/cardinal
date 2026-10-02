# localize/tdoa.py
#
# Signal processing and geometry utilities for TDOA-based localization.

from __future__ import annotations

import warnings

import numpy as np
import scipy.signal
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len, fftshift
from scipy import interpolate

from config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_np(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (km) between two points (or arrays of points)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


def dist_pointstopoint(points: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row in *points* to *point*."""
    return np.sqrt(np.sum(np.power(points - point, 2), axis=1))


def time_pointstopoint(points: np.ndarray, point: np.ndarray,
                       s: float = 344) -> np.ndarray:
    """Travel time (seconds) at speed *s* m/s from each point to *point*."""
    return dist_pointstopoint(points, point) / s


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def max_rolling(y: np.ndarray, w: int) -> np.ndarray:
    w = int(w)
    if w == 0:
        return y
    yw = np.zeros((2 * w + 1, len(y)))
    yw[0] = y
    for i in range(1, w + 1):
        yw[i][: len(y) - i]  = y[i:]
        yw[-i][i:]            = y[: len(y) - i]
    return np.max(yw, axis=0)


def argmax_rolling(y: np.ndarray, w: int) -> np.ndarray:
    w = int(w)
    if w == 0:
        return y
    yw = np.zeros((2 * w + 1, len(y)))
    yw[0] = y
    for i in range(1, w + 1):
        yw[i][: len(y) - i]  = y[i:]
        yw[-i][i:]            = y[: len(y) - i]
    return np.argmax(yw, axis=0)


def find_consecutive_ones(arr: np.ndarray) -> list[list[int]]:
    arr = np.array(arr)
    padded = np.pad(arr, 1, constant_values=0)
    diffs  = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    return [[s, e] for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# Frequency filter
# ---------------------------------------------------------------------------

class FreqFilter:
    """Interpolated frequency-domain filter built from a spectral profile."""

    def __init__(self):
        self.filter_inter = None

    @classmethod
    def from_profile(cls, freqs: np.ndarray, profile: np.ndarray,
                     rate: int) -> "FreqFilter":
        obj = cls()
        profile = profile / np.max(profile)
        obj.filter_x = freqs
        obj.filter_y = profile
        obj.filter_inter = interpolate.interp1d(
            freqs, profile, kind='linear', fill_value=0, bounds_error=False
        )
        return obj

    def filter(self, fft_x: np.ndarray, fft_y: np.ndarray) -> np.ndarray:
        return self.filter_inter(fft_x) * fft_y


# ---------------------------------------------------------------------------
# Cross-correlation (TDOA)
# ---------------------------------------------------------------------------

def fcc(y1: np.ndarray, y2: np.ndarray,
        freqfilter=None,
        rate: int = SAMPLE_RATE,
        use_phat: bool = False,
        max_lag: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute (optionally PHAT-weighted) cross-correlation of two real signals.

    Returns
    -------
    cc   : signed, normalized cross-correlation (max |cc| = 1)
    lags : time lags in seconds, from −T to +T
    qual : quality metric (std of |non-zero cc|)
    """
    n_lin = len(y1) + len(y2) - 1
    n_fft = next_fast_len(n_lin)

    fy1 = np.zeros(n_fft, dtype=np.float64)
    fy2 = np.zeros(n_fft, dtype=np.float64)
    fy1[:len(y1)] = y1
    fy2[:len(y2)] = y2

    F1    = rfft(fy1)
    F2    = rfft(fy2)
    freqs = rfftfreq(n_fft, d=1 / rate)

    if freqfilter is not None:
        F1 = freqfilter.filter(freqs, F1)
        F2 = freqfilter.filter(freqs, F2)

    R = F1 * np.conj(F2)
    if use_phat:
        R /= (np.abs(R) + np.finfo(float).eps)

    cc   = irfft(R, n=n_fft).real
    cc   = fftshift(cc)
    half = n_fft // 2
    lags = np.arange(-half, half) / rate

    if max_lag is not None:
        cc[np.abs(lags) > max_lag] = 0.0

    peak = np.max(np.abs(cc))
    if peak > 0:
        cc = cc / peak

    nz   = cc[cc != 0]
    qual = float(np.std(np.abs(nz))) if nz.size else 0.0

    return cc, lags, qual


# ---------------------------------------------------------------------------
# Spectrogram (used for QC plots in localize/localize_utils/core.py)
# ---------------------------------------------------------------------------

class Spec:
    """Log-spectrogram wrapper for plotting."""

    def __init__(self, data: np.ndarray, rate: int):
        self.rate = rate
        self.freqs, self.times, self.spec = scipy.signal.spectrogram(data, rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.call = np.log(self.spec)

    @classmethod
    def from_call(cls, call) -> "Spec":
        return cls(call.data, call.rate)

    def plotcall(self, ax=None, vmin=0, vmax=None, cmap='gray_r',
                 axes=False, lines=True):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        extent = (self.times.min(), self.times.max(),
                  self.freqs.min(), self.freqs.max())
        kwargs = dict(cmap=cmap, origin='lower', aspect='auto',
                      extent=extent, vmin=vmin)
        if vmax is not None:
            kwargs['vmax'] = vmax
        ax.imshow(self.call, **kwargs)
        if self.times.max() - self.times.min() <= 10 and lines:
            for t in range(int(self.times.min()), int(self.times.max())):
                ax.plot([t + 1, t + 1],
                        [self.freqs.min(), self.freqs.max()],
                        c='red', linewidth=1)
        if not axes:
            ax.set_xticks([])
            ax.set_yticks([])
