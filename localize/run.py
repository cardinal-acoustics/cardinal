# localize/run.py
#
# Core localization functions and parallel runner.
#
# This module implements the bl_ref API expected by
# localize/localize_utils/core.py (localize_plot_withorig).
#
# Functions:
#   get_rowdata(row, wf, ...)              load audio clips for all active recorders
#   filter_rowdata(datas, freqfilter)      apply frequency filter to audio dict
#   get_recorderlocs(row, gdf, recorders)  look up recorder (x,y,z) coordinates
#   get_ccs(recorders, datas, ...)         cross-correlations for all recorder pairs
#   get_tdoas(pairs, lags, ccs)            extract peak lag (TDOA) per pair
#   get_toas(rec_coords, x_est, c_est)     travel times from source to each recorder
#   get_aligned_rowdata(...)               load audio aligned to estimated source
#   locate_source_and_speed(...)           least-squares TDOA solver
#   compute_tdoa_errors(...)               residuals between predicted/observed TDOAs
#   plot_spectrogram(sig, fs, ax, title)   plot a log-spectrogram on an axis
#   process_row(i)                         per-row worker for parallel runner
#   init_pool(...)                         pool initializer; loads shared globals

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import least_squares

from catalog.audiofile import AudioFileSolarbar, AudioFileS4
from localize.tdoa import Spec, fcc
from config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _open_audiofile(r):
    """Open the appropriate AudioFile subclass for a catalog row."""
    if r.recorder_type == "SOLARBAR":
        return AudioFileSolarbar(r.file)
    elif r.recorder_type == "S4A":
        return AudioFileS4(r.file)
    return None


def get_rowdata(row, wf: pd.DataFrame,
                winsize: float = 5.0,
                dt_in_win: float = 1.0,
                require_gps_sync: bool = True) -> tuple[list, dict]:
    """Load audio clips for every recorder active at *row.call_datetime*.

    Parameters
    ----------
    row       : detection row with fields call_datetime, recorder_group
    wf        : wave-file catalog (from catalog/build.py)
    winsize   : total clip length in seconds
    dt_in_win : seconds before call_datetime to start the clip
    require_gps_sync : drop files whose clock was not GPS-disciplined. Defaults
        to True because such a file cannot support TDOA at all -- see below.

    Returns
    -------
    recorders : sorted list of recorder IDs
    datas     : dict mapping recorder ID → np.ndarray (audio at SAMPLE_RATE Hz)
    """
    rc = wf[
        (wf.start_datetime < row.call_datetime) &
        (wf.end_datetime   > row.call_datetime) &
        (wf.recorder_group == row.recorder_group)
    ]
    rc = rc[rc.channel != "1"]
    # birdnet.parse.filter_detections already requires gps_sync for DETECTIONS,
    # but audio was selected purely by time overlap, so an unsynced file could
    # still be handed to the solver. A BAR-LT without a fix writes the
    # `<ts>+0000_REC.wav` form: whole-second timestamps (no microseconds) and
    # no E-timestamp, so AudioFileSolarbar cannot calibrate sample_size either.
    # Quantisation alone is +/-0.5 s -- thousands of times the ~1 ms TDOA
    # budget. Measured on BARLT_00018124, House 2026-09-09: its caw onsets sat
    # 1065 ms from the other four recorders, which all agreed to within 125 ms.
    if require_gps_sync and "gps_sync" in rc.columns:
        rc = rc[rc.gps_sync == True]  # noqa: E712 -- pandas column, not a bool
    recorders = sorted(rc.recorder.tolist())
    datas = {}

    for _, r in rc.sort_values("recorder").iterrows():
        af = _open_audiofile(r)
        if af is None:
            continue
        offset = af.get_ttoffset(row.call_datetime - timedelta(seconds=dt_in_win))
        datas[r.recorder] = af.getdata(offset, offset + winsize * af.rate)

    return recorders, datas


def get_aligned_rowdata(row, wf: pd.DataFrame, recorders: list,
                        toa_offsets: np.ndarray,
                        winsize: float = 5.0,
                        dt_in_win: float = 1.0) -> dict:
    """Load audio clips time-shifted so each recorder's signal is aligned.

    Parameters
    ----------
    row         : detection row
    wf          : wave-file catalog
    recorders   : list of recorder IDs (same order as toa_offsets)
    toa_offsets : per-recorder time-of-arrival offsets (seconds) relative to
                  the earliest arrival; used to shift each clip forward
    winsize     : clip length in seconds
    dt_in_win   : seconds before call_datetime to start the unshifted clip
    """
    rc = wf[
        (wf.start_datetime < row.call_datetime) &
        (wf.end_datetime   > row.call_datetime) &
        (wf.recorder_group == row.recorder_group)
    ]
    rc = rc[rc.channel != "1"].set_index("recorder")
    datas = {}

    for i, rec in enumerate(recorders):
        if rec not in rc.index:
            continue
        r  = rc.loc[rec]
        af = _open_audiofile(r)
        if af is None:
            continue
        # Shift the clip start by toa_offset so signals align
        t_start = (row.call_datetime
                   - timedelta(seconds=dt_in_win)
                   + timedelta(seconds=float(toa_offsets[i])))
        offset = af.get_ttoffset(t_start)
        datas[rec] = af.getdata(offset, offset + winsize * af.rate)

    return datas


def filter_rowdata(datas: dict, freqfilter) -> dict:
    """Apply a FreqFilter to each audio array in *datas*.

    Parameters
    ----------
    datas      : dict mapping recorder ID → np.ndarray
    freqfilter : FreqFilter instance (or None → returns datas unchanged)

    Returns
    -------
    dict mapping recorder ID → filtered np.ndarray
    """
    if freqfilter is None:
        return datas

    from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
    out = {}
    for rec, y in datas.items():
        n    = next_fast_len(len(y))
        F    = rfft(y, n=n)
        freq = rfftfreq(n, d=1 / SAMPLE_RATE)
        F    = freqfilter.filter(freq, F)
        out[rec] = irfft(F, n=n)[:len(y)].real
    return out


# ---------------------------------------------------------------------------
# Recorder geometry
# ---------------------------------------------------------------------------

def get_recorderlocs(row, deployments: pd.DataFrame,
                     recorders: list) -> np.ndarray:
    """Return an (n, 3) array of (longitude, latitude, z) coordinates.

    Coordinates are deployment-aware: looks up the deployment active at
    row.call_datetime for each recorder, using the z coordinate
    (DSM elevation + height_above_ground) from deployments.parquet.

    Parameters
    ----------
    row         : detection row with field call_datetime
    deployments : DataFrame from catalog/deployments.py with columns
                  recorder, begin, end (NaT=active), longitude, latitude, z
    recorders   : list of recorder IDs (same order as returned by get_rowdata)

    Returns
    -------
    np.ndarray of shape (n_recorders, 3)  — columns: longitude, latitude, z
    """
    from catalog.deployments import match_deployment
    from pyproj import Transformer

    # Project lon/lat → UTM metres so Euclidean distance calculations are correct
    _tf = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)

    locs = []
    for rec in recorders:
        dep = match_deployment(rec, row.call_datetime, deployments)
        if dep is None:
            raise ValueError(
                f"No deployment found for {rec} at {row.call_datetime}. "
                f"Add it to recorder_deployments.csv and rebuild deployments.parquet."
            )
        x, y = _tf.transform(dep["longitude"], dep["latitude"])
        locs.append([x, y, float(dep["z"])])
    return np.array(locs)


# ---------------------------------------------------------------------------
# Cross-correlations
# ---------------------------------------------------------------------------

def get_ccs(recorders: list, datas: dict,
            freqfilter=None,
            rec_coords: np.ndarray | None = None,
            use_phat: bool = False,
            qualcut: float = 0.2) -> tuple[list, list, dict, dict, list, list]:
    """Compute cross-correlations for all recorder pairs.

    Parameters
    ----------
    recorders  : list of recorder IDs
    datas      : dict of audio arrays from get_rowdata()
    freqfilter : FreqFilter instance or None
    rec_coords : optional (n, 3) array; max_lag is set from inter-recorder
                 distance divided by 300 m/s when provided
    use_phat   : apply PHAT weighting in cross-correlation
    qualcut    : keep pairs with CC quality (std of |cc|) <= qualcut;
                 lower = more peaked = clearer signal

    Returns
    -------
    quals     : quality values for accepted pairs
    pairs     : accepted (i, j) index tuples
    ccs       : dict (i, j) → CC array for accepted pairs
    lags      : dict (i, j) → lag array (seconds) for accepted pairs
    allquals  : quality values for ALL pairs (before qualcut filter)
    allpairs  : ALL (i, j) tuples (before qualcut filter)
    """
    pairs, quals, ccs, lags = [], [], {}, {}
    allpairs, allquals = [], []

    for i1, r1 in enumerate(recorders):
        for i2 in range(i1 + 1, len(recorders)):
            r2   = recorders[i2]
            pair = (i1, i2)

            max_lag = 5.0
            if rec_coords is not None:
                max_lag = np.linalg.norm(rec_coords[i1] - rec_coords[i2]) / 300.0

            cc, lag, qual = fcc(datas[r1], datas[r2], freqfilter,
                                rate=SAMPLE_RATE, use_phat=use_phat, max_lag=max_lag)

            allpairs.append(pair)
            allquals.append(qual)

            if qual <= qualcut:
                quals.append(qual)
                pairs.append(pair)
                ccs[pair]  = cc
                lags[pair] = lag

    return quals, pairs, ccs, lags, allquals, allpairs


# ---------------------------------------------------------------------------
# TDOA extraction
# ---------------------------------------------------------------------------

def get_tdoas(pairs: list, lags: dict, ccs: dict) -> np.ndarray:
    """Return the peak-lag TDOA for each pair.

    Parameters
    ----------
    pairs : list of (i, j) tuples
    lags  : dict mapping (i, j) → lag array
    ccs   : dict mapping (i, j) → CC array

    Returns
    -------
    np.ndarray of TDOA values (seconds), one per pair
    """
    return np.array([lags[p][np.argmax(ccs[p])] for p in pairs])


# ---------------------------------------------------------------------------
# Source localization
# ---------------------------------------------------------------------------

def locate_source_and_speed(rec_coords, pairs, tdoas,
                             speed0: float = 343.0,
                             speed_bounds: tuple = (330, 360),
                             x0=None,
                             fit_speed: bool = True):
    """Least-squares TDOA solver estimating source position and speed of sound.

    Parameters
    ----------
    rec_coords    : (n_rec, 3) array of recorder coordinates
    pairs         : list of (i, j) index pairs
    tdoas         : observed time-difference-of-arrival for each pair (seconds)
    speed0        : initial guess for speed of sound (m/s)
    speed_bounds  : (low, high) bounds for speed of sound
    x0            : initial guess for source (x, y, z); defaults to recorder centroid
    fit_speed     : if False, hold c at *speed0* and solve for (x, y, z) only.

        Why this exists: on a near-planar 4-recorder array the (x, y, z, c)
        Jacobian is singular -- measured across the 1.06M-solution corpus on
        2026-09-21, 0% of 3- and 4-recorder solves were identifiable
        (condition number > 1e6), which is 77.7% of the archive. Dropping c
        from the fit takes 4-recorder solves to 86% identifiable and the
        median condition number from 5.7e12 to 49. The cost is that a wrong
        c now biases position directly instead of being absorbed, so only do
        this with a c you trust.

    Returns
    -------
    x_est : (3,) estimated source position
    c_est : estimated speed of sound (m/s); equals speed0 when fit_speed=False
    sol   : scipy least_squares result object
    """
    rec_coords = np.asarray(rec_coords, dtype=float)
    tdoas      = np.asarray(tdoas, dtype=float)

    if x0 is None:
        x0 = rec_coords.mean(axis=0)

    xmin, xmax = rec_coords.min(axis=0), rec_coords.max(axis=0)
    buffer = 300
    lb = np.array([xmin[0] - buffer, xmin[1] - buffer, xmin[2] - 5,
                   speed_bounds[0]], dtype=float)
    ub = np.array([xmax[0] + buffer, xmax[1] + buffer, xmax[2] + 75,
                   speed_bounds[1]], dtype=float)

    def _resid_xyz(x, c):
        # fcc(y_i, y_j) peaks at (d_i - d_j)/c, so the model is
        # dt == (d_i - d_j)/c, i.e. residual = (d_i - d_j) - c*dt.
        return np.array(
            [(np.linalg.norm(x - rec_coords[i]) -
              np.linalg.norm(x - rec_coords[j])) - c * dt
             for (i, j), dt in zip(pairs, tdoas)],
            dtype=float,
        )

    if not fit_speed:
        c_fixed = float(speed0)
        sol = least_squares(lambda v: _resid_xyz(v, c_fixed), np.asarray(x0, dtype=float),
                            bounds=(lb[:3], ub[:3]), verbose=0)
        return sol.x[:3], c_fixed, sol

    def residual(vars):
        return _resid_xyz(vars[:3], float(vars[3]))

    sol   = least_squares(residual, np.hstack([x0, speed0]),
                          bounds=(lb, ub), verbose=0)
    x_est = sol.x[:3]
    c_est = sol.x[3]
    return x_est, c_est, sol


def compute_tdoa_errors(rec_coords, pairs, tdoas,
                        x_est, c_est) -> tuple[np.ndarray, dict, np.ndarray]:
    """Compute residuals between predicted and observed TDOAs.

    Parameters
    ----------
    rec_coords : (n_rec, 3) recorder coordinates
    pairs      : list of (i, j) index pairs
    tdoas      : observed TDOAs (seconds)
    x_est      : estimated source position (3,)
    c_est      : estimated speed of sound (m/s)

    Returns
    -------
    errors     : per-pair residual array (predicted − observed)
    error_dict : dict mapping (i, j) → error
    err_t      : per-recorder summed error (signed)
    """
    rec_coords = np.asarray(rec_coords, dtype=float)
    tdoas      = np.asarray(tdoas, dtype=float)
    src        = np.asarray(x_est, dtype=float)
    c_est      = float(c_est)

    dists = np.linalg.norm(rec_coords - src[None, :], axis=1)
    toas  = dists / c_est

    errors     = np.array([toas[i] - toas[j] - tau_obs
                           for (i, j), tau_obs in zip(pairs, tdoas)],
                          dtype=float)
    error_dict = dict(zip(pairs, errors))

    err_t = np.zeros(rec_coords.shape[0])
    for (i, j), e in error_dict.items():
        err_t[i] += e
        err_t[j] -= e

    # Predicted TDOAs at the estimated position (same sign convention as fcc)
    tdoa_pred = np.array([toas[i] - toas[j] for i, j in pairs], dtype=float)

    return errors, error_dict, err_t, tdoa_pred


def get_toas(rec_coords: np.ndarray, x_est: np.ndarray,
             c_est: float) -> np.ndarray:
    """Return travel time (seconds) from *x_est* to each recorder.

    Parameters
    ----------
    rec_coords : (n, 3) recorder coordinates
    x_est      : (3,) estimated source position
    c_est      : speed of sound (m/s)
    """
    dists = np.linalg.norm(np.asarray(rec_coords) - np.asarray(x_est), axis=1)
    return dists / c_est


def plot_spectrogram(sig: np.ndarray, fs: int, ax=None,
                     title: str = "",
                     fmax: float = 10000.0) -> tuple:
    """Plot a log-power spectrogram on *ax* and return (freqs, times, Sxx).

    Uses gray_r colormap, no axis labels or ticks, frequency clipped to
    *fmax* Hz to focus on the biologically relevant range.

    Parameters
    ----------
    sig   : 1D audio array
    fs    : sample rate (Hz)
    ax    : matplotlib Axes; if None uses current axes
    title : axes title
    fmax  : upper frequency limit in Hz (default 10000)
    """
    import matplotlib.pyplot as plt
    import scipy.signal

    if ax is None:
        ax = plt.gca()
    import warnings as _w

    freqs, times, Sxx = scipy.signal.spectrogram(sig, fs=fs)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        log_Sxx = np.log(Sxx)

    fmask = freqs <= fmax
    log_plot = log_Sxx[fmask]
    extent = (times.min(), times.max(), freqs[fmask].min(), freqs[fmask].max())
    ax.imshow(log_plot, cmap="gray_r", origin="lower",
              aspect="auto", extent=extent, vmin=0,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return freqs, times, Sxx


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

RESULT_COLS = [
    "common_name", "recorder_group", "call_datetime",
    "recorders", "pairs", "quals", "x_est", "c_est", "errors",
    "cost", "optimality",
]


def process_row(i: int):
    """Per-row worker function for the parallel runner.

    Uses globals loaded by init_pool: wf, deployments, candidates, cornellspec.
    Returns a result list (matching RESULT_COLS) or None on failure.
    """
    try:
        row = candidates.iloc[i]  # noqa: F821 — loaded by init_pool
        recorders, datas = get_rowdata(row, wf)  # noqa: F821
        rec_coords = get_recorderlocs(row, deployments, recorders)  # noqa: F821

        # Species frequency template, as figure3_alignment_examples.py does.
        # Noise classes (ENGINE) have no profile; None means "no filter".
        freqfilter = None
        birdcode = getattr(row, "birdcode", None)
        if birdcode is not None and birdcode in cornellspec.index:  # noqa: F821
            freqfilter = cornellspec.loc[birdcode].maxfilt  # noqa: F821

        quals, pairs, ccs, lags, _allquals, _allpairs = get_ccs(
            recorders, datas, freqfilter, rec_coords=rec_coords)
        if len(pairs) < 4:
            return None

        tdoas            = get_tdoas(pairs, lags, ccs)
        x_est, c_est, sol = locate_source_and_speed(rec_coords, pairs, tdoas)
        errors, _, _, _  = compute_tdoa_errors(rec_coords, pairs, tdoas,
                                               x_est, c_est)
        return [
            row.common_name, row.recorder_group, row.call_datetime,
            recorders, pairs, quals, x_est, c_est, errors,
            sol.cost, sol.optimality,
        ]
    except Exception:
        return None


def init_pool(wf_path: str, deployments_path: str,
              candidates_path: str, cornellspec_path: str) -> None:
    """Load shared read-only data into each worker process exactly once.

    Sets module-level globals: wf, deployments, candidates, cornellspec.
    """
    global wf, deployments, candidates, cornellspec
    wf          = pd.read_parquet(wf_path)
    deployments = pd.read_parquet(deployments_path)
    candidates  = pd.read_parquet(candidates_path)
    # The profile table must be a pickle: its meanfilt/maxfilt columns hold
    # FreqFilter objects, which Arrow cannot serialise. freqprofiles.py writes
    # a pickle for exactly that reason, so read_parquet here never matched the
    # artifact the builder produces.
    cornellspec = (pd.read_pickle(cornellspec_path)
                   if str(cornellspec_path).endswith((".pkl", ".pickle"))
                   else pd.read_parquet(cornellspec_path))


def run_parallel(candidates_path: str,
                 wf_path: str,
                 deployments_path: str,
                 cornellspec_path: str,
                 output_path: str,
                 n_workers: int = 8,
                 row_checkpoint: int = 5000,
                 save_interval_s: int = 600) -> pd.DataFrame:
    """Run localization in parallel over all rows in the candidates table.

    Parameters
    ----------
    candidates_path  : parquet path to candidate detections (from birdnet/select.py)
    wf_path          : parquet path to wave-file catalog
    deployments_path : parquet path to deployment table (from catalog/deployments.py)
    cornellspec_path : parquet path to species frequency profile table
    output_path      : where to write the final results parquet
    n_workers        : number of worker processes
    row_checkpoint   : write a partial result every this many rows
    save_interval_s  : also write a partial result every this many seconds

    Returns
    -------
    pd.DataFrame of localization results
    """
    n_rows = len(pd.read_parquet(candidates_path))
    print(f"Localizing {n_rows} candidates with {n_workers} workers ...")

    partial_dir = Path(output_path).parent / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    results    = []
    row_count  = 0
    last_save  = time.time()

    def _save_partial(tag: str) -> None:
        path = partial_dir / f"partial_{tag}.parquet"
        pd.DataFrame(results, columns=RESULT_COLS).to_parquet(path, index=False)
        print(f"[{time.strftime('%H:%M:%S')}] Checkpoint: {len(results)} rows → {path}")

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_pool,
            initargs=(wf_path, deployments_path, candidates_path, cornellspec_path),
        ) as executor:
            for out in executor.map(process_row, range(n_rows), chunksize=250):
                row_count += 1
                if out is not None:
                    results.append(out)

                if time.time() - last_save >= save_interval_s:
                    _save_partial(f"time_{int(time.time())}")
                    last_save = time.time()

                if row_count % row_checkpoint == 0:
                    _save_partial(f"rows_{row_count}")

    finally:
        _save_partial(f"final_{row_count}")

    df = pd.DataFrame(results, columns=RESULT_COLS)
    df.to_parquet(output_path, index=False)
    print(f"Done. {len(df)} localizations written to {output_path}")
    return df
