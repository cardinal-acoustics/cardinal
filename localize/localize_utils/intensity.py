# localize/localize_utils/intensity.py
"""Intensity-vs-distance consistency score for a candidate localization.

A correct source position should show received amplitude falling off with
distance from that position (spherical spreading, amplitude ~ 1/r). This
scores how well the *observed* per-recorder amplitude for a call matches
that expectation for a given x_est, as an independent check alongside the
TDOA geometry score -- independent because it uses signal level, not
arrival-time differences.

Overlapping calls (a second individual or species sounding at the same
time) can contaminate a naive whole-clip amplitude measurement. To reduce
that, amplitude is measured in a short window that has been time-aligned
to the candidate's predicted per-recorder arrival time, not a fixed
absolute window -- an unrelated call would have to happen to fall at the
same *relative* alignment offset in every recorder simultaneously to
contaminate this score the same way it would a naive measurement.

Two entry points, sharing the same falloff fit (`_falloff_score`):

- `intensity_falloff_score` : standalone, post-hoc use -- loads audio from
  disk via localize.run.get_aligned_rowdata. Convenient for scoring rows
  from an already-written results table.
- `intensity_falloff_score_from_datas` : for use *inside* the search
  (localize_plot_withorig), slicing a short aligned window directly out of
  an already-loaded wideband audio buffer instead of hitting disk again --
  lets several candidate x_est solutions for the same event get scored at
  essentially no extra I/O cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from localize import run as bl
from localize.freqprofiles import common_name_to_birdcode


def _falloff_score(amps: dict, dists: dict) -> dict:
    """Fit log(amp) + log(dist) = const (fixed 1/r falloff slope) across
    recorders and score by the residual spread. Only one free parameter
    (the intercept) is fit -- with the ~4-6 recorders typical of an event,
    there isn't enough data to also fit slope and still have a meaningful
    residual, so the physical spherical-spreading slope of -1 is fixed
    rather than estimated.
    """
    valid = [r for r in amps if amps.get(r, 0) > 0 and dists.get(r, 0) > 0]
    if len(valid) < 2:
        return {"score": float("nan"), "n_recorders": len(valid),
                "amps": amps, "dists": dists, "residuals": {}}

    log_resid = {r: float(np.log(amps[r]) + np.log(dists[r])) for r in valid}
    mean_resid = float(np.mean(list(log_resid.values())))
    centered = {r: v - mean_resid for r, v in log_resid.items()}
    score = float(np.std(list(centered.values())))

    return {"score": score, "n_recorders": len(valid),
            "amps": amps, "dists": dists, "residuals": centered}


def _amps_from_filtered(recorders, filtered: dict) -> dict:
    amps = {}
    for r in recorders:
        y = filtered.get(r)
        if y is None or y.size == 0:
            continue
        amps[r] = float(np.max(np.abs(y)))
    return amps


def intensity_falloff_score(
        row,
        wf: pd.DataFrame,
        deployments: pd.DataFrame,
        freqprofiles: pd.DataFrame | None = None,
        win_seconds: float = 0.4,
) -> dict:
    """Score how well observed amplitude falls off with distance from x_est.

    Loads audio from disk (via localize.run.get_aligned_rowdata) -- intended
    for post-hoc scoring of rows already written to a results table. For
    scoring multiple candidates for the same event during the search itself,
    use `intensity_falloff_score_from_datas` instead to avoid repeated I/O.

    Parameters
    ----------
    row          : a result row with fields recorders, x_est, c_est,
                   call_datetime, recorder_group, common_name
    wf           : wave-file catalog (for audio loading)
    deployments  : deployments table (for recorder coordinates)
    freqprofiles : species frequency-profile table (birdcode-indexed, as
                   built by localize.freqprofiles); None to skip bandpass
    win_seconds  : aligned window length in seconds to measure amplitude in

    Returns
    -------
    dict with:
      score        : residual std of log(amp) + log(dist) across recorders
                     under a fixed 1/r falloff model (lower = more
                     consistent with a real source at x_est); NaN if fewer
                     than 2 recorders had usable signal
      n_recorders  : number of recorders with usable (nonzero) amplitude
      amps         : dict recorder -> peak amplitude in the aligned window
      dists        : dict recorder -> distance (m) from x_est
      residuals    : dict recorder -> log(amp) + log(dist) - mean_residual
    """
    recorders = list(row.recorders)
    x_est = np.asarray(row.x_est, dtype=float)
    c_est = float(row.c_est)

    rec_coords = bl.get_recorderlocs(row, deployments, recorders)
    toas = bl.get_toas(rec_coords, x_est, c_est)
    offsets = toas - toas.min()

    dt_in_win = win_seconds / 2.0
    aligned = bl.get_aligned_rowdata(
        row, wf, recorders, offsets, winsize=win_seconds, dt_in_win=dt_in_win,
    )

    freqfilter = None
    if freqprofiles is not None:
        try:
            birdcode = common_name_to_birdcode(row.common_name)
            freqfilter = freqprofiles.loc[birdcode].maxfilt
        except Exception:
            freqfilter = None

    filtered = bl.filter_rowdata(aligned, freqfilter) if freqfilter is not None else aligned

    dists = {r: float(np.linalg.norm(rec_coords[i] - x_est))
             for i, r in enumerate(recorders)}
    amps = _amps_from_filtered(recorders, filtered)

    return _falloff_score(amps, dists)


def _slice_aligned_from_datas(
        datas: dict, recorders: list, offsets: np.ndarray,
        rate: int, dt_in_win_wide: float, win_seconds: float,
) -> dict:
    """Slice a short, TOA-aligned window per recorder out of an
    already-loaded wideband buffer (e.g. localize_plot_withorig's `datas`,
    which spans [call_datetime - dt_in_win_wide, +winsize_wide)).

    Mirrors localize.run.get_aligned_rowdata's windowing convention (short
    window starts at call_datetime - win_seconds/2 + offset) but as pure
    in-memory array slicing -- no disk access.
    """
    dt_in_win_short = win_seconds / 2.0
    win_len = int(round(win_seconds * rate))
    out = {}
    for r, off in zip(recorders, offsets):
        y = datas.get(r)
        if y is None:
            continue
        k = int(round((dt_in_win_wide - dt_in_win_short + float(off)) * rate))
        if k < 0 or k + win_len > len(y):
            continue
        out[r] = y[k:k + win_len]
    return out


def intensity_falloff_score_from_datas(
        datas: dict,
        recorders: list,
        rec_coords: np.ndarray,
        x_est,
        c_est: float,
        rate: int,
        dt_in_win_wide: float,
        win_seconds: float = 0.4,
        freqfilter=None,
) -> dict:
    """Like `intensity_falloff_score`, but scores against an already-loaded
    wideband audio buffer instead of re-reading from disk.

    Intended for use inside the search itself: localize_plot_withorig
    already loads a `winsize`-second buffer (`datas`) per recorder before
    doing any peak-combination scoring, so several candidate x_est
    solutions for the same event can each be scored here with zero
    additional I/O -- only in-memory slicing and a cheap 1/r fit.

    Parameters
    ----------
    datas          : dict recorder -> wideband audio array, as returned by
                     localize.run.get_rowdata
    recorders      : list of recorder IDs (same order as rec_coords)
    rec_coords     : (n, 3) array of recorder coordinates
    x_est, c_est   : candidate source position and speed of sound
    rate           : sample rate of the arrays in `datas`
    dt_in_win_wide : the dt_in_win the wideband buffer was loaded with
                     (i.e. how many seconds before the nominal call time
                     `datas` starts)
    win_seconds    : short aligned window length to measure amplitude in
    freqfilter     : FreqFilter instance or None
    """
    x_est = np.asarray(x_est, dtype=float)
    c_est = float(c_est)

    toas = bl.get_toas(rec_coords, x_est, c_est)
    offsets = toas - toas.min()

    aligned = _slice_aligned_from_datas(datas, recorders, offsets, rate, dt_in_win_wide, win_seconds)
    filtered = bl.filter_rowdata(aligned, freqfilter) if freqfilter is not None else aligned

    dists = {r: float(np.linalg.norm(rec_coords[i] - x_est))
              for i, r in enumerate(recorders)}
    amps = _amps_from_filtered(recorders, filtered)

    return _falloff_score(amps, dists)


def pairwise_amplitude_logdiff(
        datas: dict,
        r_i: str,
        r_j: str,
        tdoa: float,
        rate: int,
        dt_in_win_wide: float,
        win_seconds: float = 0.4,
        freqfilter=None,
) -> float:
    """log(amp_i) - log(amp_j) in a short window aligned to one candidate
    pairwise TDOA -- no position estimate needed.

    `tdoa` is (t_i - t_j) in the sign convention used by localize.tdoa.fcc /
    localize.run.get_tdoas: positive means recorder i receives the signal
    *later* than j. Per locate_source_and_speed's docstring, that arrival
    difference equals exactly (d_i - d_j) / c for any consistent source
    position -- i.e. tdoa > 0 implies i is farther from the source than j,
    regardless of where the source actually is. A real source should
    therefore have the farther (later) recorder be the quieter one, so
    log(amp_i) - log(amp_j) is expected to carry the opposite sign from
    tdoa. This lets a candidate peak's amplitude pattern be checked against
    its own implied geometry before any combinatorial search or position
    solve -- see `amplitude_tdoa_consistency` for turning this into a
    signed consistency score.

    Returns NaN if either recorder's aligned window has no usable signal.
    """
    off_i = max(float(tdoa), 0.0)
    off_j = max(-float(tdoa), 0.0)
    aligned = _slice_aligned_from_datas(
        datas, [r_i, r_j], [off_i, off_j], rate, dt_in_win_wide, win_seconds,
    )
    filtered = bl.filter_rowdata(aligned, freqfilter) if freqfilter is not None else aligned
    amps = _amps_from_filtered([r_i, r_j], filtered)
    if amps.get(r_i, 0) <= 0 or amps.get(r_j, 0) <= 0:
        return float("nan")
    return float(np.log(amps[r_i]) - np.log(amps[r_j]))


def amplitude_tdoa_consistency(tdoa: float, logdiff: float) -> float:
    """Turn a pairwise_amplitude_logdiff result into a signed consistency
    score: positive means the observed amplitude ordering agrees with the
    TDOA-implied distance ordering (farther/later recorder is quieter),
    negative means it disagrees. NaN in, NaN out (unscoreable pair/peak,
    e.g. one recorder had no usable signal in the aligned window --
    treat as unknown, not as a violation, in downstream re-ranking).
    """
    if not np.isfinite(logdiff):
        return float("nan")
    return float(-np.sign(tdoa) * logdiff)
