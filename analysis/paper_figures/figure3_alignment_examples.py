#!/usr/bin/env python3
"""figure3_alignment_examples.py — TOA-aligned spectrograms across a range of
localization quality (Figure 3).

For each example call: loads audio for its recorders, computes each
recorder's predicted time-of-arrival offset from the already-localized
x_est/c_est, loads clips pre-shifted to align on the call (via
localize.run.get_aligned_rowdata -- the same primitive the main pipeline
uses), and plots the aligned per-recorder spectrograms plus their mean.
Deliberately does NOT re-run the full localize_plot_withorig combinatorial
search or its QC-figure rendering -- this only needs the alignment, not a
fresh geometry solve.

Usage:
    python analysis/paper_figures/figure3_alignment_examples.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("CARDINAL_CONFIG", "sites/ferguson.toml")

import localize.run as bl

RESULTS_PATH = "localization/results/localize_parallel.parquet"
CANDIDATES_PATH = "birdnet/candidates.parquet"
CATALOG_PATH = "wavfiles.parquet"
DEPLOYMENTS_PATH = "data/deployments.parquet"
FREQPROFILES_PATH = "localization/inputs/freqprofiles.pkl"
FS = 44100
WINSIZE = 5.0
DT_IN_WIN = 1.0
OUT_DIR = os.path.dirname(__file__)

# --- results set selection -------------------------------------------------
# Default reproduces the published figure. Pass --results to render the same
# figure from a different localization run (e.g. the fixed-c corpus); outputs
# get a suffix derived from that run's directory name, so nothing is
# overwritten. --suffix overrides the derived one.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--results", default=RESULTS_PATH,
                 help="localize_parallel.parquet to plot (default: %(default)s)")
_ap.add_argument("--suffix", default=None,
                 help="appended to output filenames; default is derived from "
                      "the results directory when --results is non-default")
_args, _ = _ap.parse_known_args()
if _args.results != RESULTS_PATH:
    RESULTS_PATH = _args.results
    _derived = os.path.basename(os.path.dirname(RESULTS_PATH))
    FIG_SUFFIX = _args.suffix if _args.suffix is not None else f"_{_derived}"
else:
    FIG_SUFFIX = _args.suffix or ""
# ---------------------------------------------------------------------------


# Temporal-support panel params, matching localize_plot_withorig's defaults
# (localize/localize_utils/core.py: support_smooth_ms, support_downsample,
# support_line_threshold).
SUPPORT_SMOOTH_MS = 10.0
SUPPORT_DOWNSAMPLE = 400
SUPPORT_LINE_THRESHOLD = 0.25
SUPPORT_MIN_PEAK_SEP_S = 0.25
SUPPORT_MAX_PEAKS = 5


def compute_temporal_support(datas, recorders, fs=FS):
    """Time-resolved cross-recorder agreement at ~0 residual lag (envelope
    product per pair, smoothed/normalized/downsampled), matching
    localize_plot_withorig's support-map computation. Returns
    (support_t, support_M) or (None, None) if unavailable.
    """
    pairs = list(combinations(recorders, 2))
    env = {}
    for rname in recorders:
        if rname in datas:
            env[rname] = np.abs(hilbert(np.asarray(datas[rname], dtype=float)))

    n_ds = SUPPORT_DOWNSAMPLE
    win = max(1, int((SUPPORT_SMOOTH_MS / 1000.0) * fs))
    kernel = np.ones(win, dtype=float) / float(win)

    M = np.zeros((len(pairs), n_ds), dtype=float)
    T = None
    for pi, (r1, r2) in enumerate(pairs):
        if r1 not in env or r2 not in env:
            continue
        a, b = env[r1], env[r2]
        nmin = min(len(a), len(b))
        if nmin <= 0:
            continue
        a, b = a[:nmin], b[:nmin]

        s = a * b
        if win > 1 and len(s) >= win:
            s = np.convolve(s, kernel, mode="same")

        denom = np.percentile(s, 99.0) if np.any(np.isfinite(s)) else 0.0
        if denom and denom > 0:
            s = s / denom
        s = np.clip(s, 0.0, 1.5)

        idx = np.linspace(0, len(s) - 1, n_ds)
        s_ds = np.interp(idx, np.arange(len(s)), s)
        M[pi, :] = s_ds

        if T is None:
            T = np.linspace(0, len(s) - 1, n_ds) / float(fs)

    return T, M


def support_peak_times(support_t, support_M):
    """Top (up to SUPPORT_MAX_PEAKS) peak times of mean cross-pair support,
    matching localize_plot_withorig's vertical-line selection."""
    if support_t is None or support_M is None:
        return []
    w = np.nanmean(support_M, axis=0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    tt = np.asarray(support_t, dtype=float)
    if w.size < 3:
        return []
    dt = float(np.median(np.diff(tt))) if tt.size > 1 else 0.0
    min_dist = int(max(1, round(SUPPORT_MIN_PEAK_SEP_S / dt))) if dt > 0 else 1
    pk, props = find_peaks(w, height=SUPPORT_LINE_THRESHOLD, distance=min_dist)
    if pk.size == 0:
        return []
    heights = props.get("peak_heights", w[pk])
    pk = pk[np.argsort(-heights)][:SUPPORT_MAX_PEAKS]
    return [float(tt[i]) for i in pk]

# Same 5 Back9 Indigo Bunting examples spanning the 2nd/25th/50th/85th/99th
# percentile of mean_error within the full 6-recorder-array subset.
CALL_IDS = [
    "INDIGOBUNTING_Back9_20240726-152833",
    "INDIGOBUNTING_Back9_20240809-185418",
    "INDIGOBUNTING_Back9_20240729-124012",
    "INDIGOBUNTING_Back9_20240804-113500",
    "INDIGOBUNTING_Back9_20240725-123033",
]


def build_panel(fig, gs_col, call_id, row, wf, deployments, cornellspec, mean_error):
    recorders = list(row["recorders"])
    x_est = np.asarray(row["x_est"], dtype=float)
    c_est = float(row["c_est"])

    rec_coords = bl.get_recorderlocs(row, deployments, recorders)
    toas = bl.get_toas(rec_coords, x_est, c_est)
    offsets = toas - np.min(toas)

    datas = bl.get_aligned_rowdata(row, wf, recorders, offsets, winsize=WINSIZE, dt_in_win=DT_IN_WIN)

    freqfilter = None
    birdcode = getattr(row, "birdcode", None)
    if birdcode is not None and birdcode in cornellspec.index:
        freqfilter = cornellspec.loc[birdcode].maxfilt
    if freqfilter is not None:
        datas = bl.filter_rowdata(datas, freqfilter)

    order = np.argsort(offsets)
    ordered_recorders = [recorders[i] for i in order]

    support_t, support_M = compute_temporal_support(datas, recorders, fs=FS)
    peak_times = support_peak_times(support_t, support_M)

    n_rows = len(ordered_recorders) + 2  # support map + mean + per-recorder
    inner = gs_col.subgridspec(n_rows, 1, hspace=0.35)

    ax_support = fig.add_subplot(inner[0])
    ax_support.set_xticks([]); ax_support.set_yticks([])
    if support_t is not None and support_M is not None:
        ax_support.imshow(
            support_M, aspect="auto", origin="lower", cmap="gray_r",
            extent=[float(support_t[0]), float(support_t[-1]), 0, support_M.shape[0]],
        )
        ax_support.set_title("Support at ~0 lag (pairs x time)", fontsize=8)
    else:
        ax_support.set_title("Support (unavailable)", fontsize=8)

    def _add_vlines(ax):
        for tpk in peak_times:
            ax.axvline(tpk, linewidth=0.8, alpha=0.8, color="tab:blue")

    sigs = []
    for i, rname in enumerate(ordered_recorders):
        ax = fig.add_subplot(inner[i + 2])
        if rname in datas:
            sig = np.asarray(datas[rname], dtype=float)
            sigs.append(sig)
            dist_m = c_est * offsets[order[i]]
            bl.plot_spectrogram(sig, FS, ax=ax, title=f"{rname}  {dist_m:.0f}m")
            _add_vlines(ax)
        else:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{rname} (missing)", fontsize=8)

    ax_mean = fig.add_subplot(inner[1])
    if sigs:
        nmin = min(len(s) for s in sigs)
        mean_sig = np.mean(np.vstack([s[:nmin] for s in sigs]), axis=0)
        bl.plot_spectrogram(mean_sig, FS, ax=ax_mean, title=f"mean_error={mean_error * 1000:.1f}ms")
        _add_vlines(ax_mean)
    ax_mean.title.set_fontweight("bold")


def main() -> None:
    res = pd.read_parquet(RESULTS_PATH,
                           columns=["call_id", "recorders", "x_est", "c_est", "mean_error"])
    res = res[res["call_id"].isin(CALL_IDS)].set_index("call_id")

    cands = pd.read_parquet(CANDIDATES_PATH)
    cands = cands[cands["call_id"].isin(CALL_IDS)].set_index("call_id")

    wf = pd.read_parquet(CATALOG_PATH)
    wf = wf[wf["recorder_group"] == "Back9"]
    wf = wf[wf["recorder_type"] == "SOLARBAR"]
    deployments = pd.read_parquet(DEPLOYMENTS_PATH)
    cornellspec = pd.read_pickle(FREQPROFILES_PATH)

    fig = plt.figure(figsize=(3.2 * len(CALL_IDS), 10))
    outer = fig.add_gridspec(1, len(CALL_IDS), wspace=0.3)

    for col, call_id in enumerate(CALL_IDS):
        row = cands.loc[call_id]
        mean_error = float(res.loc[call_id, "mean_error"])
        row = row.copy()
        row["recorders"] = res.loc[call_id, "recorders"]
        row["x_est"] = res.loc[call_id, "x_est"]
        row["c_est"] = res.loc[call_id, "c_est"]
        build_panel(fig, outer[col], call_id, row, wf, deployments, cornellspec, mean_error)

    fig.suptitle("Figure 3. TOA-aligned spectrograms across a range of localization quality\n"
                 "(Indigo Bunting, Back9; per-recorder spectrograms sorted by predicted arrival "
                 "time, distance to estimated source shown; top row = weighted mean)", fontsize=10)
    fig.subplots_adjust(top=0.88)

    out_png = os.path.join(OUT_DIR, f"figure3_alignment_examples{FIG_SUFFIX}.png")
    out_pdf = os.path.join(OUT_DIR, f"figure3_alignment_examples{FIG_SUFFIX}.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
