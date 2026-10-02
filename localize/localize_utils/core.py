# localize/localize_utils/core.py
"""Core TDOA localization module.

Contains `localize_plot_withorig`, the main entry point that runs TDOA-based
acoustic localization for a single detection row: it loads audio, computes
cross-correlation functions, selects peak TDOA candidates, scores them for
triangle consistency, estimates the sound speed, and optionally saves QC plots
(waveforms, CC functions, localization maps, support maps, and audio clips).
Helper utilities for figure saving, module resolution, and image loading are
also defined here.
"""
from __future__ import annotations
import logging
import os
import math
import traceback
import importlib
import re
import subprocess
import tempfile
from itertools import combinations, product
from typing import Any

log = logging.getLogger(__name__)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, hilbert
from scipy.io import wavfile

from localize.localize_utils.intensity import (
    intensity_falloff_score_from_datas,
    pairwise_amplitude_logdiff,
    amplitude_tdoa_consistency,
)


def _resolve_module(mod_or_str: Any):
    """Accept a module/namespace or dotted-path string and return the module."""
    if isinstance(mod_or_str, str):
        return importlib.import_module(mod_or_str)
    return mod_or_str


def _maybe_load_img(img_entry):
    """Accept either a numpy array (already loaded) or a file path string."""
    if isinstance(img_entry, str):
        return matplotlib.pyplot.imread(img_entry)
    return img_entry


def _safe_savefig(path: str, collision_strategy: str) -> str:
    """Save current matplotlib figure honoring collision strategy.

    - 'overwrite': write to the path (clobber if exists)
    - 'skip': if exists, return the existing path (no write)
    - 'unique': add -001, -002, ... before the extension

    Returns the actual path written (or existing path if 'skip').
    """
    base, ext = os.path.splitext(path)
    raster = ext.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    dpi = 150 if raster else None

    if collision_strategy == "overwrite":
        plt.savefig(path, dpi=dpi)
        return path

    if not os.path.exists(path):
        plt.savefig(path, dpi=dpi)
        return path

    if collision_strategy == "skip":
        return path

    i = 1
    while True:
        trial = f"{base}-{i:03d}{ext}"
        if not os.path.exists(trial):
            plt.savefig(trial, dpi=dpi)
            return trial
        i += 1


def _sanitize_filename(s: str) -> str:
    """Make a string safe for use as a filename across platforms."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().replace(" ", "_")
    # Replace anything that's not alnum, underscore, dash, or dot.
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    # Avoid pathological empty names
    return s or "item"


def _write_audio_mp4(
        y: np.ndarray,
        fs: int,
        out_path: str,
        *,
        collision_strategy: str = "unique",
) -> str:
    """Write a 1D mono audio array to an MP4 container (AAC) via ffmpeg.

    Notes
    -----
    - Requires `ffmpeg` to be available on PATH.
    - We write a temporary WAV then transcode to MP4 (AAC).
    """
    # Normalize/convert to int16 WAV for ffmpeg
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)

    # If float, scale to int16 safely
    if np.issubdtype(y.dtype, np.floating):
        ymax = float(np.nanmax(np.abs(y))) if y.size else 0.0
        if not np.isfinite(ymax) or ymax <= 0:
            y_i16 = np.zeros_like(y, dtype=np.int16)
        else:
            y_i16 = np.clip(y / ymax, -1.0, 1.0)
            y_i16 = (y_i16 * 32767.0).astype(np.int16)
    else:
        # If already integer, clamp to int16 range
        y_i16 = np.clip(y.astype(np.int64), -32768, 32767).astype(np.int16)

    # Decide final path honoring collision strategy
    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".mp4":
        out_path = base + ".mp4"

    if collision_strategy == "overwrite":
        final_path = out_path
    elif collision_strategy == "skip" and os.path.exists(out_path):
        return out_path
    elif not os.path.exists(out_path):
        final_path = out_path
    else:
        i = 1
        while True:
            trial = f"{base}-{i:03d}.mp4"
            if not os.path.exists(trial):
                final_path = trial
                break
            i += 1

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    # Write temp wav then transcode
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tmp.wav")
        wavfile.write(wav_path, int(fs), y_i16)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            final_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg not found on PATH; cannot write .mp4 audio clips. "
                "Install ffmpeg or disable save_audio_clips."
            ) from e
        return final_path

def localize_plot_withorig(
        row,
        *,
        # heavy deps: pass module OR dotted path string (recommended for multiprocessing)
        bl_ref,
        wf,
        rec_recgrp_locations,
        cornellspec,
        img_rgb,  # dict[recorder_group] -> ndarray OR filepath
        extent,  # dict[recorder_group] -> [xmin, xmax, ymin, ymax]
        # plotting control
        plot_mode: str = "save",  # "none" | "save" | "show"
        error_mode: str = "log",  # "raise" | "return" | "log"
        save_dir: str = "localization/qcplots",
        save_format: str = "png",  # "png", "pdf", "svg",
        unique_id: str | None = None,  # e.g., call_id for filenames
        collision_strategy: str = "unique",  # "unique" | "overwrite" | "skip"
        plot_all_candidates: bool = False,  # if True, also plot additional candidates beyond the best
        max_candidates_to_plot: int = 0,  # 0 = plot all candidates; otherwise plot up to this many
        candidate_plot_mode: str = "separate",  # "separate" (one file per candidate) | "multipage" (PDF only)
        use_phat: bool = False,
        use_freqfilter: bool = True,
        # Hold the speed of sound at row.speed instead of fitting it. On a
        # near-planar 4-recorder array the (x,y,z,c) Jacobian is singular --
        # 0% of 3- and 4-recorder solves in the 1.06M corpus were identifiable
        # (2026-09-21) -- and dropping c takes that to 86%. Requires a
        # trustworthy row.speed, since a wrong c now biases position.
        fit_speed: bool = True,
        # analysis params
        dt_in_win=1.0,
        winsize: float = 5.0,
        fs: int = 44100,
        qualcut: float = 0.1,
        memory_cap_gb: float = 10.0,
        n_best_combos: int = 1,
        include_toppeak_combo: bool = True,
        include_disjoint_best_combo: bool = False,
        force_toppeak_per_pair: bool = False,
        include_prefer_zeros_candidate: bool = False,
        # optional diagnostic: intensity-vs-distance consistency score
        compute_intensity_scores: bool = False,
        intensity_topk: int = 3,
        intensity_win_seconds: float = 0.4,
        # peak picking controls
        max_peaks_per_pair: int = 6,  # keep up to this many peaks per pair (ranked by height)
        max_pairs: int = 15,  # hard cap on n_pairs before any combinatorial search runs
                              # (see the comment where this is applied for why); validated up
                              # to 15 (a 6-recorder array). Excess pairs are dropped by quality,
                              # keeping the best (lowest qualcut value) max_pairs.
        max_outlier_pairs: int | None = None,  # see docstring: bounded-outlier search mode
        outlier_shallow_cap: int = 2,  # shallow-search depth used alongside max_outlier_pairs
        peak_min_height: float = 0.25,
        peak_min_prominence: float = 0.05,
        peak_min_distance: int = 750,
        fallback_to_max_cc_if_no_peak: bool = False,
        # position-free amplitude-vs-TDOA consistency check (see
        # localize.localize_utils.intensity.pairwise_amplitude_logdiff):
        # demotes candidate peaks whose amplitude pattern contradicts their
        # own implied distance ordering, before the combinatorial search.
        use_amplitude_peak_weighting: bool = False,
        amplitude_consistency_tol: float = 0.1,

        # support-map (time-resolved agreement) panel
        plot_support_map: bool = True,
        support_smooth_ms: float = 10.0,
        support_downsample: int = 400,
        support_use_predicted: bool = False,
        support_line_threshold: float = 0.25,

        # optional audio clip saving
        save_audio_clips: bool = False,
        audio_subdir: str = "Audio",
):
    """
    Localize a source via TDOA peak selection + triangle consistency, estimate c, and
    (optionally) produce QC plots.

    Parameters
    ----------
    bl_ref : module | str
        Either the 'bl' module/namespace itself, or a dotted import path string.
        Using a string is ideal for multiprocessing to avoid pickling large objects.
    plot_mode : {"none","save","show"}
        "none"  = compute only, no figure created.
        "save"  = headless save to PNG.
        "show"  = interactive display.
    include_disjoint_best_combo : bool, optional
        If True, also evaluates the best geometry-score combination that uses no peaks in common with the best combination (if such a combination exists).
    include_prefer_zeros_candidate : bool, optional
        If True, and the best candidate uses peak-index==1 for some pairs, also evaluates the best geometry-score
        candidate among those where those specific pairs are forced to peak-index==0. This is a targeted “use tallest peak
        for the ambiguous pairs” diagnostic.
    max_peaks_per_pair : int, optional
        Maximum number of CC peaks kept per recorder pair (ranked by peak height). Increase this to consider
        more candidate TDOA combinations. Note: the number of combinations grows multiplicatively across pairs
        UNLESS max_outlier_pairs is set (see below), which changes how this search space is built.
    max_outlier_pairs : int, optional
        If set, switches from the default "every pair searches its own top max_peaks_per_pair
        peaks independently" (a search space of max_peaks_per_pair ** n_pairs) to the union of
        two smaller, cheaper searches -- matching the physical expectation that a genuine event
        aligns correctly on *most* pairs, with at most a few thrown off by noise/multipath, but
        allowing for either "several pairs each shallowly off" or "one or two pairs deeply off"
        (they are not the same case -- a single bounded-outlier search over-restricts one of
        them; see the module's git history around 2026-08-21 for a worked example where it
        silently picked a worse-scoring answer than a plain shallow cap):
          (a) shallow: full Cartesian product over every pair's top outlier_shallow_cap peaks
              (cheap when outlier_shallow_cap is small, e.g. 2 -- this alone was previously
              exposed as max_peaks_per_pair without max_outlier_pairs set).
          (b) deep-outlier: baseline is every pair's #1 peak; combos are generated by letting at
              most max_outlier_pairs pairs deviate to any of their max_peaks_per_pair alternates,
              holding every other pair fixed at #1.
        All combos from (a) and (b) are pooled and scored together (some overlap between the two
        sets is expected and harmless). This lets max_peaks_per_pair be set generously (deep
        per-pair search) without the combinatorial blowup of every pair searching deep
        simultaneously, while not losing shallow-multi-pair solutions the way a pure
        deep-outlier-only search would. None (default) preserves the original behavior exactly.
    force_toppeak_per_pair : bool, optional
        If True, skip geometry-based peak-combination search and instead use peak #0 from each pair.
        This is useful for fast debugging and sanity checks.
    fallback_to_max_cc_if_no_peak : bool, optional
        If True, and no CC peaks pass the `find_peaks` criteria for a given pair, fall back to using the
        global maximum of the CC curve for that pair as the sole candidate peak.
    plot_all_candidates : bool, optional
        If True and plot_mode is "save" or "show", generate plots for candidate peak sets in addition to the best.
    max_candidates_to_plot : int, optional
        Maximum number of candidates to plot. 0 means plot all candidates.
    candidate_plot_mode : {"separate","multipage"}, optional
        How to emit candidate plots when plot_mode="save". "multipage" appends pages to a single PDF (requires save_format="pdf").

    save_audio_clips : bool, optional
        If True, save each recorder's 5s audio clip used for localization into `save_dir/<audio_subdir>/`.
    audio_subdir : str, optional
        Subdirectory (within `save_dir`) to place audio clips. Defaults to "Audio".

    Returns
    -------
    dict with keys:
      'common_name','recorder_group','call_datetime',
      'optimality','cost','mean_error',
      'x_est','c_est','errors',
      'tdoa_errors' (dict of per-pair errors, if available),
      'tdoa_errors_by_pair' (dict of per-pair errors derived from the errors vector),
      'tdoa_pred' (list of predicted TDOAs aligned to pairs),
      'figure' (if plot_mode="show"), 'saved_path' (if "save"),
      'candidates' (list of candidate dicts),
      'n_candidates' (int),
      'best_score' (float geometry score)
    Note: When include_disjoint_best_combo is True, the candidates list may also include an additional candidate labeled "best_disjoint" representing the best geometry-score combination with no peaks in common with the best combination, if such a combination exists.
    """
    bl = _resolve_module(bl_ref)

    out = {
        "common_name": getattr(row, "common_name", None),
        "recorder_group": getattr(row, "recorder_group", None),
        "recorders": [],
        "pairs": [],
        "quals": [],
        "call_datetime": getattr(row, "call_datetime", None),
        "optimality": np.nan,
        "cost": np.nan,
        "mean_error": np.nan,
        "x_est": None,
        "c_est": None,
        "errors": None,
        "figure": None,
        "saved_path": None,
        "audio_paths": None,
        "peak_indexes": None,
        "tdoas": None,
        "best_score": np.nan,
        "intensity_score": np.nan,
        "candidates": None,
        "n_candidates": 0,
        "n_peaks_per_pair": None,
        "total_peak_combos": None,
        "combo_sizes": None,
    }
    debug_path = None

    try:
        # ---- Data & metadata
        recorders, datas = bl.get_rowdata(row, wf, dt_in_win=dt_in_win, winsize=winsize)

        # ---- Optional: save 5s audio clips per recorder
        if save_audio_clips and plot_mode in ("save", "show", "none"):
            try:
                rg0 = getattr(row, "recorder_group", None)
                cn0 = getattr(row, "common_name", None)
                # Prefer explicit `unique_id` (often call_id) if provided
                cid0 = unique_id
                if cid0 is None:
                    cid0 = getattr(row, "call_id", None)
                if cid0 is None:
                    cid0 = getattr(row, "id", None)
                if cid0 is None and hasattr(row, "call_datetime"):
                    cid0 = row.call_datetime.strftime("%Y%m%d-%H%M%S")
                cid0 = _sanitize_filename(cid0)
                cn0 = _sanitize_filename(cn0)
                rg0 = _sanitize_filename(rg0)

                audio_dir = os.path.join(save_dir, audio_subdir)
                os.makedirs(audio_dir, exist_ok=True)

                audio_paths = {}
                for rname in recorders:
                    y = datas.get(rname, None)
                    if y is None:
                        continue
                    r_safe = _sanitize_filename(rname)
                    fname = f"{cid0}_{r_safe}.mp4"
                    fpath = os.path.join(audio_dir, fname)
                    audio_paths[rname] = _write_audio_mp4(
                        y,
                        fs,
                        fpath,
                        collision_strategy=collision_strategy,
                    )

                out["audio_paths"] = audio_paths
            except Exception as _e_audio:
                # Don't fail localization just because audio saving failed.
                out["audio_paths"] = {"error": f"{type(_e_audio).__name__}: {_e_audio}"}

        # Ensure we only include recorders that have coordinates available.
        # Some rows can include a recorder with missing location metadata, which
        # can make rec_coords shorter than the recorders list and crash get_ccs.
        rg = getattr(row, "recorder_group", None)

        if rg is not None and rec_recgrp_locations is not None:
            keep_recorders = []
            for r in recorders:
                try:
                    sel = (rec_recgrp_locations.recorder_group == rg) & (rec_recgrp_locations.recorder == r)
                    if np.any(sel):
                        keep_recorders.append(r)
                except Exception:
                    # If rec_recgrp_locations isn't a pandas-like table, fall back to keeping all.
                    keep_recorders = list(recorders)
                    break
            if len(keep_recorders) != len(recorders):
                dropped = sorted(set(recorders) - set(keep_recorders))
                log.warning("[localize_plot_withorig] Dropping recorders with missing coords: %s", dropped)
                recorders = keep_recorders
                datas = {r: datas[r] for r in recorders}

        rec_coords = bl.get_recorderlocs(row, rec_recgrp_locations, recorders)

        freqfilter = None

        if use_freqfilter:
            try:
                freqfilter = cornellspec.loc[row.birdcode].maxfilt
            except:
                freqfilter = None

        if use_freqfilter and freqfilter == None:
            use_phat = True

        # Final sanity check for downstream code (get_ccs indexes rec_coords by recorder index)
        if hasattr(rec_coords, "shape") and rec_coords.shape[0] != len(recorders):
            raise RuntimeError(
                f"Recorder/coord mismatch: len(recorders)={len(recorders)} rec_coords.shape[0]={rec_coords.shape[0]} "
                f"row={getattr(row, 'call_datetime', None)} birdcode={getattr(row, 'birdcode', None)}"
            )

        if freqfilter == None:
            datasf = datas
        else:
            datasf = bl.filter_rowdata(datas, freqfilter)

        quals, pairs, ccs, lags, allquals, allpairs = bl.get_ccs(
            recorders, datas, freqfilter=freqfilter, qualcut=qualcut, rec_coords=rec_coords, use_phat=use_phat
        )

        # Hard cap on n_pairs before any combinatorial search runs. Every
        # search mode here (full/streaming/bounded_outlier) was only ever
        # validated up to n_pairs=15 (a 6-recorder array, C(6,2)=15) -- and
        # critically, the bounded_outlier branch below has no memory-size
        # check of its own (unlike the plain "full" branch's cap_bytes
        # check), so an event with many more active recorders (e.g. a
        # merged multi-site array) can materialize a combo list of size
        # outlier_shallow_cap**n_pairs with no guard at all. Confirmed as
        # the cause of a real system-freezing memory exhaustion incident
        # (2026-08-25, an 18-recorder merged-array test). Keep only the
        # best-quality pairs (lowest qualcut value) up to max_pairs rather
        # than trying to search all of them.
        if len(pairs) > max_pairs:
            order = np.argsort(quals)[:max_pairs]
            pairs = [pairs[i] for i in order]
            quals = [quals[i] for i in order]

        recorders_used = []
        for p in pairs:
            recorders_used.extend([recorders[p[0]], recorders[p[1]]])
        out["recorders"] = sorted(set(recorders_used))

        # ---- Peak detection per pair
        ccpeaks, ccpeakpos = {}, {}
        n_peaks_per_pair = {}
        for pair in pairs:
            cc = ccs[pair];
            lag = lags[pair]
            if cc is None or lag is None or len(cc) == 0 or len(lag) == 0:
                ccpeakpos[pair] = np.array([], dtype=int)
                ccpeaks[pair] = np.array([], dtype=float)
                n_peaks_per_pair[pair] = 0
                continue
            # Find only positive peaks in the signed cross-correlation.
            # We assume physically meaningful TDOAs correspond to positive correlation peaks.
            p_pos, props_pos = find_peaks(
                cc,
                height=peak_min_height,
                prominence=peak_min_prominence,
                distance=peak_min_distance,
            )

            if p_pos.size == 0:
                # Optional fallback: if no peaks pass thresholds, use the global maximum CC sample.
                if fallback_to_max_cc_if_no_peak:
                    try:
                        idx = int(np.nanargmax(cc))
                        if 0 <= idx < len(cc):
                            ccpeakpos[pair] = np.array([idx], dtype=int)
                            ccpeaks[pair] = np.array([lag[idx]], dtype=float)
                            ccpeaks[(pair[1], pair[0])] = -ccpeaks[pair]
                            n_peaks_per_pair[pair] = 1
                            continue
                    except Exception:
                        pass
                ccpeakpos[pair] = np.array([], dtype=int)
                ccpeaks[pair] = np.array([], dtype=float)
                n_peaks_per_pair[pair] = 0
                continue

            peaks = p_pos.astype(int, copy=False)
            heights = props_pos.get("peak_heights", np.ones(peaks.size, dtype=float))

            if use_amplitude_peak_weighting and peaks.size > 1:
                r_i, r_j = recorders[pair[0]], recorders[pair[1]]
                consistency = np.array([
                    amplitude_tdoa_consistency(
                        float(lag[p]),
                        pairwise_amplitude_logdiff(
                            datas, r_i, r_j, float(lag[p]), fs, dt_in_win,
                            win_seconds=intensity_win_seconds, freqfilter=freqfilter,
                        ),
                    )
                    for p in peaks
                ])
                # Demote peaks with a clear amplitude-vs-TDOA violation below
                # every peak that isn't -- within each group, still ranked
                # by CC height. A peak we couldn't score (NaN, e.g. one
                # recorder had no usable signal in the aligned window) stays
                # with the non-violating group rather than being penalized
                # for a measurement that simply wasn't possible.
                demote = np.where(np.isfinite(consistency), consistency < -amplitude_consistency_tol, False)
                order = np.lexsort((-heights, demote.astype(int)))
            else:
                # Sort by height descending
                order = np.argsort(-heights)
            peaks = peaks[order]

            # Keep up to max_peaks_per_pair (or all if <= 0)
            if max_peaks_per_pair is not None and int(max_peaks_per_pair) > 0:
                peaks = peaks[: min(int(max_peaks_per_pair), peaks.size)]

            n_peaks_per_pair[pair] = int(peaks.size)
            ccpeakpos[pair] = peaks
            ccpeaks[pair] = lag[peaks]
            ccpeaks[(pair[1], pair[0])] = -ccpeaks[pair]

        pair_index = {pair: i for i, pair in enumerate(pairs)}

        # ---- Triangle constraints
        triangles = []
        for c in combinations(range(len(recorders)), 3):
            p1, p2, p3 = (c[0], c[1]), (c[1], c[2]), (c[0], c[2])
            if p1 in pair_index and p2 in pair_index and p3 in pair_index:
                arr = np.zeros(len(pairs), dtype=float)
                arr[pair_index[p1]] = 1.0
                arr[pair_index[p2]] = 1.0
                arr[pair_index[p3]] = -1.0
                triangles.append(arr)
        triangles = np.array(triangles, dtype=float)
        if triangles.size == 0:
            raise RuntimeError("No valid recorder triplets to form triangle constraints.")

        # ---- Enumerate peak combos & score (memory-safe)
        peak_lists, peak_id_lists = [], []
        for pair in pairs:
            vals = ccpeaks.get(pair, np.array([], dtype=float))
            if vals.size == 0:
                vals = np.array([0.0], dtype=float)  # fallback
            peak_lists.append(vals)
            peak_id_lists.append(np.arange(vals.size, dtype=int))

        n_pairs = len(pairs)
        combo_sizes = [len(v) for v in peak_lists]
        out["n_peaks_per_pair"] = {str(k): int(v) for k, v in n_peaks_per_pair.items()}
        out["combo_sizes"] = [int(x) for x in combo_sizes]
        try:
            total_combos = int(np.prod(combo_sizes)) if combo_sizes else 0
        except OverflowError:
            total_combos = float("inf")
        out["total_peak_combos"] = int(total_combos) if isinstance(total_combos, int) else total_combos

        # Rough memory estimate for materializing all combos as float32 + int64
        bytes_per_row = (4 * n_pairs) + (8 * n_pairs)
        estimated_bytes = float(total_combos) * bytes_per_row
        cap_bytes = float(memory_cap_gb) * (1024.0 ** 3)
        out["estimated_bytes"] = estimated_bytes

        triangles32 = triangles.astype(np.float32, copy=False)

        def _eval_candidate(candidate_label: str, tdoas_vec: np.ndarray, peak_idx_vec: np.ndarray,
                            geometry_score: float):
            # Solve localization for this candidate
            x_e, c_e, sol_e = bl.locate_source_and_speed(
                rec_coords, pairs, tdoas_vec,
                speed0=row.speed,
                speed_bounds=(row.speed - 2, row.speed + 2),
                fit_speed=fit_speed,
            )
            errs_e, _, _, tdoa_pred_e = bl.compute_tdoa_errors(
                rec_coords, pairs, tdoas_vec, x_e, c_e
            )
            mean_err_e = float(np.mean(np.abs(errs_e))) if errs_e is not None else np.nan
            return {
                "candidate_label": candidate_label,
                "candidate_index": None,  # filled by batch flattener (optional)
                "geometry_score": float(geometry_score),
                "peak_indexes": np.asarray(peak_idx_vec).astype(int).tolist(),
                "tdoas": np.asarray(tdoas_vec).astype(float).tolist(),
                "x_est": np.asarray(x_e).astype(float).tolist(),
                "c_est": float(c_e),
                "optimality": float(getattr(sol_e, "optimality", np.nan)),
                "cost": float(getattr(sol_e, "cost", np.nan)),
                "mean_error": mean_err_e,
                "tdoa_pred": np.asarray(tdoa_pred_e).astype(float).tolist() if tdoa_pred_e is not None else None,
            }

        # Baseline candidate: choose peak #0 for every pair ("top peak" per pair)
        baseline_peak_indexes = np.zeros(n_pairs, dtype=int)
        baseline_tdoas = np.array([peak_lists[i][0] for i in range(n_pairs)], dtype=np.float32)
        baseline_geometry_score = (
            float(np.max(np.abs(baseline_tdoas @ triangles32.T)))
            if triangles32.size else float("inf")
        )

        candidates: list[dict[str, Any]] = []

        # If requested, skip combo search entirely and just use the top peak per pair.
        if force_toppeak_per_pair:
            candidates.append(
                _eval_candidate(
                    candidate_label="toppeak_per_pair",
                    tdoas_vec=baseline_tdoas,
                    peak_idx_vec=baseline_peak_indexes,
                    geometry_score=baseline_geometry_score,
                )
            )
            scores_for_plot = np.array([candidates[0]["geometry_score"]], dtype=np.float32)
        elif max_outlier_pairs is not None:
            out["geom_method"] = "bounded_outlier"
            # Union of two cheap searches -- see the max_outlier_pairs docstring
            # entry for why neither alone is enough:
            #   (a) shallow-full: every pair independently at any of its top
            #       outlier_shallow_cap peaks (catches "several pairs each
            #       shallowly off").
            #   (b) deep-outlier: baseline (every pair at its #1 peak), with at
            #       most max_outlier_pairs pairs allowed to deviate to any of
            #       their max_peaks_per_pair alternates (catches "one or two
            #       pairs deeply off"). Holding every non-outlier pair fixed at
            #       #1 (rather than enumerating their shallow choices too) is
            #       what keeps this polynomial in n_pairs instead of
            #       exponential.
            shallow_sizes = [min(int(outlier_shallow_cap), s) for s in combo_sizes]
            # Defense in depth alongside the max_pairs cap above: this branch
            # has no other size check, and product(*shallow_sizes) can still
            # be enormous if n_pairs is large for any reason (e.g. max_pairs
            # raised by a future caller, or many active recorders). Degrade
            # to shallow_sizes=1 (shallow-full component collapses to just
            # the baseline combo) rather than materializing an uncapped list.
            MAX_SHALLOW_COMBOS = 2_000_000
            if float(np.prod(shallow_sizes)) > MAX_SHALLOW_COMBOS:
                log.warning(
                    "[localize_plot_withorig] shallow-full combo count %.3g exceeds cap "
                    "(%d) with n_pairs=%d -- collapsing to baseline-only for the shallow "
                    "component; deep-outlier component still runs normally",
                    float(np.prod(shallow_sizes)), MAX_SHALLOW_COMBOS, n_pairs,
                )
                shallow_sizes = [1 for _ in shallow_sizes]
            combo_id_rows = [tuple(row) for row in product(*(range(s) for s in shallow_sizes))]

            baseline_ids_tuple = tuple(0 for _ in range(n_pairs))
            if baseline_ids_tuple not in combo_id_rows:
                combo_id_rows.append(baseline_ids_tuple)
            # Hard running stop on total combos, independent of how the
            # theoretical count was reached (unbounded nested Python loop,
            # not a single big allocation -- a pre-computed size check
            # wouldn't catch every way this could still run away).
            MAX_TOTAL_COMBOS = 2_000_000
            _truncated = False
            for k in range(1, max(0, int(max_outlier_pairs)) + 1):
                if _truncated:
                    break
                for outlier_idxs in combinations(range(n_pairs), k):
                    if len(combo_id_rows) >= MAX_TOTAL_COMBOS:
                        log.warning(
                            "[localize_plot_withorig] deep-outlier combo count hit cap "
                            "(%d) with n_pairs=%d, max_outlier_pairs=%d -- truncating "
                            "rather than continuing to enumerate",
                            MAX_TOTAL_COMBOS, n_pairs, max_outlier_pairs,
                        )
                        _truncated = True
                        break
                    alt_ranges = [range(1, combo_sizes[i]) for i in outlier_idxs]
                    for alt_choice in product(*alt_ranges):
                        combo_ids = list(baseline_ids_tuple)
                        for pos, pair_i in enumerate(outlier_idxs):
                            combo_ids[pair_i] = alt_choice[pos]
                        combo_id_rows.append(tuple(combo_ids))

            peak_id_combos = np.array(combo_id_rows, dtype=int)
            peak_combos = np.array(
                [[peak_lists[i][idx] for i, idx in enumerate(combo_ids)] for combo_ids in combo_id_rows],
                dtype=np.float32,
            )
            out["total_peak_combos"] = len(combo_id_rows)

            comps = np.dot(peak_combos, triangles32.T)
            if comps.dtype != np.float32:
                comps = comps.astype(np.float32)
            scores = np.max(np.abs(comps), axis=1)

            order = np.argsort(scores)
            k_top = int(max(1, n_best_combos))
            topk = order[: min(k_top, len(order))]

            for rank, idx in enumerate(topk):
                candidates.append(
                    _eval_candidate(
                        candidate_label=f"rank_{rank}",
                        tdoas_vec=peak_combos[int(idx)],
                        peak_idx_vec=peak_id_combos[int(idx)],
                        geometry_score=float(scores[int(idx)]),
                    )
                )

            if include_disjoint_best_combo and len(order) > 1:
                best_idx = int(order[0])
                best_ids_vec = peak_id_combos[best_idx]
                cand_idx = None
                for j in order[1:]:
                    j = int(j)
                    if np.all(peak_id_combos[j] != best_ids_vec):
                        cand_idx = j
                        break
                if cand_idx is not None:
                    candidates.append(
                        _eval_candidate(
                            candidate_label="best_disjoint",
                            tdoas_vec=peak_combos[cand_idx],
                            peak_idx_vec=peak_id_combos[cand_idx],
                            geometry_score=float(scores[cand_idx]),
                        )
                    )

            if include_toppeak_combo:
                if not (len(topk) > 0 and np.all(peak_id_combos[int(topk[0])] == baseline_peak_indexes)):
                    candidates.append(
                        _eval_candidate(
                            candidate_label="toppeak_per_pair",
                            tdoas_vec=baseline_tdoas,
                            peak_idx_vec=baseline_peak_indexes,
                            geometry_score=baseline_geometry_score,
                        )
                    )

            scores_for_plot = np.array([c["geometry_score"] for c in candidates], dtype=np.float32)
        else:
            if estimated_bytes <= cap_bytes and total_combos > 0:
                out["geom_method"] = "full"
                # Materialize all combinations, score them, then evaluate top-k
                peak_combos = np.array(list(product(*peak_lists)), dtype=np.float32)  # (N, n_pairs)
                peak_id_combos = np.array(list(product(*peak_id_lists)), dtype=int)  # (N, n_pairs)

                comps = np.dot(peak_combos, triangles32.T)
                if comps.dtype != np.float32:
                    comps = comps.astype(np.float32)
                scores = np.max(np.abs(comps), axis=1)  # geometry score per combo

                order = np.argsort(scores)
                k = int(max(1, n_best_combos))
                topk = order[: min(k, len(order))]

                for rank, idx in enumerate(topk):
                    tdoas_vec = peak_combos[int(idx)]
                    peak_idx_vec = peak_id_combos[int(idx)]
                    candidates.append(
                        _eval_candidate(
                            candidate_label=f"rank_{rank}",
                            tdoas_vec=tdoas_vec,
                            peak_idx_vec=peak_idx_vec,
                            geometry_score=float(scores[int(idx)]),
                        )
                    )

                if include_disjoint_best_combo and len(order) > 1:
                    best_idx = int(order[0])
                    best_ids_vec = peak_id_combos[best_idx]
                    cand_idx = None
                    for j in order[1:]:
                        j = int(j)
                        if np.all(peak_id_combos[j] != best_ids_vec):
                            cand_idx = j
                            break
                    if cand_idx is not None:
                        candidates.append(
                            _eval_candidate(
                                candidate_label="best_disjoint",
                                tdoas_vec=peak_combos[cand_idx],
                                peak_idx_vec=peak_id_combos[cand_idx],
                                geometry_score=float(scores[cand_idx]),
                            )
                        )

                # Optional baseline (top peak per pair)
                if include_toppeak_combo:
                    # Avoid duplicates: if rank_0 already equals baseline peak indexes
                    if not (len(topk) > 0 and np.all(peak_id_combos[int(topk[0])] == baseline_peak_indexes)):
                        candidates.append(
                            _eval_candidate(
                                candidate_label="toppeak_per_pair",
                                tdoas_vec=baseline_tdoas,
                                peak_idx_vec=baseline_peak_indexes,
                                geometry_score=baseline_geometry_score,
                            )
                        )

                # For plotting panel: use geometry scores of evaluated candidates
                scores_for_plot = np.array([c["geometry_score"] for c in candidates], dtype=np.float32)

            else:

                out["geom_method"] = "streaming"
                # Streaming mode: we can find only the single best geometry-score combo without materializing.
                best_score = None
                best_tdoas = None
                best_ids = None

                for id_combo in product(*peak_id_lists):
                    tdoas_vec = np.array([peak_lists[i][idx] for i, idx in enumerate(id_combo)], dtype=np.float32)
                    comps_stream = tdoas_vec @ triangles32.T
                    score_stream = float(np.max(np.abs(comps_stream))) if comps_stream.size else float("inf")
                    if (best_score is None) or (score_stream < best_score):
                        best_score = score_stream
                        best_tdoas = tdoas_vec
                        best_ids = np.fromiter(id_combo, dtype=int, count=n_pairs)

                if best_tdoas is None:
                    best_tdoas = np.zeros(n_pairs, dtype=np.float32)
                    best_ids = np.zeros(n_pairs, dtype=int)
                    best_score = float("inf")

                candidates.append(
                    _eval_candidate(
                        candidate_label="rank_0",
                        tdoas_vec=best_tdoas,
                        peak_idx_vec=best_ids,
                        geometry_score=float(best_score),
                    )
                )

                if include_disjoint_best_combo:
                    best2_score = None
                    best2_tdoas = None
                    best2_ids = None
                    for id_combo in product(*peak_id_lists):
                        id_vec = np.fromiter(id_combo, dtype=int, count=n_pairs)
                        if np.any(id_vec == best_ids):
                            continue
                        tdoas_vec = np.array([peak_lists[i][idx] for i, idx in enumerate(id_combo)], dtype=np.float32)
                        comps_stream = tdoas_vec @ triangles32.T
                        score_stream = float(np.max(np.abs(comps_stream))) if comps_stream.size else float("inf")
                        if (best2_score is None) or (score_stream < best2_score):
                            best2_score = score_stream
                            best2_tdoas = tdoas_vec
                            best2_ids = id_vec
                    if best2_tdoas is not None:
                        candidates.append(
                            _eval_candidate(
                                candidate_label="best_disjoint",
                                tdoas_vec=best2_tdoas,
                                peak_idx_vec=best2_ids,
                                geometry_score=float(best2_score),
                            )
                        )

                if include_toppeak_combo:
                    candidates.append(
                        _eval_candidate(
                            candidate_label="toppeak_per_pair",
                            tdoas_vec=baseline_tdoas,
                            peak_idx_vec=baseline_peak_indexes,
                            geometry_score=baseline_geometry_score,
                        )
                    )

                scores_for_plot = np.array([c["geometry_score"] for c in candidates], dtype=np.float32)

        # Sort candidates by geometry score (ascending)
        candidates.sort(key=lambda d: d.get("geometry_score", float("inf")))

        # Use the best candidate for the primary downstream flow (backward compatible)
        best = candidates[0]

        # Optional: constrained alternative candidate.
        # If the best candidate uses peak-index==1 for some pairs, find the best geometry-score
        # candidate among those where those specific pairs are forced to peak-index==0.
        if include_prefer_zeros_candidate:
            try:
                best_ids = np.asarray(best.get("peak_indexes", []), dtype=int)
                if best_ids.size == n_pairs:
                    force_mask = (best_ids == 1)
                    if np.any(force_mask):
                        # --- Materialized path: we have peak_id_combos/peak_combos/scores in scope
                        if "peak_id_combos" in locals() and "peak_combos" in locals() and "scores" in locals():
                            # Filter combos where forced positions are 0
                            ok = np.ones(len(scores), dtype=bool)
                            cols = np.where(force_mask)[0]
                            for col in cols:
                                ok &= (peak_id_combos[:, col] == 0)
                            if np.any(ok):
                                idxs = np.where(ok)[0]
                                # pick the best geometry score among constrained set
                                best_cand_idx = int(idxs[np.argmin(scores[ok])])
                                candidates.append(
                                    _eval_candidate(
                                        candidate_label="prefer_zeros_constrained",
                                        tdoas_vec=peak_combos[best_cand_idx],
                                        peak_idx_vec=peak_id_combos[best_cand_idx],
                                        geometry_score=float(scores[best_cand_idx]),
                                    )
                                )
                        else:
                            # --- Streaming path: re-scan combos with constraint
                            best2_score = None
                            best2_tdoas = None
                            best2_ids = None
                            for id_combo in product(*peak_id_lists):
                                # enforce forced positions == 0
                                violated = False
                                for j in np.where(force_mask)[0]:
                                    if id_combo[j] != 0:
                                        violated = True
                                        break
                                if violated:
                                    continue
                                tdoas_vec = np.array([peak_lists[i][idx] for i, idx in enumerate(id_combo)],
                                                     dtype=np.float32)
                                comps_stream = tdoas_vec @ triangles32.T
                                score_stream = float(np.max(np.abs(comps_stream))) if comps_stream.size else float(
                                    "inf")
                                if (best2_score is None) or (score_stream < best2_score):
                                    best2_score = score_stream
                                    best2_tdoas = tdoas_vec
                                    best2_ids = np.fromiter(id_combo, dtype=int, count=n_pairs)
                            if best2_tdoas is not None:
                                candidates.append(
                                    _eval_candidate(
                                        candidate_label="prefer_zeros_constrained",
                                        tdoas_vec=best2_tdoas,
                                        peak_idx_vec=best2_ids,
                                        geometry_score=float(best2_score),
                                    )
                                )

                        # Re-sort candidates so downstream uses the true best geometry candidate as candidates[0]
                        candidates.sort(key=lambda d: d.get("geometry_score", float("inf")))
                        best = candidates[0]
            except Exception:
                # Never let this optional diagnostic candidate break the run
                pass

        # ---- Optional diagnostic: intensity-vs-distance consistency score.
        # _eval_candidate already solved x_est/c_est for every shortlisted
        # candidate above, so scoring the top few costs no extra TDOA
        # solving -- just an in-memory amplitude/1-r fit against the audio
        # already loaded into `datas`. Purely additive: does not change
        # which candidate is `best` (computed after any prefer_zeros_constrained
        # re-sort above, so it always reflects the final candidate list). The
        # geometry score and this score can disagree on genuinely ambiguous
        # events (see git history around 2026-08-21); this is not yet trusted
        # enough to override the geometry pick on its own, so it's recorded
        # for downstream analysis rather than used to rerank here.
        if compute_intensity_scores:
            for cand in candidates[:max(0, int(intensity_topk))]:
                try:
                    iscore = intensity_falloff_score_from_datas(
                        datas, recorders, rec_coords,
                        cand["x_est"], cand["c_est"],
                        rate=fs, dt_in_win_wide=dt_in_win,
                        win_seconds=intensity_win_seconds,
                        freqfilter=freqfilter,
                    )
                    cand["intensity_score"] = iscore["score"]
                    cand["intensity_n_recorders"] = iscore["n_recorders"]
                except Exception:
                    cand["intensity_score"] = float("nan")
                    cand["intensity_n_recorders"] = 0

        tdoas = np.asarray(best["tdoas"], dtype=np.float32)
        peak_indexes = np.asarray(best["peak_indexes"], dtype=int)

        # For the score panel, we plot the evaluated candidate scores
        scores = scores_for_plot
        best_i = 0

        # ---- Store peak-selection outputs (for downstream analysis)
        out["peak_indexes"] = best["peak_indexes"]
        out["tdoas"] = best["tdoas"]
        out["best_score"] = float(best.get("geometry_score", np.nan))
        out["intensity_score"] = best.get("intensity_score", np.nan)
        out["candidates"] = candidates
        out["n_candidates"] = int(len(candidates))
        out["pairs"] = allpairs
        out["quals"] = allquals
        # The pairs actually solved with, aligned element-for-element with
        # out["tdoas"]. `allpairs` above is every pair get_ccs formed, before
        # the quality cut and the max_pairs trim, so it is longer than the
        # tdoa vector and the mapping between them is not recoverable from
        # the two lists alone. Without this a finished run cannot be re-solved
        # from storage -- changing the speed of sound meant re-running the
        # whole corpus (34 h) when it should have been a few minutes of
        # least-squares on the saved TDOAs.
        out["pairs_used"] = [list(map(int, pr)) for pr in pairs]

        # ---- Solve localization & errors
        x_est, c_est, sol = bl.locate_source_and_speed(
            rec_coords, pairs, tdoas,
            speed0=row.speed, speed_bounds=(row.speed - 2, row.speed + 2),
            fit_speed=fit_speed,
        )
        out["x_est"] = x_est
        out["c_est"] = float(c_est)

        errors, errors_dict, err_t, tdoa_pred = bl.compute_tdoa_errors(
            rec_coords, pairs, tdoas, x_est, c_est
        )
        out["errors"] = errors
        # Expose per-pair TDOA errors for downstream analysis
        # errors: array aligned to `pairs`
        # errors_dict: mapping from pair -> error (if provided by backend)
        out["tdoa_errors"] = (
            {str(k): float(v) for k, v in errors_dict.items()} if isinstance(errors_dict, dict) else None
        )
        out["tdoa_errors_by_pair"] = (
            {str(pairs[i]): float(errors[i]) for i in range(len(pairs))}
            if errors is not None and hasattr(errors, "__len__") and len(errors) == len(pairs)
            else None
        )
        out["tdoa_pred"] = (
            np.asarray(tdoa_pred).astype(float).tolist() if tdoa_pred is not None else None
        )


        toas = bl.get_toas(rec_coords, x_est, c_est)

        # ---- Energy COM (for display alignment if we do plots)
        time_coms = []
        if freqfilter == None:
            datasf = datas
        else:
            datasf = bl.filter_rowdata(datas, freqfilter)  # reuse filtered data
        any_r = next(iter(recorders))
        n0 = len(datasf[any_r])
        t_full = np.arange(n0) / float(fs)
        for r in recorders:
            f = np.abs(datasf[r])
            t = t_full if len(f) == len(t_full) else (np.arange(len(f)) / float(fs))
            envelope = np.abs(hilbert(f))
            E = envelope ** 2
            cumE = np.cumsum(E)
            time_coms.append(0.0 if cumE[-1] == 0 else t[np.searchsorted(cumE, cumE[-1] / 2.0)])
        time_com = float(np.median(time_coms))

        # ---- Sup-title numbers
        optimality = getattr(sol, "optimality", np.nan)
        cost = getattr(sol, "cost", np.nan)
        mean_error = float(np.mean(np.abs(errors))) if errors is not None else np.nan
        out["optimality"] = float(optimality) if np.isfinite(optimality) else np.nan
        out["cost"] = float(cost) if np.isfinite(cost) else np.nan
        out["mean_error"] = mean_error

        def _render_qc_figure(candidate: dict[str, Any], scores_vec: np.ndarray, best_index_for_title: int = 0):
            # Candidate-specific fields
            cand_label = candidate.get("candidate_label", "candidate")
            cand_peak_indexes = np.asarray(candidate.get("peak_indexes", []), dtype=int)
            cand_tdoas = np.asarray(candidate.get("tdoas", []), dtype=np.float32)
            cand_x_est = np.asarray(candidate.get("x_est", x_est), dtype=float)
            cand_c_est = float(candidate.get("c_est", c_est))
            cand_tdoa_pred = candidate.get("tdoa_pred", None)
            if cand_tdoa_pred is None:
                # fall back to recompute if not present
                _, _, _, cand_tdoa_pred = bl.compute_tdoa_errors(rec_coords, pairs, cand_tdoas, cand_x_est, cand_c_est)
            cand_tdoa_pred = np.asarray(cand_tdoa_pred, dtype=float)

            # ---- Prepare aligned data for plotting (use candidate localization)
            cand_toas = bl.get_toas(rec_coords, cand_x_est, cand_c_est)
            datasa_local = bl.get_aligned_rowdata(
                row, wf, recorders, cand_toas - np.min(cand_toas), winsize=winsize, dt_in_win=dt_in_win
            )
            datasaf_local = datasa_local if (freqfilter is None) else bl.filter_rowdata(datasa_local, freqfilter)

            # ---- Build figure (constrained layout; no tight_layout())
            fig = plt.figure(figsize=(16, 10), constrained_layout=False)
            outer = GridSpec(1, 4, width_ratios=[4, 4, 4, 8], figure=fig, wspace=0.3)

            # ===== LEFT COLUMN: scores (wide) + CC minis (2 columns) =====
            left_gs = outer[0].subgridspec(2, 1, height_ratios=[2, 10], hspace=0.2, wspace=0.2)

            ax_scores = fig.add_subplot(left_gs[0])
            ax_scores.plot(np.sort(scores_vec), linewidth=1.2)
            ax_scores.set_title(
                f"CC peak scores ({cand_label})\n{cand_peak_indexes.tolist()}",
                fontsize=8,
            )
            ax_scores.set_xlabel("Combination (sorted)")
            ax_scores.set_ylabel("Max triangle residual", fontsize=8)

            cc_cols = 2
            nleft = len(pairs)
            cc_rows = max(1, math.ceil(nleft / cc_cols))
            cc_gs = left_gs[1].subgridspec(cc_rows, cc_cols, hspace=0.3, wspace=0.2)

            # Shared limits (use normalized CC range you previously set)
            cc_xlim = (-0.5, 0.5)
            ylo, yhi = 0, 1.1

            axes_cc = []
            for i in range(cc_rows * cc_cols):
                r, c = divmod(i, cc_cols)
                ax = fig.add_subplot(cc_gs[r, c])
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.6)
                axes_cc.append(ax)

            for i, pair in enumerate(pairs):
                ax = axes_cc[i]
                lag = lags[pair]
                cc = ccs[pair]
                # Plot signed CC with negative portions in gray for readability
                cc = np.asarray(cc)
                lag = np.asarray(lag)

                cc_pos = cc.astype(float, copy=True)
                cc_pos[cc_pos < 0] = np.nan
                ax.plot(lag, cc_pos, linewidth=0.9, alpha=0.6)

                # cc_neg = cc.astype(float, copy=True)
                # cc_neg[cc_neg >= 0] = np.nan
                # ax.plot(lag, cc_neg, linewidth=0.9, color="0.6")

                peakpos = ccpeakpos.get(pair, np.array([], dtype=int))
                # Plot all detected peaks as small red dots for debugging/visibility
                if peakpos is not None and len(peakpos) > 0:
                    try:
                        peakpos_clip = peakpos[(peakpos >= 0) & (peakpos < len(cc))]
                        if len(peakpos_clip) > 0:
                            ax.plot(lag[peakpos_clip], np.asarray(cc)[peakpos_clip], ".", color="red", ms=3, alpha=0.6)
                    except Exception:
                        pass
                if peakpos.size > 0 and i < len(cand_peak_indexes) and cand_peak_indexes[i] < peakpos.size:
                    px = cand_tdoas[i]
                    idx = peakpos[cand_peak_indexes[i]]
                    if 0 <= idx < len(cc):
                        py = float(np.asarray(cc)[idx])
                        ax.plot(px, py, "x", color="red", mew=1.2, ms=6)

                px_pred = cand_tdoa_pred[i]
                py_pred = 1.0
                ax.plot(px_pred, py_pred, marker="s", mfc="none", mec="green", ms=6, mew=1.0)

                ax.set_xlim(*cc_xlim)
                ax.set_ylim(ylo, yhi)
                ax.set_title(f"{recorders[pair[0]]}\n{recorders[pair[1]]}", fontsize=5, pad=1)

            for j in range(nleft, cc_rows * cc_cols):
                axes_cc[j].set_visible(False)

            # ===== Optional SUPPORT MAP: time-resolved agreement at selected/predicted lag =====
            # This is NOT the global GCC peak. It shows *when* the signals agree at the model/selected lag.
            support_t = None
            support_M = None
            if plot_support_map:
                try:
                    # Using *aligned* signals: expected residual lag between channels is ~0.
                    # Do NOT re-apply cand_tdoas/cand_tdoa_pred here (that would double-shift).
                    lag_vec = np.zeros(len(pairs), dtype=float)

                    # Build a per-recorder envelope from the *aligned, filtered* signals
                    env = {}
                    for rname in recorders:
                        sig = np.asarray(datasaf_local[rname], dtype=float)
                        env[rname] = np.abs(hilbert(sig))

                    # Parameters
                    n_ds = int(support_downsample) if int(support_downsample) > 10 else 200
                    win = int((support_smooth_ms / 1000.0) * float(fs))
                    win = max(1, win)
                    kernel = np.ones(win, dtype=float) / float(win)

                    # Compute support per pair
                    M = np.zeros((len(pairs), n_ds), dtype=float)
                    T = None

                    for pi, pair in enumerate(pairs):
                        r1 = recorders[pair[0]]
                        r2 = recorders[pair[1]]
                        e1 = env[r1]
                        e2 = env[r2]

                        # Aligned signals: compare at ~zero residual shift
                        a = e1
                        b = e2
                        t0 = 0

                        nmin = min(len(a), len(b))
                        if nmin <= 0:
                            continue
                        a = a[:nmin]
                        b = b[:nmin]

                        # Local agreement (product of envelopes), smoothed
                        s = a * b
                        if win > 1 and len(s) >= win:
                            s = np.convolve(s, kernel, mode="same")

                        # Normalize per pair for visual comparability (robust scale)
                        denom = np.percentile(s, 99.0) if np.any(np.isfinite(s)) else 0.0
                        if denom and denom > 0:
                            s = s / denom
                        s = np.clip(s, 0.0, 1.5)

                        # Downsample to n_ds points
                        if len(s) == n_ds:
                            s_ds = s
                        else:
                            idx = np.linspace(0, len(s) - 1, n_ds)
                            s_ds = np.interp(idx, np.arange(len(s)), s)

                        M[pi, :] = s_ds

                        # Time axis (seconds) aligned to original sample index
                        if T is None:
                            # map the downsampled indices back to global time
                            # approximate: (t0 + idx)/fs
                            T = (t0 + np.linspace(0, len(s) - 1, n_ds)) / float(fs)

                    support_t = T
                    support_M = M
                except Exception:
                    support_t = None
                    support_M = None

            # ===== CENTER COLUMNS: unaligned vs aligned spectrograms in register =====
            def _plot_weighted_mean_spectrogram(ax, data_dict, recorder_order, title: str):
                """Plot mean spectrogram of aligned signals, weighting each time-column by mean support."""
                ax.set_xticks([])
                ax.set_yticks([])

                # Build mean waveform across recorders (trim to common length)
                sigs = []
                for rname in recorder_order:
                    try:
                        sigs.append(np.asarray(data_dict[rname], dtype=float))
                    except Exception:
                        pass
                if len(sigs) == 0:
                    ax.set_title(f"{title} (missing)")
                    return

                nmin = int(min(len(s) for s in sigs))
                if nmin <= 0:
                    ax.set_title(f"{title} (empty)")
                    return

                stack = np.vstack([s[:nmin] for s in sigs])
                mean_sig = np.mean(stack, axis=0)

                # Get spectrogram arrays from existing helper, then re-render weighted in B/W
                f, t, Sxx = bl.plot_spectrogram(mean_sig, fs, ax=ax, title=title)
                # Capture the frequency limits chosen by bl.plot_spectrogram so our
                # re-rendered (weighted) image matches the per-recorder panels.
                ylims = ax.get_ylim()

                # If support not available, keep the plain plot
                if not (plot_support_map and support_M is not None and support_t is not None):
                    return

                try:
                    # mean support over pairs -> weight over time
                    w = np.nanmean(support_M, axis=0)
                    if not np.all(np.isfinite(w)):
                        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

                    # Normalize weights to [0,1] for stable visualization
                    wmax = float(np.max(w)) if len(w) else 0.0
                    if wmax > 0:
                        w = w / wmax

                    # Interpolate weights onto spectrogram time axis
                    # support_t and t are both in seconds-from-window-start
                    wt = np.interp(
                        np.asarray(t, dtype=float),
                        np.asarray(support_t, dtype=float),
                        np.asarray(w, dtype=float),
                    )

                    # Avoid washing out: keep a small floor so non-peak regions still show structure
                    wt = np.clip(wt, 0.0, 1.0)
                    weight_floor = 0.05
                    wt = weight_floor + (1.0 - weight_floor) * wt

                    S = np.asarray(Sxx, dtype=float)
                    if S.ndim == 2 and wt.ndim == 1 and S.shape[1] == wt.shape[0]:
                        Sw = S * wt[None, :]
                    else:
                        # Fallback: no weighting if shapes are unexpected
                        return

                    # Robust normalization for display to keep contrast
                    finite = np.isfinite(Sw)
                    if np.any(finite):
                        denom = float(np.nanpercentile(Sw[finite], 99.5))
                        if denom > 0:
                            Sw = Sw / denom
                    Sw = np.clip(Sw, 0.0, 1.2)

                    # Re-render as image (white=low, black=high)
                    ax.clear()
                    ax.imshow(
                        Sw,
                        origin="lower",
                        aspect="auto",
                        cmap="gray_r",
                        vmin=0.0,
                        vmax=1.0,
                        extent=[float(np.min(t)), float(np.max(t)), float(np.min(f)), float(np.max(f))],
                    )
                    ax.set_title(title, fontsize=8)
                    # Re-apply the same frequency axis limits as the recorder spectrogram panels
                    ax.set_ylim(ylims)
                    ax.set_xticks([])
                    ax.set_yticks([])
                except Exception:
                    # Keep unweighted plot if anything goes wrong
                    return

            def _add_dual_spectrogram_columns():
                # One grid per column; both have the same number of rows
                n_rows = len(recorders) + 2

                # Unaligned column (outer[1]): row0 blank, row1 blank, then per-recorder
                un_gs = outer[1].subgridspec(1, 1)[0].subgridspec(n_rows, 1, hspace=0.3)
                # Aligned column (outer[2]): row0 support plot, row1 weighted mean, then per-recorder
                al_gs = outer[2].subgridspec(1, 1)[0].subgridspec(n_rows, 1, hspace=0.3)

                # Sort recorders by TOA so both columns use identical order
                order = np.argsort(cand_toas)
                plt_toas = cand_toas[order]
                plt_recorders = [recorders[i] for i in order]

                # ---- Peak times from mean support over pairs (for vertical reference lines)
                support_peak_times = []
                try:
                    if plot_support_map and support_M is not None and support_t is not None:
                        w = np.nanmean(support_M, axis=0)
                        if not np.all(np.isfinite(w)):
                            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                        w = np.asarray(w, dtype=float)
                        tt = np.asarray(support_t, dtype=float)
                        if w.ndim == 1 and tt.ndim == 1 and w.size == tt.size and w.size >= 3:
                            dt = float(np.median(np.diff(tt))) if tt.size > 1 else 0.0
                            min_sep_s = 0.25
                            min_dist = int(max(1, round(min_sep_s / dt))) if dt > 0 else 1

                            # Apply threshold PER PEAK using height parameter
                            pk, props = find_peaks(
                                w,
                                height=support_line_threshold,
                                distance=min_dist,
                            )

                            if pk.size > 0:
                                heights = props.get("peak_heights", None)
                                if heights is None:
                                    heights = w[pk]
                                pk = pk[np.argsort(-heights)]
                                pk = pk[:5]
                                support_peak_times = [float(tt[i]) for i in pk]
                except Exception:
                    support_peak_times = []

                def _add_support_peak_vlines(ax):
                    """Draw vertical lines at support peak times (aligned column only)."""
                    if support_peak_times:
                        for tpk in support_peak_times:
                            ax.axvline(tpk, linewidth=0.8, alpha=0.8)

                # --- Row 0
                ax_un0 = fig.add_subplot(un_gs[0, 0])
                ax_un0.set_visible(False)

                ax_al0 = fig.add_subplot(al_gs[0, 0])
                ax_al0.set_xticks([]);
                ax_al0.set_yticks([])
                if plot_support_map and support_M is not None and support_t is not None:
                    ax_al0.imshow(
                        support_M,
                        aspect="auto",
                        origin="lower",
                        cmap="gray_r",
                        extent=[float(support_t[0]), float(support_t[-1]), 0, float(len(pairs))],
                    )
                    # _add_support_peak_vlines(ax_al0)  # (REMOVED: no vlines in support panel)
                    ax_al0.set_title("Support at ~0 lag (pairs × time)", fontsize=8)
                    # ax_al0.set_xlabel("time (s)", fontsize=7)  # Removed as requested
                    ax_al0.set_yticks([])
                else:
                    ax_al0.set_title("Support (unavailable)", fontsize=8)

                # --- Row 1
                ax_un1 = fig.add_subplot(un_gs[1, 0])
                ax_un1.set_visible(False)

                ax_al1 = fig.add_subplot(al_gs[1, 0])
                _plot_weighted_mean_spectrogram(
                    ax_al1,
                    datasaf_local,
                    plt_recorders,
                    title=f"Aligned MEAN (weighted) (n={len(plt_recorders)})",
                )
                _add_support_peak_vlines(ax_al1)

                # --- Rows 2..: per-recorder spectrograms
                for i, rname in enumerate(plt_recorders):
                    rowi = i + 2

                    ax_u = fig.add_subplot(un_gs[rowi, 0])
                    ax_u.set_xticks([]);
                    ax_u.set_yticks([])
                    _ = bl.plot_spectrogram(
                        datasf[rname], fs, ax=ax_u,
                        title=f"{rname}  {plt_toas[i]:.3f}s  {cand_c_est * plt_toas[i]:.1f}m",
                    )

                    ax_a = fig.add_subplot(al_gs[rowi, 0])
                    ax_a.set_xticks([]);
                    ax_a.set_yticks([])
                    _ = bl.plot_spectrogram(
                        datasaf_local[rname], fs, ax=ax_a,
                        title=f"Aligned {rname}  {plt_toas[i]:.3f}s  {cand_c_est * plt_toas[i]:.1f}m",
                    )
                    _add_support_peak_vlines(ax_a)

            _add_dual_spectrogram_columns()

            rg_local = row.recorder_group
            # ---- Map panel (drone-orthomosaic background + elevation profiles).
            # Requires img_rgb/extent keyed by recorder_group, AND rec_recgrp_locations
            # to carry 'source'/'geometry'/'height' columns -- a schema that predates
            # the current deployments.parquet (recorder/recorder_group/begin/end/
            # longitude/latitude/z). Guarded rather than assumed: callers that only
            # want the spectrogram/alignment panels (e.g. img_rgb={}) skip this
            # cleanly instead of crashing on a missing key or attribute.
            try:
                if rg_local not in img_rgb or rg_local not in extent:
                    raise KeyError(rg_local)
                image_gs = outer[3].subgridspec(1, 1)
                im_gs = image_gs[0].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.28)

                ax = fig.add_subplot(im_gs[0])
                bg = _maybe_load_img(img_rgb[rg_local])
                ax.imshow(bg, extent=extent[rg_local], aspect="auto")
                recplts = rec_recgrp_locations[
                    (rec_recgrp_locations.recorder_group == rg_local) & (rec_recgrp_locations.source == "dd")
                    ]
                recplts.plot(ax=ax, color="yellow", markersize=5)
                ax.scatter(cand_x_est[0], cand_x_est[1], c="red", alpha=0.75)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = fig.add_subplot(im_gs[1])
                x = recplts.geometry.x
                z = recplts.height
                ax.scatter(x, z)
                ax.scatter(cand_x_est[0], cand_x_est[2], c="red", alpha=0.75)
                ax.set_xticks([])
                ax.set_xlim(extent[rg_local][0], extent[rg_local][1])
                ax.set_ylim(float(np.min(z)), float(np.max(z)) + 50)

                ax = fig.add_subplot(im_gs[2])
                y = recplts.geometry.y
                ax.scatter(y, z)
                ax.scatter(cand_x_est[1], cand_x_est[2], c="red", alpha=0.75)
                ax.set_xlim(extent[rg_local][2], extent[rg_local][3])
                ax.set_ylim(float(np.min(z)), float(np.max(z)) + 50)
                ax.set_xticks([])
            except Exception as e:
                log.debug("Skipping map panel for %s: %s", rg_local, e)

            fig.suptitle(
                f"{out['common_name']}  {out['recorder_group']}  {out['call_datetime']}\n"
                f"candidate={cand_label}  geom={float(candidate.get('geometry_score', np.nan)):.5f}  "
                f"cost={float(candidate.get('cost', np.nan)):.2f}  opt={float(candidate.get('optimality', np.nan)):.5f}  "
                f"mean_error={float(candidate.get('mean_error', np.nan)):.4f}\n"
                f"x={cand_x_est[0]:.1f} y={cand_x_est[1]:.1f} z={cand_x_est[2]:.1f}"
            )
            return fig

        # ---- Early exit if no plots requested
        if plot_mode == "none":
            return out

        # Decide which candidates to plot
        cand_list = candidates
        if not plot_all_candidates:
            cand_list = [candidates[0]]
        if max_candidates_to_plot and int(max_candidates_to_plot) > 0:
            cand_list = cand_list[: int(max_candidates_to_plot)]

        if plot_mode == "show":
            # Show only the first candidate to avoid spamming windows
            fig = _render_qc_figure(cand_list[0], scores)
            out["figure"] = fig
            plt.show()
            return out

        # plot_mode == "save"
        os.makedirs(save_dir, exist_ok=True)
        # Use call_id (via unique_id if provided) as the sole filename stem
        ext = save_format.lower().lstrip(".")
        call_id = unique_id
        if call_id is None:
            call_id = getattr(row, "call_id", None)
        if call_id is None:
            call_id = getattr(row, "id", None)
        if call_id is None and hasattr(row, "call_datetime"):
            call_id = row.call_datetime.strftime("%Y%m%d-%H%M%S")
        call_id = _sanitize_filename(call_id)
        base = f"{call_id}"

        if candidate_plot_mode == "multipage" and ext == "pdf" and len(cand_list) > 1:
            path = os.path.join(save_dir, f"{base}.pdf")
            # collision strategy handled by writing to a unique path first
            base_path = _safe_savefig(path, collision_strategy=collision_strategy)  # reserve path
            # overwrite reserved file with multipage
            with PdfPages(base_path) as pdf:
                for cand in cand_list:
                    fig = _render_qc_figure(cand, scores)
                    pdf.savefig(fig)
                    plt.close(fig)
            out["saved_path"] = base_path
        else:
            saved_paths = []
            for idx, cand in enumerate(cand_list):
                label = str(cand.get("candidate_label", f"cand{idx}"))
                # sanitize label for filesystem
                safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
                fname = f"{base}_{idx:02d}_{safe_label}.{ext}" if plot_all_candidates else f"{base}.{ext}"
                path = os.path.join(save_dir, fname)
                fig = _render_qc_figure(cand, scores)
                saved_paths.append(_safe_savefig(path, collision_strategy=collision_strategy))
                plt.close(fig)
            out["saved_path"] = saved_paths[0] if len(saved_paths) == 1 else saved_paths

        return out

    except Exception as e:

        msg = f"localize_plot_withorig failed: {type(e).__name__}: {e}"
        if debug_path is not None:
            msg = f"{msg} (ccgrid={debug_path})"

        if error_mode == "raise":
            raise
        elif error_mode == "return":
            return {"error": msg, "ccgrid": debug_path}
        else:  # "log" (default)
            log.info(msg)
            return {"error": msg, "ccgrid": debug_path}
