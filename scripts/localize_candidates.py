#!/usr/bin/env python3
"""
localize_candidates.py — run TDOA localization over a candidates table in
parallel, with resumable checkpointing.

Usage:
    export CARDINAL_CONFIG=/path/to/Cardinal/sites/ferguson.toml

    # Calibration run first -- measure throughput on a small sample before
    # committing to the full set. Watch CPU (Activity Monitor) and disk
    # activity on your recording drives while this runs; if disk I/O looks
    # saturated well before CPU does, drop --n-workers.
    python scripts/localize_candidates.py \
        --candidates birdnet/candidates.parquet \
        --freqprofiles localization/inputs/freqprofiles.pkl \
        --deployments data/deployments.parquet \
        --catalog wavfiles.parquet --groups Back9 House NorthSide EastCreek \
        --recorder-types SOLARBAR \
        --out-dir localization/results/four_arrays_solarbar \
        --n-workers 16 --limit 200

    # Full run (safe to re-run after interruption -- resumes from checkpoint)
    python scripts/localize_candidates.py \
        --candidates birdnet/candidates.parquet \
        --freqprofiles localization/inputs/freqprofiles.pkl \
        --deployments data/deployments.parquet \
        --catalog wavfiles.parquet --groups Back9 House NorthSide EastCreek \
        --recorder-types SOLARBAR \
        --out-dir localization/results/four_arrays_solarbar \
        --n-workers 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from localize.localize_utils.batch import run_localizations_parallel

log = logging.getLogger(__name__)



def _record_invocation(args) -> None:
    """Write the exact invocation next to the results.

    A finished run used to record nothing about how it was produced -- not in
    the log, not in the parquet metadata. Reconstructing the flags for the
    2026-09-22 corpus run took ten trial runs, and the one that mattered was a
    flag that had NOT been passed (--recorder-types), which no amount of
    staring at the output would have revealed. Cheap insurance.
    """
    import json
    import os
    import shlex
    import subprocess
    import sys
    from datetime import datetime

    os.makedirs(args.out_dir, exist_ok=True)
    rec = {
        "started": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
    }
    try:
        rec["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10).stdout.strip() or None
        rec["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=10).stdout.strip())
    except Exception:
        pass
    path = os.path.join(args.out_dir, "run_invocation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"Invocation recorded: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True,
                        help="candidates parquet with a 'speed' column "
                             "(see scripts/prepare_localization_inputs.py)")
    parser.add_argument("--freqprofiles", required=True)
    parser.add_argument("--deployments", default="data/deployments.parquet")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--groups", nargs="*", default=None, metavar="GROUP",
                        help="restrict the catalog used for audio lookup to these groups "
                             "(should match how candidates were selected)")
    parser.add_argument("--recorder-types", nargs="*", default=None, metavar="TYPE",
                        help="restrict the catalog to these recorder types, e.g. SOLARBAR -- "
                             "important: without this, a recorder_group with a mix of "
                             "recorder types (e.g. Solarbar + S4A) will pull in audio from "
                             "types you didn't intend to localize with")
    parser.add_argument("--out-dir", default="localization/results")
    parser.add_argument("--winsize", type=float, default=None,
                        help="total audio clip length (seconds) loaded per recorder for TDOA "
                             "estimation (default: localize_plot_withorig's own default of "
                             "5.0s, sized for closely-spaced (~35m) arrays where max TDOA is "
                             "well under 1s). Widely-separated recorders need more: at 343 "
                             "m/s, a 1km separation implies TDOAs up to ~2.9s, so the clip must "
                             "extend far enough on both sides of the nominal call window to "
                             "contain the shifted call for the most separated pair, or the "
                             "cross-correlation simply never sees the true peak.")
    parser.add_argument("--dt-in-win", type=float, default=None,
                        help="seconds of padding before the nominal call window where the "
                             "clip starts (default: localize_plot_withorig's own default of "
                             "1.0s). Increase alongside --winsize for widely-separated "
                             "recorders -- see --winsize.")
    parser.add_argument("--qualcut", type=float, default=None,
                        help="keep a recorder pair only if its CC quality (std of |cc|) is "
                             "<= this value; lower = stricter/cleaner-only pairs kept, more "
                             "pairs dropped as low-SNR (default: localize_plot_withorig's own "
                             "default of 0.1).")
    parser.add_argument("--n-workers", type=int, default=16,
                        help="parallel worker processes. Confirmed empirically: going "
                             "meaningfully above the machine's physical core count (%d "
                             "here) causes severe contention, not just reduced throughput "
                             "-- a batch of candidates that ran fine at 16 workers hit mass "
                             "TimeoutErrors at 32 on this 24-core machine, even though each "
                             "task took under a second in isolation. Stay at or below core "
                             "count; tune down further based on a --limit calibration run."
                             % __import__("os").cpu_count())
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N candidates (for calibration runs)")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--max-peaks-per-pair", type=int, default=None,
                        help="cap on candidate cross-correlation peaks kept per recorder "
                             "pair before the peak-combination search runs (default: "
                             "localize_plot_withorig's own default of 6). Lower this to "
                             "bound the combinatorial search for highly-concordant events "
                             "(more recorders -> more pairs -> more combinations).")
    parser.add_argument("--fixed-speed", action="store_true",
                        help="hold the speed of sound at the candidate's own "
                             "temperature-derived 'speed' instead of fitting it. "
                             "On a near-planar 4-recorder array the (x,y,z,c) "
                             "Jacobian is singular -- 0%% of 3- and 4-recorder "
                             "solves in the 1.06M corpus were identifiable, 77.7%% "
                             "of the archive -- and dropping c takes 4-recorder "
                             "solves to 86%% identifiable. Only use with a speed "
                             "you trust: a wrong c now biases position directly.")
    parser.add_argument("--max-outlier-pairs", type=int, default=None,
                        help="switch to the combined shallow-full + bounded-outlier peak "
                             "search (see localize_plot_withorig's docstring): most pairs are "
                             "searched shallowly (--outlier-shallow-cap), while up to this "
                             "many pairs may additionally search deeply (--max-peaks-per-pair). "
                             "Recommended over a plain --max-peaks-per-pair cap when that alone "
                             "either misses real solutions (cap too shallow) or blows up "
                             "combinatorially / times out (cap too deep with many recorders).")
    parser.add_argument("--outlier-shallow-cap", type=int, default=2,
                        help="shallow per-pair search depth used by --max-outlier-pairs "
                             "(default 2; ignored unless --max-outlier-pairs is set)")
    parser.add_argument("--memory-cap-gb", type=float, default=1.0,
                        help="localize_plot_withorig's own per-task memory cap for "
                             "materializing the full peak-combination search (its default "
                             "is 10 GB -- with N parallel workers that's a theoretical "
                             "worst case of N*10 GB if several tasks hit large combinatorial "
                             "cases at once, which is what caused a real ~192GB RAM/swap "
                             "exhaustion on a 32-worker run. Default here (1 GB) keeps the "
                             "worst case at n_workers GB; scale with --n-workers and your "
                             "available RAM, not just left at localize_plot_withorig's default.")
    parser.add_argument("--task-timeout", type=float, default=120.0,
                        help="max seconds to wait for a single candidate's localization "
                             "before recording it as a TimeoutError and moving on "
                             "(default 120s; set to 0 to disable)")
    parser.add_argument("--no-intensity-scores", action="store_true",
                        help="skip the intensity-vs-distance diagnostic score (on by default). "
                             "A permutation-null test (2026-08-21) confirmed the real amplitude-"
                             "to-distance pairing scores meaningfully better than random "
                             "reassignments of the same amplitudes across the same distances "
                             "(~65%% of events beat the null median vs 50%% under no signal) -- "
                             "real but currently noisy signal, not yet used to override the "
                             "geometry-based pick, but persisted on every row for future use.")
    parser.add_argument("--intensity-topk", type=int, default=3,
                        help="number of top geometry-ranked candidates to also score for "
                             "intensity-vs-distance consistency (default 3; only matters if "
                             "intensity scoring is enabled)")
    parser.add_argument("--intensity-win-seconds", type=float, default=0.4,
                        help="aligned window length (seconds) used to measure per-recorder "
                             "amplitude for the intensity score (default 0.4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    candidates = pd.read_parquet(args.candidates)
    if "speed" not in candidates.columns:
        raise SystemExit(
            "candidates table has no 'speed' column -- run "
            "scripts/prepare_localization_inputs.py first"
        )
    if args.limit:
        candidates = candidates.head(args.limit)
    log.info("Localizing %d candidates", len(candidates))

    wf = pd.read_parquet(args.catalog)
    if args.groups:
        wf = wf[wf["recorder_group"].isin(args.groups)]
    if args.recorder_types:
        wf = wf[wf["recorder_type"].isin(args.recorder_types)]
    log.info("Catalog scoped to %d files, %d recorders", len(wf), wf["recorder"].nunique())

    deployments = pd.read_parquet(args.deployments)
    cornellspec = pd.read_pickle(args.freqprofiles)

    from localize.localize_utils.core import localize_plot_withorig

    _record_invocation(args)

    func_kwargs = dict(
        bl_ref="localize.run",  # dotted-path string, not the module object --
                                 # ProcessPoolExecutor can't pickle a module
        wf=wf,
        rec_recgrp_locations=deployments,
        cornellspec=cornellspec,
        img_rgb={},
        extent={},
        plot_mode="none",
        error_mode="raise",  # exceptions still get caught per-row by the batch runner
    )
    if args.max_peaks_per_pair is not None:
        func_kwargs["max_peaks_per_pair"] = args.max_peaks_per_pair
    if args.max_outlier_pairs is not None:
        func_kwargs["max_outlier_pairs"] = args.max_outlier_pairs
        func_kwargs["outlier_shallow_cap"] = args.outlier_shallow_cap
    func_kwargs["memory_cap_gb"] = args.memory_cap_gb
    if args.winsize is not None:
        func_kwargs["winsize"] = args.winsize
    if args.dt_in_win is not None:
        func_kwargs["dt_in_win"] = args.dt_in_win
    if args.qualcut is not None:
        func_kwargs["qualcut"] = args.qualcut
    func_kwargs["fit_speed"] = not args.fixed_speed
    func_kwargs["compute_intensity_scores"] = not args.no_intensity_scores
    func_kwargs["intensity_topk"] = args.intensity_topk
    func_kwargs["intensity_win_seconds"] = args.intensity_win_seconds

    results, errors = run_localizations_parallel(
        candidates,
        func=localize_plot_withorig,
        func_kwargs=func_kwargs,
        key_col="call_id",
        n_workers=args.n_workers,
        checkpoint_every=args.checkpoint_every,
        out_dir=args.out_dir,
        out_basename="localize_parallel",
        resume=True,
        task_timeout=(args.task_timeout or None),
    )

    print(f"\n{len(results)} localized, {len(errors)} errors")
    print(f"Results: {args.out_dir}/localize_parallel.parquet")
    if len(errors):
        print(f"Errors:  {args.out_dir}/localize_parallel_errors.csv")


if __name__ == "__main__":
    main()
