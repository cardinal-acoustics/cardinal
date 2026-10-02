#!/usr/bin/env python3
"""figure2_geometric_consistency.py — internal geometric consistency (panels A-C).

Reproduces the user's original QC.png script as closely as possible:
  (A) plain histogram of mean_error (seconds), fixed range [0, 0.1]
  (B) rank_0 minus toppeak_per_pair mean_error, restricted to events
      where rank_0's own mean_error < 5 ms (only asking "does geometry
      selection help on events that are otherwise already good")
  (C) best_score vs mean_error scatter, restricted to nq == nql == 15
      (a clean 6-recorder array where every pair passed quality
      filtering, none dropped) and best_score > 1e-5
  (D) change in mean_error under a stricter qualcut=0.05 rerun (vs. the
      default 0.1) on a 20,000-call_id random sample, merged against
      this same baseline run by call_id (see
      localization/results/strict_snr/). Of the 20,000 sampled
      candidates, only 2,090 (10.5%) still had >=3 usable pairs at the
      stricter cutoff -- the rest raised "No valid recorder triplets to
      form triangle constraints" and are excluded from this comparison
      entirely (survivorship: this panel only shows how much the
      *surviving* events' error changed, not how many events the
      stricter cutoff would cost).

Note: this uses the current uncapped, all-species primary results
(1,059,066 events, ~52% Dickcissel) -- reproducing the original code
exactly does not by itself resolve the compositional difference from
whatever dataset the original QC.png was built from, since the original
code has no species-balancing step either. That's a separate decision
(see conversation) from the plotting-code fidelity fixed here.

Usage:
    python analysis/paper_figures/figure2_geometric_consistency.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("CARDINAL_CONFIG", "sites/ferguson.toml")

RESULTS_PATH = "localization/results/localize_parallel.parquet"
STRICT_SNR_RESULTS_PATH = "localization/results/strict_snr/localize_parallel.parquet"
OUT_DIR = os.path.dirname(__file__)

# --- results set selection -------------------------------------------------
# Default reproduces the published figure. Pass --results to render the same
# figure from a different localization run (e.g. the fixed-c corpus); outputs
# get a suffix derived from that run's directory name, so nothing is
# overwritten. --suffix overrides the derived one.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--results", default=RESULTS_PATH,
                 help="localize_parallel.parquet to plot (default: %(default)s)")
_ap.add_argument("--strict-snr", default=None,
                 help="results parquet for panel D's stricter-qualcut rerun. "
                      "Must come from the same configuration as --results, or "
                      "the panel compares two different pipelines.")
_ap.add_argument("--suffix", default=None,
                 help="appended to output filenames; default is derived from "
                      "the results directory when --results is non-default")
_args, _ = _ap.parse_known_args()
if _args.results != RESULTS_PATH:
    RESULTS_PATH = _args.results
if getattr(_args, "strict_snr", None):
    STRICT_SNR_RESULTS_PATH = _args.strict_snr
    _derived = os.path.basename(os.path.dirname(RESULTS_PATH))
    FIG_SUFFIX = _args.suffix if _args.suffix is not None else f"_{_derived}"
else:
    FIG_SUFFIX = _args.suffix or ""
# ---------------------------------------------------------------------------



def build_figure(res: pd.DataFrame, out_suffix: str, title_suffix: str,
                  strict_snr: pd.DataFrame | None = None) -> None:
    print(f"[{out_suffix}] {len(res)} localized events")

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 8.0))

    # ---- Panel A: plain histogram of mean_error (seconds), range [0, 0.1]
    ax = axes[0][0]
    ax.hist(res["mean_error"], bins=np.linspace(0, 0.1, 100))
    ax.set_xticks([0, .01, .02, .05, .10])
    # Corrected vs. the original QC.png labels (seconds -> ms is x1000, so
    # .01 is 10ms not 1ms) -- flagged in the module docstring.
    ax.set_xticklabels(["0", "10ms", "20ms", "50ms", "100ms"])
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("Mean Error")
    ax.set_yticks([])
    ax.set_title("A. Distribution of Mean Errors", fontsize=10)

    # ---- Panel B: rank_0 vs toppeak_per_pair
    # Restricted to events with a genuine alternate (the two combos
    # actually differ) AND at least one of the two achieved a "success"
    # (< 5ms) -- symmetric on which one wins, unlike conditioning on
    # rank_0 specifically. This keeps the comparison to cases where
    # localization succeeded by at least one method, without pre-biasing
    # toward the method being evaluated.
    ax = axes[0][1]

    def _paired(cands):
        by_label = {c.get("candidate_label"): c["mean_error"] for c in cands}
        return by_label.get("rank_0"), by_label.get("toppeak_per_pair")

    pairs_be = res["candidates"].apply(_paired)
    rank0 = pairs_be.apply(lambda t: t[0])
    toppeak = pairs_be.apply(lambda t: t[1])
    both = pd.DataFrame({"rank0": rank0, "toppeak": toppeak}).dropna()
    print(f"[{out_suffix}] panel B: {len(both)} events with a genuine toppeak_per_pair alternate")

    at_least_one_good = both[["rank0", "toppeak"]].min(axis=1) < 0.005
    both = both[at_least_one_good]
    print(f"[{out_suffix}] panel B: {len(both)} of those have at least one candidate < 5ms")

    diff_b = both["rank0"] - both["toppeak"]

    ax.hist(diff_b, bins=np.linspace(-.10, .10, 100))
    ax.set_xticks([-.1, -.05, 0, .05, .1])
    ax.set_xticklabels(["-100ms", "-50ms", "0", "+50ms", "+100ms"])
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("Change in Mean Error")
    ax.set_title("B. Effect of Optimal Peak Geometry", fontsize=10)
    ax.set_yticks([])

    # ---- Panel C: best_score vs mean_error, clean 6-recorder/15-pair subset
    ax = axes[1][0]
    nq = res["pairs"].apply(len)
    nql = res["tdoas"].apply(len)
    clean = res[(nq == nql) & (nql == 15) & (res["best_score"] > 0.00001)]
    print(f"[{out_suffix}] panel C: {len(clean)} events in the clean nq==nql==15 subset")

    ax.scatter(clean["best_score"], clean["mean_error"], s=1, alpha=.09)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Triangle deviation of peaks")
    ax.set_ylabel("Mean Error")
    ax.set_title("C.", fontsize=10, loc="left")

    # ---- Panel D: effect of excluding low-S/N pairs, i.e. a stricter
    # qualcut=0.05 rerun (vs. the default 0.1) on a 20,000-call_id sample,
    # merged against this same baseline run by call_id. Mirrors the
    # original code's dff = df2.merge(df, on='call_id') where df2 is the
    # stricter run and df is the baseline: mean_error_x - mean_error_y.
    ax = axes[1][1]
    if strict_snr is not None:
        dff = strict_snr[["call_id", "mean_error"]].merge(
            res[["call_id", "mean_error"]], on="call_id", how="left",
            suffixes=("_x", "_y"),
        ).dropna()
        print(f"[{out_suffix}] panel D: {len(dff)} events re-localized under qualcut=0.05 "
              f"with a baseline match")
        diff_d = dff["mean_error_x"] - dff["mean_error_y"]
        ax.hist(diff_d, bins=np.linspace(-.010, .010, 25), density=True)
        ax.set_xticks([-.01, -.005, 0, .005, .01])
        ax.set_xticklabels(["-10ms", "-5ms", "0", "+5ms", "+10ms"])
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("Change in mean error")
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "Panel D pending:\nneeds a second run with a\nstricter pair-quality cutoff",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#898781")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("D. Effect of excluding low S/N pairs", fontsize=10)

    for ax_, label in zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]], "ABCD"):
        ax_.text(-0.1, 1.05, label, transform=ax_.transAxes, fontsize=14,
                 fontweight="bold", va="top", ha="right")

    fig.suptitle(title_suffix, fontsize=11, y=0.995)
    plt.subplots_adjust(hspace=1.0, wspace=0.25)

    out_png = os.path.join(OUT_DIR, f"figure2_geometric_consistency_ABC{out_suffix}{FIG_SUFFIX}.png")
    out_pdf = os.path.join(OUT_DIR, f"figure2_geometric_consistency_ABC{out_suffix}{FIG_SUFFIX}.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def main() -> None:
    res = pd.read_parquet(
        RESULTS_PATH,
        columns=["mean_error", "best_score", "candidates", "pairs", "tdoas", "common_name", "call_id"],
    )
    strict_snr = None
    if os.path.exists(STRICT_SNR_RESULTS_PATH):
        strict_snr = pd.read_parquet(STRICT_SNR_RESULTS_PATH, columns=["call_id", "mean_error"])

    build_figure(res, "", "All species", strict_snr=strict_snr)

    no_dcc = res[res["common_name"] != "Dickcissel"]
    build_figure(no_dcc, "_no_dickcissel", "Excluding Dickcissel", strict_snr=strict_snr)


if __name__ == "__main__":
    main()
