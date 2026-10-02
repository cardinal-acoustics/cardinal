#!/usr/bin/env python3
"""figure1_array_maps.py — recorder array maps over drone orthomosaic imagery.

Regenerates Figure 1 (spatial configuration of the four recorder arrays)
using the "CompleteSurvey" drone orthomosaic, which covers all four sites
in one image and is already in EPSG:32615 (UTM zone 15N) -- the same CRS
used for recorder deployment coordinates throughout this pipeline, so no
reprojection is needed, just a windowed crop per array.

The full orthomosaic is enormous (~30000x58000 px); this only ever reads
a small windowed crop per array via rasterio, never the full raster.

Usage:
    python analysis/paper_figures/figure1_array_maps.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("CARDINAL_CONFIG", "sites/ferguson.toml")

# Basemap for the site panels. Read from the site config
# ([paths].orthomosaic) so the figures run on any machine; override for a
# single run with --orthomosaic.
def _default_orthomosaic():
    try:
        import config as _c
        return getattr(_c, "ORTHOMOSAIC_PATH", None)
    except Exception:
        return None


ORTHOMOSAIC_PATH = _default_orthomosaic()
GROUPS = ["Back9", "NorthSide", "House", "EastCreek"]  # site 1-4, per paper numbering
SITE_LABELS = {"Back9": "Site 1 (Back9)", "NorthSide": "Site 2 (NorthSide)",
               "House": "Site 3 (House)", "EastCreek": "Site 4 (EastCreek)"}
WINDOW_M = 220.0  # fixed window size (meters) centered on each array's centroid,
                  # so every panel is the same physical size/scale for direct
                  # visual comparison -- comfortably fits the largest array
                  # (NorthSide, ~122m north-south span) with margin to spare.
OUT_DIR = os.path.dirname(__file__)

# deployments.parquet is a full multi-year history -- recorders get moved
# and replaced over time, so "most recent position per recorder ID" shows
# every unit ever deployed there (confirmed visually: NorthSide rendered
# 11 overlapping markers, not a real 4-6-recorder roster). A single
# reference-date snapshot doesn't work either: it undercounts recorders
# that were added partway through the study (e.g. a 7th Back9 unit
# installed 2025-10-10, after every snapshot date tried). Instead, use
# the actual set of recorders that contributed to at least one localized
# event in the primary results -- the most direct, principled criterion
# for "what was in this array" since it's tied to what the analysis
# itself used, not an arbitrary date. A recorder can occupy different
# positions at different sites over its lifetime (confirmed: BARLT_00008240
# moved from NorthSide to EastCreek in Sept 2024, with distinct deployment
# rows for each), so position lookups are scoped to the specific
# (site, recorder) pair, not the recorder alone.
RESULTS_PATH = "localization/results/localize_parallel.parquet"

# --- results set selection -------------------------------------------------
# Default reproduces the published figure. Pass --results to render the same
# figure from a different localization run (e.g. the fixed-c corpus); outputs
# get a suffix derived from that run's directory name, so nothing is
# overwritten. --suffix overrides the derived one.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--results", default=RESULTS_PATH,
                 help="localize_parallel.parquet to plot (default: %(default)s)")
_ap.add_argument("--groups", nargs="*", default=None,
                 help="recorder groups to plot, in order. Defaults to the "
                      "four-site paper set; pass a subset to drop a site "
                      "(e.g. --groups Back9 NorthSide House).")
_ap.add_argument("--suffix", default=None,
                 help="appended to output filenames; default is derived from "
                      "the results directory when --results is non-default")
_args, _ = _ap.parse_known_args()
if getattr(_args, "groups", None):
    GROUPS = list(_args.groups)
if _args.results != RESULTS_PATH:
    RESULTS_PATH = _args.results
    _derived = os.path.basename(os.path.dirname(RESULTS_PATH))
    FIG_SUFFIX = _args.suffix if _args.suffix is not None else f"_{_derived}"
else:
    FIG_SUFFIX = _args.suffix or ""
# ---------------------------------------------------------------------------



def _used_recorders_by_group() -> dict[str, set[str]]:
    res = pd.read_parquet(RESULTS_PATH, columns=["recorders", "recorder_group"])
    used = {}
    for group, g in res.groupby("recorder_group"):
        recs = set()
        for lst in g["recorders"]:
            if lst is not None:
                recs.update(lst)
        used[group] = recs
    return used


def load_recorder_positions() -> pd.DataFrame:
    deployments = pd.read_parquet("data/deployments.parquet")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    used = _used_recorders_by_group()

    rows = []
    for group in GROUPS:
        dep_g = deployments[deployments["recorder_group"] == group]
        for rec in sorted(used.get(group, set())):
            dep = dep_g[dep_g["recorder"] == rec]
            if dep.empty:
                continue
            d = dep.sort_values("begin").iloc[-1]  # most recent deployment record within this group
            x, y = tf.transform(d["longitude"], d["latitude"])
            rows.append({"recorder_group": group, "recorder": rec, "x": x, "y": y})
    return pd.DataFrame(rows)


def crop_orthomosaic(src, xmin, xmax, ymin, ymax):
    window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
    rgb = src.read([1, 2, 3], window=window)  # drop alpha band
    rgb = np.moveaxis(rgb, 0, -1)  # (H, W, 3)
    return rgb


def main() -> None:
    recorders = load_recorder_positions()

    # lay out as a single row when there are 3 or fewer sites, 2x2 otherwise
    n = len(GROUPS)
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes = np.atleast_1d(axes).ravel()

    with rasterio.open(ORTHOMOSAIC_PATH) as src:
        for ax, group in zip(axes, GROUPS):
            g = recorders[recorders["recorder_group"] == group]
            cx, cy = g["x"].mean(), g["y"].mean()
            half = WINDOW_M / 2.0
            xmin, xmax = cx - half, cx + half
            ymin, ymax = cy - half, cy + half

            rgb = crop_orthomosaic(src, xmin, xmax, ymin, ymax)
            ax.imshow(rgb, extent=(xmin, xmax, ymin, ymax), origin="upper")
            ax.set_aspect("equal")
            ax.scatter(g["x"], g["y"], s=80, facecolors="none", edgecolors="#ff3b3b",
                       linewidths=2, zorder=3)
            for _, r in g.iterrows():
                ax.annotate(r["recorder"].replace("BARLT_", ""), (r["x"], r["y"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=7,
                            color="white",
                            path_effects=None)
            # len(g) counts every recorder ever deployed at the site, not the
            # number running at once -- Back9's 8563 started after 8648 ended,
            # so the cumulative count exceeds the concurrent one the text quotes.
            ax.set_title(f"{SITE_LABELS[group]}  ({len(g)} recorder positions)",
                         fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            # scale bar
            bar_len = 50.0
            bx0 = xmin + 0.05 * (xmax - xmin)
            by0 = ymin + 0.05 * (ymax - ymin)
            ax.plot([bx0, bx0 + bar_len], [by0, by0], color="white", lw=3)
            ax.text(bx0 + bar_len / 2, by0 + 0.02 * (ymax - ymin), "50 m",
                    color="white", ha="center", fontsize=8)

    _nw = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(
        len(GROUPS), str(len(GROUPS)))
    fig.suptitle("Figure 1. Spatial configuration of passive acoustic monitoring "
                 f"arrays at {_nw} field sites", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = os.path.join(OUT_DIR, f"figure1_array_maps{FIG_SUFFIX}.png")
    out_pdf = os.path.join(OUT_DIR, f"figure1_array_maps{FIG_SUFFIX}.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
