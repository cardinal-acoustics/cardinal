#!/usr/bin/env python3
"""figure4_localization_density.py — spatial density of high-quality localizations.

Regenerates Figure 4: localizations with mean residual TDOA error < 1 ms,
overlaid as a density map (not raw scatter -- House alone has ~478k such
points, dominated by Dickcissel, so individual markers would just render
as a solid blob) on the drone orthomosaic, one panel per site.

Usage:
    python analysis/paper_figures/figure4_localization_density.py
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
GROUPS = ["Back9", "NorthSide", "House", "EastCreek"]
SITE_LABELS = {"Back9": "Site 1 (Back9)", "NorthSide": "Site 2 (NorthSide)",
               "House": "Site 3 (House)", "EastCreek": "Site 4 (EastCreek)"}
RESULTS_PATH = "localization/results/localize_parallel.parquet"
# "Good" is an open question -- mean_error alone doesn't guarantee a correct
# solution (see the geometry-score-fuzziness finding earlier this session),
# but it's the paper's existing stated criterion and a reasonable starting
# filter. Worth revisiting with best_score/intensity_score once that's
# settled.
MEAN_ERROR_THRESHOLD = 0.001  # 1 ms
# Tight window focused on the array and its immediate surroundings -- the
# point is showing how localizations map onto local landscape features
# (treelines, structures), not comprehensive coverage of every solution
# the pipeline produced, however far afield.
WINDOW_M = 300.0
MAX_POINTS_PER_PANEL = 3000
OUT_DIR = os.path.dirname(__file__)

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



def load_recorder_centroids() -> dict:
    """Centroid of each site's recorders actually used in the primary run
    (see figure1_array_maps.py for why -- ties the map center to what the
    analysis used, not a nominal deployment schedule)."""
    res = pd.read_parquet(RESULTS_PATH, columns=["recorders", "recorder_group"])
    deployments = pd.read_parquet("data/deployments.parquet")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)

    centroids = {}
    for group, g in res.groupby("recorder_group"):
        recs = set()
        for lst in g["recorders"]:
            if lst is not None:
                recs.update(lst)
        dep_g = deployments[(deployments["recorder_group"] == group) & (deployments["recorder"].isin(recs))]
        dep_g = dep_g.sort_values("begin").drop_duplicates(subset=["recorder"], keep="last")
        x, y = tf.transform(dep_g["longitude"].to_numpy(), dep_g["latitude"].to_numpy())
        centroids[group] = (x.mean(), y.mean())
    return centroids


def crop_orthomosaic(src, xmin, xmax, ymin, ymax):
    window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
    rgb = src.read([1, 2, 3], window=window)
    return np.moveaxis(rgb, 0, -1)


def main() -> None:
    res = pd.read_parquet(RESULTS_PATH, columns=["x_est", "mean_error", "recorder_group"])
    hq = res[res["mean_error"] < MEAN_ERROR_THRESHOLD].copy()
    hq["x"] = hq["x_est"].apply(lambda v: v[0])
    hq["y"] = hq["x_est"].apply(lambda v: v[1])
    print(f"high-quality localizations (mean_error < {MEAN_ERROR_THRESHOLD*1000:.0f} ms): "
          f"{len(hq)} / {len(res)}")

    centroids = load_recorder_centroids()

    windows = {}
    for group in GROUPS:
        cx, cy = centroids[group]
        half = WINDOW_M / 2.0
        windows[group] = (cx - half, cx + half, cy - half, cy + half)

    # single row for three or fewer sites, 2x2 otherwise
    n_sites = len(GROUPS)
    if n_sites <= 3:
        fig, axes = plt.subplots(1, n_sites, figsize=(5.4 * n_sites, 5.8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = np.atleast_1d(axes).ravel()

    with rasterio.open(ORTHOMOSAIC_PATH) as src:
        for ax, group in zip(axes, GROUPS):
            xmin, xmax, ymin, ymax = windows[group]

            rgb = crop_orthomosaic(src, xmin, xmax, ymin, ymax)
            ax.imshow(rgb, extent=(xmin, xmax, ymin, ymax), origin="upper")
            ax.set_aspect("equal")

            g = hq[hq["recorder_group"] == group]
            g = g[(g["x"] > xmin) & (g["x"] < xmax) & (g["y"] > ymin) & (g["y"] < ymax)]
            # At these point counts (up to ~470k for House), even very low
            # alpha saturates to a solid fill regardless of marker style --
            # that's a density problem, not a style problem. Subsample so
            # individual open circles stay visible; the goal is showing the
            # spatial pattern, not plotting every point.
            g_plot = g.sample(n=min(MAX_POINTS_PER_PANEL, len(g)), random_state=0) if len(g) else g
            ax.scatter(g_plot["x"], g_plot["y"], s=25, facecolors="none", edgecolors="#ff3b3b",
                      linewidths=0.7, alpha=0.35)

            ax.set_title(f"{SITE_LABELS[group]}  (n={len(g)} localizations)", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            bar_len = 100.0
            bx0 = xmin + 0.05 * (xmax - xmin)
            by0 = ymin + 0.05 * (ymax - ymin)
            ax.plot([bx0, bx0 + bar_len], [by0, by0], color="white", lw=3)
            ax.text(bx0 + bar_len / 2, by0 + 0.02 * (ymax - ymin), "100 m",
                    color="white", ha="center", fontsize=8)

    fig.suptitle(f"Figure 4. Spatial density of high-quality localizations "
                 f"(mean residual TDOA error < {MEAN_ERROR_THRESHOLD*1000:.0f} ms)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = os.path.join(OUT_DIR, f"figure4_localization_density{FIG_SUFFIX}.png")
    out_pdf = os.path.join(OUT_DIR, f"figure4_localization_density{FIG_SUFFIX}.pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
