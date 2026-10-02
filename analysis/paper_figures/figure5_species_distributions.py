#!/usr/bin/env python3
"""figure5_species_distributions.py — species-specific spatial distributions.

Regenerates Figure 5: localization patterns for five focal species across
all four sites, using the same visual conventions as figure4 (same 300m
window per site, same open-circle low-alpha rendering, same mean_error
threshold, same point subsampling cap) so the two figures read as one
consistent system.

Usage:
    python analysis/paper_figures/figure5_species_distributions.py
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
SPECIES = ["Indigo Bunting", "American Crow", "Swamp Sparrow",
           "Dickcissel", "Yellow-billed Cuckoo"]
RESULTS_PATH = "localization/results/localize_parallel.parquet"
MEAN_ERROR_THRESHOLD = 0.001  # 1 ms -- same as figure4
WINDOW_M = 300.0               # same as figure4
MAX_POINTS_PER_PANEL = 3000    # same as figure4
MIN_POINTS_TO_PLOT = 20        # below this, mark the panel "no data" rather than plot noise
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
    """Same criterion as figure1/figure4: centroid of recorders that
    actually contributed to a localized event in the primary run."""
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
    res = pd.read_parquet(RESULTS_PATH, columns=["x_est", "mean_error", "recorder_group", "common_name"])
    hq = res[(res["mean_error"] < MEAN_ERROR_THRESHOLD) & (res["common_name"].isin(SPECIES))].copy()
    hq["x"] = hq["x_est"].apply(lambda v: v[0])
    hq["y"] = hq["x_est"].apply(lambda v: v[1])

    centroids = load_recorder_centroids()
    windows = {}
    for group in GROUPS:
        cx, cy = centroids[group]
        half = WINDOW_M / 2.0
        windows[group] = (cx - half, cx + half, cy - half, cy + half)

    # sites as rows, species as columns
    fig, axes = plt.subplots(len(GROUPS), len(SPECIES),
                             figsize=(4 * len(SPECIES), 4 * len(GROUPS)),
                             squeeze=False)

    with rasterio.open(ORTHOMOSAIC_PATH) as src:
        for i, group in enumerate(GROUPS):
            for j, species in enumerate(SPECIES):
                ax = axes[i, j]
                xmin, xmax, ymin, ymax = windows[group]
                rgb = crop_orthomosaic(src, xmin, xmax, ymin, ymax)
                ax.imshow(rgb, extent=(xmin, xmax, ymin, ymax), origin="upper")
                ax.set_aspect("equal")

                g = hq[(hq["common_name"] == species) & (hq["recorder_group"] == group)]
                g = g[(g["x"] > xmin) & (g["x"] < xmax) & (g["y"] > ymin) & (g["y"] < ymax)]

                if len(g):
                    g_plot = g.sample(n=min(MAX_POINTS_PER_PANEL, len(g)), random_state=0)
                    ax.scatter(g_plot["x"], g_plot["y"], s=25, facecolors="none",
                              edgecolors="#ff3b3b", linewidths=0.7, alpha=0.35)

                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(species, fontsize=12, style="italic")
                if j == 0:
                    ax.set_ylabel(SITE_LABELS[group], fontsize=12)
                # an empty panel means the species was not recorded there,
                # not that localization failed -- say so rather than "n=0"
                label = f"n={len(g)}" if len(g) else "not detected"
                ax.text(0.03, 0.03, label, transform=ax.transAxes, fontsize=8,
                        color="white", va="bottom", ha="left",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.4, pad=0.2))

    fig.suptitle("Figure 5. Species-specific spatial distributions of high-quality localizations "
                f"(mean residual TDOA error < {MEAN_ERROR_THRESHOLD*1000:.0f} ms)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = os.path.join(OUT_DIR, f"figure5_species_distributions{FIG_SUFFIX}.png")
    out_pdf = os.path.join(OUT_DIR, f"figure5_species_distributions{FIG_SUFFIX}.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
