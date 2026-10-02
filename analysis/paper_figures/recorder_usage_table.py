#!/usr/bin/env python3
"""recorder_usage_table.py — when each recorder was in use at each site.

Built from the actual recorded audio catalog (wavfiles.parquet), not the
nominal deployment schedule (data/deployments.parquet) -- the catalog
reflects when a recorder actually produced data, gaps and all, rather
than when it was scheduled/intended to be active. Cross-references
against the primary localization run to flag which recorders actually
contributed to a localized event.

Usage:
    python analysis/paper_figures/recorder_usage_table.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("CARDINAL_CONFIG", "sites/ferguson.toml")

GROUPS = ["Back9", "NorthSide", "House", "EastCreek"]
RESULTS_PATH = "localization/results/localize_parallel.parquet"
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



def main() -> None:
    wf = pd.read_parquet("wavfiles.parquet")
    wf = wf[(wf["recorder_type"] == "SOLARBAR") & (wf["recorder_group"].isin(GROUPS))]

    rows = []
    for (group, rec), g in wf.groupby(["recorder_group", "recorder"]):
        rows.append({
            "recorder_group": group,
            "recorder": rec,
            "first_recorded": g["start_datetime"].min(),
            "last_recorded": g["end_datetime"].max(),
            "n_files": len(g),
            "total_hours": g["duration"].sum() / 3600.0 if "duration" in g.columns else None,
        })
    table = pd.DataFrame(rows)

    res = pd.read_parquet(RESULTS_PATH, columns=["recorders", "recorder_group"])
    used_by_group = {}
    for group, g in res.groupby("recorder_group"):
        recs = set()
        for lst in g["recorders"]:
            if lst is not None:
                recs.update(lst)
        used_by_group[group] = recs
    table["used_in_primary_run"] = table.apply(
        lambda r: r["recorder"] in used_by_group.get(r["recorder_group"], set()), axis=1
    )

    # Order sites to match the paper's site numbering, chronological within site
    site_order = {g: i for i, g in enumerate(GROUPS)}
    table["_site_order"] = table["recorder_group"].map(site_order)
    table = table.sort_values(["_site_order", "first_recorded"]).drop(columns="_site_order")

    out_csv = os.path.join(OUT_DIR, f"recorder_usage_table{FIG_SUFFIX}.csv")
    table.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(table)} recorder-site rows)")
    print()
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
