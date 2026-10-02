# birdnet/pipeline.py
#
# CLI for the full BirdNET detection pipeline:
#
#   Step 1: load_detections  — walk _bn_grid_wprobs/ parquets, explode probs
#   Step 2: attach_labels    — join species names and site-species flag
#   Step 3: join_catalog     — join with wavfiles catalog, compute call_datetime
#   Step 4: filter           — GPS-sync, site species, optional date range
#   Step 5: concordance      — pivot + threshold counts + active recorders
#   Step 6: candidates       — select localization candidates
#
# Each step is checkpointed so the pipeline can be resumed.  Pass --force to
# recompute all steps from scratch.
#
# Usage:
#   python -m birdnet.pipeline                        # run all steps
#   python -m birdnet.pipeline --force                # recompute everything
#   python -m birdnet.pipeline --date-start 2024-01-01 --date-end 2025-01-01
#   python -m birdnet.pipeline --groups Back9 NorthSide

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from birdnet.parse import (
    load_detections, attach_labels, join_catalog,
    filter_detections, compute_concordance, count_active_recorders,
    save_checkpoint, load_checkpoint, save_concordance, load_concordance,
)
from birdnet.select import select_candidates
from config import (
    CATALOG_PATH, BIRDNET_LABELS_PATH, SITE_SPECIES_PATH,
    CONCORDANCE_THRESHOLDS, CHECKPOINT_DIR,
)

log = logging.getLogger(__name__)


def run_pipeline(
    catalog_path: str = CATALOG_PATH,
    checkpoint_dir: str = CHECKPOINT_DIR,
    labels_path: str = BIRDNET_LABELS_PATH,
    site_species_path: str | None = SITE_SPECIES_PATH,
    audio_dirs: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    recorder_groups: list[str] | None = None,
    recorder_types: list[str] | None = None,
    require_gps_sync: bool = True,
    strategy: str = "flexible",
    min_max_prob: float | None = None,
    min_active_recorders: int | None = None,
    min_concordant_recs: int | None = None,
    max_per_species: int | None = None,
    force: bool = False,
    candidates_path: str = "birdnet/candidates.parquet",
) -> pd.DataFrame:
    """Run the full detection pipeline, checkpointing each step.

    Returns the candidates DataFrame.
    """
    CP = Path(checkpoint_dir)
    CP.mkdir(parents=True, exist_ok=True)

    if force:
        log.info("--force: clearing checkpoints")
        shutil.rmtree(CP, ignore_errors=True)
        CP.mkdir(parents=True, exist_ok=True)

    # ---- Step 1+2: load detections + attach labels ----
    bnf = load_checkpoint(CP / "detections_raw.parquet")
    if bnf is None:
        log.info("Step 1: loading detections from _bn_grid_wprobs/ ...")
        bnf = load_detections(audio_dirs)
        log.info("Step 2: attaching labels ...")
        bnf = attach_labels(bnf, labels_path=labels_path,
                             site_species_path=site_species_path)
        save_checkpoint(bnf, CP / "detections_raw.parquet")

    # ---- Step 3+4: join catalog + filter ----
    wf = _load_catalog(catalog_path)
    if recorder_types:
        # Keep wf scoped to the same recorder types as bnfi -- otherwise
        # count_active_recorders (step 5b) would count uptime from recorder
        # types excluded from detection entirely (e.g. S4A when running
        # Solarbar-only), inflating the active-recorder denominator.
        wf = wf[wf.recorder_type.isin(recorder_types)]
        log.info("Catalog restricted to recorder_types=%s: %d rows", recorder_types, len(wf))

    bnfi = load_checkpoint(CP / "detections.parquet")
    if bnfi is None:
        log.info("Step 3: joining catalog ...")
        bnfi = join_catalog(bnf, wf)
        del bnf   # free memory

        log.info("Step 4: filtering detections (require_gps_sync=%s) ...",
                 require_gps_sync)
        bnfi = filter_detections(
            bnfi,
            require_gps_sync=require_gps_sync,
            date_start=date_start,
            date_end=date_end,
            recorder_types=recorder_types,
        )
        if recorder_groups:
            bnfi = bnfi[bnfi.recorder_group.isin(recorder_groups)]
            log.info("Filtered to groups %s: %d rows", recorder_groups, len(bnfi))

        save_checkpoint(bnfi, CP / "detections.parquet")

    # ---- Step 5: concordance + active recorders ----
    concordance = load_concordance(CP / "concordance")
    if concordance is None:
        log.info("Step 5: computing concordance ...")
        concordance = compute_concordance(bnfi)
        log.info("Step 5b: counting active recorders ...")
        concordance = count_active_recorders(concordance, wf)
        save_concordance(concordance, CP / "concordance")

    # ---- Step 6: select candidates ----
    log.info("Step 6: selecting candidates (strategy=%s) ...", strategy)
    candidates = select_candidates(
        concordance, bnfi,
        strategy=strategy,
        recorder_groups=recorder_groups,
        min_max_prob=min_max_prob,
        min_active_recorders=min_active_recorders,
        min_concordant_recs=min_concordant_recs,
        max_per_species=max_per_species,
    )
    candidates.to_parquet(candidates_path, index=False)
    log.info("Candidates written: %s  (%d rows)", candidates_path, len(candidates))

    return candidates


def _load_catalog(catalog_path: str) -> pd.DataFrame:
    """Load wavfiles catalog (parquet or pkl)."""
    from pathlib import Path
    p = Path(catalog_path)
    if p.suffix == ".parquet":
        return pd.read_parquet(catalog_path)
    return pd.read_pickle(catalog_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BirdNET detection pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with default settings
  python -m birdnet.pipeline

  # Recompute everything from scratch
  python -m birdnet.pipeline --force

  # Restrict to a date range and specific groups
  python -m birdnet.pipeline --date-start 2024-01-01 --date-end 2025-01-01 \\
      --groups Back9 NorthSide

  # Use strict candidate selection
  python -m birdnet.pipeline --strategy strict
""",
    )
    parser.add_argument("--catalog",     default=CATALOG_PATH,
                        help="wavfiles catalog parquet (default: CATALOG_PATH)")
    parser.add_argument("--checkpoints", default=CHECKPOINT_DIR,
                        help="checkpoint directory (default: CHECKPOINT_DIR)")
    parser.add_argument("--audio-dirs",  nargs="*", default=None, metavar="DIR",
                        help="audio root dirs to scan for _bn_grid_wprobs/ "
                             "(default: AUDIO_TOPDIRS from config)")
    parser.add_argument("--date-start",  default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--date-end",    default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--groups",      nargs="*", default=None, metavar="GROUP",
                        help="recorder groups to include (default: all)")
    parser.add_argument("--recorder-types", nargs="*", default=None, metavar="TYPE",
                        help="recorder types to include, e.g. SOLARBAR "
                             "(default: all types)")
    parser.add_argument("--no-gps-sync", action="store_true",
                        help="include recordings without a GPS fix "
                             "(default: GPS sync required)")
    parser.add_argument("--strategy",    default="flexible",
                        choices=["flexible", "strict"])
    parser.add_argument("--min-prob",    type=float, default=None,
                        help="minimum peak probability across recorders "
                             "(flexible strategy, default: FLEXIBLE_MIN_MAX_PROB)")
    parser.add_argument("--min-active",  type=int, default=None,
                        help="minimum active recorders at detection time "
                             "(default: MIN_ACTIVE_RECORDERS)")
    parser.add_argument("--min-concordant", type=int, default=None,
                        help="minimum recorders detecting at p50 "
                             "(flexible strategy, default: FLEXIBLE_MIN_CONCORDANT_RECS)")
    parser.add_argument("--max-per-species", type=int, default=None,
                        help="cap on candidates per species/group "
                             "(flexible strategy, default: FLEXIBLE_MAX_PER_SPECIES)")
    parser.add_argument("--candidates",  default="birdnet/candidates.parquet",
                        help="output path for candidates parquet")
    parser.add_argument("--force",       action="store_true",
                        help="clear checkpoints and recompute all steps")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        catalog_path         = args.catalog,
        checkpoint_dir       = args.checkpoints,
        audio_dirs           = args.audio_dirs,
        date_start           = args.date_start,
        date_end             = args.date_end,
        recorder_groups      = args.groups,
        recorder_types       = args.recorder_types,
        require_gps_sync     = not args.no_gps_sync,
        strategy             = args.strategy,
        min_max_prob         = args.min_prob,
        min_active_recorders = args.min_active,
        min_concordant_recs  = args.min_concordant,
        max_per_species      = args.max_per_species,
        force                = args.force,
        candidates_path      = args.candidates,
    )


if __name__ == "__main__":
    main()
