#!/usr/bin/env python3
"""
build_cornell_profiles.py — build per-species frequency profiles from the
Cornell Guide to Bird Sounds reference clip library, for comparison against
localize.freqprofiles.build_from_detections (field-detection-based profiles).

Filenames follow "<Species> <NN> <description> <location>.mp3", e.g.
"Blue Jay 09 Calls US-NY.mp3" -> species "Blue Jay".

Usage:
    python scripts/build_cornell_profiles.py \
        --clips-dir /path/to/reference/clips \
        --out localization/inputs/freqprofiles.pkl \
        --max-clips-per-species 5
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localize.freqprofiles import build_from_reference_clips, save_profiles

_PATTERN = re.compile(r"^(.+?)\s+\d{2}\s+")


def parse_cornell_species(filepath: str) -> str | None:
    fn = os.path.basename(filepath)
    m = _PATTERN.match(fn)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-clips-per-species", type=int, default=5,
                        help="cap clips per species for speed (most species have "
                             "5-9 available; default 5 keeps a first test run fast)")
    parser.add_argument("--hp", type=float, default=500.0)
    parser.add_argument("--lp", type=float, default=15000.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    profiles = build_from_reference_clips(
        args.clips_dir, parse_cornell_species,
        max_clips_per_species=args.max_clips_per_species,
        hp=args.hp, lp=args.lp,
    )
    save_profiles(profiles, args.out)
    print(f"\nBuilt Cornell-reference profiles for {len(profiles)} species -> {args.out}")


if __name__ == "__main__":
    main()
