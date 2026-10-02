#!/usr/bin/env python3
"""
prepare_localization_inputs.py — build the two inputs localization needs
beyond the candidates themselves: per-species frequency profiles and a
per-call speed of sound.

Usage:
    export CARDINAL_CONFIG=/path/to/Cardinal/sites/ferguson.toml
    python scripts/prepare_localization_inputs.py \
        --candidates birdnet/candidates.parquet \
        --detections birdnet/checkpoints/detections.parquet \
        --out-dir localization/inputs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import SITE_LAT, SITE_LON, SITE_TIMEZONE
from localize.freqprofiles import build_from_detections, save_profiles
from localize.temperature import fetch_temperature, attach_speed, save_temperature

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--detections", required=True,
                        help="the detections.parquet checkpoint written by birdnet.pipeline")
    parser.add_argument("--out-dir", default="localization/inputs")
    parser.add_argument("--top-n", type=int, default=20,
                        help="clips per species for frequency-profile building")
    parser.add_argument("--min-prob", type=float, default=0.9)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = pd.read_parquet(args.candidates)
    log.info("Loaded %d candidates spanning %s to %s",
             len(candidates), candidates["call_datetime"].min(), candidates["call_datetime"].max())

    # ---- Frequency profiles -------------------------------------------------
    freqprofiles_path = os.path.join(args.out_dir, "freqprofiles.pkl")
    log.info("Building frequency profiles (top_n=%d, min_prob=%.2f) from %s ...",
             args.top_n, args.min_prob, args.detections)
    detections = pd.read_parquet(args.detections)
    profiles = build_from_detections(detections, top_n=args.top_n, min_prob=args.min_prob)
    save_profiles(profiles, freqprofiles_path)
    log.info("Built profiles for %d species -> %s", len(profiles), freqprofiles_path)

    # ---- Temperature / speed of sound ---------------------------------------
    temperature_path = os.path.join(args.out_dir, "temperature.pkl")
    date_start = candidates["call_datetime"].min().strftime("%Y-%m-%d")
    date_end   = candidates["call_datetime"].max().strftime("%Y-%m-%d")
    log.info("Fetching historical temperature %s to %s ...", date_start, date_end)
    temps = fetch_temperature(SITE_LAT, SITE_LON, date_start, date_end, tz=str(SITE_TIMEZONE))
    save_temperature(temps, temperature_path)

    candidates_ws = attach_speed(candidates, temps)
    candidates_ws_path = args.candidates.replace(".parquet", "_with_speed.parquet")
    candidates_ws.to_parquet(candidates_ws_path, index=True)
    log.info("Candidates with speed -> %s", candidates_ws_path)

    print(f"\nDone.\n"
          f"  Frequency profiles: {freqprofiles_path}\n"
          f"  Temperature/speed:  {temperature_path}\n"
          f"  Candidates+speed:   {candidates_ws_path}")


if __name__ == "__main__":
    main()
