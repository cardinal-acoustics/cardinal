"""Re-derive each candidate's speed of sound with the corrected model.

The corpus was prepared with ERA5 dry-air c. Three things were wrong with
that and are now fixed in localize.temperature: the grid (HRRR at 3 km beats
ERA5 at 9 km here, measured against the ATMOS 41 stations and independently
against the tone generators), its night warm bias (~2.3 degC, from the
stations and from 47k trail-camera night readings), and humidity (+1.0 to
+1.6 m/s, omitted entirely -- larger than the ERA5-to-HRRR difference).

This only rewrites the `speed` column, so the re-run differs from the
original in c and nothing else.

The fetch is chunked by year: Open-Meteo's historical-forecast archive does
not serve a multi-year span in one request.

Usage:
    python scripts/reattach_speed.py \\
        --in birdnet/candidates.parquet \\
        --out birdnet/candidates_hrrr.parquet
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
from localize.temperature import fetch_temperature
from config import SITE_LAT, SITE_LON, SITE_TIMEZONE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="hrrr")
    ap.add_argument("--no-humidity", action="store_true")
    ap.add_argument("--night-bias", type=float, default=None,
                    help="default: the value in localize.temperature")
    ap.add_argument("--day-bias", type=float, default=None)
    args = ap.parse_args()

    C = pd.read_parquet(args.inp)
    t = pd.to_datetime(C.call_datetime, utc=True)
    print(f"{len(C)} candidates, {t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d}")
    print(f"old speed: {C.speed.min():.2f} to {C.speed.max():.2f}, "
          f"median {C.speed.median():.2f}")

    kw = dict(tz=str(SITE_TIMEZONE), source=args.source,
              humidity=not args.no_humidity)
    if args.night_bias is not None:
        kw["night_bias"] = args.night_bias
    if args.day_bias is not None:
        kw["day_bias"] = args.day_bias

    parts = []
    for year in range(t.min().year, t.max().year + 1):
        s = max(pd.Timestamp(f"{year}-01-01"), t.min().tz_localize(None) - pd.Timedelta(days=2))
        e = min(pd.Timestamp(f"{year}-12-31"), t.max().tz_localize(None) + pd.Timedelta(days=2))
        if s > e:
            continue
        w = fetch_temperature(SITE_LAT, SITE_LON, f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}", **kw)
        print(f"  {year}: {len(w)} hours from {w.source.iloc[0]}, "
              f"c {w.speed.min():.1f}-{w.speed.max():.1f}")
        parts.append(w)
    W = pd.concat(parts, ignore_index=True).sort_values("dt")
    W = W.drop_duplicates("dt")
    W["dt_utc"] = W.dt.dt.tz_convert("UTC")

    C = C.copy()
    C["_t"] = t
    C = C.sort_values("_t")
    merged = pd.merge_asof(
        C[["_t"]], W[["dt_utc", "temp_c", "rh", "speed"]].rename(
            columns={"dt_utc": "_t", "temp_c": "temp_c_new", "speed": "speed_new"}),
        on="_t", direction="nearest", tolerance=pd.Timedelta("1h"))
    merged.index = C.index
    miss = merged.speed_new.isna().sum()
    if miss:
        print(f"WARNING: {miss} candidates found no weather within 1 h; "
              f"keeping their original speed")
    C["speed_old"] = C.speed
    C["speed"] = merged.speed_new.fillna(C.speed)
    C["temp_c"] = merged.temp_c_new.fillna(C.temp_c)
    C["rh"] = merged.rh
    C = C.drop(columns=["_t"]).sort_index()

    d = C.speed - C.speed_old
    print(f"\nnew speed: {C.speed.min():.2f} to {C.speed.max():.2f}, "
          f"median {C.speed.median():.2f}")
    print(f"change: mean {d.mean():+.3f}  median {d.median():+.3f}  "
          f"sd {d.std():.3f}  range {d.min():+.2f} to {d.max():+.2f} m/s")
    hr = pd.to_datetime(C.call_datetime, utc=True).dt.tz_convert(
        "America/Chicago").dt.hour
    night = (hr >= 21) | (hr < 6)
    print(f"  night mean {d[night].mean():+.3f}   day mean {d[~night].mean():+.3f}")
    print(f"expected position shift at 0.231 m per m/s: "
          f"median {abs(d.median())*0.231:.3f} m")

    C.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(C)} rows)")


if __name__ == "__main__":
    main()
