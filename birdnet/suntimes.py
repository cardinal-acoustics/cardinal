# birdnet/suntimes.py
#
# Build a sunrise/sunset table for the site.
# Run once; output is used by analysis notebooks for diel activity plots.
#
# Usage:
#   python -m birdnet.suntimes --output birdnet/suntimes.parquet

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from config import SITE_LAT, SITE_LON, SITE_TIMEZONE

log = logging.getLogger(__name__)


def build_sun_table(lat: float | None = None,
                    lon: float | None = None,
                    date_start: str = "2023-10-01",
                    date_end:   str = "2026-01-01") -> pd.DataFrame:
    """Compute sunrise and sunset times for every date in [date_start, date_end).

    Requires the `suntime` package (pip install suntime).

    Parameters
    ----------
    lat, lon   : site coordinates; default to SITE_LAT/SITE_LON from config
    date_start : ISO date string, inclusive
    date_end   : ISO date string, exclusive

    Returns
    -------
    DataFrame with columns: date, sunrise, sunset  (timezone-aware, SITE_TIMEZONE)
    """
    try:
        from suntime import Sun
    except ImportError:
        raise ImportError("pip install suntime")

    if lat is None:
        lat = SITE_LAT
    if lon is None:
        lon = SITE_LON

    sun  = Sun(lat, lon)
    rows = []
    d    = date.fromisoformat(date_start)
    end  = date.fromisoformat(date_end)

    while d < end:
        try:
            sr = sun.get_sunrise_time(d).astimezone(SITE_TIMEZONE)
            ss = sun.get_sunset_time(d).astimezone(SITE_TIMEZONE)
            rows.append({"date": d, "sunrise": sr, "sunset": ss})
        except Exception as e:
            log.warning("Sun time failed for %s: %s", d, e)
        d += timedelta(days=1)

    df = pd.DataFrame(rows)
    log.info("Sun table: %d days (%s to %s)", len(df), date_start, date_end)
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build site sunrise/sunset table.")
    parser.add_argument("--output",     default="birdnet/suntimes.parquet")
    parser.add_argument("--date-start", default="2023-10-01")
    parser.add_argument("--date-end",   default="2026-01-01")
    parser.add_argument("--lat",  type=float, default=None)
    parser.add_argument("--lon",  type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    df = build_sun_table(lat=args.lat, lon=args.lon,
                         date_start=args.date_start, date_end=args.date_end)
    df.to_parquet(args.output, index=False)
    print(f"Written: {args.output}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
