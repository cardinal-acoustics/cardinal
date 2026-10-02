# localize/temperature.py
#
# Fetch historical air temperature for a site and convert it to speed of
# sound, for use as the initial guess (and tight search bound) passed to
# localize_plot_withorig's TDOA solver via row.speed.
#
# Source: Open-Meteo's historical weather archive API
# (https://open-meteo.com/en/docs/historical-weather-api) -- no API key,
# no signup, no rate limit for non-commercial use. Backed by reanalysis
# data (ERA5), so it covers essentially any past date globally.
#
# Usage:
#   from localize.temperature import fetch_temperature, attach_speed
#   temps = fetch_temperature(lat, lon, "2024-06-12", "2024-06-12", tz="America/Chicago")
#   candidates = attach_speed(candidates, temps)   # adds a 'speed' column
#
#   python -m localize.temperature --lat 33.7271 --lon -93.3066 \
#       --start 2024-06-01 --end 2024-06-30 --tz America/Chicago \
#       --out demo/outputs/temperature.pkl

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# HRRR: 3 km, and better than ERA5's 9 km here. Measured against the five
# ATMOS 41 stations, night bias +2.3 vs +3.4 degC, sd 1.7 vs 2.2, r 0.88 vs
# 0.71; and the tone generators -- which measure c acoustically rather than
# deriving it -- correlate better with HRRR at all three arrays. CONUS only,
# so non-US sites fall back to ERA5 automatically.
FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HRRR_MODEL = "gfs_hrrr"
HRRR_BOX = (21.0, 53.0, -134.0, -60.0)     # lat_min, lat_max, lon_min, lon_max

# Both grids run warm at night at this site. METER stations put HRRR +2.3 degC
# over 21:00-06:00 and near zero by day; 47k trail-camera night readings from
# 2021-2026 give -2.2 degC independently. Applied as a flat offset per period
# rather than a fitted curve -- the two instruments agree on the size, not on
# a shape.
NIGHT_HOURS = (21, 6)
DEFAULT_NIGHT_BIAS_C = -2.3
DEFAULT_DAY_BIAS_C = -0.3


def speed_of_sound(temp_c: np.ndarray | float,
                   rh_pct: np.ndarray | float | None = None) -> np.ndarray | float:
    """Speed of sound (m/s) at a temperature (deg C) and optional humidity.

    Dry air is the ideal-gas formula c = sqrt(gamma * R_air * T_kelvin),
    gamma=1.4, R_air=287.05 J/(kg*K).

    Humidity is NOT negligible here, contrary to what this function assumed
    for a long time. Water vapour is lighter than the N2/O2 it displaces, so
    humid air is faster. Measured against 7,775 ATMOS 41 readings at this
    site, the correction is +1.31 m/s on average and up to +2.06 -- larger
    than the entire ERA5-to-HRRR difference. Arkansas summer nights sit at
    85% RH, not the temperate mid-range where "a few tenths" holds. It is
    also seasonal (+1.6 m/s in July, +0.7 in November), so omitting it
    injects a spurious annual cycle.

    The first-order term used here gives +1.25 m/s at 30 degC / 84% RH; the
    textbook 331.3 + 0.606*T + 0.0124*RH approximation gives +1.04 for the
    same conditions, so treat the humid value as good to a few tenths, not
    to the second decimal.

    Pass rh_pct=None for the dry-air value (what the 2026 corpus run used).
    """
    t = np.asarray(temp_c, dtype=float)
    c_dry = np.sqrt(1.4 * 287.05 * (t + 273.15))
    if rh_pct is None:
        return c_dry
    rh = np.asarray(rh_pct, dtype=float)
    es = 0.61078 * np.exp(17.27 * t / (t + 237.3))      # kPa, Tetens
    x_w = np.clip(rh, 0.0, 100.0) / 100.0 * es / 101.325
    return c_dry * (1.0 + 0.165 * x_w)


def _is_conus(lat: float, lon: float) -> bool:
    la0, la1, lo0, lo1 = HRRR_BOX
    return la0 <= lat <= la1 and lo0 <= lon <= lo1


def apply_diurnal_bias(df: pd.DataFrame, night_bias: float, day_bias: float,
                       tz: str) -> pd.DataFrame:
    """Subtract the grid's measured warm bias, by time of day."""
    hours = df["dt"].dt.tz_convert(tz).dt.hour
    n0, n1 = NIGHT_HOURS
    night = (hours >= n0) | (hours < n1)
    df["temp_c"] = df["temp_c"] + np.where(night, night_bias, day_bias)
    return df


def fetch_temperature(lat: float, lon: float,
                      start_date: str, end_date: str,
                      tz: str = "UTC",
                      source: str = "hrrr",
                      humidity: bool = True,
                      night_bias: float | None = DEFAULT_NIGHT_BIAS_C,
                      day_bias: float | None = DEFAULT_DAY_BIAS_C) -> pd.DataFrame:
    """Fetch hourly historical temperature for a site and date range.

    Parameters
    ----------
    lat, lon   : site coordinates (WGS84)
    start_date, end_date : "YYYY-MM-DD", inclusive
    tz         : IANA timezone name for the returned timestamps (matches
                 SITE_TIMEZONE in cardinal.toml)

    Returns
    -------
    DataFrame with columns:
        dt      tz-aware hourly timestamp
        temp_c  air temperature (deg C)
        speed   speed of sound (m/s)
    """
    import requests

    # Request UTC rather than the site's local timezone: Open-Meteo returns a
    # naive wall-clock label per hour, and over a multi-year range that hits
    # real DST transitions -- a repeated local label at fall-back with only
    # one row (not a genuine duplicate pair), which pandas can't disambiguate
    # from the label alone. UTC has no DST, so it's unambiguous by
    # construction; we localize to UTC (always safe) and convert to the site
    # timezone afterward. Pad the requested range by a day on each side so
    # the UTC window still fully covers the requested *local* calendar dates.
    start_pad = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_pad   = (pd.Timestamp(end_date)   + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    want = ["temperature_2m"] + (["relative_humidity_2m"] if humidity else [])
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_pad,
        "end_date": end_pad,
        "hourly": ",".join(want),
        "timezone": "UTC",
    }

    used = source
    if source == "hrrr" and not _is_conus(lat, lon):
        log.warning("fetch_temperature: %.3f,%.3f is outside HRRR's CONUS "
                    "domain; falling back to ERA5", lat, lon)
        used = "era5"
    if used == "hrrr":
        try:
            resp = requests.get(FORECAST_URL,
                                params={**params, "models": HRRR_MODEL},
                                timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            # the historical-forecast archive lags real time by a few days and
            # does not reach back before HRRR's own record
            log.warning("fetch_temperature: HRRR request failed (%s); "
                        "falling back to ERA5", exc)
            used = "era5"
    if used != "hrrr":
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "dt": pd.to_datetime(hourly["time"]),
        "temp_c": hourly["temperature_2m"],
    })
    if humidity and "relative_humidity_2m" in hourly:
        df["rh"] = hourly["relative_humidity_2m"]
    df["dt"] = df["dt"].dt.tz_localize("UTC").dt.tz_convert(tz)
    df = df.dropna(subset=["temp_c"]).reset_index(drop=True)

    df["temp_c_raw"] = df["temp_c"]
    if night_bias is not None or day_bias is not None:
        df = apply_diurnal_bias(df, night_bias or 0.0, day_bias or 0.0, tz)
    df["speed"] = speed_of_sound(df["temp_c"], df["rh"] if "rh" in df else None)
    df["source"] = used

    log.info("fetch_temperature: %d hourly records from %s, %s to %s, "
             "%.1f-%.1f degC, c %.1f-%.1f m/s", len(df), used, start_date,
             end_date, df["temp_c"].min(), df["temp_c"].max(),
             df["speed"].min(), df["speed"].max())
    return df


def attach_speed(candidates: pd.DataFrame, temps: pd.DataFrame,
                 call_dt_col: str = "call_datetime") -> pd.DataFrame:
    """Attach a per-row 'speed' column by matching each candidate to the
    nearest hourly temperature reading.

    Parameters
    ----------
    candidates : detections/candidates table with a tz-aware datetime column
    temps      : output of fetch_temperature()
    call_dt_col: name of the datetime column in *candidates* to match on
    """
    left = candidates.sort_values(call_dt_col)
    right = temps.sort_values("dt")
    merged = pd.merge_asof(
        left, right[["dt", "temp_c", "speed"]],
        left_on=call_dt_col, right_on="dt", direction="nearest",
    )
    merged.index = left.index  # merge_asof drops the original index; restore it
    return merged.drop(columns=["dt"]).reindex(candidates.index)


def save_temperature(temps: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temps.to_pickle(path)
    log.info("Temperature data saved: %s (%d rows)", path, len(temps))


def load_temperature(path: str) -> pd.DataFrame | None:
    import os
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical hourly temperature / speed of sound for a site.")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--tz", default="UTC")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")
    temps = fetch_temperature(args.lat, args.lon, args.start, args.end, tz=args.tz)
    save_temperature(temps, args.out)


if __name__ == "__main__":
    main()
