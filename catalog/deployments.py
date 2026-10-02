# catalog/deployments.py
#
# Build and maintain the recorder deployment table.
#
# Each row describes one deployment of one recorder:
#   recorder, recorder_group, begin, end (NaT = still active),
#   latitude, longitude, elevation_m (from DSM), height_above_ground,
#   z (= elevation_m + height_above_ground)
#
# GPS coordinates come from:
#   1. Survey GPS in the deployment CSV (preferred)
#   2. Median lat/lon from WAV filenames for that recorder + date range
#
# z is derived by sampling a DSM at (lat, lon) and adding height_above_ground.
# The recorder_group in the CSV overrides the catalog's spatial-join assignment.
#
# Usage:
#   python -m catalog.deployments
#
# Output: data/deployments.parquet (path configurable via DEPLOYMENTS_PATH in config)

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load deployment CSV
# ---------------------------------------------------------------------------

def load_deployments(csv_path: str) -> pd.DataFrame:
    """Load the recorder deployment spreadsheet.

    Required columns: recorder, recorder_group, begin, end,
                      height_above_ground, Longitude, Latitude
    Optional: status, mount, notes, and GPS quality fields (ignored).
    """
    df = pd.read_csv(csv_path)

    # Normalise column names to lowercase for internal use
    rename = {
        "Longitude": "longitude",
        "Latitude":  "latitude",
    }
    df = df.rename(columns=rename)

    from config import SITE_TIMEZONE
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce").dt.tz_localize(SITE_TIMEZONE)
    df["end"]   = pd.to_datetime(df["end"],   errors="coerce").dt.tz_localize(SITE_TIMEZONE)

    # Keep only the columns we need
    keep = ["recorder", "recorder_group", "begin", "end",
            "status", "mount", "height_above_ground",
            "longitude", "latitude"]
    df = df[[c for c in keep if c in df.columns]]

    # AN ACTIVE DEPLOYMENT HAS NO END DATE. Three rows once carried both, and
    # the consequence was invisible: every file a recorder produced after that
    # spurious end fell outside all of its windows and lost its
    # recorder_group. It accounted for 4,242 orphaned catalog rows, and nothing
    # reported it -- apply_deployment_metadata logs a count, not a cause.
    # Normalised here rather than left to the CSV, so the invariant holds
    # however the table is edited.
    bad = df["status"].eq("active") & df["end"].notna()
    if bad.any():
        for _, r in df[bad].iterrows():
            log.warning("active deployment with an end date: %s %s ends %s "
                        "-- clearing it (set status if it really ended)",
                        r["recorder"], r.get("recorder_group", "?"),
                        r["end"].date())
        df.loc[bad, "end"] = pd.NaT

    # Empty lat/lon strings → NaN
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")

    log.info("Loaded %d deployments from %s", len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# Fill missing GPS from WAV filenames
# ---------------------------------------------------------------------------

def fill_missing_gps(deployments: pd.DataFrame,
                     catalog: pd.DataFrame) -> pd.DataFrame:
    """For deployments lacking survey GPS, use median lat/lon from WAV filenames.

    Matches catalog rows to deployment by recorder and overlapping date range.
    Solarbar filenames embed GPS; S4A filenames do not (those deployments will
    remain without coordinates if neither the CSV nor WAV filenames provide them).
    """
    needs_gps = deployments["latitude"].isna() | deployments["longitude"].isna()
    if not needs_gps.any():
        return deployments

    if "latitude" not in catalog.columns or "longitude" not in catalog.columns:
        log.warning("Catalog has no lat/lon columns — cannot fill missing GPS")
        return deployments

    dep = deployments.copy()

    for idx in dep[needs_gps].index:
        row  = dep.loc[idx]
        rec  = row["recorder"]
        beg  = row["begin"]
        end  = row["end"]   # NaT = still active

        mask = catalog["recorder"] == rec
        if beg is not pd.NaT:
            mask &= catalog["start_datetime"] >= beg
        if end is not pd.NaT and pd.notna(end):
            mask &= catalog["start_datetime"] <= end

        sub = catalog[mask].dropna(subset=["latitude", "longitude"])
        if sub.empty:
            log.debug("No WAV GPS for %s %s–%s", rec, beg, end)
            continue

        dep.loc[idx, "latitude"]  = sub["latitude"].median()
        dep.loc[idx, "longitude"] = sub["longitude"].median()
        log.info("Filled GPS for %s %s–%s from %d WAV files",
                 rec, beg, end, len(sub))

    return dep


# ---------------------------------------------------------------------------
# DSM elevation lookup
# ---------------------------------------------------------------------------

def sample_dsm(lat: float, lon: float, dsm_path: str) -> float | None:
    """Sample the DSM at a single (lat, lon) point and return elevation in metres.

    The DSM must be in a geographic CRS (EPSG:4326) with orthometric heights
    (EGM96). Returns None if the point is outside the raster or at a nodata cell.
    """
    try:
        import rasterio
        with rasterio.open(dsm_path) as src:
            row, col = src.index(lon, lat)
            # Bounds check
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            data = src.read(1)
            val  = float(data[row, col])
            if src.nodata is not None and abs(val - src.nodata) < 1.0:
                return None
            return val
    except Exception as e:
        log.warning("DSM sample failed at (%.5f, %.5f): %s", lat, lon, e)
        return None


def add_elevations(deployments: pd.DataFrame, dsm_path: str) -> pd.DataFrame:
    """Sample the DSM at each deployment's (lat, lon) and compute z.

    Adds columns:
        elevation_m  : ground elevation from DSM (metres above EGM96 geoid)
        z            : elevation_m + height_above_ground
    """
    dep = deployments.copy()

    elevations = []
    for _, row in dep.iterrows():
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            elev = sample_dsm(row["latitude"], row["longitude"], dsm_path)
        else:
            elev = None
        elevations.append(elev)

    dep["elevation_m"] = elevations

    hag = pd.to_numeric(dep.get("height_above_ground", 0), errors="coerce").fillna(0)
    dep["z"] = dep["elevation_m"] + hag

    n_ok  = dep["elevation_m"].notna().sum()
    n_bad = len(dep) - n_ok
    log.info("DSM elevations: %d succeeded, %d failed (outside raster or no coords)",
             n_ok, n_bad)
    return dep


# ---------------------------------------------------------------------------
# Match a file date to a deployment
# ---------------------------------------------------------------------------

def apply_deployment_metadata(catalog: pd.DataFrame,
                              deployments: pd.DataFrame) -> pd.DataFrame:
    """Override recorder_group, latitude, and longitude in *catalog* using
    the deployment table.

    For each deployment row, finds all catalog files for that recorder within
    the deployment date range and updates:
      - recorder_group : deployment group overrides spatial-join result
      - latitude       : survey GPS (more accurate than recorder built-in GPS)
      - longitude      : survey GPS

    The deployment table is authoritative for group and coordinates, ensuring
    the catalog is consistent with what is used for localization. A file
    whose recorder appears in the deployment table but whose timestamp falls
    outside every documented window for that recorder (e.g. pre-deployment
    power-on testing at a staging location) gets its recorder_group nulled
    out, rather than silently keeping the earlier spatial-join guess -- that
    guess reflects wherever the recorder physically was at that moment, not
    a real field deployment, and left uncorrected it can masquerade as
    genuine activity in whatever group the staging location's GPS zone
    happens to match.

    Parameters
    ----------
    catalog     : wavfiles catalog with columns recorder, start_datetime,
                  recorder_group, latitude, longitude
    deployments : deployment table from build_deployment_table()

    Returns
    -------
    Updated catalog DataFrame.
    """
    result = catalog.copy()
    n_updated = 0
    matched_any = pd.Series(False, index=result.index)

    for _, dep in deployments.iterrows():
        mask = result["recorder"] == dep["recorder"]
        if pd.notna(dep.get("begin")):
            mask &= result["start_datetime"] >= dep["begin"]
        if pd.notna(dep.get("end")):
            # `end` IS INCLUSIVE OF ITS WHOLE DAY. The dates in
            # recorder_deployments.csv are days, not instants, so an end of
            # 2024-05-20 means "through the end of the 20th". Comparing against
            # midnight orphaned everything a recorder produced on its own final
            # day -- 360 files across the archive, and it silently defeated a
            # deliberate end-date correction on S4A20227, whose last five files
            # at NorthSide were recorded that evening.
            # Safe because every consecutive deployment pair for a given
            # recorder has at least a one-day gap, so no window can overlap.
            mask &= result["start_datetime"] < dep["end"] + pd.Timedelta(days=1)

        if not mask.any():
            continue

        matched_any |= mask

        grp = dep.get("recorder_group")
        if grp and pd.notna(grp):
            result.loc[mask, "recorder_group"] = grp

        lat = dep.get("latitude")
        lon = dep.get("longitude")
        if pd.notna(lat) and pd.notna(lon):
            result.loc[mask, "latitude"]  = float(lat)
            result.loc[mask, "longitude"] = float(lon)

        n_updated += mask.sum()

    log.info("apply_deployment_metadata: updated %d catalog rows", n_updated)

    known_recorder = result["recorder"].isin(deployments["recorder"].unique())
    outside_window = known_recorder & ~matched_any
    if outside_window.any():
        log.info(
            "apply_deployment_metadata: nulling recorder_group for %d rows "
            "outside every documented deployment window for their recorder",
            outside_window.sum(),
        )
        result.loc[outside_window, "recorder_group"] = None
    return result


# Keep old name as alias for backwards compatibility
apply_deployment_groups = apply_deployment_metadata


def match_deployment(recorder: str,
                     file_date: pd.Timestamp,
                     deployments: pd.DataFrame) -> pd.Series | None:
    """Return the deployment row for a recorder at a given date, or None."""
    mask = deployments["recorder"] == recorder
    mask &= deployments["begin"] <= file_date
    # end NaT = still active (no upper bound)
    active = deployments["end"].isna()
    mask &= active | (deployments["end"] >= file_date)

    matches = deployments[mask]
    if matches.empty:
        return None
    if len(matches) > 1:
        log.debug("Multiple deployments for %s at %s — using first", recorder, file_date)
    return matches.iloc[0]


# ---------------------------------------------------------------------------
# Build full deployment table
# ---------------------------------------------------------------------------

def build_deployment_table(csv_path: str,
                            dsm_path: str,
                            catalog: pd.DataFrame | None = None,
                            output_path: str | None = None) -> pd.DataFrame:
    """Load deployments, fill missing GPS, compute z, optionally save.

    Parameters
    ----------
    csv_path    : path to recorder_deployments.csv
    dsm_path    : path to DSM GeoTIFF (WGS84/EGM96)
    catalog     : wavfiles catalog — used to fill missing GPS from WAV filenames
    output_path : where to write the result parquet (None = don't save)
    """
    deps = load_deployments(csv_path)

    if catalog is not None:
        deps = fill_missing_gps(deps, catalog)

    n_missing = (deps["latitude"].isna() | deps["longitude"].isna()).sum()
    if n_missing:
        log.warning("%d deployment(s) still have no coordinates after GPS fill", n_missing)

    deps = add_elevations(deps, dsm_path)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        deps.to_parquet(output_path, index=False)
        log.info("Deployment table written: %s  (%d rows)", output_path, len(deps))

    return deps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    from config import CATALOG_PATH

    _DEFAULT_CSV = str(Path(__file__).parent.parent /
                        "user_files/recorder_info/recorder_deployments.csv")
    _DEFAULT_DSM = str(Path(__file__).parent.parent /
                        "user_files/dem/DEM_State_3DEP_TIF_AR_1m_x45y37_ferg_WGS84_EGM96.tif")
    _DEFAULT_OUT = "data/deployments.parquet"

    parser = argparse.ArgumentParser(
        description="Build recorder deployment table with z coordinates.")
    parser.add_argument("--csv",     default=_DEFAULT_CSV,
                        help="deployment CSV (default is this site's -- override for your own site)")
    parser.add_argument("--dsm",     default=_DEFAULT_DSM,
                        help="elevation raster (default is this site's -- override for your own site)")
    parser.add_argument("--catalog", default=CATALOG_PATH,
                        help="wavfiles catalog for GPS fill (parquet or pkl)")
    parser.add_argument("--output",  default=_DEFAULT_OUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load catalog for GPS fill
    catalog = None
    if args.catalog and os.path.exists(args.catalog):
        if args.catalog.endswith(".parquet"):
            catalog = pd.read_parquet(args.catalog)
        else:
            catalog = pd.read_pickle(args.catalog)
        log.info("Catalog loaded: %d rows", len(catalog))
    else:
        log.warning("No catalog found — GPS fill from WAV filenames disabled")

    deps = build_deployment_table(args.csv, args.dsm, catalog, args.output)

    print(f"\nDeployment table ({len(deps)} rows):")
    print(deps[["recorder", "recorder_group", "begin", "end",
                "latitude", "longitude", "elevation_m", "z"]].to_string())


if __name__ == "__main__":
    main()
