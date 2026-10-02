# catalog/build.py
#
# Build and update the audio file catalog.
#
# Two-phase process:
#   1. index_directory()  — walk an audio directory, write a per-folder
#                           wavfiles.parquet cache.  Re-run to pick up new
#                           files; folders with an existing cache are skipped
#                           unless --force is given.
#   2. build_catalog()    — aggregate all per-folder caches into one master
#                           catalog parquet file.
#
# Convention:
#   - A 'donotindex.txt' file in any directory causes that subtree to be skipped.
#   - A 'meta.csv' (Kaleidoscope sidecar) anywhere up the tree supplies GPS
#     and corrected timestamps for S4A recorders.
#   - The 'timestamp_source' column records whether the timestamp came from
#     the filename ('filename') or the Kaleidoscope sidecar ('kal_meta').
#
# Usage:
#   python -m catalog.build --index                          # use AUDIO_TOPDIRS from config
#   python -m catalog.build --index /path/to/audio/drive1       # one specific drive
#   python -m catalog.build --catalog                        # aggregate all caches
#   python -m catalog.build --index --catalog                # full rebuild
#   python -m catalog.build --index --force /path/to/audio/drive1  # re-index even if cached

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import wave

from catalog.parse import parse_filename, iswave, S4A_UTC_OFFSET
from config import SITE_TIMEZONE, AUDIO_TOPDIRS, RECORDER_ZONES_PATH, CATALOG_PATH

log = logging.getLogger(__name__)

CACHE_FILENAME = "wavfiles.parquet"


# ---------------------------------------------------------------------------
# WAV metadata
# ---------------------------------------------------------------------------

def read_wav_metadata(file_path: str) -> pd.Series:
    """Read low-level WAV header metadata without loading audio data."""
    try:
        with wave.open(file_path, 'rb') as f:
            p = f.getparams()
            return pd.Series({
                'channels':         p.nchannels,
                'sample_width':     p.sampwidth,
                'rate':             p.framerate,
                'datalen':          p.nframes,
                'compression_type': p.compname,
            })
    except Exception as e:
        log.warning("Bad WAV metadata %s: %s", file_path, e)
        return pd.Series({
            'channels':         np.nan,
            'sample_width':     np.nan,
            'rate':             np.nan,
            'datalen':          np.nan,
            'compression_type': '',
        })


# ---------------------------------------------------------------------------
# Kaleidoscope sidecar (meta.csv) handling
# ---------------------------------------------------------------------------

def _setkmetadatetime(row: pd.Series) -> datetime:
    """Convert Kaleidoscope date + time columns to a tz-aware datetime.

    Kaleidoscope copies the recorder's own wall clock, so the sidecar carries
    the same fixed UTC-5 offset as the filename. Both paths use the one
    constant in catalog.parse -- see the timezone note at the top of that
    module for the cross-correlation measurement behind it.
    """
    try:
        naive = row['date'] + row['time']
        return naive.replace(tzinfo=S4A_UTC_OFFSET).astimezone(SITE_TIMEZONE)
    except Exception:
        return pd.NaT


def splitw4vfn(row: pd.Series, base_dir: str) -> pd.Series:
    """Map a Kaleidoscope W4V/WAV row back to the matching S4A WAV filenames."""
    fn = row['IN FILE']
    pattern = re.compile(
        r"(S4A\d{5})_(?:[01]_)?(\d{8})(.)(\d{6})(?:_000)?\.(?:wav|w4v)$"
    )
    match = pattern.search(fn)
    if not match:
        return pd.Series([None, None, None, None])

    recorder, fndate, gps_char, fntime = match.groups()
    fndt    = datetime.strptime(f"{fndate}{fntime}", "%Y%m%d%H%M%S")
    gpssync = (gps_char == '$')

    wavfile0 = wavfile1 = ""
    for td in range(4):
        timestr = (fndt + timedelta(seconds=td)).strftime("%H%M%S")
        candidate = os.path.join(base_dir, row['FOLDER'],
                                 f"{recorder}_0_{fndate}_{timestr}_000.wav")
        if os.path.exists(candidate):
            wavfile0 = f"{recorder}_0_{fndate}_{timestr}_000.wav"
            wavfile1 = f"{recorder}_1_{fndate}_{timestr}_000.wav"
            break

    return pd.Series([recorder, gpssync, wavfile0, wavfile1])


def loadkmeta(meta_csv: str) -> pd.DataFrame | None:
    """Load a Kaleidoscope meta.csv and return a tidy DataFrame.

    Returns None if the file cannot be read.
    """
    try:
        km = pd.read_csv(meta_csv, encoding_errors='ignore')
        km['FOLDER']          = km['FOLDER'].fillna("")
        km['date']            = pd.to_datetime(km.DATE)
        km['time']            = pd.to_timedelta(km.TIME)
        km['kmeta_datetime']  = km.apply(_setkmetadatetime, axis=1)
        km['kmeta_longitude'] = km.LONGITUDE
        km['kmeta_latitude']  = km.LATITUDE

        base_dir = os.path.dirname(meta_csv)
        km[['recorder', 'gps', 'wavfile0', 'wavfile1']] = km.apply(
            lambda r: splitw4vfn(r, base_dir), axis=1
        )
        km = km.rename(columns={
            'IN FILE':  'sourcefile',
            'DURATION': 'kmeta_duration',
            'recorder': 'kmeta_recorder',
            'gps':      'kmeta_gps',
        })
        return km[['sourcefile', 'kmeta_duration', 'kmeta_datetime',
                   'kmeta_recorder', 'kmeta_gps',
                   'kmeta_longitude', 'kmeta_latitude',
                   'wavfile0', 'wavfile1']]
    except Exception as e:
        log.warning("Could not load meta.csv %s: %s", meta_csv, e)
        return None


def _find_meta_csv(dirpath: str, topdir: str) -> str | None:
    """Walk up the directory tree from dirpath looking for the nearest meta.csv."""
    for parent in [dirpath] + [str(p) for p in Path(dirpath).parents]:
        candidate = os.path.join(parent, "meta.csv")
        if os.path.exists(candidate):
            return candidate
        if parent == topdir:
            break
    return None


# ---------------------------------------------------------------------------
# Timezone normalization
# ---------------------------------------------------------------------------

def _to_site_tz(series: pd.Series) -> pd.Series:
    """Coerce a datetime series to SITE_TIMEZONE, handling both aware and naive.

    Tz-naive input is treated as UTC then converted, which avoids DST ambiguity
    errors at fall-back transitions (e.g. 1:00 AM appearing twice in local time).
    """
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(SITE_TIMEZONE)
    # Interpret naive datetimes as UTC, then convert — never localize directly,
    # which fails on ambiguous DST boundary times.
    return s.dt.tz_localize("UTC").dt.tz_convert(SITE_TIMEZONE)


# ---------------------------------------------------------------------------
# Phase 1: index one directory tree
# ---------------------------------------------------------------------------

def index_directory(topdir: str, regions: gpd.GeoDataFrame,
                    force: bool = False) -> None:
    """Walk topdir and write a wavfiles.parquet into each subdirectory with WAV files.

    Parameters
    ----------
    topdir  : root directory to walk
    regions : GeoDataFrame with a 'group' column and polygon geometry
    force   : re-index directories that already have a cache
    """
    indexed = skipped = errors = 0

    for dirpath, dirnames, filenames in os.walk(topdir):
        if "donotindex.txt" in filenames:
            dirnames.clear()
            log.debug("Skipping (donotindex.txt): %s", dirpath)
            continue

        cache = os.path.join(dirpath, CACHE_FILENAME)
        if not force and os.path.exists(cache):
            skipped += 1
            log.debug("Already cached, skipping: %s", dirpath)
            continue

        wavpaths = [os.path.join(dirpath, f) for f in filenames if iswave(f)]
        if not wavpaths:
            continue

        try:
            wf = pd.DataFrame([parse_filename(w) for w in wavpaths])
            meta = wf.filepath.apply(read_wav_metadata)
            wf[meta.columns] = meta
            wf['duration'] = wf['datalen'] / wf['rate']
            wf = wf.drop(columns=['filepath', 'dirpath'])
            wf['timestamp_source'] = 'filename'

            # Merge Kaleidoscope sidecar if present
            meta_csv = _find_meta_csv(dirpath, topdir)
            if meta_csv:
                log.debug("Using meta.csv: %s", meta_csv)
                km = loadkmeta(meta_csv)
                if km is not None:
                    kmm = km.melt(
                        id_vars=[c for c in km.columns
                                 if c not in ('wavfile0', 'wavfile1')],
                        value_vars=['wavfile0', 'wavfile1'],
                        var_name='_ab', value_name='filename',
                    ).drop(columns='_ab')
                    wf = wf.merge(kmm, on='filename', how='left')

                    has_kal = wf['kmeta_datetime'].notna()
                    wf.loc[has_kal, 'gps_sync']          = wf.loc[has_kal, 'kmeta_gps']
                    wf.loc[has_kal, 'start_datetime']     = wf.loc[has_kal, 'kmeta_datetime']
                    wf.loc[has_kal, 'longitude']          = wf.loc[has_kal, 'kmeta_longitude']
                    wf.loc[has_kal, 'latitude']           = wf.loc[has_kal, 'kmeta_latitude']
                    wf.loc[has_kal, 'timestamp_source']   = 'kal_meta'
                    wf = wf.drop(columns=[c for c in wf.columns
                                          if c.startswith('kmeta_')])

            wf = wf[wf.start_datetime.notna()]
            wf = wf[wf.datalen > 0]

            wf['start_datetime'] = _to_site_tz(wf['start_datetime'])
            wf['end_datetime']   = wf['start_datetime'] + pd.to_timedelta(
                wf['duration'], unit='s'
            )

            # Assign recorder_group via spatial join
            valid_coords = wf.latitude.notna() & wf.longitude.notna()
            wf['recorder_group'] = pd.NA
            if valid_coords.any():
                wf_gpd = gpd.GeoDataFrame(
                    wf[valid_coords],
                    geometry=gpd.points_from_xy(
                        wf.loc[valid_coords, 'longitude'],
                        wf.loc[valid_coords, 'latitude'],
                    ),
                    crs=regions.crs,
                )
                joined = gpd.sjoin(wf_gpd, regions[['group', 'geometry']],
                                   how='left', predicate='within')
                wf.loc[valid_coords, 'recorder_group'] = joined['group'].values

            wf.sort_values('start_datetime').to_parquet(cache, index=False)
            log.info("Indexed %4d files → %s", len(wf), dirpath)
            indexed += 1

        except Exception as e:
            log.error("Error indexing %s: %s", dirpath, e, exc_info=True)
            errors += 1

    log.info("index_directory done — indexed: %d, skipped: %d, errors: %d",
             indexed, skipped, errors)


# ---------------------------------------------------------------------------
# Phase 2: aggregate per-folder caches into master catalog
# ---------------------------------------------------------------------------

def build_catalog(topdirs: list[str] | None = None,
                  catalog_path: str | None = None,
                  deployments_path: str | None = None) -> pd.DataFrame:
    """Aggregate all per-folder wavfiles.parquet files into a master catalog.

    Parameters
    ----------
    topdirs          : directories to search; defaults to AUDIO_TOPDIRS from config
    catalog_path     : output path; defaults to CATALOG_PATH from config
    deployments_path : optional path to deployments.parquet; if provided, the
                       recorder_group column is updated using the deployment table
                       (deployment groups override the zones spatial-join result)

    Returns
    -------
    pd.DataFrame  The deduplicated, cleaned catalog.
    """
    if topdirs is None:
        topdirs = AUDIO_TOPDIRS
    if catalog_path is None:
        catalog_path = CATALOG_PATH

    frames: list[pd.DataFrame] = []
    bad:    list[str]          = []

    for topdir in topdirs:
        for dirpath, dirnames, filenames in os.walk(topdir):
            if "donotindex.txt" in filenames:
                dirnames.clear()
                continue
            cache = os.path.join(dirpath, CACHE_FILENAME)
            if os.path.exists(cache):
                try:
                    chunk = pd.read_parquet(cache)
                    chunk['dirpath'] = dirpath
                    frames.append(chunk)
                except Exception as e:
                    log.warning("Could not read %s: %s", cache, e)
                    bad.append(cache)

    if not frames:
        raise RuntimeError(
            "No wavfiles.parquet caches found — run with --index first."
        )

    wf = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d rows from %d cache(s)", len(wf), len(frames))

    # Reconstruct full file path
    wf['file'] = wf.apply(
        lambda r: str(Path(r['dirpath']) / r['filename']), axis=1
    )

    # Deduplicate (same SD card copied across drives — keep first alphabetically)
    before = len(wf)
    wf = wf.sort_values('dirpath')
    wf = wf[~wf.duplicated(subset=['recorder', 'filename'], keep='first')]
    log.info("Dropped %d duplicates", before - len(wf))

    # Drop S4A stereo second channel
    wf = wf[wf.channel != "1"]

    # Normalize timestamps
    wf['start_datetime'] = (pd.to_datetime(wf['start_datetime'], utc=True)
                              .dt.tz_convert(SITE_TIMEZONE))
    wf['end_datetime'] = (wf['start_datetime']
                          + pd.to_timedelta(wf['duration'], unit='s'))

    cols = ['file', 'recorder', 'recorder_type', 'channel', 'gps_sync',
            'start_datetime', 'end_datetime', 'duration',
            'latitude', 'longitude', 'recorder_group', 'timestamp_source']
    wf = wf[[c for c in cols if c in wf.columns]]

    # Apply deployment-based recorder_group overrides if provided
    if deployments_path and os.path.exists(deployments_path):
        from catalog.deployments import apply_deployment_metadata
        # WARN IF THE PARQUET IS OLDER THAN THE CSV IT DERIVES FROM. This read
        # a three-month-old deployments.parquet while the CSV had been edited
        # minutes earlier, so two full catalog rebuilds produced byte-identical
        # orphan counts (14,427 both times) and the edits looked ineffective
        # rather than unread. Nothing in the log said which table was in use.
        try:
            from config import DEPLOYMENTS_CSV_PATH as _csv
        except Exception:
            _csv = os.path.join("user_files", "recorder_info",
                                "recorder_deployments.csv")
        if os.path.exists(_csv) and \
                os.path.getmtime(_csv) > os.path.getmtime(deployments_path):
            log.warning(
                "%s is NEWER than %s -- the catalog is about to use a stale "
                "deployment table. Run `python -m catalog.deployments` first.",
                _csv, deployments_path)
        deps = pd.read_parquet(deployments_path)
        log.info("deployment table: %s (%d rows, built %s)", deployments_path,
                 len(deps),
                 __import__("datetime").datetime.fromtimestamp(
                     os.path.getmtime(deployments_path)).strftime("%Y-%m-%d %H:%M"))
        wf = apply_deployment_metadata(wf, deps)
        log.info("recorder_group and GPS updated from deployment table")

    wf.to_parquet(catalog_path, index=False)
    log.info("Catalog written: %s  (%d files)", catalog_path, len(wf))

    if bad:
        log.warning("%d cache(s) could not be read: %s", len(bad), bad)

    return wf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_regions(zones_path: str) -> gpd.GeoDataFrame:
    regions = gpd.read_file(zones_path)
    regions = regions[regions.name.str.contains("RecorderZone")]
    regions['group'] = regions.name.str.replace("RecorderZone_", "")
    return regions[['group', 'geometry']]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and maintain the audio file catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all drives from config, then build master catalog
  python -m catalog.build --index --catalog

  # Index a single drive only
  python -m catalog.build --index /path/to/audio/drive1

  # Re-index a drive (ignore existing caches), then rebuild catalog
  python -m catalog.build --index --force /path/to/audio/drive1 --catalog

  # Use a non-default zones file
  python -m catalog.build --index --zones /path/to/recorder_zones.json
""",
    )
    parser.add_argument(
        "--index", nargs="*", metavar="DIR",
        help="Phase 1: index directories. Pass one or more paths to override "
             "AUDIO_TOPDIRS from config, or omit to use config.",
    )
    parser.add_argument(
        "--catalog", nargs="?", const=CATALOG_PATH, metavar="OUTPUT",
        help="Phase 2: aggregate caches into master catalog parquet. "
             "Optionally specify output path (default: %(const)s).",
    )
    parser.add_argument(
        "--zones", default=RECORDER_ZONES_PATH, metavar="PATH",
        help="Path to recorder_zones GeoJSON (default: %(default)s).",
    )
    parser.add_argument(
        "--deployments", default="data/deployments.parquet", metavar="PATH",
        help="Deployment table parquet for recorder_group overrides "
             "(default: data/deployments.parquet). Pass 'none' to skip.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-index directories that already have a cache.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.index is not None:
        topdirs = args.index if args.index else AUDIO_TOPDIRS
        regions = _load_regions(args.zones)
        for td in topdirs:
            log.info("Indexing %s ...", td)
            index_directory(td, regions, force=args.force)

    if args.catalog is not None:
        topdirs = args.index if args.index else None
        deps_path = None if args.deployments.lower() == "none" else args.deployments
        build_catalog(topdirs=topdirs, catalog_path=args.catalog,
                      deployments_path=deps_path)


if __name__ == "__main__":
    main()
