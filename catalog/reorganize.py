# catalog/reorganize.py
#
# Audit and rename recorder deployment directories to the canonical convention:
#
#   {topdir}/{YYYYMMDD}/{RecorderGroup}/{RecorderID}_{StartDate}_{EndDate}/
#
# The YYYYMMDD level is the field-collection date (already in the path).
# The RecorderGroup level is whatever is between YYYYMMDD and the recorder dir.
# The recorder directory is renamed; everything inside it is left unchanged.
#
# Solarbar SD card example:
#   Before: NorthSide/BARLT_00008051/
#             BARLT_00008051/           ← SD card audio directory (unchanged)
#             Reclog.csv
#             GPS_log.gpx
#   After:  NorthSide/BARLT_00008051_20240521_20240714/
#             BARLT_00008051/           ← untouched
#             Reclog.csv
#             GPS_log.gpx
#
# Safety: --dry-run is the default.  Pass --execute to rename for real.
# A rename manifest (JSON) is written before any renames so they can be undone.
#
# Usage:
#   python -m catalog.reorganize --scan /path/to/audio/drive1
#   python -m catalog.reorganize --scan /Volumes/... --execute
#   python -m catalog.reorganize --undo manifest_20240601_120000.json

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, date
from pathlib import Path

from catalog.parse import parse_filename, iswave

log = logging.getLogger(__name__)

# Recorder ID patterns
_SOLARBAR_RE = re.compile(r"^BARLT_\d{8}$")
_S4A_RE      = re.compile(r"^S4A\d{5}.*$")


def _is_recorder_dir(name: str) -> bool:
    """Return True if *name* looks like a recorder ID directory."""
    return bool(_SOLARBAR_RE.match(name) or _S4A_RE.match(name))


def _is_collection_date(name: str) -> bool:
    """Return True if *name* is an 8-digit date (YYYYMMDD)."""
    if len(name) != 8 or not name.isdigit():
        return False
    try:
        datetime.strptime(name, "%Y%m%d")
        return True
    except ValueError:
        return False


def _looks_like_sdcard_id(name: str) -> bool:
    """Return True if *name* looks like an SD card ID rather than a recorder group.

    SD card IDs are all-uppercase alphanumeric (with optional underscore separators),
    e.g. LI5ZAXUH, 3PLCI21P, 8QMRXH3B, S4_MMK8722A.
    Real group names (Back9, NorthSide, EastCreek) always contain lowercase letters.
    """
    if not name:
        return False
    # Strip trailing conflict suffix (_1, _2, ...)
    base = re.sub(r"_\d+$", "", name)
    # Real group names have at least one lowercase letter
    if any(c.islower() for c in base):
        return False
    # Must contain at least one letter (not just a date or number)
    if not any(c.isalpha() for c in base):
        return False
    # Must look like a hash or code: all-caps letters, digits, underscores only
    if not re.match(r"^[A-Z0-9_]{4,20}$", base):
        return False
    return True


def _recorder_id_from_wavs(dirpath: str) -> str | None:
    """Return the recorder ID found in WAV filenames under *dirpath*, or None.

    Used for S4A directories where the recorder name does not appear as a
    subdirectory name but is embedded in every WAV filename.
    Returns the most common recorder ID found.
    """
    from collections import Counter
    counts: Counter = Counter()
    for root, _, files in os.walk(dirpath):
        for fn in files:
            if not iswave(fn):
                continue
            try:
                r = parse_filename(os.path.join(root, fn))
                if r.recorder:
                    counts[r.recorder] += 1
            except Exception:
                pass
        if counts:
            break   # one directory's worth is enough
    return counts.most_common(1)[0][0] if counts else None


def _wav_date_range(dirpath: str) -> tuple[date | None, date | None]:
    """Return (earliest, latest) recording date found in WAV files under *dirpath*."""
    dates = []
    for root, _, files in os.walk(dirpath):
        for fn in files:
            if not iswave(fn):
                continue
            try:
                r = parse_filename(os.path.join(root, fn))
                if r.start_datetime is not None and str(r.start_datetime) != "NaT":
                    d = r.start_datetime
                    dates.append(d.date() if hasattr(d, "date") else d)
            except Exception:
                pass
    if not dates:
        return None, None
    return min(dates), max(dates)


def _group_from_gps(dirpath: str, zones) -> str | None:
    """Extract GPS from WAV filenames in *dirpath* and spatially join with *zones*."""
    if zones is None:
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import statistics
        lats, lons = [], []
        for root, _, files in os.walk(dirpath):
            for fn in files:
                if not iswave(fn):
                    continue
                try:
                    r = parse_filename(os.path.join(root, fn))
                    if r.latitude is not None and r.longitude is not None:
                        lats.append(float(r.latitude))
                        lons.append(float(r.longitude))
                except Exception:
                    pass
            if lats:
                break
        if not lats:
            return None
        lat, lon = statistics.median(lats), statistics.median(lons)
        pt     = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=zones.crs)
        joined = gpd.sjoin(pt, zones[["group", "geometry"]], how="left", predicate="within")
        group  = joined["group"].iloc[0]
        return str(group) if str(group) != "nan" else None
    except Exception:
        return None


def _canonical_name(recorder_id: str,
                    start: date | None,
                    end: date | None,
                    suffix: str = "") -> str:
    """Build the canonical directory name for a recorder deployment."""
    start_s = start.strftime("%Y%m%d") if start else "UNKNOWN"
    end_s   = end.strftime("%Y%m%d")   if end   else "UNKNOWN"
    name    = f"{recorder_id}_{start_s}_{end_s}"
    if suffix:
        name += f"_{suffix}"
    return name


def _load_catalog(catalog_path: str | None):
    """Load the wavfiles catalog, return a DataFrame or None."""
    if not catalog_path or not os.path.exists(catalog_path):
        return None
    try:
        import pandas as pd
        if catalog_path.endswith(".parquet"):
            return pd.read_parquet(catalog_path)
        return pd.read_pickle(catalog_path)
    except Exception as e:
        log.warning("Could not load catalog from %s: %s", catalog_path, e)
        return None


def _group_from_catalog(dirpath: str, catalog,
                        recorder_id: str | None = None) -> str | None:
    """Look up the recorder_group for a deployment directory.

    Two strategies, in order:
    1. Path-based: find catalog rows whose file path starts with dirpath.
       Fast and exact but fails when the catalog is stale (files moved).
    2. Recorder-ID-based: find catalog rows for recorder_id regardless of path.
       Path-independent — works even for S4A dirs with stale catalogs.
    """
    if catalog is None:
        return None

    def _majority(series):
        vals = series.dropna()
        if vals.empty:
            return None
        counts = vals.value_counts()
        if len(counts) > 1:
            log.info("Multiple groups %s — using majority: %s",
                     counts.index.tolist(), counts.index[0])
        return str(counts.index[0])

    # 1. Path-based lookup
    try:
        mask = catalog["file"].str.startswith(dirpath + os.sep)
        if mask.any():
            return _majority(catalog.loc[mask, "recorder_group"])
    except Exception as e:
        log.debug("Path-based catalog lookup failed for %s: %s", dirpath, e)

    # 2. Recorder-ID-based lookup (path-independent)
    if recorder_id and "recorder" in catalog.columns:
        try:
            mask = catalog["recorder"] == recorder_id
            if mask.any():
                return _majority(catalog.loc[mask, "recorder_group"])
        except Exception as e:
            log.debug("Recorder-ID catalog lookup failed for %s: %s", recorder_id, e)

    return None


def scan(topdir: str,
         catalog_path: str | None = None,
         zones_path: str | None = None) -> list[dict]:
    """Walk *topdir* and return a list of proposed renames.

    Each entry is a dict with keys:
        current   : current absolute path
        proposed  : proposed absolute path
        recorder  : recorder ID
        start     : earliest recording date (YYYY-MM-DD string or None)
        end       : latest recording date
        collection: YYYYMMDD collection-date directory name
        group     : recorder group directory name
        status    : "ok" | "already_canonical" | "no_wavs" | "conflict"
    """
    topdir  = str(Path(topdir).resolve())
    catalog = _load_catalog(catalog_path)
    if catalog is not None:
        log.info("Loaded catalog: %d rows", len(catalog))

    # Load zones for GPS-based group fallback (handles stale/missing catalog entries)
    zones = None
    if zones_path and os.path.exists(zones_path):
        try:
            import geopandas as gpd
            z = gpd.read_file(zones_path)
            z = z[z.name.str.contains("RecorderZone")]
            z["group"] = z.name.str.replace("RecorderZone_", "", regex=False)
            zones = z[["group", "geometry"]]
        except Exception as e:
            log.warning("Could not load zones: %s", e)

    # Known recorder group names — if a directory's parent is one of these,
    # the directory itself is the deployment dir (not its parent).
    known_groups: set[str] = set()
    if catalog is not None and "recorder_group" in catalog.columns:
        known_groups = set(catalog["recorder_group"].dropna().unique())

    proposals: list[dict] = []
    seen_proposed: dict[str, int] = {}   # proposed_path → count (for conflict handling)

    # Already-canonical pattern: RecorderID_StartDate_EndDate[_N]
    _canonical_re = re.compile(
        r"^(BARLT_\d{8}|S4A\d{5}[^_]*)_\d{8}_\d{8}(_\d+)?$"
    )
    seen_inner: set[str] = set()  # avoid processing the same to_rename twice

    for dirpath, dirnames, filenames in os.walk(topdir):
        dirnames.sort()

        inner_name = os.path.basename(dirpath)

        # We look for INNER directories whose name is exactly a recorder ID.
        # These are the SD card audio directories (BARLT_XXXXXXXX or S4A2XXXXX).
        # Their PARENT is what we actually rename.
        if not _is_recorder_dir(inner_name):
            continue

        # Determine what to rename.
        # If the inner dir's parent is a known RecorderGroup or a YYYYMMDD date,
        # the inner dir IS the deployment dir — rename it directly.
        # Otherwise the parent (e.g. BARLT_suffix, B1A, SD-card-ID) is the
        # deployment dir and that's what we rename.
        parent_of_inner      = os.path.dirname(dirpath)
        parent_of_inner_name = os.path.basename(parent_of_inner)

        if (parent_of_inner_name in known_groups
                or _is_collection_date(parent_of_inner_name)):
            # Inner dir sits directly under a group or date — rename inner itself
            to_rename = dirpath
        else:
            # Parent might be a wrapper (SD-card ID, position label, etc.)
            # BUT if the parent's OWN parent is a YYYYMMDD date and the parent
            # is not a recorder ID, the parent is a group-level fallback dir
            # (e.g. RecorderData) — rename inner, not the group dir.
            grandparent_of_inner      = os.path.dirname(parent_of_inner)
            grandparent_of_inner_name = os.path.basename(grandparent_of_inner)
            if (_is_collection_date(grandparent_of_inner_name)
                    and not _is_recorder_dir(parent_of_inner_name)):
                to_rename = dirpath
            else:
                to_rename = parent_of_inner

        to_rename_name = os.path.basename(to_rename)
        recorder_id    = inner_name   # always from the exact inner dir name

        if to_rename in seen_inner:
            continue
        seen_inner.add(to_rename)

        # Walk up from to_rename to find RecorderGroup and YYYYMMDD.
        p1      = os.path.dirname(to_rename)
        p1_name = os.path.basename(p1)
        p2      = os.path.dirname(p1)
        p2_name = os.path.basename(p2)

        if _is_collection_date(p1_name):
            collection_date = p1_name
            date_dir        = p1
            current_group   = None
        elif _is_collection_date(p2_name):
            collection_date = p2_name
            date_dir        = p2
            current_group   = p1_name
        else:
            log.debug("Skipping %s — can't find YYYYMMDD ancestor", to_rename)
            continue

        # Determine target group from catalog, current path, or fallback
        target_group = (_group_from_catalog(to_rename, catalog, recorder_id)
                        or _group_from_gps(to_rename, zones)
                        or (current_group if current_group and not _looks_like_sdcard_id(current_group) else None)
                        or "RecorderData")

        # If name is already canonical AND already in the right group, skip
        if _canonical_re.match(to_rename_name):
            if current_group == target_group:
                proposals.append({
                    "current": to_rename, "proposed": to_rename,
                    "recorder": recorder_id, "start": None, "end": None,
                    "collection": collection_date,
                    "current_group": current_group or "(none)",
                    "target_group":  target_group,
                    "status": "already_canonical",
                })
                dirnames.clear()
                continue
            # Name is canonical but in wrong group — fall through to propose a move

        # Find WAV date range from everything inside to_rename
        start, end = _wav_date_range(to_rename)
        if start is None:
            proposals.append({
                "current": to_rename, "proposed": None,
                "recorder": recorder_id, "start": None, "end": None,
                "collection": collection_date,
                "current_group": current_group or "(none)",
                "target_group":  target_group,
                "status": "no_wavs",
            })
            continue

        # Build proposed path: {date_dir}/{target_group}/{RecorderID_Start_End}
        canonical     = _canonical_name(recorder_id, start, end)
        base_proposed = os.path.join(date_dir, target_group, canonical)

        proposed = base_proposed
        count    = seen_proposed.get(base_proposed, 0)
        if count > 0:
            proposed = os.path.join(
                date_dir, target_group,
                _canonical_name(recorder_id, start, end, str(count + 1))
            )
        seen_proposed[base_proposed] = count + 1

        status = "ok"
        if proposed != base_proposed:
            status = "conflict"
        elif to_rename == proposed:
            status = "already_canonical"

        proposals.append({
            "current":       to_rename,
            "proposed":      proposed,
            "recorder":      recorder_id,
            "start":         start.isoformat() if start else None,
            "end":           end.isoformat()   if end   else None,
            "collection":    collection_date,
            "current_group": current_group or "(none)",
            "target_group":  target_group,
            "status":        status,
        })

        # Don't recurse further — to_rename is being handled as a unit
        dirnames.clear()

    # ---- Second pass: SD card ID dirs with WAV files but no inner recorder dir ----
    # Handles S4A recorders whose WAV files sit in KAL/Data subdirs, not in a
    # directory named after the recorder.
    for dirpath, dirnames, filenames in os.walk(topdir):
        dirnames.sort()
        dirname = os.path.basename(dirpath)

        # Only look at SD card ID directories not already handled
        if not _looks_like_sdcard_id(dirname):
            continue
        if dirpath in seen_inner:
            continue

        # Find YYYYMMDD and group
        parent      = os.path.dirname(dirpath)
        parent_name = os.path.basename(parent)
        grandparent      = os.path.dirname(parent)
        grandparent_name = os.path.basename(grandparent)

        if _is_collection_date(parent_name):
            collection_date, date_dir, current_group = parent_name, parent, None
        elif _is_collection_date(grandparent_name):
            collection_date, date_dir, current_group = grandparent_name, grandparent, parent_name
        else:
            continue

        # Extract recorder ID from WAV filenames
        recorder_id = _recorder_id_from_wavs(dirpath)
        if not recorder_id:
            continue

        seen_inner.add(dirpath)

        target_group = (_group_from_catalog(dirpath, catalog, recorder_id)
                        or _group_from_gps(dirpath, zones)
                        or (current_group if current_group and not _looks_like_sdcard_id(current_group) else None)
                        or "RecorderData")

        if _canonical_re.match(dirname) and current_group == target_group:
            proposals.append({"current": dirpath, "proposed": dirpath,
                               "recorder": recorder_id, "start": None, "end": None,
                               "collection": collection_date,
                               "current_group": current_group or "(none)",
                               "target_group": target_group,
                               "status": "already_canonical"})
            dirnames.clear()
            continue
        # If canonical name but wrong group, fall through to propose a move

        start, end = _wav_date_range(dirpath)
        if not start:
            proposals.append({"current": dirpath, "proposed": None,
                               "recorder": recorder_id, "start": None, "end": None,
                               "collection": collection_date,
                               "current_group": current_group or "(none)",
                               "target_group": target_group,
                               "status": "no_wavs"})
            continue

        canonical     = _canonical_name(recorder_id, start, end)
        base_proposed = os.path.join(date_dir, target_group, canonical)
        count = seen_proposed.get(base_proposed, 0)
        proposed = base_proposed if count == 0 else os.path.join(
            date_dir, target_group, _canonical_name(recorder_id, start, end, str(count + 1)))
        seen_proposed[base_proposed] = count + 1

        proposals.append({
            "current": dirpath, "proposed": proposed,
            "recorder": recorder_id,
            "start": start.isoformat(), "end": end.isoformat(),
            "collection": collection_date,
            "current_group": current_group or "(none)",
            "target_group": target_group,
            "status": "ok" if proposed == base_proposed else "conflict",
        })
        dirnames.clear()

    # ---- Third pass: canonical-named dirs in wrong group ----
    # Catches S4A dirs (no inner recorder subdir) and any dir manually renamed
    # to canonical form but left in RecorderData or wrong group.
    for dirpath, dirnames, filenames in os.walk(topdir):
        dirnames.sort()
        dirname = os.path.basename(dirpath)

        if not _canonical_re.match(dirname):
            continue
        if dirpath in seen_inner:
            dirnames.clear()
            continue

        # Extract recorder ID from the canonical name
        parts = dirname.split("_")
        if dirname.startswith("BARLT"):
            recorder_id = f"{parts[0]}_{parts[1]}"
        else:
            recorder_id = parts[0]   # S4A2XXXXX

        # Determine hierarchy
        parent      = os.path.dirname(dirpath)
        parent_name = os.path.basename(parent)
        gp          = os.path.dirname(parent)
        gp_name     = os.path.basename(gp)

        if _is_collection_date(parent_name):
            collection_date, date_dir, current_group = parent_name, parent, None
        elif _is_collection_date(gp_name):
            collection_date, date_dir, current_group = gp_name, gp, parent_name
        else:
            continue

        target_group = (_group_from_catalog(dirpath, catalog, recorder_id)
                        or _group_from_gps(dirpath, zones)
                        or (current_group if current_group
                            and not _looks_like_sdcard_id(current_group) else None)
                        or "RecorderData")

        if current_group == target_group:
            seen_inner.add(dirpath)
            dirnames.clear()
            continue   # truly in the right place

        # Propose move to correct group
        seen_inner.add(dirpath)
        base_proposed = os.path.join(date_dir, target_group, dirname)
        count = seen_proposed.get(base_proposed, 0)
        proposed = base_proposed if count == 0 else os.path.join(
            date_dir, target_group,
            _canonical_name(recorder_id,
                             date.fromisoformat(parts[-2]),
                             date.fromisoformat(parts[-1]),
                             str(count + 1)))
        seen_proposed[base_proposed] = count + 1

        proposals.append({
            "current":       dirpath,
            "proposed":      proposed,
            "recorder":      recorder_id,
            "start":         parts[-2],
            "end":           parts[-1],
            "collection":    collection_date,
            "current_group": current_group or "(none)",
            "target_group":  target_group,
            "status":        "ok" if proposed == base_proposed else "conflict",
        })
        dirnames.clear()

    return proposals


def report(proposals: list[dict]) -> None:
    """Print proposed renames in OLDPATH → NEWPATH format with brief notes."""
    ok       = [p for p in proposals if p["status"] == "ok"]
    conflict = [p for p in proposals if p["status"] == "conflict"]
    no_wavs  = [p for p in proposals if p["status"] == "no_wavs"]
    canon    = [p for p in proposals if p["status"] == "already_canonical"]

    print(f"\n{len(ok)+len(conflict)} to rename  |  "
          f"{len(canon)} already canonical  |  "
          f"{len(no_wavs)} skipped (no WAVs)\n")

    for p in ok + conflict:
        notes = []
        if p["status"] == "conflict":
            notes.append("conflict — suffixed to avoid duplicate")
        if p["current_group"] != p["target_group"] and p["current_group"] != "(none)":
            notes.append(f"group reassigned from {p['current_group']}")
        note_str = f"  # {', '.join(notes)}" if notes else ""

        print(p["current"])
        print(f"  → {p['proposed']}{note_str}")
        print()

    if no_wavs:
        print("Skipped (no WAV files found):")
        for p in no_wavs:
            print(f"  {p['current']}")
        print()

    if canon:
        print(f"Already canonical ({len(canon)} dirs) — no change needed.")


def execute(proposals: list[dict],
            manifest_dir: str = ".") -> str:
    """Execute the renames listed in *proposals*.

    Writes a manifest JSON file first so renames can be undone.
    Returns the manifest file path.
    """
    to_rename = [p for p in proposals if p["status"] in ("ok", "conflict")]
    if not to_rename:
        log.info("Nothing to rename.")
        return ""

    # Process deepest paths first so a parent move never orphans its children
    to_rename = sorted(to_rename,
                       key=lambda p: p["current"].count(os.sep),
                       reverse=True)

    # Write manifest
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(manifest_dir, f"rename_manifest_{ts}.json")
    with open(manifest_path, "w") as f:
        json.dump(to_rename, f, indent=2, default=str)
    log.info("Manifest written: %s", manifest_path)

    errors = 0
    for p in to_rename:
        src, dst = p["current"], p["proposed"]
        if src == dst:
            continue
        if os.path.exists(dst):
            log.error("Destination already exists, skipping: %s", dst)
            errors += 1
            continue
        try:
            # Create the target group directory if it doesn't exist yet
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            log.info("Moved: %s → %s", src, dst)
        except Exception as e:
            log.error("Failed to move %s: %s", src, e)
            errors += 1

    log.info("Done — %d renamed, %d errors", len(to_rename) - errors, errors)
    return manifest_path


def undo(manifest_path: str) -> None:
    """Reverse all renames recorded in a manifest file."""
    with open(manifest_path) as f:
        entries = json.load(f)

    errors = 0
    for p in reversed(entries):   # reverse order to handle nested cases
        src, dst = p["proposed"], p["current"]
        if not os.path.exists(src):
            log.warning("Source no longer exists, skipping: %s", src)
            continue
        if os.path.exists(dst):
            log.error("Undo destination already exists, skipping: %s", dst)
            errors += 1
            continue
        try:
            os.rename(src, dst)
            log.info("Undone: %s → %s",
                     os.path.basename(src), os.path.basename(dst))
        except Exception as e:
            log.error("Failed to undo %s: %s", src, e)
            errors += 1

    log.info("Undo complete — %d errors", errors)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit and rename recorder deployment directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — show what would be renamed
  python -m catalog.reorganize --scan /path/to/audio/drive1

  # Execute renames (writes manifest first)
  python -m catalog.reorganize --scan /path/to/audio/drive1 --execute

  # Undo a previous rename run
  python -m catalog.reorganize --undo rename_manifest_20240601_120000.json
""",
    )
    parser.add_argument("--scan",      metavar="DIR",
                        help="Top-level audio directory to scan.")
    parser.add_argument("--catalog", metavar="FILE",
                        help="wavfiles catalog (.parquet or .pkl). "
                             "Defaults to CATALOG_PATH from config.",
                        default=None)
    parser.add_argument("--zones", metavar="FILE",
                        help="recorder_zones.json for GPS-based group fallback "
                             "when catalog lookup fails (stale catalog after moves). "
                             "Defaults to RECORDER_ZONES_PATH from config.",
                        default=None)
    parser.add_argument("--execute",   action="store_true",
                        help="Execute renames (default is dry-run only).")
    parser.add_argument("--undo",      metavar="MANIFEST",
                        help="JSON manifest from a previous --execute run.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.undo:
        undo(args.undo)

    elif args.scan:
        from config import CATALOG_PATH, RECORDER_ZONES_PATH
        catalog_path = args.catalog or CATALOG_PATH
        zones_path   = args.zones   or RECORDER_ZONES_PATH
        proposals    = scan(args.scan, catalog_path=catalog_path, zones_path=zones_path)
        report(proposals)

        if args.execute:
            print("\nExecuting renames ...")
            manifest = execute(proposals)
            if manifest:
                print(f"Manifest: {manifest}  (use --undo to reverse)")
        else:
            print("\n(Dry run — pass --execute to rename for real)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
