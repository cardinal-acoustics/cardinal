# birdnet/parse.py
#
# Load, join, and filter BirdNET detection output.
#
# Pipeline:
#   load_detections()        load raw BirdNET parquets from an output directory
#   attach_labels()          join species names and site-species flag
#   join_catalog()           join with wave-file catalog; compute call_datetime
#   filter_detections()      GPS-sync, valid times, site species
#   compute_concordance()    pivot to wide; count recorders above each threshold
#   count_active_recorders() sweep-line count of how many recorders were live
#
# Checkpointing (skip recomputation when output already exists):
#   save_checkpoint(df, path)         save a DataFrame to parquet
#   load_checkpoint(path)             load parquet → DataFrame, or None if absent
#   save_concordance(concordance, dir) save concordance dict to a directory
#   load_concordance(dir)             reload concordance dict, or None if absent
#
# Typical usage with checkpoints:
#   bnf = load_checkpoint(CHECKPOINT_DIR / "detections_raw.parquet")
#   if bnf is None:
#       bnf = attach_labels(load_detections())
#       save_checkpoint(bnf, CHECKPOINT_DIR / "detections_raw.parquet")
#
#   bnfi = load_checkpoint(CHECKPOINT_DIR / "detections.parquet")
#   if bnfi is None:
#       bnfi = filter_detections(join_catalog(bnf, wf))
#       save_checkpoint(bnfi, CHECKPOINT_DIR / "detections.parquet")
#
#   concordance = load_concordance(CHECKPOINT_DIR / "concordance")
#   if concordance is None:
#       concordance = count_active_recorders(compute_concordance(bnfi), wf)
#       save_concordance(concordance, CHECKPOINT_DIR / "concordance")

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BIRDNET_OUTPUT_DIR, BIRDNET_LABELS_PATH, SITE_SPECIES_PATH,
    SITE_TIMEZONE, CONCORDANCE_THRESHOLDS, CHECKPOINT_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load raw BirdNET output
# ---------------------------------------------------------------------------

def load_detections(audio_dirs: list[str] | str | None = None) -> pd.DataFrame:
    """Walk audio directories and load all BirdNET sidecar parquets.

    BirdNET output is stored as sidecar files alongside the audio:
        {audio_dir}/**/_bn_grid_wprobs/emb_*.parquet

    Each parquet has columns:
        file, abs_start_iso, win_start, win_end, embedding, probs

    where *probs* is a list of [class_idx, prob] pairs (only detections
    above threshold, may be empty).  This function explodes *probs* into
    one row per (file, window, class) and drops the *embedding* column.

    Parameters
    ----------
    audio_dirs : top-level audio directory or list of directories to walk.
                 Defaults to AUDIO_TOPDIRS from config.
    """
    from config import AUDIO_TOPDIRS

    if audio_dirs is None:
        audio_dirs = AUDIO_TOPDIRS
    if isinstance(audio_dirs, str):
        audio_dirs = [audio_dirs]

    frames = []
    n_parquets = 0
    for topdir in audio_dirs:
        for dirpath, dirnames, filenames in os.walk(topdir):
            # Only descend into _bn_grid_wprobs directories
            if os.path.basename(dirpath) == "_bn_grid_wprobs":
                for fn in sorted(filenames):
                    if fn.startswith("emb_") and fn.endswith(".parquet"):
                        path = os.path.join(dirpath, fn)
                        try:
                            frames.append(pd.read_parquet(
                                path, columns=["file", "abs_start_iso",
                                               "win_start", "win_end", "probs"]
                            ))
                            n_parquets += 1
                        except Exception as e:
                            log.warning("Could not read %s: %s", path, e)
                dirnames.clear()   # don't recurse into _bn_grid_wprobs

    if not frames:
        raise RuntimeError(
            "No _bn_grid_wprobs/emb_*.parquet files found. "
            "Check AUDIO_TOPDIRS in config.py."
        )

    raw = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d window rows from %d parquets", len(raw), n_parquets)

    # Explode probs: each [class_idx, prob] pair becomes its own row.
    # Windows with no detections (probs=[]) are dropped.
    raw = raw[raw["probs"].apply(lambda x: len(x) > 0)]
    raw = raw.explode("probs").reset_index(drop=True)
    raw["class_idx"] = raw["probs"].apply(lambda x: int(x[0]))
    raw["prob"]      = raw["probs"].apply(lambda x: float(x[1]))
    raw = raw.drop(columns="probs")

    log.info("Exploded to %d detection rows", len(raw))
    return raw


# ---------------------------------------------------------------------------
# Attach species labels and site-species flag
# ---------------------------------------------------------------------------

def attach_labels(bnf: pd.DataFrame,
                  labels_path: str | None = None,
                  site_species_path: str | None = None) -> pd.DataFrame:
    """Join BirdNET species labels and a site-specific species list onto *bnf*.

    Adds columns: species, common_name, site_species (bool).

    Parameters
    ----------
    labels_path       : BirdNET label file — one "ScientificName_Common Name"
                        per line.  Defaults to BIRDNET_LABELS_PATH from config.
    site_species_path : site species list — same format as labels_path.
                        Defaults to SITE_SPECIES_PATH from config.
    """
    if labels_path is None:
        labels_path = BIRDNET_LABELS_PATH
    if site_species_path is None:
        site_species_path = SITE_SPECIES_PATH

    # BirdNET label file: each line is "ScientificName_Common Name"
    # Split on the first underscore only — scientific names may contain spaces
    # but common names may also contain underscores in rare cases.
    raw = pd.read_csv(labels_path, header=None, names=["raw"])
    split = raw["raw"].str.split("_", n=1, expand=True)
    spec = pd.DataFrame({"species": split[0], "common_name": split[1]})
    spec["class_idx"] = spec.index

    # Site species list: same format (ScientificName_Common Name); extract common name.
    # If no list is configured, all species are treated as site species.
    if site_species_path:
        with open(site_species_path) as f:
            site_names = {
                line.strip().split("_", 1)[1]
                for line in f
                if line.strip() and "_" in line
            }
        spec["site_species"] = spec["common_name"].isin(site_names)
    else:
        spec["site_species"] = True

    bnf = bnf.merge(spec[["class_idx", "species", "common_name", "site_species"]],
                    on="class_idx", how="left")
    log.info("Labels attached — %d site species flagged",
             spec["site_species"].sum())
    return bnf


# ---------------------------------------------------------------------------
# Join wave-file catalog
# ---------------------------------------------------------------------------

def join_catalog(bnf: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    """Join BirdNET detections with the wave-file catalog on filename.

    Joins on the WAV filename's basename rather than the full path stored
    in the BirdNET sidecar. The sidecar's 'file' column is whatever path
    existed when BirdNET was run; if catalog/reorganize.py later renamed or
    restructured the containing directories (a normal, expected operation),
    the full path goes stale even though the recording itself didn't move.
    The filename itself (it encodes recorder + start/end timestamp + GPS)
    is not renamed by reorganize.py, so it survives as a stable join key.

    Computes:
        call_datetime   — absolute moment of the detection window
        call_datetime_r — call_datetime rounded to nearest second
        date            — date portion
        time_in_day     — fractional hour (0–24)

    Parameters
    ----------
    bnf : raw BirdNET detections (output of load_detections / attach_labels)
    wf  : wave-file catalog (output of catalog/build.py)
    """
    import os

    bnf = bnf.copy()
    wf = wf.copy()
    bnf["_basename"] = bnf["file"].apply(os.path.basename)
    wf["_basename"] = wf["file"].apply(os.path.basename)

    dup_basenames = wf["_basename"].duplicated()
    if dup_basenames.any():
        log.warning(
            "%d catalog file(s) share a basename with another catalog "
            "entry -- keeping only the first of each to avoid duplicate "
            "joins", dup_basenames.sum(),
        )
        wf = wf[~dup_basenames]

    missing = set(bnf["_basename"].unique()) - set(wf["_basename"].unique())
    if missing:
        log.warning("%d BirdNET file(s) not found in catalog", len(missing))

    bnfi = bnf.drop(columns=["file"]).merge(wf, on="_basename", how="left").drop(columns=["_basename"])

    # Absolute detection time = file start + window offset.
    # Round in UTC before converting to local time — rounding a tz-aware
    # America/Chicago datetime fails during DST transitions (e.g. fall-back
    # at 1:00 AM when the same local time occurs twice).
    call_dt_utc = pd.to_datetime(
        bnfi["start_datetime"] + pd.to_timedelta(bnfi["win_start"], unit="s"),
        utc=True,
    )
    bnfi["call_datetime"]   = call_dt_utc.dt.tz_convert(SITE_TIMEZONE)
    bnfi["call_datetime_r"] = call_dt_utc.dt.round("1s").dt.tz_convert(SITE_TIMEZONE)
    bnfi["date"]            = bnfi["call_datetime"].dt.date
    bnfi["time_in_day"]     = (bnfi["call_datetime"].dt.hour
                                + bnfi["call_datetime"].dt.minute / 60)

    bnfi = bnfi.sort_values("call_datetime")
    log.info("Joined catalog: %d rows", len(bnfi))
    return bnfi


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_detections(bnfi: pd.DataFrame,
                      require_gps_sync: bool = True,
                      site_species_only: bool = True,
                      date_start: str | None = None,
                      date_end:   str | None = None,
                      recorder_types: list[str] | None = None) -> pd.DataFrame:
    """Filter the joined detections table.

    Parameters
    ----------
    require_gps_sync  : drop rows where gps_sync is False
    site_species_only : drop rows where site_species is False
    date_start/end    : optional ISO date strings ("YYYY-MM-DD") to restrict range
    recorder_types    : optional list, e.g. ["SOLARBAR"] -- drop rows from any
                        other recorder_type
    """
    before = len(bnfi)

    bnfi = bnfi[bnfi["call_datetime"].notna()]
    if require_gps_sync:
        bnfi = bnfi[bnfi["gps_sync"] == True]
    if site_species_only:
        bnfi = bnfi[bnfi["site_species"] == True]
    if date_start:
        bnfi = bnfi[bnfi["call_datetime"] >= pd.Timestamp(date_start, tz=SITE_TIMEZONE)]
    if date_end:
        bnfi = bnfi[bnfi["call_datetime"] <  pd.Timestamp(date_end,   tz=SITE_TIMEZONE)]
    if recorder_types:
        bnfi = bnfi[bnfi["recorder_type"].isin(recorder_types)]

    log.info("filter_detections: %d → %d rows", before, len(bnfi))
    return bnfi


# ---------------------------------------------------------------------------
# Concordance pivot and threshold counts
# ---------------------------------------------------------------------------

def compute_concordance(bnfi: pd.DataFrame,
                        thresholds: list[float] | None = None,
                        ) -> dict:
    """Pivot detections and compute per-recorder-group concordance counts.

    Returns a dict with keys:
        "pivot"          — wide DataFrame (index: (common_name, call_datetime_r),
                           columns: MultiIndex (recorder, recorder_group))
        "counts_{t}"     — for each threshold t: DataFrame indexed by
                           (common_name, call_datetime_r) with one column per
                           recorder_group counting recorders above threshold
        "maxes"          — max prob per recorder_group + global allmax column

    Parameters
    ----------
    thresholds : list of probability thresholds, e.g. [0.9, 0.75, 0.5].
                 Defaults to CONCORDANCE_THRESHOLDS from config.
    """
    if thresholds is None:
        thresholds = CONCORDANCE_THRESHOLDS

    pivot = bnfi.pivot_table(
        index=["common_name", "call_datetime_r"],
        columns=["recorder", "recorder_group"],
        values="prob",
        aggfunc="max",
    )
    log.info("Pivot shape: %s", pivot.shape)

    result: dict = {"pivot": pivot}

    # Concordance counts at each threshold
    for t in thresholds:
        key = f"counts_p{int(t * 100)}"
        counts = {}
        for rg in pivot.columns.get_level_values("recorder_group").unique():
            rg_cols = pivot.xs(rg, level="recorder_group", axis=1)
            counts[rg] = (rg_cols >= t).sum(axis=1)
        result[key] = pd.DataFrame(counts)

    # Max prob per recorder_group
    maxes = {}
    for rg in pivot.columns.get_level_values("recorder_group").unique():
        rg_cols = pivot.xs(rg, level="recorder_group", axis=1)
        maxes[rg] = rg_cols.max(axis=1)
    maxes_df = pd.DataFrame(maxes)
    maxes_df["allmax"] = maxes_df.max(axis=1)
    result["maxes"] = maxes_df

    return result


# ---------------------------------------------------------------------------
# Active-recorder counting
# ---------------------------------------------------------------------------

def _collapse_recorder_intervals(wf: pd.DataFrame,
                                  gap_tolerance_s: float = 10.0) -> pd.DataFrame:
    """Merge adjacent/overlapping recording intervals per (recorder_group, recorder).

    Intervals separated by less than *gap_tolerance_s* are treated as continuous.
    """
    gap = pd.Timedelta(seconds=gap_tolerance_s)
    rows = []
    for (rg, rec), grp in wf.groupby(["recorder_group", "recorder"]):
        grp = grp.sort_values("start_datetime")
        cur_start = grp.iloc[0]["start_datetime"]
        cur_end   = grp.iloc[0]["end_datetime"]
        for _, row in grp.iloc[1:].iterrows():
            if row["start_datetime"] <= cur_end + gap:
                cur_end = max(cur_end, row["end_datetime"])
            else:
                rows.append({"recorder_group": rg, "recorder": rec,
                             "start_datetime": cur_start, "end_datetime": cur_end})
                cur_start, cur_end = row["start_datetime"], row["end_datetime"]
        rows.append({"recorder_group": rg, "recorder": rec,
                     "start_datetime": cur_start, "end_datetime": cur_end})
    return pd.DataFrame(rows)


def count_active_recorders(concordance: dict,
                            wf: pd.DataFrame,
                            gap_tolerance_s: float = 10.0) -> dict:
    """Add an 'active_recorders' entry to *concordance*.

    For each (common_name, call_datetime_r, recorder_group) triple, counts
    how many recorders were actively recording at that moment.

    Uses a sweep-line approach via pd.merge_asof for efficiency.

    Parameters
    ----------
    concordance     : dict returned by compute_concordance()
    wf              : wave-file catalog
    gap_tolerance_s : passed to _collapse_recorder_intervals()
    """
    wf_simple = _collapse_recorder_intervals(
        wf[wf.recorder_group.notna()], gap_tolerance_s
    )

    counts_p90 = concordance[f"counts_p{int(CONCORDANCE_THRESHOLDS[0] * 100)}"]
    calls = counts_p90.reset_index()[["common_name", "call_datetime_r"]]

    active: dict[str, pd.Series] = {}

    for rg in counts_p90.columns:
        rg_intervals = wf_simple[wf_simple.recorder_group == rg].copy()
        if rg_intervals.empty:
            active[rg] = pd.Series(0, index=counts_p90.index)
            continue

        rg_calls = calls.copy()
        rg_calls["recorder_group"] = rg
        rg_calls = rg_calls.sort_values("call_datetime_r")

        # Build +1/-1 event stream
        starts = rg_intervals[["start_datetime"]].copy()
        starts["delta"] = 1
        starts = starts.rename(columns={"start_datetime": "t"})
        ends = rg_intervals[["end_datetime"]].copy()
        ends["delta"] = -1
        ends = ends.rename(columns={"end_datetime": "t"})
        events = pd.concat([starts, ends]).sort_values("t")
        events["n_active"] = events["delta"].cumsum()

        # Merge asof on int64 nanoseconds — avoids pandas rejecting two
        # datetime64[tz] columns that have the same timezone name but
        # different backends (pytz vs zoneinfo), which happens when the
        # catalog was built with an older code path.
        rg_calls_sorted = rg_calls.sort_values("call_datetime_r").copy()
        rg_calls_sorted["_ns"] = rg_calls_sorted["call_datetime_r"].astype("int64")
        events_ns = events[["t", "n_active"]].copy()
        events_ns["_ns"] = events_ns["t"].astype("int64")

        merged = pd.merge_asof(
            rg_calls_sorted,
            events_ns[["_ns", "n_active"]],
            on="_ns",
            direction="backward",
        )
        merged = merged.drop(columns="_ns")
        merged = merged.set_index(["common_name", "call_datetime_r"])
        active[rg] = merged["n_active"].fillna(0).astype(int)

    concordance["active_recorders"] = pd.DataFrame(active).reindex(counts_p90.index)
    log.info("Active recorder counts computed for %d groups", len(active))
    return concordance


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(df: pd.DataFrame, path: str | Path) -> None:
    """Save *df* to a parquet file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    log.info("Checkpoint saved: %s  (%d rows)", path, len(df))


def load_checkpoint(path: str | Path) -> pd.DataFrame | None:
    """Load a parquet checkpoint, or return None if it does not exist."""
    path = Path(path)
    if not path.exists():
        log.info("No checkpoint at %s — will recompute", path)
        return None
    df = pd.read_parquet(path)
    log.info("Checkpoint loaded: %s  (%d rows)", path, len(df))
    return df


def save_concordance(concordance: dict, directory: str | Path) -> None:
    """Save all concordance DataFrames to *directory* as parquet files.

    The 'pivot' key is skipped (too large; recomputed cheaply from detections).
    One file is written per key, e.g. counts_p90.parquet, maxes.parquet, etc.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in concordance.items():
        if key == "pivot":
            continue
        if not isinstance(value, pd.DataFrame):
            log.warning("Skipping non-DataFrame concordance key: %s", key)
            continue
        out = directory / f"{key}.parquet"
        value.to_parquet(out, index=True)
        log.info("Concordance saved: %s", out)


def load_concordance(directory: str | Path) -> dict | None:
    """Reload a concordance dict from *directory*.

    Returns None if the directory does not exist or is missing expected files.
    The 'pivot' key is not restored (not saved); downstream code that only
    needs counts/maxes/active_recorders will work fine without it.
    """
    directory = Path(directory)
    if not directory.exists():
        log.info("No concordance checkpoint at %s — will recompute", directory)
        return None

    concordance = {}
    for parquet_file in sorted(directory.glob("*.parquet")):
        key = parquet_file.stem
        try:
            concordance[key] = pd.read_parquet(parquet_file)
            log.info("Concordance loaded: %s", parquet_file)
        except Exception as e:
            log.warning("Could not load %s: %s", parquet_file, e)

    if not concordance:
        log.warning("Concordance directory exists but contains no parquets: %s",
                    directory)
        return None

    return concordance
