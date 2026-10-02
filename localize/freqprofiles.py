# localize/freqprofiles.py
#
# Build per-species frequency profiles for use as FreqFilter templates during
# TDOA cross-correlation (see localize/tdoa.py FreqFilter, and the `cornellspec`
# argument of localize/localize_utils/core.py's localize_plot_withorig).
#
# Two sources, combinable:
#   build_from_reference_clips()  Average spectrogram magnitude across a folder
#                                  of labeled reference recordings (e.g. a
#                                  Macaulay Library / xeno-canto / Cornell Guide
#                                  to Bird Sounds download).  Reference libraries
#                                  rarely cover every species you encounter in
#                                  the field, so...
#   build_from_detections()       ...this fills the gap: it builds the same kind
#                                  of profile from your own highest-confidence
#                                  BirdNET field detections instead of reference
#                                  audio.
#   merge_profiles()              Combine the two, preferring one source but
#                                  filling in species missing from it with the
#                                  other.
#
# Output schema (both builders): DataFrame indexed by birdcode (uppercase,
# non-alpha characters stripped from the common name) with columns:
#   freqs      spectrogram frequency bins (Hz), array
#   specmean   mean magnitude-spectrogram profile across clips, array
#   specmax    max  magnitude-spectrogram profile across clips, array
#   n_clips    number of clips averaged
#   meanfilt   FreqFilter built from specmean
#   maxfilt    FreqFilter built from specmax
#
# This matches what localize_plot_withorig expects from `cornellspec`:
#   freqfilter = cornellspec.loc[row.birdcode].maxfilt
#
# Usage:
#   python -m localize.freqprofiles detections --detections birdnet/checkpoints/detections.parquet \
#       --out localize/freqprofiles_detections.pkl
#   python -m localize.freqprofiles merge --primary ref.pkl --fallback det.pkl \
#       --out localize/freqprofiles.pkl

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal

from localize.tdoa import FreqFilter
from birdnet.run import apply_bandpass

log = logging.getLogger(__name__)

PROFILE_COLUMNS = ["freqs", "specmean", "specmax", "n_clips", "meanfilt", "maxfilt"]


def common_name_to_birdcode(name: str) -> str:
    """Normalize a common name to the uppercase, alpha-only birdcode used as index."""
    import re
    return re.sub(r"[^A-Za-z]", "", str(name)).upper()


def _spectrogram_profile(y: np.ndarray, rate: int,
                         nperseg: int = 1024, noverlap: int = 512,
                         hp: float = 500.0, lp: float = 15000.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, mean-magnitude-over-time) for one audio clip.

    A bandpass filter is applied first to keep low-frequency noise from
    dominating the averaged profile.  The default highpass (500 Hz) is
    tighter than birdnet/run.py's general-purpose 150 Hz: field recordings
    tend to carry a very consistent low-frequency resonance (wind buffeting,
    recorder self-noise) below ~300 Hz that is *more* frequency-stable across
    clips than natural pitch variation in bird calls, so it can win a
    per-bin aggregation across clips even when it's quieter in absolute
    terms than the vocalization. Since virtually no target passerine song
    falls below 500 Hz, the tighter cutoff removes it outright rather than
    relying on aggregation statistics to filter it out. Lower this if you
    need to profile species with genuinely low-frequency vocalizations
    (e.g. some owls, doves, grouse).
    """
    y = apply_bandpass(y, rate, hp=hp, lp=lp)
    freqs, _, Sxx = scipy.signal.spectrogram(
        y, rate, nperseg=nperseg, noverlap=noverlap, mode="magnitude", scaling="density",
    )
    profile = Sxx.mean(axis=1)
    peak = profile.max()
    if peak > 0:
        profile = profile / peak
    return freqs, profile


def _profiles_to_table(rows: list[dict]) -> pd.DataFrame:
    """Group per-clip profiles by birdcode and build FreqFilter columns.

    Each clip's profile is normalized to its own peak before stacking (see
    _spectrogram_profile), which means every clip has exactly one bin equal
    to 1.0 -- its own loudest frequency, signal or noise.  A plain per-bin
    max across clips ("specmax") would therefore let a single noise-
    contaminated clip (wind, handling) plant a spurious peak anywhere.
    We use robust statistics instead: median (specmean) and 90th percentile
    (specmax) across clips, so a minority of bad clips can't dominate.
    """
    if not rows:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    df = pd.DataFrame(rows)
    out_rows = {}
    for birdcode, g in df.groupby("birdcode"):
        freqs = g.iloc[0]["freqs"]
        stacked = np.stack(g["profile"].to_list())
        specmean = np.median(stacked, axis=0)
        specmax = np.percentile(stacked, 90, axis=0)
        rate = g.iloc[0]["rate"]
        out_rows[birdcode] = {
            "freqs": freqs,
            "specmean": specmean,
            "specmax": specmax,
            "n_clips": len(g),
            "meanfilt": FreqFilter.from_profile(freqs, specmean, rate),
            "maxfilt": FreqFilter.from_profile(freqs, specmax, rate),
        }
    out = pd.DataFrame.from_dict(out_rows, orient="index")
    out.index.name = "birdcode"
    return out[PROFILE_COLUMNS]


# ---------------------------------------------------------------------------
# Source 1: reference clip library
# ---------------------------------------------------------------------------

def build_from_reference_clips(
        clips_dir: str,
        parse_species,
        rate: int | None = None,
        nperseg: int = 1024,
        noverlap: int = 512,
        hp: float = 500.0,
        lp: float = 15000.0,
        max_clips_per_species: int | None = None,
) -> pd.DataFrame:
    """Build per-species frequency profiles from a directory of reference clips.

    Parameters
    ----------
    clips_dir     : directory to walk recursively for audio files (.wav/.mp3/.flac)
    parse_species : callable(filepath) -> common name string, or None to skip
                    that file.  Reference libraries vary in naming convention
                    (Macaulay Library downloads, Cornell Guide to Bird Sounds,
                    xeno-canto, ...), so the caller supplies the parser.
    rate          : resample target for all clips (must be consistent across
                    clips of the same species so their spectrograms share a
                    frequency axis and can be averaged).  Defaults to
                    config.SAMPLE_RATE.
    hp, lp        : bandpass edges (Hz) applied before profiling. Lower `hp`
                    for species with genuinely low-frequency vocalizations
                    (owls, doves, grouse); reference clips are usually clean
                    enough that the default matters less than for field data.
    max_clips_per_species : cap the number of clips averaged per species.
    """
    import librosa
    from config import SAMPLE_RATE
    rate = rate or SAMPLE_RATE

    rows = []
    counts: dict[str, int] = {}
    exts = {".wav", ".mp3", ".flac"}

    for root, _, files in os.walk(clips_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            fp = os.path.join(root, fn)
            name = parse_species(fp)
            if not name:
                continue
            birdcode = common_name_to_birdcode(name)

            if max_clips_per_species is not None and counts.get(birdcode, 0) >= max_clips_per_species:
                continue

            try:
                y, sr = librosa.load(fp, sr=rate, mono=True)
            except Exception as e:
                log.warning("skip %s: %s", fp, e)
                continue
            if y.size == 0:
                continue

            freqs, profile = _spectrogram_profile(y, sr, nperseg=nperseg, noverlap=noverlap, hp=hp, lp=lp)
            rows.append({"birdcode": birdcode, "freqs": freqs, "profile": profile, "rate": sr})
            counts[birdcode] = counts.get(birdcode, 0) + 1

    log.info("build_from_reference_clips: %d clips across %d species", len(rows), len(counts))
    return _profiles_to_table(rows)


# ---------------------------------------------------------------------------
# Source 2: your own field detections (fills gaps left by the reference library)
# ---------------------------------------------------------------------------

def build_from_detections(
        detections: pd.DataFrame,
        top_n: int = 20,
        clip_seconds: float = 3.0,
        min_prob: float = 0.7,
        nperseg: int = 1024,
        noverlap: int = 512,
        hp: float = 500.0,
        lp: float = 15000.0,
) -> pd.DataFrame:
    """Build per-species frequency profiles from your own BirdNET field detections.

    Parameters
    ----------
    detections : a detections table such as the birdnet/checkpoints/detections.parquet
                 checkpoint (output of birdnet.parse.filter_detections) -- must have
                 columns common_name, prob, recorder_type, file, call_datetime.
    top_n       : highest-confidence detections per species to average.
    clip_seconds: length of audio clip to load around each detection.
    min_prob    : ignore detections below this probability.
    hp, lp      : bandpass edges (Hz) applied before profiling. The default
                  500 Hz highpass is tighter than birdnet/run.py's
                  general-purpose 150 Hz -- field recordings carry a very
                  frequency-stable low-frequency noise floor (wind, recorder
                  self-noise) that can otherwise dominate the profile even
                  though it's quieter than the vocalization in absolute
                  terms. Lower this for species with genuinely low-frequency
                  calls (owls, doves, grouse).
    """
    from catalog.audiofile import AudioFileSolarbar, AudioFileS4

    det = detections[detections["prob"] >= min_prob]
    top = (det.sort_values("prob", ascending=False)
              .groupby("common_name", group_keys=False)
              .head(top_n))

    rows = []
    counts: dict[str, int] = {}
    file_cache: dict[str, object] = {}

    for _, r in top.iterrows():
        try:
            af = file_cache.get(r.file)
            if af is None:
                if r.recorder_type == "SOLARBAR":
                    af = AudioFileSolarbar(r.file)
                elif r.recorder_type == "S4A":
                    af = AudioFileS4(r.file)
                else:
                    continue
                file_cache[r.file] = af

            offset = af.get_ttoffset(r.call_datetime)
            y = af.getdata(offset, offset + clip_seconds * af.rate)
            if not np.any(y):
                continue

            freqs, profile = _spectrogram_profile(y, af.rate, nperseg=nperseg, noverlap=noverlap, hp=hp, lp=lp)
            birdcode = common_name_to_birdcode(r.common_name)
            rows.append({"birdcode": birdcode, "freqs": freqs, "profile": profile, "rate": af.rate})
            counts[birdcode] = counts.get(birdcode, 0) + 1
        except Exception as e:
            log.warning("skip detection %s @ %s: %s", getattr(r, "file", "?"),
                       getattr(r, "call_datetime", "?"), e)
            continue

    log.info("build_from_detections: %d clips across %d species", len(rows), len(counts))
    return _profiles_to_table(rows)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_profiles(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Combine two profile tables: keep *primary*'s rows, and fill in any
    species missing from *primary* using rows from *fallback*.
    """
    missing = fallback.index.difference(primary.index)
    if len(missing):
        log.info("merge_profiles: filling %d species from fallback source", len(missing))
    return pd.concat([primary, fallback.loc[missing]])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_profiles(profiles: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profiles.to_pickle(path)
    log.info("Frequency profiles saved: %s (%d species)", path, len(profiles))


def load_profiles(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-species frequency profiles for TDOA localization.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ref = sub.add_parser("reference", help="build profiles from a reference clip library")
    p_ref.add_argument("--clips-dir", required=True)
    p_ref.add_argument("--out", required=True)
    p_ref.add_argument("--max-clips-per-species", type=int, default=None)
    p_ref.add_argument("--hp", type=float, default=500.0)
    p_ref.add_argument("--lp", type=float, default=15000.0)

    p_det = sub.add_parser("detections", help="build profiles from your own field detections")
    p_det.add_argument("--detections", required=True, help="path to a detections parquet")
    p_det.add_argument("--out", required=True)
    p_det.add_argument("--top-n", type=int, default=20)
    p_det.add_argument("--min-prob", type=float, default=0.7)
    p_det.add_argument("--clip-seconds", type=float, default=3.0)
    p_det.add_argument("--hp", type=float, default=500.0)
    p_det.add_argument("--lp", type=float, default=15000.0)

    p_merge = sub.add_parser("merge", help="merge two profile tables")
    p_merge.add_argument("--primary", required=True)
    p_merge.add_argument("--fallback", required=True)
    p_merge.add_argument("--out", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    if args.command == "reference":
        raise SystemExit(
            "The 'reference' command needs a --parse-species callable, which "
            "varies by library naming convention. Call "
            "localize.freqprofiles.build_from_reference_clips(...) directly "
            "from a script instead of via this CLI."
        )
    elif args.command == "detections":
        det = pd.read_parquet(args.detections)
        profiles = build_from_detections(
            det, top_n=args.top_n, min_prob=args.min_prob, clip_seconds=args.clip_seconds,
            hp=args.hp, lp=args.lp,
        )
        save_profiles(profiles, args.out)
    elif args.command == "merge":
        primary = load_profiles(args.primary)
        fallback = load_profiles(args.fallback)
        if primary is None or fallback is None:
            raise SystemExit("both --primary and --fallback must exist")
        merged = merge_profiles(primary, fallback)
        save_profiles(merged, args.out)


if __name__ == "__main__":
    main()
