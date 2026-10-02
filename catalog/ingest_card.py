#!/usr/bin/env python3
"""ingest_card.py -- copy one recorder's SD card to the archive, verified.

Nothing in this repo copied from a card. catalog/reorganize.py handles the step
AFTER, auditing and renaming deployment directories into

    {archive}/{collection date}/{site}/{RecorderID}_{start}_{end}/

so this writes straight into that convention and leaves reorganize.py nothing
to fix.

It copies MORE than the audio, deliberately. The firmware history comes from
logfile_<serial>_SD*.txt, the battery and microphone history from Reclog.csv,
and the S4A power history from *_Summary.txt. Those are small, they are the only
record of how the deployment went, and they are easy to lose by copying only
*.wav -- which is what the one existing ad-hoc script on the archive volume did.

Safety, in order of how much it matters:

  * NOTHING IS EVER DELETED OR MODIFIED ON THE CARD. The card is opened
    read-only and no destructive call appears in this file.
  * --dry-run is the default. --execute is required to write.
  * Every file is verified after copying, by size and by SHA-256 of both
    copies. A file that does not verify is reported and left in place for
    inspection rather than silently retried.
  * Resumable. A file already at the destination with a matching hash is
    skipped, so an interrupted transfer can simply be re-run.

Cards are usually inserted four at a time, so --card is optional: with no
--card every mounted volume is examined and each one that looks like a recorder
card is queued. The site is looked up per recorder from
user_files/recorder_info/recorder_deployments.csv, which already maps serial to
recorder_group, so it only has to be given by hand for a recorder that has never
been deployed before -- which this season means the six new units.

Run it as a plain script, not with -m: the catalog package's __init__ loads
cardinal.toml, and this needs nothing but the standard library, so it should
work on a field laptop with no environment set up.

Usage:
    python catalog/ingest_card.py --archive /path/to/audio/archive
    python catalog/ingest_card.py --archive ... --execute
    python catalog/ingest_card.py --archive ... --site-map BARLT_00012345=Back9
    python catalog/ingest_card.py --card "/Volumes/<SD CARD>" --site Back9 ...
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from datetime import date, datetime

SOLARBAR_DIR_RE = re.compile(r"^BARLT_(\d{8})$")
S4A_FILE_RE = re.compile(r"^(S4A\d{5})")
WAV_TS_RE = re.compile(r"S(\d{8})T\d{6}")             # Solarbar: S20240912T...
S4A_TS_RE = re.compile(r"_(\d{8})_\d{6}_\d{3}\.wav$")  # S4A: ..._20240912_...
META_NAMES = ("reclog.csv", "gps_log.csv", "gps_log.gpx", "loclog.txt",
              "wavfiles.parquet")
CHUNK = 8 << 20
UNKNOWN_SITE = "Unknown"
ARCHIVE_VOLS = ("8M2A", "8M2B", "8M2C", "8M2D")     # never treat these as cards
DEPLOY_CSV = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "user_files", "recorder_info",
    "recorder_deployments.csv")


def site_from_deployments(rec, path=DEPLOY_CSV):
    """Most recent site recorded for this recorder, or None if it is new.

    Six of 28 recorders have been deployed at more than one site, so the LATEST
    deployment is used rather than any of them; a card that has just come out of
    the field belongs with the most recent deployment.
    """
    if not os.path.exists(path):
        return None
    import csv
    best, best_begin = None, ""
    ser = re.search(r"(\d{8})", rec)
    tag = ser.group(1) if ser else rec
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            r = (row.get("recorder") or "")
            if tag not in r:
                continue
            b = (row.get("begin") or "")
            if b >= best_begin:
                best, best_begin = (row.get("recorder_group") or "").strip(), b
    return best or None


def find_cards(volumes_root="/Volumes"):
    """Mounted volumes that look like recorder cards."""
    out = []
    if not os.path.isdir(volumes_root):
        return out
    for name in sorted(os.listdir(volumes_root)):
        if name in ARCHIVE_VOLS:
            continue
        path = os.path.join(volumes_root, name)
        if not os.path.isdir(path) or os.path.islink(path):
            continue
        try:
            rec, kind = identify(path)
        except (PermissionError, OSError):
            continue
        if rec:
            out.append((path, rec, kind))
    return out


def sha256(path, chunk=CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def identify(card):
    """Recorder id and kind, from the card's own contents."""
    for entry in sorted(os.listdir(card)):
        if SOLARBAR_DIR_RE.match(entry) and os.path.isdir(os.path.join(card, entry)):
            return entry, "BARLT"
    for root, _, files in os.walk(card):
        for f in files:
            m = S4A_FILE_RE.match(f)
            if m:
                return m.group(1), "S4A"
    return None, None


def audio_dates(card):
    """First and last recording date, from the filenames rather than mtimes.

    File modification times survive a copy badly and are not what the archive
    convention means; the recorders both stamp the date into the filename, so
    that is what is used.
    """
    seen = set()
    for root, _, files in os.walk(card):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            m = WAV_TS_RE.search(f) or S4A_TS_RE.search(f)
            if m:
                try:
                    seen.add(datetime.strptime(m.group(1), "%Y%m%d").date())
                except ValueError:
                    pass
    return (min(seen), max(seen)) if seen else (None, None)


def plan(card, dest_root):
    """Every file to copy, as (source, destination) pairs."""
    jobs = []
    for root, dirs, files in os.walk(card):
        dirs[:] = [d for d in dirs if d not in (".Spotlight-V100", ".Trashes",
                                                ".fseventsd", "System Volume Information")]
        rel = os.path.relpath(root, card)
        for f in files:
            if f.startswith(".") or f == ".DS_Store":
                continue
            keep = (f.lower().endswith(".wav")
                    or f.lower() in META_NAMES
                    or f.lower().startswith("logfile_")
                    or f.lower().endswith("_summary.txt")
                    or f.lower().endswith(".txt"))
            if not keep:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(dest_root, "" if rel == "." else rel, f)
            jobs.append((src, dst))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=None,
                    help="one mounted SD card path; omit to find all of them")
    ap.add_argument("--site", default=None,
                    help="site for every card in this run; by default each "
                         "recorder's site is looked up from the deployment table")
    ap.add_argument("--site-map", default="",
                    help="per-recorder sites, RECORDER=Site,RECORDER=Site -- "
                         "needed for a recorder not yet in the deployment table")
    ap.add_argument("--archive", required=True,
                    help=".../Carbonranch/<property>/Audio")
    ap.add_argument("--collected", default=date.today().strftime("%Y%m%d"),
                    help="collection date for the archive path (YYYYMMDD)")
    ap.add_argument("--execute", action="store_true",
                    help="actually copy; without this nothing is written")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the SHA-256 check (not recommended)")
    args = ap.parse_args()

    smap = {}
    for tok in filter(None, (x.strip() for x in args.site_map.split(","))):
        k, v = tok.split("=", 1)
        smap[k.strip()] = v.strip()

    if args.card:
        rec, kind = identify(args.card)
        if rec is None:
            sys.exit(f"no BARLT_######## directory and no S4A##### file on "
                     f"{args.card}; is this a recorder card?")
        cards = [(args.card, rec, kind)]
    else:
        cards = find_cards()
        if not cards:
            sys.exit("no recorder cards found under /Volumes")
    print(f"{len(cards)} card(s) found\n")

    queue, problems = [], []
    for path, rec, kind in cards:
        site = smap.get(rec) or args.site or site_from_deployments(rec)
        # An unknown site must NOT stop the copy. Getting the data off the card
        # is the urgent part and the only part that cannot be redone later;
        # where it was deployed can be filled in afterwards, and moving the
        # directory then is a rename, which catalog/reorganize.py already does.
        unknown = site is None
        if unknown:
            site = UNKNOWN_SITE
        d0, d1 = audio_dates(path)
        if d0 is None:
            problems.append((rec, path, "no datable .wav filenames -- refusing "
                                        "to guess the deployment window"))
            continue
        queue.append((path, rec, kind, site, d0, d1, unknown))

    for rec, path, why in problems:
        print(f"SKIP {rec} ({path}): {why}")
    if problems and not queue:
        sys.exit(1)

    grand = 0
    for path, rec, kind, site, d0, d1, unknown in queue:
        dep = f"{rec}_{d0:%Y%m%d}_{d1:%Y%m%d}"
        dest = os.path.join(args.archive, args.collected, site, dep)
        jobs = plan(path, dest)
        sz = sum(os.path.getsize(s_) for s_, _ in jobs)
        grand += sz
        nw = sum(1 for s_, _ in jobs if s_.lower().endswith(".wav"))
        mark = "  <- site unknown, fill in later" if unknown else ""
        print(f"{rec:<16} {site:<12} {d0} to {d1}  {nw:5d} wav "
              f"{len(jobs)-nw:3d} meta  {sz/1e9:6.2f} GB{mark}")
        print(f"    {path}  ->  {dest}")
    print(f"\ntotal {grand/1e9:.2f} GB across {len(queue)} card(s)")
    unk = [q[1] for q in queue if q[6]]
    if unk:
        print(f"{len(unk)} card(s) going to {UNKNOWN_SITE}/: "
              f"{', '.join(unk)}")
        print(f"  add them to recorder_deployments.csv when you know where they "
              f"were, then move the directory (or use catalog/reorganize.py).")
    if not args.execute:
        print("\nDRY RUN -- nothing written. Re-run with --execute.")
        return

    bad = {}
    for path, rec, kind, site, d0, d1, unknown in queue:
        print(f"\n=== {rec} ({site}) ===", flush=True)
        f = ingest_one(path, rec, site, d0, d1, args)
        if f:
            bad[rec] = f
    print(f"\n{len(queue)} card(s) done")
    if bad:
        for rec, f in bad.items():
            print(f"  {rec}: {f} file(s) failed verification -- DO NOT ERASE")
        sys.exit(1)
    print("every file on every card verified.")
    return


def ingest_one(card, rec, site, d0, d1, args):
    """Copy one card, verifying every file. Never writes to the card."""
    dep = f"{rec}_{d0:%Y%m%d}_{d1:%Y%m%d}"
    dest_root = os.path.join(args.archive, args.collected, site, dep)
    jobs = plan(card, dest_root)
    total = sum(os.path.getsize(s_) for s_, _ in jobs)

    os.makedirs(dest_root, exist_ok=True)
    copied = skipped = failed = 0
    done_bytes = 0
    for i, (src, dst) in enumerate(jobs, 1):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ssz = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == ssz:
            if args.no_verify or sha256(dst) == sha256(src):
                skipped += 1
                done_bytes += ssz
                continue
        shutil.copy2(src, dst)
        ok = os.path.getsize(dst) == ssz and (
            args.no_verify or sha256(dst) == sha256(src))
        if ok:
            copied += 1
        else:
            failed += 1
            print(f"  VERIFY FAILED, left in place for inspection: {dst}")
        done_bytes += ssz
        if i % 25 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] {done_bytes/1e9:.2f}/{total/1e9:.2f} GB "
                  f"copied={copied} skipped={skipped} failed={failed}",
                  flush=True)
    print(f"\n{copied} copied, {skipped} already present and verified, "
          f"{failed} failed")
    if failed:
        # do not exit: with four cards in a run, one bad card must not abandon
        # the other three, and the count is reported again at the end
        print(f"  {failed} file(s) did not verify -- this card is NOT safe "
              f"to reuse")
    else:
        print("  all files verified; card unmodified and safe to erase once "
              "you are satisfied")
    return failed


if __name__ == "__main__":
    main()
