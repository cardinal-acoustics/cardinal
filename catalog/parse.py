# catalog/parse.py
#
# Filename parsing and metadata extraction for Solarbar and S4A recorders.
#
# NOTE on S4A timezone: S4A recorders log timestamps in a FIXED UTC-5 offset
# year-round -- they are set to Central *Daylight* time and never fall back.
# This was measured, not assumed, on 2026-09-19: the amplitude envelope of an
# S4A hour was cross-correlated against a 3 h window of a co-located GPS-synced
# BAR-LT, which dates itself in explicit UTC.  Across three site pairs
# (S4A20237/8655 Back9, S4A20258/8570 NorthSide, S4A20233/8099 House), all 18
# confident trials (z >= 6) put the true audio 3600.1 s BEFORE the catalogued
# time -- 10/10 in CST and 8/8 in CDT, none at zero lag.
#
# The offset is therefore constant, not a DST artefact: the previous UTC-6
# assumption made every S4A timestamp exactly one hour late in both seasons.
# Since the S4A clock does not observe DST, a fixed offset is the correct
# model; do not substitute a DST-aware zone here.
#
# The catalog also carries a 'timestamp_source' column ('filename' or
# 'kal_meta').  Both paths showed the same one-hour error, so the two sources
# agree with each other and the column does not discriminate good from bad.

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import SITE_TIMEZONE


# ---------------------------------------------------------------------------
# File-type helpers
# ---------------------------------------------------------------------------

def iswave(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() == ".wav"


def isflac(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() == ".flac"


# ---------------------------------------------------------------------------
# Recorder-ID extraction
# ---------------------------------------------------------------------------

def getrecorder(dirpath: str) -> str | None:
    """Return the BARLT recorder ID embedded in a directory path, or None."""
    pattern = r'BARLT_\d{8}'
    matches = re.findall(pattern, dirpath)
    if len(set(matches)) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


# The S4A clock's true fixed offset from UTC. See the module note above for the
# measurement that fixed this at -5; catalog.build imports it so the filename
# and Kaleidoscope paths can never drift apart again.
S4A_UTC_OFFSET = timezone(timedelta(hours=-5))


def return_s4_time(timestring: str) -> datetime:
    """Parse an S4A timestamp string to a timezone-aware datetime.

    S4A recorders record in a fixed UTC-5 offset regardless of DST, so the
    returned datetime carries that fixed offset. Callers that need
    America/Chicago (with DST) must convert.
    """
    naive = datetime.strptime(timestring, "%Y%m%d_%H%M%S")
    return naive.replace(tzinfo=S4A_UTC_OFFSET)


# ---------------------------------------------------------------------------
# Core filename parser
# ---------------------------------------------------------------------------

def parse_filename(filepath: str) -> pd.Series:
    """Parse recorder metadata from an audio filepath.

    Returns a pd.Series with columns:
        filepath, dirpath, filename, recorder, recorder_type, channel,
        gps_sync, start_datetime, end_datetime, latitude, longitude
    """
    dirpath, filename = os.path.split(filepath)

    start_datetime = pd.NaT
    end_datetime   = pd.NaT
    latitude       = None
    longitude      = None
    recorder       = ''
    recorder_type  = ''
    gpssync        = False
    channel        = None
    matched        = False

    # --- Solarbar: S + end timestamps + GPS in filename ---
    if not matched:
        pattern = re.compile(
            r"S(\d{8}T\d{6}\.\d{6}\+\d{4})_E(\d{8}T\d{6}\.\d{6}\+\d{4})_"
            r"([+-]\d+\.\d+)([+-]\d+\.\d+)\.(?:wav|flac)"
        )
        match = pattern.search(filename)
        if match:
            s, e, lat, lon = match.groups()
            start_datetime = datetime.strptime(s, "%Y%m%dT%H%M%S.%f%z").astimezone(SITE_TIMEZONE)
            end_datetime   = datetime.strptime(e, "%Y%m%dT%H%M%S.%f%z").astimezone(SITE_TIMEZONE)
            latitude, longitude = float(lat), float(lon)
            recorder = getrecorder(dirpath)
            recorder_type = "SOLARBAR"
            gpssync = True
            matched = True

    # --- Solarbar: S + end timestamps + GPS + segment suffix ---
    if not matched:
        pattern = re.compile(
            r"S(\d{8}T\d{6}\.\d{6}\+\d{4})_E(\d{8}T\d{6}\.\d{6}\+\d{4})_"
            r"([+-]\d+\.\d+)([+-]\d+\.\d+)\_00000_000\.(?:wav|flac)"
        )
        match = pattern.search(filename)
        if match:
            s, e, lat, lon = match.groups()
            start_datetime = datetime.strptime(s, "%Y%m%dT%H%M%S.%f%z").astimezone(SITE_TIMEZONE)
            end_datetime   = datetime.strptime(e, "%Y%m%dT%H%M%S.%f%z").astimezone(SITE_TIMEZONE)
            latitude, longitude = float(lat), float(lon)
            recorder = getrecorder(dirpath)
            recorder_type = "SOLARBAR"
            gpssync = True
            matched = True

    # --- Solarbar: start timestamp + REC + GPS ---
    if not matched:
        pattern = re.compile(
            r"(\d{8}T\d{6}\+\d{4})_REC_"
            r"([+-]\d+\.\d+)([+-]\d+\.\d+)\.(?:wav|flac)"
        )
        match = pattern.search(filename)
        if match:
            s, lat, lon = match.groups()
            start_datetime = datetime.strptime(s, "%Y%m%dT%H%M%S%z").astimezone(SITE_TIMEZONE)
            latitude, longitude = float(lat), float(lon)
            recorder = getrecorder(dirpath)
            recorder_type = "SOLARBAR"
            gpssync = False
            matched = True

    # --- Solarbar: start timestamp + REC (no GPS) ---
    if not matched:
        pattern = re.compile(r"(\d{8}T\d{6}\+\d{4})_REC\.(?:wav|flac)")
        match = pattern.search(filename)
        if match:
            s = match.group(1)
            start_datetime = datetime.strptime(s, "%Y%m%dT%H%M%S%z").astimezone(SITE_TIMEZONE)
            recorder = getrecorder(dirpath)
            recorder_type = "SOLARBAR"
            gpssync = False
            matched = True

    # --- S4A: standard segment filename ---
    if not matched:
        pattern = re.compile(r"(S4A\d{5})_(\d{1})_(\d{8}_\d{6})_\d{3}\.(?:wav|flac)")
        match = pattern.search(filename)
        if match:
            recorder, channel, s = match.groups()
            start_datetime = return_s4_time(s)
            recorder_type = "S4A"
            gpssync = False
            matched = True

    # --- S4A: GPS-synced filename with $ delimiter ---
    if not matched:
        pattern = re.compile(r"(S4A\d{5})_(\d{8})\$(\d{6})\.(?:wav|w4v|flac)$")
        match = pattern.search(filename)
        if match:
            recorder, date_str, time_str = match.groups()
            start_datetime = return_s4_time(f"{date_str}_{time_str}")
            recorder_type = "S4A"
            gpssync = True
            matched = True

    # --- S4A: GPS-synced filename with _ delimiter ---
    if not matched:
        pattern = re.compile(r"(S4A\d{5})_(\d{8}_\d{6})\.(?:wav|w4v|flac)$")
        match = pattern.search(filename)
        if match:
            recorder, s = match.groups()
            start_datetime = return_s4_time(s)
            recorder_type = "S4A"
            gpssync = True
            matched = True

    indices = [
        'filepath', 'dirpath', 'filename', 'recorder', 'recorder_type',
        'channel', 'gps_sync', 'start_datetime', 'end_datetime',
        'latitude', 'longitude',
    ]
    return pd.Series(
        [filepath, dirpath, filename, recorder, recorder_type, channel,
         gpssync, start_datetime, end_datetime, latitude, longitude],
        index=indices,
    )
