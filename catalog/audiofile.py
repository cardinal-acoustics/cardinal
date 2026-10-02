# catalog/audiofile.py
#
# Audio file classes for Solarbar and S4A recorders.
# These wrap the catalog metadata (from parse.py) with on-demand audio loading.

from __future__ import annotations

import contextlib
import os
import re
import warnings
import wave
from datetime import datetime, timedelta

import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile

from catalog.parse import parse_filename
from config import SAMPLE_RATE


class RecorderInfo:
    """Lightweight container for recorder identity and location."""

    def __init__(self, recorder_id: str, recorder_type: str | None = None,
                 location=None):
        self.recorder_id = recorder_id
        self.type = recorder_type
        self.location = location


class AudioFile:
    """Base class for a single audio file.

    Subclasses set dtstart, dtend, lat, lon, recorder, recorder_type.
    Audio data is loaded lazily via loaddata() and released via releasedata().
    """

    def __init__(self, filepath: str, uselibrosa: bool = False):
        self.filepath = filepath
        self.dir, self.file = os.path.split(filepath)
        self.filename, self.ext = os.path.splitext(self.file)
        self.uselibrosa = uselibrosa
        self.rate = librosa.get_samplerate(filepath)

        self.recorder: str | None = None
        self.recorder_type: str = "undefined"
        self.dtstart = None
        self.dtend = None
        self.lat = None
        self.lon = None
        self.data = None

    def loaddata(self):
        if self.uselibrosa or self.ext != ".wav":
            self.data, self.rate = librosa.load(self.filepath, sr=None)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.rate, self.data = scipy.io.wavfile.read(self.filepath, mmap=True)
                except Exception:
                    self.rate, self.data = scipy.io.wavfile.read(self.filepath, mmap=False)

    def releasedata(self):
        del self.data
        self.data = None

    def get_offset(self, dt: datetime) -> float:
        """Return sample offset for a given datetime."""
        return (dt - self.dtstart).total_seconds() * self.rate

    def getdata(self, start_index: int, end_index: int) -> np.ndarray:
        if not isinstance(self.data, np.ndarray):
            self.loaddata()

        start_index = int(round(start_index))
        end_index   = int(round(end_index))
        retdata = np.zeros(end_index - start_index)

        if self.data.ndim == 1:
            chunk = self.data[start_index:min(end_index, len(self.data))]
        else:
            chunk = self.data[0][start_index:min(end_index, len(self.data))]
        retdata[:len(chunk)] = chunk

        if len(retdata) != (end_index - start_index):
            warnings.warn("Data out of range")

        if self.rate != SAMPLE_RATE:
            retdata = librosa.resample(retdata, orig_sr=self.rate, target_sr=SAMPLE_RATE)

        self.releasedata()
        return retdata

    def getdata_filesecs(self, start_sec: float, end_sec: float) -> np.ndarray:
        return self.getdata(int(start_sec * self.rate), int(end_sec * self.rate))

    def getdata_startend(self, start_dt: datetime, end_dt: datetime) -> np.ndarray:
        return self.getdata_filesecs(
            (start_dt - self.dtstart).total_seconds(),
            (end_dt   - self.dtstart).total_seconds(),
        )


class AudioFileSolarbar(AudioFile):
    """Audio file from a Solarbar recorder.

    Timestamp and GPS are parsed directly from the filename.
    """

    def __init__(self, filepath: str, location=None, uselibrosa: bool = False):
        super().__init__(filepath, uselibrosa)

        with contextlib.closing(wave.open(filepath, 'r')) as f:
            self.datalen  = f.getnframes()
            self.rate     = f.getframerate()
            self.duration = self.datalen / float(self.rate)

        fnp = parse_filename(filepath)
        self.recorder      = fnp.recorder
        self.recorder_type = fnp.recorder_type
        self.dtstart       = fnp.start_datetime
        self.dtend         = fnp.end_datetime

        if pd.isna(self.dtend):
            self.dtend = self.dtstart + timedelta(seconds=self.duration)

        self.location    = location
        self.lat         = fnp.latitude
        self.lon         = fnp.longitude
        self.sample_size = (self.dtend - self.dtstart).total_seconds() / self.datalen

    def get_tt(self, inputtime):
        if isinstance(inputtime, datetime):
            offset = (inputtime - self.dtstart).total_seconds() * self.rate
        elif isinstance(inputtime, pd.Timestamp):
            offset = (pd.to_datetime(inputtime) - self.dtstart).total_seconds() * self.rate
        else:
            offset = inputtime
        return self.dtstart + timedelta(seconds=offset * self.sample_size)

    def get_ttoffset(self, tt) -> float:
        return (tt - self.dtstart).total_seconds() / self.sample_size

    def get_ttsample(self, tt, duration: float) -> np.ndarray:
        tto = self.get_ttoffset(tt)
        return self.getdata(tto, tto + duration * self.rate)


class AudioFileS4(AudioFile):
    """Audio file from an S4A recorder.

    Timestamp is parsed from the filename (fixed UTC-6, see catalog/parse.py
    for DST caveats).  GPS coordinates must be supplied externally (from the
    Kaleidoscope meta.csv sidecar).
    """

    def __init__(self, filepath: str, location=None, lat: float | None = None,
                 lon: float | None = None, uselibrosa: bool = False):
        super().__init__(filepath, uselibrosa=uselibrosa)

        with contextlib.closing(wave.open(filepath, 'r')) as f:
            self.datalen  = f.getnframes()
            self.rate     = f.getframerate()
            self.duration = self.datalen / float(self.rate)

        # Extract recorder ID from path
        self.recorder = None
        delimiters = ['/', '_', '.']
        pattern = '|'.join(map(re.escape, delimiters))
        for part in re.split(pattern, filepath):
            if part.startswith('S4A'):
                self.recorder = part
                break
        self.recorder_type = "S4A"
        self.location = location

        fnp = parse_filename(filepath)
        self.dtstart     = fnp.start_datetime
        self.dtend       = self.dtstart + timedelta(seconds=self.duration)
        self.lat         = lat
        self.lon         = lon
        self.sample_size = 1.0 / self.rate

    def get_tt(self, inputtime):
        if isinstance(inputtime, datetime):
            return inputtime
        if isinstance(inputtime, pd.Timestamp):
            return pd.to_datetime(inputtime)
        return self.dtstart + timedelta(seconds=inputtime * self.sample_size)

    def get_ttoffset(self, tt) -> float:
        return (tt - self.dtstart).total_seconds() / self.sample_size

    def get_ttsample(self, tt, duration: float) -> np.ndarray:
        tto = self.get_ttoffset(tt)
        return self.getdata(tto, tto + duration * self.rate)
