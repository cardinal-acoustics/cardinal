# Example: House array, 9 July 2025, 18:00–18:59 local

One evening hour from the House array — four recorders, 58.8 minutes — with
everything needed to localize it and check the answer.

```
242 calls localized below 1 ms residual, 7 species

   90  Dickcissel
   61  Painted Bunting
   45  Indigo Bunting
   26  Eastern Meadowlark
   18  Blue Grosbeak
    1  Scissor-tailed Flycatcher
    1  Mourning Dove
```

Three *Passerina* species singing in the same hour, alongside grassland birds,
which makes the habitat separation visible: the buntings and grosbeak localize
to treelines and edges, the dickcissels and meadowlarks to open field.

## Getting the audio

The recordings are a release asset rather than part of the repository, so a
plain `git clone` stays small.

```bash
gh release download example-house-2025-07-09 \
    --repo cardinal-acoustics/cardinal --dir examples/house_2025-07-09
tar -xzf examples/house_2025-07-09/audio.tar.gz -C examples/house_2025-07-09/
```

That unpacks `audio/BARLT_000086xx/*.flac`, 783 MB.

## What is here already

| file | contents |
|---|---|
| `metadata/audio_files.csv` | the four files, their frame counts and true sample rates |
| `metadata/deployments.csv` | surveyed recorder positions for the window |
| `metadata/candidates.csv` | the 302 candidate events the detector found |
| `metadata/expected_localizations.csv` | the 276 positions we obtained, with residuals |

`expected_localizations.csv` is what the paper reports for this window. Use it
to check your own run rather than as an input.

## Why the filenames look like that

```
S20250709T230000.000007+0000_E20250709T235845.000015+0000_+33.74824-93.48986.flac
```

Start and end timestamps to the microsecond, plus the recorder's GPS fix. The
catalog recovers each file's **true** sample rate as `frames / (end - start)`,
and these recorders do not run at a nominal 44,100 Hz:

```
BARLT_00008099   44099.3659 Hz
BARLT_00008604   44098.2016 Hz
BARLT_00008634   44098.3351 Hz
BARLT_00008662   44098.8890 Hz
```

A spread of 1.2 Hz, about 27 ppm — some 96 ms of relative drift over this one
hour. Localization needs arrival times good to a millisecond, so that per-file
calibration is not a detail; without it the method does not work at all. The
timestamps here were recomputed at each source file's own true rate when the
segment was cut, so the calibration is preserved exactly (drift under 10 µHz).

If you re-encode or rename these files, carry the timestamps with them.

## Running it

```bash
export CARDINAL_CONFIG=sites/ferguson.toml

python -m catalog.build --index examples/house_2025-07-09/audio
python scripts/localize_candidates.py \
    --candidates examples/house_2025-07-09/metadata/candidates.csv \
    --freqprofiles localization/inputs/freqprofiles.pkl \
    --deployments examples/house_2025-07-09/metadata/deployments.csv \
    --catalog wavfiles.parquet \
    --groups House --fixed-speed \
    --out-dir examples/house_2025-07-09/results
```

Then compare your `mean_error` distribution and positions against
`expected_localizations.csv`. Small differences in the last digits are
expected — the solver is iterative — but the median residual should land near
0.06 ms and roughly 88% of events should fall below 1 ms.

Frequency profiles are not included here; build them with
`scripts/build_cornell_profiles.py` or supply your own.
