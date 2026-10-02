# Cardinal

A pipeline for detecting and acoustically localizing bird vocalizations from
multi-recorder passive acoustic monitoring arrays.

Given a folder of WAV recordings from an array of recorders with known GPS
positions, Cardinal will:

1. **Catalog** every recording (recorder, timestamp, GPS fix, recorder group)
2. **Run [BirdNET](https://github.com/kahst/BirdNET-Analyzer)** to detect species
   in every recording
3. **Cross-check detections across recorders** ("concordance") to find events
   multiple recorders agree on -- a strong signal against false positives, and
   a requirement for localization
4. **Localize** each event in 3D using time-difference-of-arrival (TDOA)
   between recorders, fitting source position and the local speed of sound

It's built for researchers running their own recorder arrays (Solarbar and
Wildlife Acoustics SM4/S4A units are supported natively; other recorders can be
added by extending `catalog/parse.py` and `catalog/audiofile.py`), not tied to
any specific field site.

## Quickstart

The fastest way to see the whole pipeline work is the demo notebook, which
runs it end to end on a small real dataset (six recorders, one day of dawn-to-
dusk recording) with no site setup required beyond downloading the audio:

```bash
pip install -e ".[birdnet-tflite]"     # or "[birdnet-tensorflow]"
python demo/assemble_demo_data.py --dest demo/demo_audio
jupyter notebook demo/Cardinal_Demo.ipynb
```

See [`demo/Cardinal_Demo.ipynb`](demo/Cardinal_Demo.ipynb) for the full
walkthrough, including a real localized bird call by the end.

## Installation

```bash
pip install -e ".[birdnet-tflite]"
```

BirdNET inference needs exactly one TFLite backend -- use `birdnet-tflite`
(lighter weight) or `birdnet-tensorflow` depending on what installs cleanly on
your platform. Both extras also pull in
[`birdnet_analyzer`](https://github.com/kahst/BirdNET-Analyzer), which supplies
the pretrained model and label files that `config.py` auto-discovers.

Requires Python >= 3.10.

## Configuring a site

Cardinal is driven by a `cardinal.toml` file -- your recording directories,
site coordinates, timezone, and detection/concordance thresholds all live
there rather than in code.

```bash
cp cardinal.toml.example cardinal.toml
# edit cardinal.toml: [audio] topdirs, [site] lat/lon/timezone, etc.
```

`config.py` looks for this file in order:

1. The path in the `CARDINAL_CONFIG` environment variable
2. `./cardinal.toml` (current working directory)
3. `~/cardinal.toml`

See `cardinal.toml.example` for every available option, documented inline.
`demo/cardinal.toml` is a second example already configured for the demo
dataset.

## Running the pipeline on your own data

```bash
# 1. Build the audio catalog
python -m catalog.build --index --catalog

# 2. Run BirdNET inference (writes sidecar parquets next to your audio)
python -m birdnet.run /path/to/your/audio --include-probs

# 3. Filter, compute concordance, and select localization candidates
python -m birdnet.pipeline

# 4. Localize candidates -- see demo/Cardinal_Demo.ipynb Step 7 for the API,
#    or localize/localize_utils/batch.py's run_localizations_batch / _parallel
#    for a batch runner over many candidates at once
```

Each step is checkpointed, so re-running after an interruption picks up where
it left off; pass `--force` to recompute from scratch.

### Recorder deployment geometry

TDOA localization needs precise recorder positions (and elevation) at the time
of each recording. Build this table once from a deployment CSV and an
elevation raster (DEM/DSM) covering your site:

```bash
python -m catalog.deployments --csv your_deployments.csv --dsm your_dem.tif
```

### Per-species frequency profiles

Cross-correlation works better when weighted toward the frequency band a
species actually sings in. `localize/freqprofiles.py` builds these profiles
either from a reference clip library (Macaulay Library, xeno-canto, the
Cornell Guide to Bird Sounds) or directly from your own field detections --
useful since reference libraries rarely cover every species you record. The
two sources can be merged, preferring reference clips where available and
falling back to field detections elsewhere.

### Speed of sound

`localize/temperature.py` fetches real historical hourly temperature for your
site and date range from the [Open-Meteo](https://open-meteo.com/) historical
weather archive (no API key needed) and converts it to speed of sound for the
TDOA solver's initial guess.

## Project structure

```
config.py                  Site configuration loader (reads cardinal.toml)
cardinal.toml.example     Template config -- copy to cardinal.toml

catalog/                   Audio cataloging
  parse.py                   Filename parsing (Solarbar and S4A conventions)
  audiofile.py                Lazy-loading audio file wrappers
  build.py                    Catalog builder (index -> aggregate)
  deployments.py               Recorder deployment table (GPS, elevation)
  reorganize.py                Directory-naming audit/rename tool

birdnet/                   BirdNET detection pipeline
  run.py                      TFLite inference (writes sidecar parquets)
  parse.py                     Load/join/filter detections, concordance
  select.py                     Candidate selection (flexible/strict strategies)
  pipeline.py                    End-to-end checkpointed pipeline runner
  suntimes.py                     Sunrise/sunset table builder

localize/                  TDOA-based acoustic localization
  tdoa.py                     Signal processing (cross-correlation, filtering)
  freqprofiles.py              Per-species frequency-weighting profiles
  temperature.py                 Speed-of-sound from historical weather data
  run.py                          Core localization functions + batch runner
  localize_utils/core.py           Full QC localization + plotting entry point

demo/                      Runnable worked example (see Quickstart above)
regions/                   Site GeoJSON zone definitions
```

## License

MIT -- see [LICENSE](LICENSE).

---

## This repository and the localization paper

This is the code behind *Automated localization of calling birds with small
passive acoustic arrays in complex soundscapes* (Eisen, Brown & Sanz Matias).

### Reproducing the figures

The figure scripts read a localization results file and write to
`analysis/paper_figures/`:

```bash
export CARDINAL_CONFIG=sites/ferguson.toml

python analysis/paper_figures/figure1_array_maps.py \
    --results path/to/localize_parallel.parquet \
    --groups Back9 NorthSide House --suffix _threesites
```

All of them accept `--results`, `--groups`, `--suffix`, and the map-based ones
(`figure1`, `figure4`, `figure5`) accept `--orthomosaic` to point at a GeoTIFF
basemap. With three or fewer sites they lay out as a single row.

`perturbation_control.py` reruns the recorder-position sensitivity analysis
reported in the paper, and `recorder_usage_table.py` regenerates the
deployment table.

### Running the pipeline from scratch

```bash
catalog.build                       # index audio into wavfiles.parquet
python -m birdnet.run               # BirdNET inference
python -m birdnet.pipeline          # detections -> candidates
python scripts/prepare_localization_inputs.py
python scripts/localize_candidates.py --candidates ... --freqprofiles ... \
    --deployments data/deployments.parquet --catalog wavfiles.parquet \
    --groups Back9 House NorthSide --fixed-speed
```

Sound speed comes from `localize/temperature.py`, which pulls hourly 2 m air
temperature and relative humidity from the HRRR 3 km analysis via the
Open-Meteo historical-forecast archive and applies the humidity correction
`c = c_dry (1 + 0.165 x_w)`. The paper uses a fixed per-event speed
(`--fixed-speed`): allowing it to vary leaves a four-recorder geometry poorly
conditioned, since a change in speed trades off against a radial shift in
source position.

### Data

Audio, detections and localization results are far too large for version
control and are deposited separately; see the Data and Code Availability
statement in the paper. `sites/ferguson.toml` has the site identity used in
the paper, with audio paths and the basemap left blank since those are
machine-specific.

### What is not here

This repository contains the localization pipeline and the analyses reported
in the paper. Exploratory work that did not inform the published results is
kept out deliberately.
