# config.py — loads site configuration from cardinal.toml
#
# Search order for the config file:
#   1. Path in the CARDINAL_CONFIG environment variable
#   2. cardinal.toml in the current working directory
#   3. cardinal.toml in the user's home directory
#
# Copy cardinal.toml.example to cardinal.toml and edit it before running
# any part of the pipeline.

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Locate and load the TOML file
# ---------------------------------------------------------------------------

def _find_config() -> Path:
    env = os.environ.get("CARDINAL_CONFIG")
    if env:
        p = Path(env)
        if p.exists():
            return p
        sys.exit(f"[config] CARDINAL_CONFIG points to a missing file: {env}")

    for candidate in (Path.cwd() / "cardinal.toml", Path.home() / "cardinal.toml"):
        if candidate.exists():
            return candidate

    sys.exit(
        "[config] No cardinal.toml found.\n"
        "Copy cardinal.toml.example to cardinal.toml and edit it, or set\n"
        "CARDINAL_CONFIG to the path of your config file."
    )


with open(_find_config(), "rb") as _f:
    _cfg = tomllib.load(_f)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(section: str, key: str, default=None):
    return _cfg.get(section, {}).get(key, default)


def _require(section: str, key: str):
    val = _cfg.get(section, {}).get(key)
    if val is None:
        sys.exit(f"[config] Missing required key [{section}].{key} in cardinal.toml")
    return val


# ---------------------------------------------------------------------------
# BirdNET model auto-discovery (used when model_path / labels_path are blank)
# ---------------------------------------------------------------------------

def _find_birdnet_model() -> str:
    """Locate the BirdNET FP32 TFLite model from the installed birdnet_analyzer package.

    FP32 is used because the default tensor indices (546=logits, 545=embedding)
    are calibrated for the FP32 model.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("birdnet_analyzer")
        if spec and spec.submodule_search_locations:
            base = list(spec.submodule_search_locations)[0]
            candidate = os.path.join(base, "checkpoints", "V2.4",
                                     "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    return "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"


def _find_birdnet_labels() -> str:
    try:
        import importlib.util
        spec = importlib.util.find_spec("birdnet_analyzer")
        if spec and spec.submodule_search_locations:
            base = list(spec.submodule_search_locations)[0]
            candidate = os.path.join(base, "checkpoints", "V2.4",
                                     "BirdNET_GLOBAL_6K_V2.4_Labels.txt")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    return "birdnet/BirdNet-Models/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


# ---------------------------------------------------------------------------
# Exported names (same as before — no other file needs to change)
# ---------------------------------------------------------------------------

SITE_TIMEZONE = ZoneInfo(_require("site", "timezone"))
SITE_LAT      = float(_require("site", "lat"))
SITE_LON      = float(_require("site", "lon"))

AUDIO_TOPDIRS = _require("audio", "topdirs")
SAMPLE_RATE   = int(_get("audio", "sample_rate", 44100))

CATALOG_PATH        = _get("paths", "catalog",        "wavfiles.parquet")
RECORDER_ZONES_PATH = _get("paths", "recorder_zones", "regions/recorder_zones.json")
CHECKPOINT_DIR      = _get("paths", "checkpoint_dir", "birdnet/checkpoints")
# Drone orthomosaic used as the basemap in the paper figures. Optional: the
# figures fall back to a plain background when it is not set.
ORTHOMOSAIC_PATH    = _get("paths", "orthomosaic", "") or None
DETECTIONS_PATH     = "birdnet/detections.parquet"   # not yet user-configurable
BIRDNET_OUTPUT_DIR  = "birdnet/probs_out"             # not yet user-configurable

_site_species = _get("paths", "site_species", "")
SITE_SPECIES_PATH = _site_species if _site_species else None

_model  = _get("birdnet", "model_path",  "")
_labels = _get("birdnet", "labels_path", "")
BIRDNET_MODEL_PATH  = _model  if _model  else _find_birdnet_model()
BIRDNET_LABELS_PATH = _labels if _labels else _find_birdnet_labels()

CONCORDANCE_THRESHOLDS       = _get("thresholds", "concordance_probs",           [0.9, 0.75, 0.5])
MIN_ACTIVE_RECORDERS         = int(_get("thresholds", "min_active_recorders",    4))
FLEXIBLE_MIN_MAX_PROB        = float(_get("thresholds", "flexible_min_max_prob", 0.9))
FLEXIBLE_MIN_CONCORDANT_RECS = int(_get("thresholds", "flexible_min_concordant_recs", 3))
FLEXIBLE_MAX_PER_SPECIES     = int(_get("thresholds", "flexible_max_per_species", 1000))
STRICT_AGREEMENT_FRACTION    = float(_get("thresholds", "strict_agreement_fraction", 1.0))
STRICT_MIN_PROB              = float(_get("thresholds", "strict_min_prob",       0.9))
