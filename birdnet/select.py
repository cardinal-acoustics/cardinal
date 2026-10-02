# birdnet/select.py
#
# Select candidate detections for localization from concordance data.
#
# Two strategies are available:
#
#   "flexible" (default)
#       Per recorder_group, keeps events where:
#         - max prob across recorders >= min_max_prob
#         - number of active recorders >= min_active_recorders
#         - number of recorders at p50 >= min_concordant_recs
#         - optionally filtered by date range
#       Caps output at max_per_species per (species, recorder_group).
#       This is the recommended path for most runs.
#
#   "strict"
#       Keeps events where a given fraction of active recorders all agree
#       at a single probability threshold (default: all recorders at p90).
#       More conservative; useful for high-precision spot checks.
#
# Usage:
#   candidates = select_candidates(concordance, bnfi, strategy="flexible")
#   candidates = select_candidates(concordance, bnfi, strategy="strict")

from __future__ import annotations

import logging
import re

import pandas as pd

from config import (
    CONCORDANCE_THRESHOLDS,
    MIN_ACTIVE_RECORDERS,
    FLEXIBLE_MIN_MAX_PROB,
    FLEXIBLE_MIN_CONCORDANT_RECS,
    FLEXIBLE_MAX_PER_SPECIES,
    STRICT_AGREEMENT_FRACTION,
    STRICT_MIN_PROB,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def common_name_to_birdcode(common_name: str) -> str:
    """Convert a species common name to an uppercase alphanumeric code.

    E.g. "Ruby-crowned Kinglet" → "RUBYCROWNEDKINGLET"
         "Chuck-will's-widow"   → "CHUCKWILLSWIDOW"
    """
    return re.sub(r"[^A-Za-z0-9]", "", common_name).upper()


def _make_call_id(row) -> str:
    birdcode = common_name_to_birdcode(row["common_name"])
    rg       = str(row["recorder_group"])
    ts       = pd.Timestamp(row["call_datetime_r"]).strftime("%Y%m%d-%H%M%S")
    return f"{birdcode}_{rg}_{ts}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_candidates(concordance: dict,
                      bnfi: pd.DataFrame,
                      strategy: str = "flexible",
                      recorder_groups: list[str] | None = None,
                      date_start: str | None = None,
                      date_end:   str | None = None,
                      # flexible parameters (override config)
                      min_max_prob:        float | None = None,
                      min_active_recorders: int  | None = None,
                      min_concordant_recs:  int  | None = None,
                      max_per_species:      int  | None = None,
                      # strict parameters (override config)
                      strict_min_prob:           float | None = None,
                      strict_agreement_fraction: float | None = None,
                      ) -> pd.DataFrame:
    """Select localization candidate events from concordance data.

    Parameters
    ----------
    concordance : dict returned by birdnet.parse.count_active_recorders()
    bnfi        : filtered detections table (from birdnet.parse.filter_detections())
                  used to pull call_datetime for output rows
    strategy    : "flexible" (default) or "strict"
    recorder_groups : list of recorder_group names to process;
                      None = all groups in concordance data
    date_start/end  : optional ISO date strings to restrict the output range

    Flexible-specific parameters (all override config.py defaults):
        min_max_prob         — min peak prob across recorders in the group
        min_active_recorders — min recorders recording at call time
        min_concordant_recs  — min recorders at p50 threshold
        max_per_species      — cap on candidates per (species, recorder_group)

    Strict-specific parameters:
        strict_min_prob           — threshold all recorders must meet
        strict_agreement_fraction — fraction of active recorders required (1.0 = all)

    Returns
    -------
    pd.DataFrame with columns:
        common_name, call_datetime_r, recorder_group, birdcode, call_id,
        maxp, ngoodrecs, n_active_recorders
    """
    if strategy not in ("flexible", "strict"):
        raise ValueError(f"strategy must be 'flexible' or 'strict', got {repr(strategy)}")

    # Resolve parameters
    min_max_prob         = min_max_prob         or FLEXIBLE_MIN_MAX_PROB
    min_active_recorders = min_active_recorders or MIN_ACTIVE_RECORDERS
    min_concordant_recs  = min_concordant_recs  or FLEXIBLE_MIN_CONCORDANT_RECS
    max_per_species      = max_per_species      or FLEXIBLE_MAX_PER_SPECIES
    strict_min_prob           = strict_min_prob           or STRICT_MIN_PROB
    strict_agreement_fraction = strict_agreement_fraction or STRICT_AGREEMENT_FRACTION

    t_high = CONCORDANCE_THRESHOLDS[0]   # p90
    t_low  = CONCORDANCE_THRESHOLDS[-1]  # p50
    key_high = f"counts_p{int(t_high * 100)}"
    key_low  = f"counts_p{int(t_low  * 100)}"

    maxes            = concordance["maxes"]
    counts_high      = concordance[key_high]
    counts_low       = concordance[key_low]
    active_recorders = concordance["active_recorders"]

    all_groups = maxes.columns.drop("allmax", errors="ignore").tolist()
    if recorder_groups is None:
        recorder_groups = all_groups

    # Build a lookup: (common_name, call_datetime_r) → call_datetime
    # so we can attach the un-rounded time to the output
    call_dt_lookup = (
        bnfi[["common_name", "call_datetime_r", "call_datetime"]]
        .drop_duplicates(subset=["common_name", "call_datetime_r"])
        .set_index(["common_name", "call_datetime_r"])["call_datetime"]
    )

    parts = []

    for rg in recorder_groups:
        if rg not in maxes.columns:
            log.warning("recorder_group %r not found in concordance data", rg)
            continue

        df = pd.DataFrame({
            "maxp":              maxes[rg],
            "ngoodrecs_high":    counts_high.get(rg, pd.Series(0, index=maxes.index)),
            "ngoodrecs":         counts_low.get(rg,  pd.Series(0, index=maxes.index)),
            "n_active_recorders": active_recorders.get(rg, pd.Series(0, index=maxes.index)),
        }).reset_index()   # brings common_name, call_datetime_r into columns

        if strategy == "flexible":
            df = df[df["maxp"]               >= min_max_prob]
            df = df[df["n_active_recorders"] >= min_active_recorders]
            df = df[df["ngoodrecs"]          >= min_concordant_recs]

        elif strategy == "strict":
            # All (or a specified fraction of) active recorders must agree
            required = df["n_active_recorders"] * strict_agreement_fraction
            df = df[df["n_active_recorders"] >= min_active_recorders]
            df = df[df["ngoodrecs_high"] >= required.apply(lambda x: max(1, round(x)))]

        if date_start:
            df = df[df["call_datetime_r"] >= pd.Timestamp(date_start, tz="UTC")]
        if date_end:
            df = df[df["call_datetime_r"] <  pd.Timestamp(date_end,   tz="UTC")]

        if df.empty:
            continue

        df["recorder_group"] = rg
        df["birdcode"]       = df["common_name"].apply(common_name_to_birdcode)
        df["call_id"]        = df.apply(_make_call_id, axis=1)

        # Attach un-rounded call_datetime where available
        df = df.join(call_dt_lookup, on=["common_name", "call_datetime_r"], how="left")

        if strategy == "flexible" and max_per_species:
            df = (df.sort_values(["ngoodrecs", "maxp"], ascending=False)
                    .groupby("common_name", group_keys=False)
                    .head(max_per_species))

        parts.append(df)
        log.info("%-12s  %-10s  %4d candidates", rg, strategy, len(df))

    out_cols = ["common_name", "birdcode", "call_datetime_r", "call_datetime",
                "recorder_group", "call_id", "maxp",
                "ngoodrecs", "ngoodrecs_high", "n_active_recorders"]

    if not parts:
        log.warning("No candidates selected")
        return pd.DataFrame(columns=out_cols)
    result = pd.concat(parts, ignore_index=True)
    result = result[[c for c in out_cols if c in result.columns]]
    result = result.sort_values(["recorder_group", "common_name", "call_datetime_r"])

    log.info("Total candidates: %d  (strategy=%s)", len(result), strategy)
    return result
