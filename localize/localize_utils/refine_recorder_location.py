"""refine_recorder_location.py

Utilities for refining recorder positions using fixed-source acoustic localizations.

Core idea (leave-one-recorder-out):
  1) Drop a recorder r_drop from each call and localize the source using remaining
     recorders/pairs/tdoas (using the same localization routine as core.py:
     bl.locate_source_and_speed).
  2) With sources fixed, optimize the dropped recorder position so that predicted
     TDOAs between (drop, other) match observed TDOAs.

This file is designed to be imported from Jupyter.

Typical usage:

    import localize.localize_utils.core as bl
    from localize.localize_utils.refine_recorder_location import (
        build_call_obs_list_from_df,
        localize_sources_leave_one_out,
        refine_dropped_recorder_fixed_sources,
    )

    call_obs_list = build_call_obs_list_from_df(df, ...)
    call_obs_by_id = {o.call_id: o for o in call_obs_list}

    sources = localize_sources_leave_one_out(bl, call_obs_list, rec_pos0, drop_rid="REC123")
    x_hat, res = refine_dropped_recorder_fixed_sources(
        sources, call_obs_by_id, rec_pos0, drop_rid="REC123",
        prior_sigma_m=None
    )

Notes:
- TDOAs are treated as fixed observations once a peak-choice has been made.
- To reduce selection bias, you can build call_obs_list using only the top peak per pair.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import pandas as pd

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("scipy is required for refine_recorder_location.py") from e


Pair = tuple[int, int]


@dataclass
class CallObs:
    """A single call observation with fixed, chosen TDOAs.

    Attributes
    ----------
    call_id:
        Unique call identifier.
    recorders:
        Recorder IDs used in this call, defining the local index space.
    pairs:
        List of local index pairs (i, j) into `recorders`.
    tdoas:
        Observed TDOAs (seconds), aligned 1:1 with `pairs`.
    """

    call_id: str
    recorders: list[str]
    pairs: list[Pair]
    tdoas: np.ndarray


@dataclass
class SourceSolution:
    """A localized source solution for a call (computed without a dropped recorder)."""

    call_id: str
    x_src: np.ndarray
    c_est: float
    recorders_used: list[str]
    pairs_used: list[Pair]


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    # numpy array / pandas series
    try:
        return list(x)
    except Exception:
        return [x]


def _coerce_pairs_to_local_indices(
    recorders: Sequence[str],
    raw_pairs: Sequence[tuple[Any, Any]],
) -> list[Pair]:
    """Coerce pairs to local integer indices.

    Accepts either:
      - (int, int) local indices
      - (str, str) recorder IDs present in `recorders`

    Returns a list of (int, int) pairs.
    """
    name_to_i = {rid: i for i, rid in enumerate(recorders)}
    out: list[Pair] = []
    for a, b in raw_pairs:
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            out.append((int(a), int(b)))
        else:
            if a not in name_to_i or b not in name_to_i:
                raise ValueError(f"Pair ({a!r},{b!r}) not resolvable within recorders list")
            out.append((name_to_i[a], name_to_i[b]))
    return out


def build_call_obs_list_from_df(
    df: "pd.DataFrame",
    *,
    call_id_col: str = "call_id",
    recorders_col: str = "recorders",
    pairs_col: str = "pairs",
    tdoas_col: str = "tdoas",
    verbose: bool = True,
) -> list[CallObs]:
    """Build a list of CallObs from a DataFrame.

    The dataframe should have columns containing:
      - call_id (string)
      - recorders (list[str])
      - pairs (list[tuple[int,int]] OR list[tuple[str,str]])
      - tdoas (list[float] seconds), aligned with pairs

    Rows that cannot be parsed are skipped.
    """
    if pd is None:
        raise ImportError("pandas is required for build_call_obs_list_from_df")

    out: list[CallObs] = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            call_id = str(row[call_id_col])
            recorders = [str(r) for r in _as_list(row[recorders_col])]
            raw_pairs = _as_list(row[pairs_col])
            raw_tdoas = _as_list(row[tdoas_col])

            if len(recorders) < 3:
                raise ValueError("<3 recorders")
            if len(raw_pairs) != len(raw_tdoas):
                raise ValueError("pairs/tdoas length mismatch")

            pairs = _coerce_pairs_to_local_indices(recorders, raw_pairs)
            tdoas = np.asarray(raw_tdoas, dtype=float)

            out.append(CallObs(call_id=call_id, recorders=recorders, pairs=pairs, tdoas=tdoas))

        except Exception as e:
            skipped += 1
            if verbose:
                print(f"Skipping row {idx}: {e}")

    if verbose:
        print(f"Built {len(out)} CallObs (skipped {skipped}).")

    return out


def drop_recorder_from_call(obs: CallObs, drop_rid: str) -> CallObs | None:
    """Return a new CallObs with `drop_rid` removed and pairs/tdoas reindexed.

    Returns None if fewer than 3 recorders remain or too few pairs remain.
    """
    if drop_rid not in obs.recorders:
        return obs

    keep = [r for r in obs.recorders if r != drop_rid]
    if len(keep) < 3:
        return None

    old_to_new = {rid: i for i, rid in enumerate(keep)}

    new_pairs: list[Pair] = []
    new_tdoas: list[float] = []

    for (i, j), t in zip(obs.pairs, obs.tdoas):
        ri, rj = obs.recorders[i], obs.recorders[j]
        if ri == drop_rid or rj == drop_rid:
            continue
        new_pairs.append((old_to_new[ri], old_to_new[rj]))
        new_tdoas.append(float(t))

    # Need enough constraints to localize; 3 pairs is a bare minimum.
    if len(new_pairs) < 3:
        return None

    return CallObs(call_id=obs.call_id, recorders=keep, pairs=new_pairs, tdoas=np.asarray(new_tdoas, dtype=float))


def localize_source_from_callobs(
    bl: Any,
    obs: CallObs,
    rec_pos: Mapping[str, np.ndarray],
    *,
    speed0: float = 343.2,
    speed_bounds: tuple[float, float] = (330.0, 360.0),
) -> tuple[np.ndarray, float, Any]:
    """Localize a fixed source from a CallObs using bl.locate_source_and_speed.

    Mirrors the core.py call:
        x_est, c_est, sol = bl.locate_source_and_speed(rec_coords, pairs, tdoas, ...)
    """
    rec_coords = np.vstack([np.asarray(rec_pos[rid], dtype=float) for rid in obs.recorders])
    tdoas = np.asarray(obs.tdoas, dtype=float)

    x_est, c_est, sol = bl.locate_source_and_speed(
        rec_coords,
        obs.pairs,
        tdoas,
        speed0=speed0,
        speed_bounds=speed_bounds,
    )
    return np.asarray(x_est, dtype=float), float(c_est), sol


def localize_sources_leave_one_out(
    bl: Any,
    call_obs_list: Sequence[CallObs],
    rec_pos: Mapping[str, np.ndarray],
    *,
    drop_rid: str,
    speed0: float = 343.2,
    speed_bounds: tuple[float, float] = (330.0, 360.0),
    require_min_recorders: int = 4,
    require_min_pairs: int = 6,
    verbose: bool = True,
) -> list[SourceSolution]:
    """Localize sources for many calls while dropping one recorder.

    Parameters
    ----------
    drop_rid:
        Recorder to exclude from the localization.
    require_min_recorders:
        Minimum number of recorders required AFTER dropping drop_rid.
    require_min_pairs:
        Minimum number of pairs required AFTER dropping drop_rid.

    Returns
    -------
    List[SourceSolution]
        Source positions computed without drop_rid.
    """
    out: list[SourceSolution] = []
    n_skip = 0

    for obs in call_obs_list:
        obs_sub = drop_recorder_from_call(obs, drop_rid)
        if obs_sub is None:
            n_skip += 1
            continue

        if len(obs_sub.recorders) < require_min_recorders:
            n_skip += 1
            continue
        if len(obs_sub.pairs) < require_min_pairs:
            n_skip += 1
            continue

        # Ensure coordinates exist for all remaining recorders
        if any(r not in rec_pos for r in obs_sub.recorders):
            n_skip += 1
            continue

        try:
            x_src, c_est, _sol = localize_source_from_callobs(
                bl,
                obs_sub,
                rec_pos,
                speed0=speed0,
                speed_bounds=speed_bounds,
            )
        except Exception:
            n_skip += 1
            continue

        out.append(
            SourceSolution(
                call_id=obs.call_id,
                x_src=x_src,
                c_est=c_est,
                recorders_used=list(obs_sub.recorders),
                pairs_used=list(obs_sub.pairs),
            )
        )

    if verbose:
        print(f"Localized {len(out)} sources (skipped {n_skip}) with drop_rid={drop_rid!r}.")

    return out


def refine_dropped_recorder_fixed_sources(
    sources: Sequence[SourceSolution],
    call_obs_by_id: Mapping[str, CallObs],
    rec_pos: Mapping[str, np.ndarray],
    *,
    drop_rid: str,
    c: float = 343.2,
    robust_loss: str = "huber",
    f_scale: float = 0.003,
    prior_sigma_m: float | None = None,
    verbose: bool = True,
):
    """Refine the position of a single dropped recorder using fixed source positions.

    Parameters
    ----------
    sources:
        Source solutions computed WITHOUT using drop_rid.
    call_obs_by_id:
        Mapping call_id -> full CallObs (that includes drop_rid when available).
    rec_pos:
        Mapping recorder_id -> xyz coordinate.
    drop_rid:
        Recorder to optimize.
    c:
        Speed of sound (m/s).
    prior_sigma_m:
        Optional weak prior sigma (meters) to keep solution near starting coord.
        Set to None to avoid any pull-back.

    Returns
    -------
    (x_hat, result)
        x_hat is a length-3 ndarray, result is scipy OptimizeResult.
    """
    x0 = np.asarray(rec_pos[drop_rid], dtype=float)

    def residuals(x_drop: np.ndarray) -> np.ndarray:
        x_drop = np.asarray(x_drop, dtype=float)
        res: list[float] = []

        for s in sources:
            c_call = float(s.c_est) if np.isfinite(getattr(s, "c_est", np.nan)) else float(c)
            obs_full = call_obs_by_id.get(s.call_id)
            if obs_full is None:
                continue
            if drop_rid not in obs_full.recorders:
                continue

            recs = obs_full.recorders
            idx_drop = recs.index(drop_rid)

            x_src = np.asarray(s.x_src, dtype=float)
            d_drop = float(np.linalg.norm(x_src - x_drop))

            # Add residuals for all pairs involving the dropped recorder.
            for (i, j), t_obs in zip(obs_full.pairs, obs_full.tdoas):
                t_obs = float(t_obs)

                if i == idx_drop:
                    other = recs[j]
                    if other not in rec_pos:
                        continue
                    d_other = float(np.linalg.norm(x_src - np.asarray(rec_pos[other], dtype=float)))
                    t_pred = (d_drop - d_other) / c_call
                    res.append(t_pred - t_obs)

                elif j == idx_drop:
                    other = recs[i]
                    if other not in rec_pos:
                        continue
                    d_other = float(np.linalg.norm(x_src - np.asarray(rec_pos[other], dtype=float)))
                    t_pred = (d_other - d_drop) / c_call
                    res.append(t_pred - t_obs)

        # Optional weak prior to keep it sane
        if prior_sigma_m is not None and prior_sigma_m > 0:
            sgm = float(prior_sigma_m)
            res.extend(((x_drop - x0) / sgm).tolist())

        return np.asarray(res, dtype=float)

    # Make sure we have enough residuals to constrain 3D
    r0 = residuals(x0)
    if r0.size < 10:
        raise RuntimeError(
            f"Not enough constraints to refine {drop_rid!r}: got {r0.size} residuals from {len(sources)} sources."
        )

    result = least_squares(
        residuals,
        x0,
        loss=robust_loss,
        f_scale=float(f_scale),
        verbose=2 if verbose else 0,
    )

    return np.asarray(result.x, dtype=float), result

def refine_dropped_recorder_fixed_sources_xy(
    sources: Sequence[SourceSolution],
    call_obs_by_id: Mapping[str, CallObs],
    rec_pos: Mapping[str, np.ndarray],
    *,
    drop_rid: str,
    c: float = 343.2,
    robust_loss: str = "huber",
    f_scale: float = 0.003,
    prior_sigma_m_xy: float | None = None,
    verbose: bool = True,
):
    """
    Like refine_dropped_recorder_fixed_sources, but optimizes ONLY (x,y) and keeps z fixed.

    This is recommended when recorder heights are reliable and the array is roughly planar,
    making z weakly identifiable from TDOA constraints.
    """
    x0_full = np.asarray(rec_pos[drop_rid], dtype=float)
    z0 = float(x0_full[2])
    x0 = x0_full[:2].copy()

    def residuals(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x_drop = np.array([xy[0], xy[1], z0], dtype=float)
        res: list[float] = []

        for s in sources:
            c_call = float(s.c_est) if np.isfinite(getattr(s, "c_est", np.nan)) else float(c)
            obs_full = call_obs_by_id.get(s.call_id)
            if obs_full is None:
                continue
            if drop_rid not in obs_full.recorders:
                continue

            recs = obs_full.recorders
            idx_drop = recs.index(drop_rid)

            x_src = np.asarray(s.x_src, dtype=float)
            d_drop = float(np.linalg.norm(x_src - x_drop))

            for (i, j), t_obs in zip(obs_full.pairs, obs_full.tdoas):
                t_obs = float(t_obs)

                if i == idx_drop:
                    other = recs[j]
                    if other not in rec_pos:
                        continue
                    d_other = float(np.linalg.norm(x_src - np.asarray(rec_pos[other], dtype=float)))
                    t_pred = (d_drop - d_other) / c_call
                    res.append(t_pred - t_obs)

                elif j == idx_drop:
                    other = recs[i]
                    if other not in rec_pos:
                        continue
                    d_other = float(np.linalg.norm(x_src - np.asarray(rec_pos[other], dtype=float)))
                    t_pred = (d_other - d_drop) / c_call
                    res.append(t_pred - t_obs)

        # Optional weak prior on XY only
        if prior_sigma_m_xy is not None and prior_sigma_m_xy > 0:
            sgm = float(prior_sigma_m_xy)
            res.extend(((xy - x0) / sgm).tolist())

        return np.asarray(res, dtype=float)

    r0 = residuals(x0)
    if r0.size < 6:
        raise RuntimeError(
            f"Not enough constraints to refine XY for {drop_rid!r}: got {r0.size} residuals from {len(sources)} sources."
        )

    result = least_squares(
        residuals,
        x0,
        loss=robust_loss,
        f_scale=float(f_scale),
        verbose=2 if verbose else 0,
    )

    x_hat = np.array([result.x[0], result.x[1], z0], dtype=float)
    return x_hat, result

def refine_recorder_leave_one_out_pipeline(
    bl: Any,
    call_obs_list: Sequence[CallObs],
    rec_pos: Mapping[str, np.ndarray],
    *,
    drop_rid: str,
    speed0: float = 343.2,
    speed_bounds: tuple[float, float] = (330.0, 360.0),
    c: float = 343.2,
    robust_loss: str = "huber",
    f_scale: float = 0.003,
    prior_sigma_m: float | None = None,
    require_min_recorders: int = 4,
    require_min_pairs: int = 6,
    verbose: bool = True,
):
    """Convenience end-to-end pipeline for one dropped recorder.

    1) Localize sources without drop_rid
    2) Refine drop_rid position using fixed sources

    Returns:
      (x_hat, result, sources)
    """
    call_obs_by_id = {o.call_id: o for o in call_obs_list}

    sources = localize_sources_leave_one_out(
        bl,
        call_obs_list,
        rec_pos,
        drop_rid=drop_rid,
        speed0=speed0,
        speed_bounds=speed_bounds,
        require_min_recorders=require_min_recorders,
        require_min_pairs=require_min_pairs,
        verbose=verbose,
    )

    x_hat, res = refine_dropped_recorder_fixed_sources(
        sources,
        call_obs_by_id,
        rec_pos,
        drop_rid=drop_rid,
        c=c,
        robust_loss=robust_loss,
        f_scale=f_scale,
        prior_sigma_m=prior_sigma_m,
        verbose=verbose,
    )

    return x_hat, res, sources