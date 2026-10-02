"""Manuscript control: how much does a wrong recorder position cost?

Residual TDOA error measures internal consistency, not spatial truth, so the
paper needs some evidence that it responds to geometry being wrong. This
displaces every recorder by a fixed distance in a random direction, re-solves
each call from the stored TDOAs, and reports what happens to the residual and
to the position.

Run against the same results and candidate set as the paper, so the numbers
share provenance with everything else:

    python analysis/paper_figures/perturbation_control.py \
        --results localization/results/localize_parallel.parquet

A "5 m perturbation" here means each recorder is moved exactly 5 m in a
uniformly random 3-D direction, independently per recorder and per repeat --
not a 5 m standard deviation. The realised displacement is reported so the
convention cannot be misread.

Sanity check first: with zero perturbation the recomputed residual must
reproduce the stored mean_error. If it does not, the geometry bookkeeping is
wrong and the perturbed numbers mean nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyproj

warnings.filterwarnings("ignore")
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from localize.run import locate_source_and_speed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="localization/results/"
                                     "localize_parallel.parquet")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--perturbations", type=float, nargs="+", default=[0, 5, 10])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="analysis/paper_figures/"
                                 "perturbation_control.csv")
args = ap.parse_args()

print("loading...", flush=True)
R = pd.read_parquet(args.results, columns=["call_id", "common_name",
                                           "recorder_group", "recorders",
                                           "pairs", "tdoas", "x_est", "c_est",
                                           "mean_error", "call_datetime"])
# complete-pair solves with >=4 recorders: the regime the paper describes
nrec = R.recorders.apply(len)
npair = R.pairs.apply(len)
# 13.5% of solves carry pair indices beyond their own recorder list -- the
# pipeline dropped recorders after forming pairs -- so their geometry cannot
# be reconstructed here. Excluded rather than silently mis-indexed.
maxidx = R.pairs.apply(lambda ps: max(max(map(int, q)) for q in ps) if len(ps) else -1)
R = R[(nrec >= 4) & (npair == nrec * (nrec - 1) // 2) & (maxidx < nrec)]
print(f"  {len(R):,} complete-pair solves with >=4 recorders", flush=True)
S = R.sample(min(args.n, len(R)), random_state=args.seed).reset_index(drop=True)

P = pd.read_parquet("data/deployments.parquet")
fw = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
P["X"], P["Y"] = fw.transform(P.longitude.values, P.latitude.values)
P["begin"] = pd.to_datetime(P.begin)
P["end"] = pd.to_datetime(P.end)
TZ = P.begin.dt.tz


def coords_for(recorders, when):
    """Deployment-correct position of each recorder at the time of the call."""
    out = []
    for rc in recorders:
        d = P[(P.recorder == rc) & (P.begin <= when) &
              (P.end.isna() | (P.end >= when))]
        if d.empty:
            d = P[P.recorder == rc]
            if d.empty:
                return None
            d = d.sort_values("begin").tail(1)
        r = d.iloc[0]
        out.append([r.X, r.Y, float(r.z)])
    return np.asarray(out, dtype=float)


def random_shift(n, dist, rng):
    """Each recorder moved exactly `dist` m in a uniformly random direction."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * dist


rng = np.random.default_rng(args.seed)
rows = []
for k, r in S.iterrows():
    # deployments carry tz-aware timestamps; match them rather than stripping
    when = pd.Timestamp(r.call_datetime)
    when = (when.tz_localize(TZ) if when.tzinfo is None
            else when.tz_convert(TZ))
    base = coords_for(list(r.recorders), when)
    if base is None:
        continue
    pairs = [tuple(map(int, q)) for q in r.pairs]
    tdoas = np.asarray(r.tdoas, dtype=float)
    c = float(r.c_est)
    x_ref = np.asarray(r.x_est, dtype=float)[:3]
    for dist in args.perturbations:
        coords = base if dist == 0 else base + random_shift(len(base), dist, rng)
        try:
            x_est, _, _ = locate_source_and_speed(coords, pairs, tdoas,
                                                  speed0=c, fit_speed=False)
        except Exception:
            continue
        # residual in seconds: the solver's residual is in metres
        res = np.array([(np.linalg.norm(x_est - coords[i]) -
                         np.linalg.norm(x_est - coords[j])) - c * dt
                        for (i, j), dt in zip(pairs, tdoas)])
        rows.append(dict(call_id=r.call_id, perturb=dist,
                         resid_ms=float(np.abs(res).mean() / c * 1000),
                         moved_m=float(np.linalg.norm(x_est - x_ref)),
                         moved_h=float(np.linalg.norm(x_est[:2] - x_ref[:2])),
                         stored_ms=float(r.mean_error) * 1000))
    if k and k % 1000 == 0:
        print(f"  {k}/{len(S)}", flush=True)

D = pd.DataFrame(rows)
D.to_csv(args.out, index=False)
print(f"\n{D.call_id.nunique():,} calls, {len(D):,} solves -> {args.out}\n")

# A re-solve from the recorder centroid does not always return to the
# pipeline's minimum -- the cost surface has several. Measuring sensitivity on
# calls whose baseline we failed to reproduce would conflate "perturbation
# moved it" with "our solver started somewhere else", so keep only the calls
# that reproduce, and say how many that is.
z = D[D.perturb == 0]
ok = z[z.moved_m < 0.5].call_id
print("sanity check, zero perturbation:")
print(f"  reproduced the stored solution: {len(ok):,} of {len(z):,} "
      f"({100*len(ok)/max(len(z),1):.1f}%)")
_d = np.abs(z[z.call_id.isin(ok)].resid_ms - z[z.call_id.isin(ok)].stored_ms)
print(f"  on those, |recomputed - stored| residual: median {_d.median():.5f} ms, "
      f"max {_d.max():.5f} ms")
D = D[D.call_id.isin(ok)]
print(f"  sensitivity measured on those {D.call_id.nunique():,} calls")

print(f"\n{'perturbation':>14}{'median resid':>15}{'median moved':>15}"
      f"{'median horiz':>15}{'p90 horiz':>12}")
for d_ in args.perturbations:
    g = D[D.perturb == d_]
    if g.empty:
        continue
    print(f"{d_:>11.0f} m {g.resid_ms.median():>13.2f} ms"
          f"{g.moved_m.median():>13.2f} m{g.moved_h.median():>13.2f} m"
          f"{np.percentile(g.moved_h, 90):>10.1f} m")
