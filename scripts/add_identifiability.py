"""Score how well each TDOA solution's geometry actually constrains position.

The residual does not measure this. Across the corpus the correlation
between `mean_error` and how far a solution moves when the speed of sound is
corrected by ~1 m/s is r = 0.010 -- none at all -- and the worst case (54 m
of movement) survives the tightest residual cut. A solution can fit its
measurements perfectly and still be free to slide along a direction the
geometry does not pin down.

What does measure it is the conditioning of the Jacobian. With c held fixed
the residual for pair (i, j) is

    r = (|x - R_i| - |x - R_j|) - c * dt

whose derivative with respect to the source position is the difference of
the two unit vectors from the source to the recorders:

    dr/dx = u_i - u_j

Stack those over pairs and the singular values say how strongly each spatial
direction is constrained. The ratio of largest to smallest -- the condition
number -- is large exactly when some direction is nearly free, which is the
situation that makes a position sensitive to any small error, in c or
anywhere else.

Writes one row per solution with the condition number, the singular values,
and the worst-constrained direction, so a run can be filtered on geometry
rather than on fit.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyproj

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)


def jacobian(x: np.ndarray, rec: np.ndarray, pairs) -> np.ndarray:
    """d(residual)/d(position) for each pair: u_i - u_j."""
    d = x[None, :] - rec
    n = np.linalg.norm(d, axis=1, keepdims=True)
    u = d / np.maximum(n, 1e-9)
    return np.array([u[i] - u[j] for i, j in pairs], dtype=float)


def score(x, rec, pairs) -> dict:
    # `pairs` can index recorders that `recorders` no longer lists -- the
    # stored pair list predates whatever pruning the solve did, and which
    # entries survived is not recorded. Keep only in-range pairs.
    n = len(rec)
    pairs = [(i, j) for i, j in pairs if i < n and j < n]
    if len(pairs) < 3:
        return dict(cond=np.inf, s_min=0.0, s_max=np.nan, worst_axis=-1,
                    worst_z=np.nan, n_pairs_used=len(pairs))
    J = jacobian(np.asarray(x, float), np.asarray(rec, float), pairs)
    try:
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        return dict(cond=np.inf, s_min=0.0, s_max=np.nan, worst_axis=np.nan)
    s_min = float(S[-1])
    cond = float(S[0] / s_min) if s_min > 0 else np.inf
    # which axis the least-constrained direction mostly points along
    worst = np.abs(Vt[-1])
    return dict(cond=cond, s_min=s_min, s_max=float(S[0]),
                worst_axis=int(np.argmax(worst)), worst_z=float(worst[2]),
                n_pairs_used=len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="localize_parallel.parquet from a run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cond-max", type=float, default=1e4,
                    help="solutions above this are reported as not identifiable")
    args = ap.parse_args()

    cols = ["recorder_group", "recorders", "pairs", "tdoas", "x_est", "c_est",
            "call_datetime", "mean_error"]
    D = pd.read_parquet(args.run, columns=cols)
    if args.limit:
        D = D.head(args.limit)
    print(f"{len(D)} solutions")

    P = pd.read_parquet(f"{ROOT}/data/deployments.parquet").dropna(
        subset=["latitude", "longitude", "z"])
    fw = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    xy = np.array([fw.transform(r.longitude, r.latitude) for _, r in P.iterrows()])
    P["X"], P["Y"] = xy[:, 0], xy[:, 1]
    P["begin_d"] = pd.to_datetime(P.begin).dt.tz_localize(None)
    P["end_d"] = pd.to_datetime(P.end).dt.tz_localize(None)
    # one lookup table keyed by recorder; deployments rarely move, and when
    # they do the date decides
    by_rec = {r: g for r, g in P.groupby("recorder")}

    def coords(recs, when):
        out = []
        for rec in recs:
            g = by_rec.get(rec)
            if g is None:
                return None
            s = g[(g.begin_d <= when) &
                  (g.end_d.isna() | (g.end_d + pd.Timedelta(days=1) >= when))]
            r = (s if len(s) else g).iloc[0]
            out.append([r.X, r.Y, float(r.z)])
        return np.array(out, dtype=float)

    rows = []
    for i, r in enumerate(D.itertuples(index=False)):
        when = pd.Timestamp(r.call_datetime).tz_convert("UTC").tz_localize(None)
        rec = coords(list(r.recorders), when)
        if rec is None:
            continue
        # `pairs` is the full pair list; `tdoas` may be a subset after outlier
        # rejection, and which pairs survived is not recorded. Conditioning
        # needs the pairs actually used, so fall back to all pairs and flag it.
        pairs = [tuple(int(v) for v in p) for p in r.pairs]
        exact = len(pairs) == len(r.tdoas)
        s = score(r.x_est, rec, pairs)
        s.update(group=r.recorder_group, nrec=len(r.recorders),
                 npairs=len(pairs), mean_error=r.mean_error,
                 call_datetime=r.call_datetime, pairs_exact=exact)
        rows.append(s)
        if (i + 1) % 100000 == 0:
            print(f"  {i+1}/{len(D)}", flush=True)

    R = pd.DataFrame(rows)
    out = args.out or os.path.join(os.path.dirname(args.run), "identifiability.csv")
    R.to_csv(out, index=False, float_format="%.6g")
    print(f"\nwrote {out}: {len(R)} rows")

    ok = np.isfinite(R.cond)
    print(f"\ncondition number: median {R.cond[ok].median():.1f}  "
          f"p90 {R.cond[ok].quantile(.9):.1f}  p99 {R.cond[ok].quantile(.99):.3g}  "
          f"infinite {(~ok).sum()}")
    print(f"identifiable (cond < {args.cond_max:g}): "
          f"{(R.cond < args.cond_max).sum()} ({(R.cond < args.cond_max).mean():.1%})")
    print("\nby recorder count:")
    print(R.groupby("nrec").agg(
        n=("cond", "size"), med=("cond", "median"),
        p90=("cond", lambda s: s.quantile(.9)),
        frac_ok=("cond", lambda s: (s < args.cond_max).mean())).round(3).to_string())
    print("\nby array:")
    print(R.groupby("group").agg(
        n=("cond", "size"), med=("cond", "median"),
        frac_ok=("cond", lambda s: (s < args.cond_max).mean())).round(3).to_string())
    if "worst_z" in R:
        print(f"\nthe least-constrained direction is mostly vertical "
              f"(|z| > 0.8) in {(R.worst_z > 0.8).mean():.0%} of solutions")


if __name__ == "__main__":
    main()
