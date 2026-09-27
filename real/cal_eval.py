"""Honest accuracy of a session calibration: 5-fold cross-validation on its (3D fingertip, pixel) pairs.

Each fold fits the camera on 80% of the pairs (PnP start, robust 6-DOF fit at a fixed field of view, or with the focal
length free) and measures the held-out 20%: the pixel error and, more usefully, how far the table point under each
held-out pixel lands from where it should (mm, at the object-centre height). Reported per region of the table.
Usage: .venv/bin/python -m real.cal_eval real/cal/top_camera_wide.json [--fovy 57] [--write]
--write refits on all pairs with the best setting and saves <json>_ok.json (accepted) for run_real.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real.selfcal_real import pnp_camera
from run3.phase2_photoreal import mjexport
from run8.agent.agent import apply_fix, fit_camera_robust, pixel_to_table


def fit(P, U, fovy):
    c0 = pnp_camera(P, U, 448, 448)
    if c0 is None:
        return None
    if fovy:
        c0 = {**c0, "fovy_deg": float(fovy)}
    x, inl, rms = fit_camera_robust(c0, P, U, focal=not fovy)
    return apply_fix(c0, x)


def table_err_mm(cam, p3, uv, z=.036):
    """Distance on the plane at the marker's own height between the marker and the back-projection of its pixel."""
    q = pixel_to_table(cam, uv[0], uv[1], p3[2])
    return None if q is None else float(np.hypot(q[0] - p3[0], q[1] - p3[1]) * 1000)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cal"); ap.add_argument("--fovy", type=float, nargs="*", default=[50, 57, 64, 0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    r = json.loads(Path(a.cal).read_text()); pr = [p for p in r["pairs"] if p["uv"] is not None]
    P = [np.asarray(p["p3"]) for p in pr]; U = [np.asarray(p["uv"], float) for p in pr]; n = len(P)
    print(json.dumps({"pairs": n, "x": [round(min(p[0] for p in P), 3), round(max(p[0] for p in P), 3)],
                      "y": [round(min(p[1] for p in P), 3), round(max(p[1] for p in P), 3)],
                      "z": [round(min(p[2] for p in P), 3), round(max(p[2] for p in P), 3)]}))
    rng = np.random.default_rng(0); order = rng.permutation(n); folds = np.array_split(order, 5); best = None
    for fovy in a.fovy:
        px, mm, rows = [], [], []
        for k in range(5):
            test = set(folds[k].tolist()); tr = [i for i in range(n) if i not in test]
            cam = fit([P[i] for i in tr], [U[i] for i in tr], fovy)
            if cam is None:
                continue
            for i in test:
                pr_ = mjexport.project(cam, P[i]); e = float(np.hypot(*(np.asarray(pr_) - U[i])))
                m = table_err_mm(cam, P[i], U[i]); px.append(e); mm.append(m if m is not None else np.nan); rows.append((P[i], e, m))
        if not px:
            continue
        px, mm = np.array(px), np.array(mm)
        reg = {}
        for name, sel in (("near x<.2", lambda p: p[0] < .2), ("far x>=.2", lambda p: p[0] >= .2),
                          ("right y<0", lambda p: p[1] < 0), ("left y>=0", lambda p: p[1] >= 0)):
            v = [m for p, e, m in rows if sel(p) and m is not None]
            if v:
                reg[name] = round(float(np.median(v)), 1)
        res = {"fovy": fovy or "free", "heldout_px_median": round(float(np.median(px)), 2), "heldout_px_p90": round(float(np.percentile(px, 90)), 2),
               "heldout_mm_median": round(float(np.nanmedian(mm)), 1), "heldout_mm_p90": round(float(np.nanpercentile(mm, 90)), 1), "by_region_mm_median": reg}
        print(json.dumps(res))
        if best is None or res["heldout_mm_median"] < best[0]["heldout_mm_median"]:
            best = (res, fovy)
    if a.write and best:
        cam = fit(P, U, best[1]); out = Path(a.cal).with_name(Path(a.cal).stem + "_ok.json")
        out.write_text(json.dumps({**r, "camera": cam, "accepted": True, "cv": best[0],
                                   "method": r.get("method", "") + f"; accepted after 5-fold CV (fovy {best[1] or 'free'})"}, indent=1))
        print("wrote", out, "camera pos", np.round(cam["pos"], 3).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
