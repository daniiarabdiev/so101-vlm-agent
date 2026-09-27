"""Offline comparison of self-calibration detector/fit variants on replayed detections (run8.diag.selfcal_replay).
Metric: the table-plane error of the corrected camera = for a grid of table points, the distance between the true point and
where the corrected camera maps its true pixel (mm; median over the grid per episode). What matters for pointing.
Variants:
  current   nearest blob to the prediction within 140 px (size >= 6, spread <= 9 px); soft-L1 fit on the first 12 poses
  reject    current + iterative outlier rejection (drop residuals > max(4 px, 3 x median), refit; >= 6 inliers)
  reassoc   reject + re-association: after the first fit, re-pick each pose's blob nearest the refined prediction (<= 25 px)
  guarded   reassoc; if < 9 inliers or inlier RMS > 2.5 px, add up to 6 more poses; if it still fails or the correction is
            implausible (> 8 deg or > 100 mm), report "failed" (arm day: stop and recalibrate) and keep the belief camera
Usage: python -m run8.diag.selfcal_fits REPLAY.jsonl [OUT.md] [--poses K] [--names a,b,...]   (K poses per fit, default 12)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from run3.phase2_photoreal import mjexport
from run3.phase3_closed_loop.runner import pixel_to_table
from run8.agent.agent import apply_fix, fit_camera

POSES = 12
GRID = [(x, y) for x in np.linspace(.10, .32, 6) for y in np.linspace(-.16, .16, 6)]


def table_err_mm(true: dict, cam: dict, z: float = .02) -> float:
    e = []
    for x, y in GRID:
        uv = mjexport.project(true, [x, y, z]); xy = pixel_to_table(cam, *uv, z)
        e.append(1e3 * float(np.hypot(xy[0] - x, xy[1] - y)) if xy is not None else 1e3)
    return float(np.median(e))


def merge(blobs, radius=12.0):
    """Blobs whose centroids lie within `radius` px are one marker split by an occluding edge: area-weighted merge."""
    out = []
    for b in sorted(blobs, key=lambda b: -b["n"]):
        for m in out:
            if np.hypot(b["u"] - m["u"], b["v"] - m["v"]) <= radius:
                n = m["n"] + b["n"]
                for key in ("u", "v", "h", "s"):
                    if key in m and key in b:
                        m[key] = (m[key] * m["n"] + b[key] * b["n"]) / n
                m["n"] = n; m["std"] = max(m["std"], b["std"]); break
        else:
            out.append(dict(b))
    return out


def moving(ep_poses, k, radius=3.0, frac=0.4):
    """The blobs of pose k that are not static scene colour: a blob found at the same pixel (within `radius`) in at least
    `frac` of the other poses' images belongs to the scene, not to the marker, which moves with the arm."""
    others = [p["blobs"] for j, p in enumerate(ep_poses) if j != k]
    out = []
    for b in ep_poses[k]["blobs"]:
        seen = sum(any(np.hypot(b["u"] - o["u"], b["v"] - o["v"]) <= radius for o in bl) for bl in others)
        if seen < frac * len(others):
            out.append(b)
    return out


def pick(blobs, pred, window, min_n=6, max_std=9.0, sized=False, color=False):
    if color:  # blob mean colour: the marker is saturated cyan (mean hue ~127, sat ~136); robot-edge / teal-table blobs
        blobs = [b for b in merge(blobs) if b.get("h", 0) >= 120 and b.get("s", 0) >= 65]  # are ~112 / ~59
    if sized:  # size-aware: merge split parts, ignore speckles (the marker is 100-280 px at this camera; C3 replay)
        blobs = [b for b in merge(blobs) if b["n"] >= 40]
    best = None
    for b in blobs:
        if b["n"] < min_n or b["std"] > max_std:
            continue
        d = float(np.hypot(b["u"] - pred[0], b["v"] - pred[1]))
        if d <= window and (best is None or d < best[0]):
            best = (d, (b["u"], b["v"]))
    return None if best is None else best[1]


def residuals(cam, x, pts, pix):
    c = apply_fix(cam, x)
    return np.array([np.hypot(*(np.asarray(mjexport.project(c, p)) - np.asarray(uv))) for p, uv in zip(pts, pix)])


def fit_reject(cam, pts, pix, focal=False):
    idx = list(range(len(pts)))
    x, _ = fit_camera(cam, pts, pix, focal=focal)
    for _ in range(4):
        r = residuals(cam, x, [pts[i] for i in idx], [pix[i] for i in idx])
        keep = [i for i, ri in zip(idx, r) if ri <= max(4.0, 3 * float(np.median(r)))]
        if len(keep) == len(idx) or len(keep) < 6:
            break
        idx = keep; x, _ = fit_camera(cam, [pts[i] for i in idx], [pix[i] for i in idx], focal=focal)
    r = residuals(cam, x, [pts[i] for i in idx], [pix[i] for i in idx])
    return x, idx, float(np.sqrt(np.mean(r ** 2)))


def variant(ep, name):
    cam = ep["belief_cam"]; poses = ep["poses"]
    sized = name.startswith(("sized", "static", "color"))
    static = name.startswith("static"); color = name.startswith("color")
    focal = name.endswith("+f")
    name = name.replace("sized+", "").replace("static+", "").replace("color+", "").replace("+f", "")
    use = poses[:POSES]
    mov = {id(p): moving(use, k) for k, p in enumerate(use)} if static else {}
    def detect(ps, c, window):
        pts, pix = [], []
        for p in ps:
            uv = pick(mov.get(id(p), p["blobs"]), mjexport.project(c, p["p3"]), window, sized=sized, color=color)
            if uv is not None:
                pts.append(p["p3"]); pix.append(uv)
        return pts, pix
    pts, pix = detect(use, cam, 140)
    if len(pts) < 6:
        return None, {"inliers": len(pts), "failed": True}
    if name == "current":
        x, rms = fit_camera(cam, pts, pix, focal=focal)
        return x, {"inliers": len(pts), "rms": rms}
    x, idx, rms = fit_reject(cam, pts, pix, focal)
    if name == "reject":
        return x, {"inliers": len(idx), "rms": rms}
    def reassoc(ps, x):
        c = apply_fix(cam, x); pts2, pix2 = detect(ps, c, 25)
        if len(pts2) < 6:
            return x, list(range(len(pts2))), 99.0, pts2
        x2, idx2, rms2 = fit_reject(cam, pts2, pix2, focal)
        return x2, idx2, rms2, pts2
    x, idx, rms, _ = reassoc(use, x)
    if name == "reassoc":
        return x, {"inliers": len(idx), "rms": rms}
    extra = False
    if len(idx) < 9 or rms > 2.5:
        extra = True; x, idx, rms, _ = reassoc(poses, x)
    bad = len(idx) < (9 if POSES >= 18 else 6) or rms > 2.5 or np.degrees(np.linalg.norm(x[:3])) > 8 or 1e3 * np.linalg.norm(x[3:]) > 100
    return (None if bad else x), {"inliers": len(idx), "rms": rms, "extra_poses": extra, "failed": bool(bad)}


def main() -> int:
    global POSES
    args = sys.argv[1:]
    if "--poses" in args:
        i = args.index("--poses"); POSES = int(args[i + 1]); del args[i:i + 2]
    names = ("current", "reject", "guarded", "sized+current", "sized+reject", "sized+guarded", "sized+reject+f", "sized+guarded+f")
    if "--names" in args:
        i = args.index("--names"); names = tuple(args[i + 1].split(",")); del args[i:i + 2]
    sys.argv = [sys.argv[0]] + args
    eps = [json.loads(l) for l in open(sys.argv[1])]
    res = {n: [] for n in names}
    base = []
    for ep in eps:
        base.append(table_err_mm(ep["true_cam"], ep["belief_cam"]))
        for n in names:
            x, info = variant(ep, n)
            cam = ep["belief_cam"] if x is None else apply_fix(ep["belief_cam"], x)
            res[n].append({**info, "err": table_err_mm(ep["true_cam"], cam), "task": ep["task"], "seed": ep["seed"]})
    L = [f"# Self-calibration variants on {len(eps)} replayed C3 episodes (3° / 30 mm error; {POSES} poses; table-plane error, mm)", "",
         f"Uncorrected: median {np.median(base):.1f} mm, max {np.max(base):.1f} mm.", "",
         "| Variant | median | 90th pct | max | episodes > 5 mm | > 10 mm | reported failed |", "|---|---|---|---|---|---|---|"]
    for n in names:
        e = np.array([r["err"] for r in res[n]])
        L.append(f"| {n} | {np.median(e):.2f} | {np.percentile(e, 90):.2f} | {e.max():.1f} | {int((e > 5).sum())} | {int((e > 10).sum())} | "
                 f"{sum(bool(r.get('failed')) for r in res[n])} |")
    g = res[names[-1]]
    L += ["", f"- {names[-1]}: extra poses used in {sum(bool(r.get('extra_poses')) for r in g)} episodes; failed = {[(r['task'], r['seed']) for r in g if r.get('failed')]}",
          f"- {names[0]}, worst 8: {sorted([(round(r['err'], 1), r['task'], r['seed'], round(r.get('rms', 0), 1)) for r in res[names[0]]], reverse=True)[:8]}",
          f"- {names[-1]}, worst 8: {sorted([(round(r['err'], 1), r['task'], r['seed'], round(r.get('rms', 0), 1)) for r in g], reverse=True)[:8]}"]
    txt = "\n".join(L) + "\n"; print(txt)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
