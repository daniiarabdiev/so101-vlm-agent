"""Joint offsets + top camera from one calibration run (selfcal_real.py / fullcal.py output with "q" per photo).

The arm model's forward kinematics were centimetres off on the real arm (grasp heights, table contact, calibration
residuals growing with reach; 2026-09-27). Zero-point offsets of shoulder_lift, elbow_flex and wrist_flex are fitted
together with the camera pose (6-DOF, field of view fixed or free) and a correction of the painted marker's position on
the fingertip, so that the marker, placed by the corrected kinematics, reprojects onto where the camera saw it.
(Pan and roll offsets are not observable from one camera: a pan offset is a rotation of the whole scene, absorbed by the
camera pose.) Offsets are absolute: raw joint readings + offset = model joint, never on top of the installed file.

Models compared by 5-fold cross-validation (held-out pixel error, and table-plane mm at the marker's own height):
  cam       camera only, fovy fixed
  kin2      + 3 joint offsets + marker shift in the gripper frame's x, y (the 2026-09-27 15:00 model)
  kin3      + marker shift in z as well (the full-territory sweep photographs the gripper at very different tilts, where
            the paint's position along the finger matters)
  kin3f     + the field of view free
Also a region hold-out (train without the right-hand sector, pan > 30 deg; test on it): does the fit extrapolate?
Errors are reported per height layer, base-angle sector and distance.
Usage: .venv/bin/python -m real.kin_cal real/cal/full_<time>.json [--fovy 57] [--model kin3f] [--write]
--write saves real/cal/joint_offsets.json and <json>_kin.json (the camera for the corrected kinematics) from the chosen
model (default: the lowest held-out median mm).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

import run8.env.world as RW
from real.arm import ArmConfig, RealArm
from real.backend import SimBackend
from real.joints import OFFSETS_DEG, real_to_sim
from real.selfcal_real import jaw_tip_local, pnp_camera
from run3.phase2_photoreal import mjexport
from run7.env.world import sample_scene
from run8.agent.agent import apply_fix, pixel_to_table

JOINTS = (1, 2, 3)   # shoulder_lift, elbow_flex, wrist_flex
MODELS = {"cam": dict(joints=False, mz=False, fov=False), "kin2": dict(joints=True, mz=False, fov=False),
          "kin3": dict(joints=True, mz=True, fov=False), "kin3f": dict(joints=True, mz=True, fov=True)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cal"); ap.add_argument("--fovy", type=float, default=57.0)
    ap.add_argument("--table-z", type=float, default=-0.020); ap.add_argument("--write", action="store_true")
    ap.add_argument("--model", choices=list(MODELS), default=None, help="model to write (default: best held-out median mm)")
    ap.add_argument("--models", default="cam,kin2,kin3,kin3f")
    ap.add_argument("--marker-body", default="", help="body carrying the paint (default: the sweep's, else the fixed jaw 'gripper')")
    ap.add_argument("--merge", default="", help="another calibration JSON (same camera) whose pairs join this fit")
    ap.add_argument("--contacts", default="", help="real/height_probe.py probe.json: table touches (the flat table pins the heights)")
    ap.add_argument("--contact-mm", type=float, default=1.5, help="weight: one contact mm counts like this many px... (sigma, mm)")
    a = ap.parse_args()
    r = json.loads(Path(a.cal).read_text()); rows = [p for p in r["pairs"] if p["uv"] is not None and p.get("q")]
    if a.merge:
        rows += [{**p, "pose": {**p.get("pose", {}), "merged": True}} for p in json.loads(Path(a.merge).read_text())["pairs"] if p["uv"] is not None and p.get("q")]
    Q = np.array([p["q"] for p in rows], float); U = np.array([p["uv"] for p in rows], float); n = len(rows)
    meta = [p.get("pose", {}) for p in rows]
    C = []
    if a.contacts:   # valid touches only: the table stopped the fingers (stall) near the table (a stall 5 cm up = the arm stuck)
        C = [np.asarray(c["q"], float) for c in json.loads(Path(a.contacts).read_text())["rows"] if c.get("touched") and c["lowest_z_mm"] < 20]
    arm = RealArm(SimBackend(sample_scene(60000, "place_in")), ArmConfig(table_z=a.table_z), mirror=False)
    body = a.marker_body or r.get("args", {}).get("marker_body", "gripper")
    RW.MARKER_BODY = body; m0 = jaw_tip_local(arm, body, float(np.median(Q[:, 5]))).copy()

    def marker(q, dq, dm):   # absolute offsets: raw readings + dq, never the installed joint_offsets.json (2026-09-27: fitting
        # on top of the installed offsets and writing only dq would have dropped them)
        # dq is in radians, the readings in degrees (until 17:30 on 2026-09-27 dq was added to the degrees directly while every
        # report and the written file converted it as radians: the installed offsets were 57.3x the fitted ones)
        qq = np.asarray(q, float).copy(); qq[list(JOINTS)] += np.degrees(dq)
        arm.kin.data.qpos[:6] = real_to_sim(qq) - np.r_[np.radians(OFFSETS_DEG), 0.0]; mujoco.mj_kinematics(arm.kin.model, arm.kin.data)
        RW.MARKER_LOCAL[:] = m0 + dm
        return np.asarray(RW.marker_world(arm.kin)) - [0, 0, a.table_z]

    def lowest(q, dq):   # lowest point of the two jaws (m, table coordinates) with model joints = raw + dq
        qq = np.asarray(q, float).copy(); qq[list(JOINTS)] += np.degrees(dq) - OFFSETS_DEG[list(JOINTS)]
        low = arm._low_points(qq)
        return min(low["gripper"], low["moving_jaw_so101_v1"])

    def unpack(x, spec):
        k = 6; dq = np.zeros(3); dm = np.zeros(3); dfov = 0.0
        if spec["joints"]:
            dq = x[k:k + 3]; dm[:2] = x[k + 3:k + 5]; k += 5
        if spec["mz"]:
            dm[2] = x[k]; k += 1
        if spec["fov"]:
            dfov = x[k]; k += 1
        return dq, dm, dfov

    def solve(idx, spec):
        P0 = [marker(Q[i], np.zeros(3), np.zeros(3)) for i in idx]
        c0 = pnp_camera(P0, [U[i] for i in idx], 448, 448); c0 = {**c0, "fovy_deg": float(a.fovy)}
        k = 6 + 5 * spec["joints"] + spec["mz"] + spec["fov"]
        xs = [.01] * 6 + ([.02] * 3 + [.005] * 2 if spec["joints"] else []) + ([.005] if spec["mz"] else []) + ([1.0] if spec["fov"] else [])
        use_c = bool(C) and spec["joints"]
        if use_c:   # one more unknown: the real table's height in the model (m)
            k += 1; xs = xs + [.005]

        def cam_of(x):
            _dq, _dm, dfov = unpack(x, spec); c = apply_fix(c0, x[:6]); c["fovy_deg"] = float(a.fovy + dfov); return c

        def res(x):
            dq, dm, _dfov = unpack(x, spec); cam = cam_of(x)
            out = [np.asarray(mjexport.project(cam, marker(Q[i], dq, dm))) - U[i] for i in idx]
            reg = [dq * 30.0, dm * 200.0] if spec["joints"] else []   # weak priors: offsets of a few degrees / 5 mm
            if use_c:   # every touch: the lowest jaw point on the flat table (height x[-1])
                reg.append(np.array([(lowest(c, dq) - x[-1]) * 1000 / a.contact_mm for c in C]))
            return np.concatenate(out + reg)
        # bounds: the paint is on the fingertip (+-25 mm of the model's tip; a one-height fit on 2026-09-27 put it 14 cm
        # away and the camera 20 cm further, trading distance for zoom), joint offsets +-20 deg, field of view +-25 deg (the elbow
        # sat on a 10 deg bound in every model, 2026-09-27 17:10; the table touches pin the heights)
        lo = [-np.inf] * 6 + ([-.35] * 3 + [-.025] * 2 if spec["joints"] else []) + ([-.025] if spec["mz"] else []) + ([-25.0] if spec["fov"] else [])
        lo = lo + ([-.05] if use_c else [])
        sol = least_squares(res, np.zeros(k), loss="soft_l1", f_scale=4.0, x_scale=xs, bounds=(lo, [-v for v in lo]))
        dq, dm, _dfov = unpack(sol.x, spec)
        solve.table_h = float(sol.x[-1]) if use_c else None
        return cam_of(sol.x), dq, dm

    def errors(test, cam, dq, dm):
        px, mm = [], []
        for i in test:
            p = marker(Q[i], dq, dm); e = float(np.hypot(*(np.asarray(mjexport.project(cam, p)) - U[i]))); px.append(e)
            t = pixel_to_table(cam, U[i][0], U[i][1], p[2]); mm.append(np.nan if t is None else float(np.hypot(t[0] - p[0], t[1] - p[1]) * 1000))
        return px, mm

    def summary(px, mm):
        return {"px_median": round(float(np.median(px)), 2), "px_p90": round(float(np.percentile(px, 90)), 2),
                "mm_median": round(float(np.nanmedian(mm)), 1), "mm_p90": round(float(np.nanpercentile(mm, 90)), 1)}

    def sector(i):
        p = meta[i].get("pan")
        return None if p is None else ("left <-15" if p < -15 else ("ahead" if p <= 30 else "right >30"))

    def dist(i):
        r_ = meta[i].get("r")
        return None if r_ is None else ("near <.2" if r_ < .2 else ("mid" if r_ < .3 else "far >=.3"))

    def regions(ids, mm):
        out = {}
        for g, f in (("z", lambda i: meta[i].get("z")), ("pan", sector), ("r", dist)):
            d = {}
            for i, m_ in zip(ids, mm):
                k = f(i)
                if k is not None:
                    d.setdefault(str(k), []).append(m_)
            if d:
                out[g] = {k: round(float(np.nanmedian(v)), 1) for k, v in sorted(d.items())}
        return out

    print(json.dumps({"pairs_with_joints": n, "installed_offsets_deg": OFFSETS_DEG.round(3).tolist(), "fit": "absolute offsets from raw readings"}))
    order = np.random.default_rng(0).permutation(n); folds = np.array_split(order, 5); results = {}
    for name in a.models.split(","):
        spec = MODELS[name]; ids, px, mm = [], [], []
        for f in folds:
            test = f.tolist(); tr = [i for i in range(n) if i not in set(test)]
            cam, dq, dm = solve(tr, spec); e, m_ = errors(test, cam, dq, dm); ids += test; px += e; mm += m_
        res = {"model": name, **summary(px, mm), "by_region_mm_median": regions(ids, mm)}
        right = [i for i in range(n) if meta[i].get("pan", 0) > 30]
        if len(right) >= 8 and n - len(right) >= 20:
            cam, dq, dm = solve([i for i in range(n) if i not in set(right)], spec); e, m_ = errors(right, cam, dq, dm)
            res["holdout_right_sector"] = {"n": len(right), **summary(e, m_)}
        if C:   # the touches under this model, fitted on all photos (+ touches if the model has joints)
            cam_, dq_, dm_ = solve(list(range(n)), spec); z = np.array([lowest(c, dq_) for c in C]) * 1000
            res["touch_heights_mm"] = {"spread_sd": round(float(z.std()), 1), "range": [round(float(z.min()), 1), round(float(z.max()), 1)],
                                       "table_h_mm": None if solve.table_h is None else round(solve.table_h * 1000, 1),
                                       "offsets_deg": np.round(np.degrees(dq_), 2).tolist(), "fovy": round(cam_["fovy_deg"], 1)}
        results[name] = res; print(json.dumps(res), flush=True)
    best = a.model or min(results, key=lambda k: results[k]["mm_median"])
    cam, dq, dm = solve(list(range(n)), MODELS[best])
    fit_px, fit_mm = errors(list(range(n)), cam, dq, dm)
    worst = sorted(range(n), key=lambda i: -fit_px[i])[:8]
    print(json.dumps({"chosen": best, "joint_offsets_deg": dict(zip(("shoulder_lift", "elbow_flex", "wrist_flex"), np.round(np.degrees(dq), 2).tolist())),
                      "marker_shift_mm": np.round(dm * 1000, 1).tolist(), "fovy_deg": round(cam["fovy_deg"], 2), "camera_pos": np.round(cam["pos"], 3).tolist(),
                      "table_h_mm": None if solve.table_h is None else round(solve.table_h * 1000, 1),
                      "touches_mm": None if not C else np.round(np.array([lowest(c, dq) for c in C]) * 1000, 1).tolist(),
                      "fit_all": summary(fit_px, fit_mm), "worst_poses": [{"i": i, "px": round(fit_px[i], 1), "pose": {k: meta[i].get(k) for k in ("pan", "r", "z", "kind")}} for i in worst]}))
    if a.write:
        off = np.zeros(5); off[list(JOINTS)] = np.degrees(dq)
        if solve.table_h is not None:   # the touches say where the real table is: move the table frame there (the camera with it)
            cam = {**cam, "pos": [float(cam["pos"][0]), float(cam["pos"][1]), float(cam["pos"][2]) - solve.table_h]}
        (Path(__file__).parent / "cal" / "joint_offsets.json").write_text(json.dumps(
            {"offsets_deg": off.round(3).tolist(), "order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
             "source": a.cal, "model": best, "meaning": "model joint = real reading + offset"}, indent=1))
        out = Path(a.cal).with_name(Path(a.cal).stem + "_kin.json")
        out.write_text(json.dumps({**r, "camera": cam, "accepted": True, "joint_offsets_deg": off.tolist(), "marker_shift_mm": (dm * 1000).tolist(),
                                   "marker_body": body, "merged": a.merge or None, "contacts": a.contacts or None,
                                   "table_h_m": solve.table_h, "table_z_world": None if solve.table_h is None else a.table_z + solve.table_h,
                                   "cv": results, "method": f"camera + joint offsets (real/kin_cal.py, model {best})"}, indent=1)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
