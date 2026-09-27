"""Table-touch height check over the whole territory: where does the model think the fingers are when the table stops them?

At each spot of a polar grid (real base angle x distance from the turning axis) the arm moves above the spot at `--travel-z`,
lowers the fingers as a grasp does (--grasp-to-table: no touch detection, no gravity push; the table stopping the arm counts
as contact), reads the joints, and computes the lowest gripper point from the model with those joints. On a flat table
that point should be at 0 (table coordinates, --table-z); what it reads instead is the model's height error at that spot
(> 0: the real fingers are lower than the model thinks). A side-camera frame at each touch is saved for checking by eye.
The table must be clear.
Usage: .venv/bin/python -m real.height_probe real/runs/height_<time> [--pans ...] [--radii ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from real.arm import PROFILES, ArmConfig, MotionError, RealArm
from real.joints import body_real_deg


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--pans", default="-33,-15,3,21,39,51"); ap.add_argument("--radii", default="0.16,0.22,0.28,0.34")
    ap.add_argument("--travel-z", type=float, default=.10); ap.add_argument("--table-z", type=float, default=-0.0098)
    ap.add_argument("--profile", default="fast"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cam-server", default="http://127.0.0.1:8766"); ap.add_argument("--side", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--depth", type=float, default=-.04, help="descent target below the model table (m): the table must stop the fingers first")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = ArmConfig(table_z=a.table_z, **PROFILES[a.profile])
    if a.dry:
        from real.backend import SimBackend
        from run7.env.world import sample_scene
        arm = RealArm(SimBackend(sample_scene(60000, "place_in")), cfg, mirror=False)
    else:
        from real.backend import SocketBackend
        from real.cameras import RealCameras
        arm = RealArm(SocketBackend(port=a.port), cfg)
        rc = RealCameras({"side": a.side}, {}, server=a.cam_server, mirror=("side",))
    ax = float(arm.kin.data.xanchor[arm.kin.model.joint("shoulder_pan").id][0])
    pans = [float(v) for v in a.pans.split(",")]; radii = [float(v) for v in a.radii.split(",")]
    spots = []
    for ri, r in enumerate(radii):   # serpentine
        for p in (pans if ri % 2 == 0 else pans[::-1]):
            phi = p; ok = False
            for _ in range(4):   # aim the real base angle
                xy = [ax + r * np.cos(np.radians(phi)), -r * np.sin(np.radians(phi))]
                sol = arm.solve([*xy, .02], arm.yaw)
                if sol["success"]:
                    q = sol["joint_pos"][:5]
                else:
                    q, err, tilt = arm._ik_free([*xy, .02], arm.yaw, arm.kin.data.qpos[:5])
                    if err > 3.0 or tilt > arm.tilt_limit(.02):
                        break
                dp = p - float(body_real_deg(q)[0]); ok = True
                if abs(dp) < .3:
                    break
                phi += dp
            if ok:
                spots.append({"pan": p, "r": r, "xy": [float(v) for v in xy]})
    print(json.dumps({"spots": len(spots)}), flush=True)
    if a.dry:
        print(json.dumps(spots[:4])); return 0
    rows = []
    try:
        arm._sync()
        if np.max(np.abs(np.asarray(arm.q[:5]) - np.asarray(arm.REST_POSE))) < 8.0:
            arm.unfold()
        for k, s in enumerate(spots):
            try:
                p0 = arm.ee()
                if p0[2] < a.travel_z - .005:
                    arm._move([p0[0], p0[1], a.travel_z], settle=False)
                arm._move([*s["xy"], a.travel_z], settle=False)
                d = arm._descend(a.depth, detect=False, lead=False)   # the table stops it: a stall = contact (the follower pauses at 6 deg lag)
                time.sleep(.3); arm._sync(); q = np.asarray(arm.q, float).copy()
                low = arm._low_points(q); lowest = min(low, key=low.get)
                frame = rc.frames(("side",))["side"]
                cv2.imwrite(str(out / f"side_{k:02d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                ee = arm.ee()
                row = {"k": k, **s, "q": q.tolist(), "ee": [round(float(v), 4) for v in ee], "stalled": bool(d.get("stall", False)), "touched": bool(d.get("touched", False)),
                       "reached": bool(d.get("reached", False)), "lowest_link": lowest, "lowest_z_mm": round(low[lowest] * 1000, 1),
                       "low_points_mm": {kk: round(v * 1000, 1) for kk, v in low.items()}}
                rows.append(row); print(json.dumps({kk: row[kk] for kk in ("k", "pan", "r", "lowest_link", "lowest_z_mm", "stalled", "touched", "reached")}), flush=True)
                arm._move([ee[0], ee[1], ee[2] + .01], settle=False)
            except MotionError as exc:
                print(json.dumps({"k": k, "skipped": s, "why": str(exc)[:160]}), flush=True)
                try:
                    p0 = arm.ee(); arm._move([p0[0], p0[1], max(p0[2], a.travel_z)], settle=False)
                except MotionError:
                    pass
        try:
            arm.retreat(); arm.fold()
        except MotionError as exc:
            print(json.dumps({"final_move_skipped": str(exc)[:160]}), flush=True)
    finally:
        (out / "probe.json").write_text(json.dumps({"table_z": a.table_z, "rows": rows}, indent=1))
        arm.close(); arm.b.close(); rc.close()
    z = [r["lowest_z_mm"] for r in rows]
    if z:
        print(json.dumps({"touches": len(z), "lowest_point_mm": {"median": round(float(np.median(z)), 1), "min": min(z), "max": max(z)}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
