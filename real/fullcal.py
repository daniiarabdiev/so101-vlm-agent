"""Full-territory calibration sweep: top camera + joint zero points from photos of the painted fingertip (2026-09-27).

The earlier session calibrations (selfcal_real.py) only photographed poses where the gripper could point straight down at
8.5 cm and that were within 40 deg of straight ahead, a random 60 of the grid: the data stopped at x 0.36 m, |y| 0.20 m and
5-13 cm high. The far and right-hand table, where tasks then failed, was extrapolated. This sweep visits a polar grid over
the whole allowed range instead: every base turn the planner allows, 12-44 cm out, from grasp height to the carry height,
with the gripper tilting where it cannot point straight down (as the tasks do), in a serpentine order (short moves).
At each pose: the wrist tilts so the top camera sees the fingertip, the arm must be still (joints steady for 0.3 s), then
top + side frames are taken and the joints read again (their mean goes with the photo).
Output: the selfcal_real.py JSON (pairs with p3 / uv / q) for real/kin_cal.py and real/cal_eval.py; per pose an overlay
(found marker = green circle; where the installed calibration predicts it = red cross) for checking by eye.
Usage: .venv/bin/python -m real.fullcal real/cal/full_<time>.json --save-frames real/runs/full_<time> [--dry]
--dry plans on the model only (no arm, no cameras): feasible poses per layer and the order.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

import run8.env.world as RW
from real.arm import PROFILES, ArmConfig, MotionError, RealArm
from real.joints import GRIP_CLOSED_PCT
from real.joints import body_real_deg, real_to_sim
from real.selfcal_real import detect_color, jaw_tip_local, tip_tilt
from run3.phase2_photoreal import mjexport


def plan(arm, pans, radii, heights, max_roll: float, min_lift: float = -45.0) -> tuple[list[dict], dict]:
    """Feasible poses in visiting order. `pans` are real base-joint angles (deg; + = right, y < 0) and `radii` distances
    from the base's turning axis (which is 3.9 cm in front of the table origin, and the arm plane is offset 1.8 cm from
    it): each pose is solved iteratively until the planned base joint equals the requested angle, so the grid covers the
    joint range itself, up to the planning limits. Per base angle one gripper yaw keeps the wrist roll near 0 (the gripper
    camera jams against the arm at large rolls). A pose is kept if the gripper reaches it pointing down or tilting within
    the arm's limits (arm.tilt_limit), with |roll| <= max_roll and every joint inside the planning bounds."""
    m, d = arm.kin.model, arm.kin.data
    ax = float(d.xanchor[m.joint("shoulder_pan").id][0])

    def ik(t, yaw, seed):
        sol = arm.solve(t, yaw)
        if sol["success"]:
            return np.asarray(sol["joint_pos"][:5], float), "down"
        q, err, tilt = arm._ik_free(t, yaw, seed)
        return (q, f"tilt {tilt:.0f}") if err <= 3.0 and tilt <= arm.tilt_limit(t[2]) else (None, None)

    def aim(p, r, z, yaw, seed):
        phi = p
        for _ in range(4):
            t = [ax + r * np.cos(np.radians(phi)), -r * np.sin(np.radians(phi)), z]
            q, kind = ik(t, yaw, seed)
            if q is None:
                return None, None, None
            dp = p - float(body_real_deg(q)[0])
            if abs(dp) < .3:
                break
            phi += dp
        return t, q, kind

    yaw_of = {}
    for p in pans:
        best = None
        for yy in np.radians(np.arange(-180, 180, 5)):
            t = [ax + .20 * np.cos(np.radians(p)), -.20 * np.sin(np.radians(p)), .10]
            sol = arm.solve(t, float(yy))
            if sol["success"]:
                roll = abs(float(body_real_deg(sol["joint_pos"])[4]))
                if best is None or roll < best[0]:
                    best = (roll, float(yy))
        yaw_of[p] = best[1] if best else float(-np.radians(p))
    poses, seed = [], arm.kin.data.qpos[:5].copy()
    for li, z in enumerate(heights):
        rs = radii if li % 2 == 0 else radii[::-1]
        for ri, r in enumerate(rs):
            ps = pans if (li * len(radii) + ri) % 2 == 0 else pans[::-1]
            for p in ps:
                y = yaw_of[p]
                t, q, kind = aim(p, r, z, y, seed)
                if q is None:
                    continue
                qr = body_real_deg(q)
                try:   # the wall rules (camera-wall exclusion: left of -25 deg and within 18 cm of the base; 2026-09-27 a
                    arm._check_wall(np.r_[qr, 50.0])   # validation pose ended there and every later move was refused)
                except MotionError:
                    continue
                if qr[1] < min_lift:   # upper arm leaning back further than the carries do: the elbow nears the back wall
                    continue
                if abs(qr[4]) > max_roll or (arm.bounds is not None and not np.all((qr >= arm.bounds[:, 0] + 1.9) & (qr <= arm.bounds[:, 1] - 1.9))):
                    continue
                seed = q
                poses.append({"pan": float(p), "r": float(r), "z": float(z), "target": [float(v) for v in t], "yaw": y, "kind": kind,
                              "q_plan": [round(float(v), 2) for v in qr]})
    xy = np.array([p["target"][:2] for p in poses]) if poses else np.zeros((1, 2))
    stats = {"poses": len(poses), "per_height": {str(z): sum(1 for p in poses if p["z"] == z) for z in heights},
             "tilted": sum(1 for p in poses if p["kind"] != "down"),
             "real_pan_range": [min(p["q_plan"][0] for p in poses), max(p["q_plan"][0] for p in poses)] if poses else None,
             "x_range": [round(float(xy[:, 0].min()), 3), round(float(xy[:, 0].max()), 3)], "y_range": [round(float(xy[:, 1].min()), 3), round(float(xy[:, 1].max()), 3)],
             "max_r_per_pan": {str(p): max([q["r"] for q in poses if q["pan"] == p] or [0]) for p in pans}}
    return poses, stats


def wait_still(arm, window_s: float = .3, tol_deg: float = .25, timeout_s: float = 3.0):
    """Poll the joints until the body joints stay within tol_deg for window_s; (still?, last reading)."""
    hist, t0 = [], time.monotonic()
    while True:
        q = np.asarray(arm.b.read(), float); t = time.monotonic(); hist.append((t, q))
        hist = [h for h in hist if t - h[0] <= window_s]
        if t - t0 >= window_s and np.ptp(np.array([h[1][:5] for h in hist]), axis=0).max() < tol_deg:
            return True, q
        if t - t0 > timeout_s:
            return False, q
        time.sleep(.04)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--pans", default="-37,-29,-21,-13,-5,3,11,19,27,35,43,51", help="real base-joint angles (deg; + = right, y < 0)")
    ap.add_argument("--radii", default="0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40", help="distances from the turning axis (m)")
    ap.add_argument("--heights", default="0.18,0.11,0.05", help="tool heights above the table (m), visited in this order")
    ap.add_argument("--cams", default="top=3,side=1"); ap.add_argument("--rotate", default="top=270")
    ap.add_argument("--crop", default="top=0:120:1440:1560"); ap.add_argument("--mirror", default="side")
    ap.add_argument("--cam-server", default="http://127.0.0.1:8766"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--marker-hsv", default="116,16,50,60"); ap.add_argument("--table-z", type=float, default=-0.0098)
    ap.add_argument("--profile", default="fast"); ap.add_argument("--show-tip", default="0.55,-0.01")
    ap.add_argument("--max-roll", type=float, default=45.0)
    ap.add_argument("--grip", choices=["closed", "open"], default="closed", help="gripper state for the photos (the paint is on the moving jaw)")
    ap.add_argument("--marker-body", default="moving_jaw_so101_v1", help="body carrying the paint (gripper = fixed jaw)")
    ap.add_argument("--min-lift", type=float, default=-45.0, help="lowest shoulder_lift (real deg): the carries lean back to ~-40")
    ap.add_argument("--camera-json", default="real/cal/current_camera.json", help="installed calibration (its prediction is drawn)")
    ap.add_argument("--save-frames", default=None); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pans = [float(v) for v in a.pans.split(",")]; radii = [float(v) for v in a.radii.split(",")]
    heights = [float(v) for v in a.heights.split(",")]
    cfg = ArmConfig(table_z=a.table_z, **PROFILES[a.profile])
    if a.dry:
        from real.backend import SimBackend
        from run7.env.world import sample_scene
        arm = RealArm(SimBackend(sample_scene(60000, "place_in")), cfg, mirror=False)
        poses, stats = plan(arm, pans, radii, heights, a.max_roll, a.min_lift)
        print(json.dumps(stats))
        for p in poses[:5] + poses[-3:]:
            print(json.dumps(p))
        return 0

    from real.backend import SocketBackend
    from real.cameras import RealCameras
    cams = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in a.cams.split(","))
    rot = {k: int(v) for k, v in (kv.split("=") for kv in a.rotate.split(",") if kv)}
    crops = {k: tuple(int(x) for x in v.split(":")) for k, v in (kv.split("=") for kv in a.crop.split(",") if kv)}
    rc = RealCameras(cams, {}, crops=crops, server=a.cam_server, rot=rot, mirror=tuple(v for v in a.mirror.split(",") if v))
    backend = SocketBackend(port=a.port); arm = RealArm(backend, cfg)
    RW.MARKER_BODY = a.marker_body; RW.MARKER_LOCAL[:] = jaw_tip_local(arm, a.marker_body, GRIP_CLOSED_PCT)

    def _term(*_):   # a stop (kill) still saves the photos taken so far and closes the arm connection
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    cal_now = json.loads(Path(a.camera_json).read_text()); cam_now = cal_now.get("camera", cal_now)
    if cal_now.get("marker_body") == a.marker_body and cal_now.get("marker_shift_mm"):   # the fitted paint position
        sh = np.asarray(cal_now["marker_shift_mm"], float) / 1000; RW.MARKER_LOCAL[:] += np.r_[sh, np.zeros(3 - len(sh))]
    hsv = tuple(int(v) for v in a.marker_hsv.split(",")); show = [float(v) for v in a.show_tip.split(",")]
    out_dir = Path(a.save_frames or Path(a.out).with_suffix("")); out_dir.mkdir(parents=True, exist_ok=True)
    poses, stats = plan(arm, pans, radii, heights, a.max_roll, a.min_lift)
    print(json.dumps({"plan": stats}), flush=True)
    rows, fails, t_start = [], 0, time.time()

    def save():
        found = [r for r in rows if r["uv"] is not None]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"method": "full-territory sweep (real/fullcal.py)", "args": vars(a), "plan": stats,
                                           "found": len(found), "poses": len(rows), "table_z": a.table_z, "camera": cam_now,
                                           "accepted": False, "pairs": rows}, indent=1))
    try:
        arm._sync()
        if np.max(np.abs(np.asarray(arm.q[:5]) - np.asarray(arm.REST_POSE))) > 8.0:   # not parked: fold first, like run_real
            p0 = arm.ee()
            if p0[2] < .10:
                arm._move([p0[0], p0[1], .12])
            arm.fold()
        print(json.dumps({"unfold": arm.unfold()}), flush=True)
        if a.grip == "closed":   # an empty close stops the painted moving jaw against the fixed one: a fixed, repeatable spot
            arm.grip_closed_cmd = True; g = arm._grip(GRIP_CLOSED_PCT, max_s=5.0)
            print(json.dumps({"gripper_closed_pct": round(float(g), 1)}), flush=True)
        for i, ps in enumerate(poses):
            t = ps["target"]; arm.yaw = ps["yaw"]
            try:
                arm._move(t, settle=False); arm._sync(); q_down = np.asarray(arm.q, float).copy()   # the photo waits for stillness
                tilted = tip_tilt(arm, show)
                if tilted is not None:
                    arm._track([np.r_[tilted[:5], np.nan]], settle=False)
                still, qa = wait_still(arm)
                time.sleep(.12)   # the frame must be newer than the stillness check
                f = rc.frames(("top", "side")) if "side" in cams else {"top": rc.frames(("top",))["top"]}
                qb = np.asarray(arm.b.read(), float)
                if tilted is not None:
                    arm._track([np.r_[q_down[:5], np.nan]], settle=False)
            except MotionError as exc:
                fails += 1; print(json.dumps({"pose": i, "skipped": t, "why": str(exc)[:160]}), flush=True)
                if fails >= 6:
                    raise
                continue
            except RuntimeError as exc:   # camera fetch
                print(json.dumps({"pose": i, "camera_error": str(exc)[:160]}), flush=True); continue
            fails = 0
            q_photo = (qa + qb) / 2
            arm.kin.data.qpos[:6] = real_to_sim(q_photo); mujoco.mj_kinematics(arm.kin.model, arm.kin.data)
            p3 = np.asarray(RW.marker_world(arm.kin)) - [0, 0, a.table_z]
            top = f["top"]; uv = detect_color(Image.fromarray(top), hsv, min_px=20)
            pred = mjexport.project(cam_now, p3)
            n = len(rows)
            Image.fromarray(top).save(out_dir / f"pose_{n:03d}.png")
            if "side" in f:
                cv2.imwrite(str(out_dir / f"side_{n:03d}.jpg"), cv2.cvtColor(f["side"], cv2.COLOR_RGB2BGR))
            ov = cv2.cvtColor(top, cv2.COLOR_RGB2BGR).copy()
            if pred is not None:
                cv2.drawMarker(ov, (int(pred[0]), int(pred[1])), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            if uv is not None:
                cv2.circle(ov, (int(uv[0]), int(uv[1])), 9, (0, 255, 0), 2)
            cv2.putText(ov, f"{n} pan {ps['pan']:.0f} r {ps['r']:.2f} z {ps['z']:.2f} {ps['kind']}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.imwrite(str(out_dir / f"ov_{n:03d}.jpg"), ov)
            err_px = None if (uv is None or pred is None) else float(np.hypot(uv[0] - pred[0], uv[1] - pred[1]))
            rows.append({"p3": [float(v) for v in p3], "uv": None if uv is None else [float(v) for v in uv], "q": [float(v) for v in q_photo],
                         "q_spread_deg": float(np.max(np.abs(qa[:5] - qb[:5]))), "still": bool(still), "tilted": tilted is not None,
                         "pose": ps, "pred_installed": None if pred is None else [float(v) for v in pred], "err_installed_px": err_px})
            print(json.dumps({"n": n, "of": len(poses), "pan": ps["pan"], "r": ps["r"], "z": ps["z"], "kind": ps["kind"], "uv": rows[-1]["uv"],
                              "err_installed_px": None if err_px is None else round(err_px, 1), "still": still,
                              "elapsed_s": round(time.time() - t_start)}), flush=True)
            if n % 10 == 0:
                save()
        try:
            arm.retreat()
        except MotionError as exc:   # the straight retreat can be refused (a joint jump); folding interpolates joints instead
            print(json.dumps({"retreat_skipped": str(exc)[:160]}), flush=True)
        try:
            arm.fold(); print(json.dumps({"end": "folded at the base pose"}), flush=True)
        except MotionError as exc:
            print(json.dumps({"fold_refused": str(exc)[:160]}), flush=True)
    finally:
        save()
        arm.close(); backend.close(); rc.close()
    found = [r for r in rows if r["uv"] is not None]
    e = [r["err_installed_px"] for r in rows if r["err_installed_px"] is not None]
    print(json.dumps({"done": True, "poses": len(rows), "found": len(found), "planned": len(poses),
                      "installed_calibration_err_px": {"median": round(float(np.median(e)), 1), "p90": round(float(np.percentile(e, 90)), 1)} if e else None,
                      "minutes": round((time.time() - t_start) / 60, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
