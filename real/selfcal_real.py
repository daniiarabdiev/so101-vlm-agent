"""Session calibration of the top camera from the wrist sticker (run8 DECISIONS #16-20; run8/ARM_DAY.md section 2).

The arm visits the 18-pose wide grid (5-10 cm high, gripper down), a top frame is taken at each pose, the sticker is found
with the robust detector (size-aware, split parts merged), and a 7-DOF fit (camera pose + focal length) with outlier
rejection corrects a rough starting guess of the camera. The acceptance gate (>= 9 inliers, RMS <= 2.5 px, correction
<= 8 deg / 100 mm) decides: if not accepted, fix the cause and repeat. Never run the agent on a rejected calibration.
Output JSON: {"camera": {pos, xmat, fovy_deg, width, height}, "accepted", "inliers", "rms_px", "pairs": [...]} for
real/run_real.py --camera-json.

The starting guess (--guess JSON, or the simulator's nominal top camera): pos (m, robot base frame), xmat (row-major
3x3 camera-to-world rotation, MuJoCo convention: the camera looks along its local -z, image "up" is local +y), fovy_deg.
A tape-measured guess within ~5 deg / 5 cm is enough. With the simulator's camera convention the robot base is at the
left edge of the image.
Usage (real):  .venv/bin/python -m real.selfcal_real real/cal/top_camera.json --backend socket --cams top=2 [--undistort top=real/cal/top_intr.npz] --guess GUESS.json --confirm
Dry run (sim): .venv/bin/python -m real.selfcal_real /tmp/cam.json --backend sim --sim-cam-err 3,30 --sim-true-fovy 56

Arm-day variant (2026-09-26), no tape measure and no sticker:
  --marker-tip   the marker is a small piece of coloured tape wrapped round the tip of the fixed finger; its position in the
                 gripper frame comes from the robot model (fixed-jaw tip, raised 7 mm to the visible tape centre)
  --marker-hsv   H,dH,minS,minV of the tape in PIL HSV (0-255); --probe prints the colours of one top frame to choose them
  --no-guess     no rough camera pose: the marker is the single clear blob of that colour, the fit starts from PnP
                 (RANSAC over focal lengths) and the same acceptance gate applies relative to that start
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

import run8.env.world as RW
from real.arm import ArmConfig, MotionError, RealArm
from real.joints import real_to_sim
from run3.phase2_photoreal import mjexport
from run8.agent.agent import SELFCAL_GRID_WIDE, apply_fix, detect_marker, fit_camera_robust
from run7.agent.agent import rotation


def marker_px_expected(cam: dict, p3) -> dict:
    f = cam["height"] / (2 * np.tan(np.radians(cam["fovy_deg"]) / 2))
    r_px = RW.MARKER_RADIUS * f / float(np.linalg.norm(np.asarray(cam["pos"]) - p3))
    return {"min_px": int(0.15 * np.pi * r_px ** 2), "merge_px": max(6.0, 1.3 * r_px)}


def detect_color(img: Image.Image, hsv_spec, predict=None, window: float = 140.0, min_px: int = 8, merge_px: float = 10.0):
    """Centroid of the marker tape: pixels within dH of hue H (wrapping) and above minS / minV, grouped into blobs (blobs
    within merge_px merged). With a prediction: the nearest blob within window. Without one: the largest blob, and only if
    it is unambiguous (the next one is under half its size); otherwise None."""
    h0, dh, s0, v0 = hsv_spec
    hsv = np.asarray(img.convert("HSV")).astype(np.int32)
    dhue = np.abs(hsv[..., 0] - h0); dhue = np.minimum(dhue, 256 - dhue)
    lab, n = ndimage.label((dhue <= dh) & (hsv[..., 1] >= s0) & (hsv[..., 2] >= v0))
    blobs = []
    for c in range(1, n + 1):
        ys, xs = np.nonzero(lab == c); blobs.append([float(xs.mean() + .5), float(ys.mean() + .5), len(xs)])
    merged = []
    for b in sorted(blobs, key=lambda b: -b[2]):
        for m in merged:
            if np.hypot(b[0] - m[0], b[1] - m[1]) <= merge_px:
                k = m[2] + b[2]; m[0] = (m[0] * m[2] + b[0] * b[2]) / k; m[1] = (m[1] * m[2] + b[1] * b[2]) / k; m[2] = k; break
        else:
            merged.append(list(b))
    blobs = sorted([b for b in merged if b[2] >= min_px], key=lambda b: -b[2])
    if not blobs:
        return None
    if predict is not None:
        d = [(float(np.hypot(b[0] - predict[0], b[1] - predict[1])), b) for b in blobs]
        d = [x for x in d if x[0] <= window]
        return None if not d else tuple(min(d, key=lambda x: x[0])[1][:2])
    if len(blobs) > 1 and blobs[1][2] > .5 * blobs[0][2]:
        return None
    return tuple(blobs[0][:2])


def tip_marker_local(arm) -> np.ndarray:
    """Visible centre of tape wrapped round the fixed finger's tip, in the gripper body frame (from the robot model: the
    lowest points of the gripper-body visual mesh with the gripper pointing down, raised 7 mm)."""
    k = arm.kin; m, d = k.model, k.data; gid = m.body("gripper").id
    saved = d.qpos.copy()
    try:
        sol = arm.solve([.22, 0.0, .07], arm.yaw); d.qpos[:5] = sol["joint_pos"][:5]; mujoco.mj_forward(m, d)
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == gid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                pts.append(d.geom_xpos[g] + m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T)
        W = np.concatenate(pts); lo = W[:, 2].min(); tip = W[W[:, 2] < lo + .004].mean(0) + [0, 0, .007]
        return d.xmat[gid].reshape(3, 3).T @ (tip - d.xpos[gid])
    finally:
        d.qpos[:] = saved; mujoco.mj_forward(m, d)


def jaw_tip_local(arm, body: str = "gripper", grip_pct: float = 15.0) -> np.ndarray:
    """Like tip_marker_local for either jaw: the lowest visual-mesh points of `body` ("gripper" = fixed jaw,
    "moving_jaw_so101_v1" = moving jaw) with the gripper pointing down at this opening, raised 7 mm, in that body's frame.
    2026-09-27: the owner's green paint is on the MOVING jaw's tip (photos with the gripper open put it ~90 px from the
    fixed-jaw model; with the jaw modelled, closed and open photos agree)."""
    k = arm.kin; m, d = k.model, k.data; gid = m.body(body).id
    saved = d.qpos.copy()
    try:
        sol = arm.solve([.22, 0.0, .07], arm.yaw); d.qpos[:5] = sol["joint_pos"][:5]
        d.qpos[5] = real_to_sim(np.r_[np.zeros(5), grip_pct])[5]; mujoco.mj_forward(m, d)
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == gid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                pts.append(d.geom_xpos[g] + m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T)
        W = np.concatenate(pts); lo = W[:, 2].min(); tip = W[W[:, 2] < lo + .004].mean(0) + [0, 0, .007]
        return d.xmat[gid].reshape(3, 3).T @ (tip - d.xpos[gid])
    finally:
        d.qpos[:] = saved; mujoco.mj_forward(m, d)


def tip_tilt(arm, toward, deg: float = 35.0):
    """Joints with wrist_flex changed by +-deg so the fingers point more towards `toward` (x, y, table coordinates), if that
    keeps every link 1.5 cm above the table and inside the joint bounds; None otherwise."""
    q0 = np.asarray(arm.q, float); best = None
    for dq in (-deg, deg):
        q = q0.copy(); q[3] += dq
        if arm.bounds is not None and not (arm.bounds[3, 0] + 2 <= q[3] <= arm.bounds[3, 1] - 2):
            continue
        if min(arm._low_points(q).values()) < .015:
            continue
        d = arm.kin._scratch; d.qpos[:] = arm.kin.data.qpos; d.qpos[:6] = real_to_sim(q); mujoco.mj_kinematics(arm.kin.model, d)
        fingers = -d.xmat[arm.gripper_body].reshape(3, 3)[:, 2]; tip = d.site_xpos[arm.site]
        to = np.asarray(toward, float) - tip[:2]; score = float(fingers[:2] @ (to / max(np.linalg.norm(to), 1e-9)))
        if best is None or score > best[0]:
            best = (score, q)
    return None if best is None or best[0] <= 0 else best[1]


def pnp_camera(pts, pix, width: int, height: int):
    """Camera (MuJoCo convention) from 3D-2D pairs without any guess: RANSAC PnP for several focal lengths, best RMS."""
    P, U = np.asarray(pts, np.float64), np.asarray(pix, np.float64); best = None
    for fovy in (30, 38, 45, 52, 60, 70, 80, 95):
        f = height / (2 * np.tan(np.radians(fovy) / 2)); K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(P, U, K, None, reprojectionError=8.0, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(P[inl[:, 0]], U[inl[:, 0]], K, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        Rcv = cv2.Rodrigues(rvec)[0]
        cam = {"pos": (-Rcv.T @ tvec).ravel().tolist(), "xmat": (Rcv.T @ np.diag([1., -1., -1.])).ravel().tolist(),
               "fovy_deg": float(fovy), "width": width, "height": height}
        r = [np.hypot(*(np.asarray(mjexport.project(cam, p) or (1e9, 1e9)) - u)) for p, u in zip(P, U)]
        score = (len(inl), -float(np.median(r)))
        if best is None or score > best[0]:
            best = (score, cam)
    return None if best is None else best[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backend", choices=["socket", "sim"], default="sim"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cams", default="top=2"); ap.add_argument("--undistort", default="")
    ap.add_argument("--cam-server", default="http://127.0.0.1:8766"); ap.add_argument("--rotate", default="", help="e.g. top=90")
    ap.add_argument("--crop", default="", help="top=x0:y0:x1:y1 square crop after rotation (must match run_real.py)")
    ap.add_argument("--guess", default=None); ap.add_argument("--poses", type=int, default=18)
    ap.add_argument("--marker-local", default=None, help="measured sticker position in the wrist frame, mm: x,y,z")
    ap.add_argument("--table-z", type=float, default=0.0); ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--sim-cam-err", default="3,30"); ap.add_argument("--sim-true-fovy", type=float, default=None)
    ap.add_argument("--save-frames", default=None)
    ap.add_argument("--marker-tip", action="store_true"); ap.add_argument("--marker-hsv", default="")
    ap.add_argument("--no-guess", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--zmax", type=float, default=.10, help="highest calibration pose (m); 0.085 on the real arm")
    ap.add_argument("--find-yaw", action="store_true", help="at the ready pose, turn the gripper until the tape is seen")
    ap.add_argument("--wide2", action="store_true", help="grid over the whole work area: x .14-.32, y +-.14, z .045-.085")
    ap.add_argument("--avoid", default="", help="x,y,r;... skip poses within r (m) of these table points (objects left on the table)")
    ap.add_argument("--ymin", type=float, default=-1.0, help="skip poses with y below this (e.g. a container on that side)")
    ap.add_argument("--hop-z", type=float, default=None, help="travel between poses via this height (m) instead of straight lines")
    ap.add_argument("--fix-fovy", type=float, default=None, help="known field of view (deg): fit pose only (e.g. 57 for the iPhone 4:3)")
    ap.add_argument("--grid", default="", help="custom grid 'x1,x2,..;y1,..;z1,..' (m), e.g. to add the far work area")
    ap.add_argument("--merge", default="", help="earlier calibration JSON whose found pairs join this fit (same camera, same table_z)")
    ap.add_argument("--static-mask", default="", help=".npy mask (448x448) of marker-coloured background spots to ignore")
    ap.add_argument("--marker-shift-mm", default="", help="x,y correction of the tip marker centre (real/kin_cal.py fit), mm")
    ap.add_argument("--marker-body", default="moving_jaw_so101_v1", help="body carrying the painted tip (gripper = fixed jaw)")
    ap.add_argument("--profile", default="commissioning", help="speed preset of real/arm.py PROFILES (the arm server's caps must match)")
    ap.add_argument("--max-pan", type=float, default=None, help="skip poses whose direction from the base exceeds this (deg): walls")
    ap.add_argument("--roll-neutral", action="store_true", help="per pose, the gripper yaw that keeps wrist_roll nearest 0 "
                    "(turned at the hop height): the wrist camera jams against the arm at large rolls")
    ap.add_argument("--show-tip", default="", help="x,y (m): at each pose tilt the wrist ~35 deg so the fingertip swings towards this "
                    "point (the camera's rough position) for the photo, then straighten (a steep camera cannot see the tip of a "
                    "gripper pointing straight down)")
    a = ap.parse_args()
    if a.marker_local:
        RW.MARKER_LOCAL[:] = np.asarray([float(v) for v in a.marker_local.split(",")]) / 1000
    hsv_spec = tuple(int(v) for v in a.marker_hsv.split(",")) if a.marker_hsv else None
    from real.arm import PROFILES
    cfg = ArmConfig(confirm=a.confirm, table_z=a.table_z, **PROFILES[a.profile])
    true_cam = None
    if a.backend == "sim":
        from real.backend import SimBackend
        from run7.env.world import sample_scene
        scene = sample_scene(60000, "place_in"); backend = SimBackend(scene)
        w = backend.world; cid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        nominal = mjexport.camera_pose(w.model, w.data, "overhead")
        if a.sim_true_fovy:
            w.model.cam_fovy[cid] = a.sim_true_fovy
        true_cam = mjexport.camera_pose(w.model, w.data, "overhead")
        deg, mm = (float(v) for v in a.sim_cam_err.split(","))
        rng = np.random.default_rng(5); ax = rng.normal(size=3); d = rng.normal(size=3); d /= np.linalg.norm(d)
        guess = {**nominal, "xmat": (rotation(ax, deg) @ np.asarray(nominal["xmat"]).reshape(3, 3)).ravel().tolist(),
                 "pos": (np.asarray(nominal["pos"]) + d * mm / 1000).tolist()}   # the believed (wrong) camera
        arm = RealArm(backend, cfg, scene=scene, mirror=True)
        from run5.env.render import flat_views
        grab = lambda: flat_views(w, ("top",))["top"]  # noqa: E731
    else:
        from real.backend import SocketBackend
        from real.cameras import RealCameras
        cams = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in a.cams.split(","))
        und = dict(kv.split("=") for kv in a.undistort.split(",") if kv)
        rot = {k: int(v) for k, v in (kv.split("=") for kv in a.rotate.split(",") if kv)}
        crops = {k: tuple(int(x) for x in v.split(":")) for k, v in (kv.split("=") for kv in a.crop.split(",") if kv)}
        rc = RealCameras({"top": cams["top"]}, und, crops=crops, server=a.cam_server or None, rot=rot)
        backend = SocketBackend(port=a.port); arm = RealArm(backend, cfg)
        g = json.loads(Path(a.guess).read_text()) if a.guess else None
        guess = g.get("camera", g) if g else mjexport.camera_pose(arm.model, arm.data, "overhead")
        guess = {"width": 448, "height": 448, **guess}
        grab = lambda: rc.frames(("top",))["top"]  # noqa: E731
    if a.marker_tip:
        RW.MARKER_BODY = a.marker_body; RW.MARKER_LOCAL[:] = jaw_tip_local(arm, a.marker_body)   # 2026-09-27: the paint is on the moving jaw
        if a.marker_shift_mm:   # the painted band's measured centre (real/kin_cal.py), in the gripper frame's x, y
            RW.MARKER_LOCAL[:2] += np.asarray([float(v) for v in a.marker_shift_mm.split(",")]) / 1000
        print(json.dumps({"marker": "fixed-finger tip tape", "gripper_frame_mm": (1000 * RW.MARKER_LOCAL).round(1).tolist()}))
    if a.probe:   # one frame, no motion: where is the tape, and how many blobs of its colour
        img = Image.fromarray(grab()); Path("real/runs/snap").mkdir(parents=True, exist_ok=True); img.save("real/runs/snap/probe_top.png")
        print(json.dumps({"probe_uv": detect_color(img, hsv_spec) if hsv_spec else None, "saved": "real/runs/snap/probe_top.png"}))
        arm.close(); backend.close(); return 0

    rng = np.random.default_rng([17, 0]); pts, pix, rows, fails = [], [], [], 0
    grid = SELFCAL_GRID_WIDE
    if a.wide2:
        grid = [(x, y, z) for x in (.14, .19, .24, .28, .32) for y in (-.14, -.07, 0.0, .07, .14) for z in (.045, .065, .085)]
    if a.grid:
        gx, gy, gz = ([float(v) for v in part.split(",")] for part in a.grid.split(";"))
        grid = [(x, y, z) for x in gx for y in gy for z in gz]
    avoid = [tuple(float(v) for v in t.split(",")) for t in a.avoid.split(";") if t]
    grid = [g for g in grid if g[1] >= a.ymin and all(np.hypot(g[0] - ax, g[1] - ay) > ar for ax, ay, ar in avoid)]
    try:
        if a.backend == "socket":   # the first move of a session starts from the folded rest pose: guarded unfold
            print(json.dumps({"unfold": arm.unfold()}), flush=True)
        if a.find_yaw:   # turn the gripper at the ready pose until the camera sees the tape; calibrate at that yaw
            y0, found = arm.yaw, None
            for dy in (0, 90, -90, 45, -45, 135, -135):   # never a full half-turn: the wrist cables
                y = float(np.arctan2(np.sin(y0 + np.radians(dy)), np.cos(y0 + np.radians(dy))))
                if not arm.solve(arm.ee(), y)["success"]:
                    continue
                if dy:
                    arm.rotate(y)
                uv = detect_color(Image.fromarray(grab()), hsv_spec or (124, 12, 45, 90), min_px=20)
                print(json.dumps({"yaw_try_deg": dy, "tape_uv": uv}), flush=True)
                if uv is not None:
                    found = y; break
            if found is None:
                print(json.dumps({"accepted": False, "reason": "tape not visible at any gripper yaw"})); return 2
            arm.yaw = found
        for idx in rng.permutation(len(grid)):
            if len(pts) >= a.poses:
                break
            x, y, z = grid[idx]
            if z > a.zmax + 1e-9:
                continue
            if a.max_pan is not None and abs(np.degrees(np.arctan2(y, x))) > a.max_pan:
                continue
            if a.roll_neutral:   # the yaw whose (top-down) solution keeps the wrist roll nearest 0 at this pose
                best = None
                for yy in np.radians(np.arange(-180, 180, 15)):
                    sol = arm.solve([x, y, min(z, .085)], float(yy))
                    if sol["success"] and (best is None or abs(np.degrees(sol["joint_pos"][4])) < best[0]):
                        best = (abs(np.degrees(sol["joint_pos"][4])), float(yy))
                if best is None:
                    continue
                target_yaw = best[1]
            yaw_here = target_yaw if a.roll_neutral else arm.yaw
            if not arm.solve([x, y, z], yaw_here)["success"]:   # top-down, or (above 6 cm) tilting within max_tilt_deg
                if not z > .06:
                    continue
                _q, err_mm, tilt = arm._ik_free([x, y, z], yaw_here, arm.kin.data.qpos[:5])
                if err_mm > 3.0 or tilt > arm.cfg.max_tilt_deg:
                    continue
            arm._confirm(f"calibration pose {len(rows) + 1}/{a.poses}: ({x:.2f}, {y:.2f}, {z:.3f}) m") if a.confirm else None
            print(json.dumps({"pose": len(rows) + 1, "target": [x, y, z]}), flush=True)
            try:
                if a.hop_z:   # up, (turn the wrist), across, down: never sweep low through objects left on the table
                    p0 = arm.ee(); arm._move([p0[0], p0[1], max(a.hop_z, p0[2])])
                    if a.roll_neutral:
                        arm._turn_in_place(target_yaw)
                    arm._move([x, y, a.hop_z])
                arm._move([x, y, z]); arm._sync()
                q_down = np.asarray(arm.q, float).copy(); tilted = None
                if a.show_tip:
                    tilted = tip_tilt(arm, [float(v) for v in a.show_tip.split(",")])
                    if tilted is not None:
                        arm._track([np.r_[tilted[:5], np.nan]], settle=False); arm._sync()
            except MotionError as exc:   # skip the pose (the arm holds where it stopped) and carry on; three in a row -> stop
                fails += 1; print(json.dumps({"skipped": [x, y, z], "why": str(exc)}), flush=True)
                if fails >= 5:
                    raise
                continue
            fails = 0
            q_photo = [float(v) for v in arm.q]
            p3 = RW.marker_world(arm.kin)  # sticker position from forward kinematics of the measured joints
            p3 = np.asarray(p3) - [0, 0, cfg.table_z]
            img = grab()
            if a.save_frames:
                Path(a.save_frames).mkdir(parents=True, exist_ok=True); Image.fromarray(img).save(f"{a.save_frames}/pose_{len(rows):02d}.png")
            if a.show_tip and tilted is not None:   # straighten again before the next hop (keeps the paths top-down)
                arm._track([np.r_[q_down[:5], np.nan]], settle=False); arm._sync()
            if a.no_guess:
                if a.static_mask:   # blank background spots of the marker colour (hand_fit.py writes this mask)
                    img = img.copy(); img[np.load(a.static_mask)] = 0
                uv = detect_color(Image.fromarray(img), hsv_spec or (124, 12, 45, 90), min_px=20)   # not servo LEDs
            elif hsv_spec:
                uv = detect_color(Image.fromarray(img), hsv_spec, predict=mjexport.project(guess, p3), **marker_px_expected(guess, p3))
            else:
                uv = detect_marker(Image.fromarray(img), predict=mjexport.project(guess, p3), **marker_px_expected(guess, p3))
            rows.append({"p3": [float(v) for v in p3], "uv": None if uv is None else [float(v) for v in uv],
                         "q": [float(v) for v in arm.q] if q_photo is None else q_photo})   # joints at the photo (for joint-offset fits)
            if uv is not None:
                pts.append(p3); pix.append(uv)
        try:
            if a.hop_z:   # finish high and away from the objects
                p0 = arm.ee(); arm._move([p0[0], p0[1], max(a.hop_z, p0[2])]); arm._move([.18, .08, a.hop_z])
            else:
                arm._move([.23, 0.0, .085])
        except MotionError as exc:   # the calibration data is complete; the arm just holds where it is
            print(json.dumps({"final_move_skipped": str(exc)}), flush=True)
    finally:
        arm.close(); backend.close()
    if a.merge:   # pairs found by an earlier run with the same camera and table height
        for pr in json.loads(Path(a.merge).read_text())["pairs"]:
            if pr["uv"] is not None:
                pts.append(np.asarray(pr["p3"])); pix.append(pr["uv"]); rows.append({**pr, "merged": a.merge})
    if len(pts) < 6:
        print(json.dumps({"accepted": False, "reason": f"sticker found in only {len(pts)} of {len(rows)} poses"})); return 2
    if a.no_guess:
        start = pnp_camera(pts, pix, guess["width"], guess["height"])
        if start is None:
            print(json.dumps({"accepted": False, "reason": "PnP found no consistent camera"})); return 2
        guess = start
    if a.fix_fovy:
        guess = {**guess, "fovy_deg": float(a.fix_fovy)}
    x, inl, rms = fit_camera_robust(guess, pts, pix, focal=not a.fix_fovy)
    rot, tr = float(np.degrees(np.linalg.norm(x[:3]))), float(1000 * np.linalg.norm(x[3:6]))
    ok = len(inl) >= 9 and rms <= 2.5 and rot <= 8 and tr <= 100
    cam = apply_fix(guess, x)
    res = {"accepted": ok, "inliers": len(inl), "found": len(pts), "poses": len(rows), "rms_px": round(rms, 2), "correction_deg": round(rot, 2),
           "correction_mm": round(tr, 1), "fovy_deg": round(cam["fovy_deg"], 2), "camera": cam, "pairs": rows}
    if true_cam is not None:
        from run8.diag.selfcal_fits import table_err_mm
        res["sim_table_error_mm"] = {"before": round(table_err_mm(true_cam, guess), 1), "after": round(table_err_mm(true_cam, cam), 2)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("camera", "pairs")}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
