"""Run one command with the Run 8 agent (V4f: LoRA v4 + the Run 8 features) on the real SO-101, or on the simulator.

Real arm (two terminals, both from vlm_policy/):
  1) ${SO101_PYTHON:-python3} real/arm_server.py        (Enter there = STOP)
  2) .venv/bin/python -m real.run_real "take the yellow ball and put it into the container" --backend socket \
        --base https://<pod>-8000.proxy.runpod.net --camera-json real/cal/top_camera.json --cams top=2,side=0,wrist=1 \
        --table-z 0.000 --confirm
Dry runs (no hardware):
  .venv/bin/python -m real.run_real "" --backend sim --sim-task place_in --sim-seed 60000 --oracle --fast   (no model)
  .venv/bin/python -m real.run_real "" --backend sim --base https://<pod>-8000.proxy.runpod.net           (real model)

The agent is unchanged; three hooks connect it: its World is the RealArm (real/arm.py), its images come from the cameras
(real/cameras.py; sim: the simulator's views), and its top-camera model is the session calibration (real/selfcal_real.py).
Not used on the real arm by default: in-episode self-calibration (done once per session instead) and wrist refinement (it
assumes the simulator's wrist-camera mount; enable with --wrist-refine only after measuring the real mount).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from pathlib import Path

import run8.agent.agent as A
from real.arm import PROFILES, ArmConfig, RealArm
from run8.agent.run_cell import build_agent, make_parser


class Recorder:
    """Real-time episode video for write-ups: top | side | gripper views tiled, captioned with the command, elapsed time and
    the agent's latest step (from progress.jsonl), ~8 fps from the camera server (real/cam_server.py)."""

    def __init__(self, out: Path, server: str, idx: dict, top_rot: int, top_crop, command: str, fps: float = 8.0):
        import threading
        self.out, self.server, self.idx, self.rot, self.crop, self.command, self.fps = out, server, idx, top_rot, top_crop, command, fps
        self.stop, self.t0, self.frames = threading.Event(), time.time(), 0
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    def _grab(self, view):
        import urllib.request
        import cv2
        import numpy as np
        try:
            data = urllib.request.urlopen(f"{self.server}/frame/{self.idx[view]}?max_age=1&fmt=jpg&q=85", timeout=2).read()
            f = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return np.zeros((480, 480, 3), np.uint8)
        if view == "top":
            f = cv2.rotate(f, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rot]) if self.rot else f
            if self.crop:
                x0, y0, x1, y1 = self.crop; f = f[y0:y1, x0:x1]
        h, w = f.shape[:2]
        return cv2.resize(f, (int(480 * w / h), 480))

    def _step(self):
        try:
            last = (self.out / "progress.jsonl").read_text().strip().splitlines()[-1]
            r = json.loads(last); return f"step {r['step'] + 1}: {r.get('executed') or '...'}"
        except Exception:  # noqa: BLE001
            return "starting"

    def _loop(self):
        import cv2
        import numpy as np
        writer, next_t = None, time.time()
        while not self.stop.is_set():
            tile = np.hstack([self._grab(v) for v in ("top", "side", "wrist")])
            bar = np.zeros((44, tile.shape[1], 3), np.uint8)
            cv2.putText(bar, f'"{self.command}"   t = {time.time() - self.t0:5.1f} s   {self._step()}', (10, 30), 0, .8, (255, 255, 255), 2)
            frame = np.vstack([bar, tile])
            if writer is None:
                writer = cv2.VideoWriter(str(self.out / "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (frame.shape[1], frame.shape[0]))
            while next_t <= time.time():   # keep the video in real time: repeat the frame if grabbing was slow
                writer.write(frame); self.frames += 1; next_t += 1 / self.fps
            time.sleep(max(0.0, next_t - time.time()))
        if writer is not None:
            writer.release()

    def close(self):
        self.stop.set(); self.th.join(timeout=5)


def install_real_reach(arm, agent, pull_max: float = .07):
    """The agent's reach check is the simulator's top-down IK map; with carries above the top-down reach (--carry-z 0.15) it
    called every spot on the table out of reach (2026-09-26: the container was never tried). Replace it with the real arm:
    the gripper pointing down within the real joint range, or the tilting IK (small tilts below 6 cm). A drop spot
    (any relation) that is out of reach is pulled towards the workspace centre to the nearest reachable spot (at most pull_max; the
    agent's release check still has to see the gripper over the container before letting go)."""
    import numpy as np

    yaws = np.linspace(-np.pi, np.pi, 12, endpoint=False)

    def real_reach_ok(xy, zs, yaw=0.0, tol=0.0):
        for z in zs:
            ok = any(arm.solve([float(xy[0]), float(xy[1]), float(z)], float(y))["success"] for y in yaws)
            if not ok:
                _q, err, tilt = arm._ik_free([float(xy[0]), float(xy[1]), float(z)], arm.yaw, arm.kin.data.qpos[:5])
                ok = err <= 3.0 and tilt <= arm.tilt_limit(z)
            if not ok:
                return False
        return True
    A.reach_ok = real_reach_ok

    locate = agent.locate

    def locate_reachable(env, subs, k, what, top):
        xy, info = locate(env, subs, k, what, top)
        if xy is None or what != "place":   # every drop spot (18:23: a spot "on the table next to the container" at 38 cm was not
            return xy, info                  # checked, the carry stalled and the ball was dropped from 18 cm)
        zs = (arm.cfg.carry_z, .085, .05)
        if real_reach_ok(xy, zs):
            return xy, info
        xy0 = np.asarray(xy, float)
        for s_ in np.arange(.01, pull_max + 1e-9, .01):   # nearest reachable spot: rings of growing radius, 16 directions
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                cand = xy0 + s_ * np.array([np.cos(t), np.sin(t)])
                if real_reach_ok(cand, zs):
                    info = {**info, "reach_pull": {"from": xy0.tolist(), "to": cand.tolist(), "m": round(float(s_), 3)}}
                    print("[reach] drop spot pulled", json.dumps(info["reach_pull"]), flush=True)
                    return cand, info
        return xy, info
    agent.locate = locate_reachable


def background_shift(ref_path: str, server: str, idx: int, rot: int, crop, max_px: float = 25.0):
    """Similarity transform (2x3) taking pixels of the reference top view to the current one, from ORB features of the
    static scene (walls, table edge; RANSAC ignores the moved arm and objects). None if the views do not match."""
    import urllib.request
    import cv2
    import numpy as np
    from PIL import Image
    from real.cameras import Cam
    ref = cv2.cvtColor(np.asarray(Image.open(ref_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    raw = cv2.imdecode(np.frombuffer(urllib.request.urlopen(f"{server}/frame/{idx}?max_age=1", timeout=6).read(), np.uint8), cv2.IMREAD_COLOR)
    c = Cam.__new__(Cam); c.K = None; c.size = ref.shape[0]; c.rot = rot; c.mirror = False; c.expect = None; c.crop = crop
    now = cv2.cvtColor(c.process(raw), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(4000); k1, d1 = sift.detectAndCompute(ref, None); k2, d2 = sift.detectAndCompute(now, None)
    m = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < .7 * b.distance]   # ORB on wood grain mismatched
    if len(m) < 12:
        return None, {"inliers": 0, "matches": len(m)}
    H, inl = cv2.estimateAffinePartial2D(np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k2[x.trainIdx].pt for x in m]),
                                         method=cv2.RANSAC, ransacReprojThreshold=2)
    if H is None or inl.sum() < 15 or np.hypot(*H[:, 2]) > max_px:
        return None, {"inliers": 0 if inl is None else int(inl.sum())}
    return H, {"inliers": int(inl.sum()), "shift_px": [round(float(v), 1) for v in H[:, 2]],
               "rot_deg": round(float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))), 2)}


def crop_map(src, dst, size: int = 448):
    """Pixel (u, v) in the size x size view of crop box src -> the same raw point in the view of crop box dst."""
    if not src or not dst or tuple(src) == tuple(dst):
        return lambda u, v: (u, v)
    ss, sd = (src[2] - src[0]) / size, (dst[2] - dst[0]) / size
    return lambda u, v: ((src[0] + u * ss - dst[0]) / sd, (src[1] + v * ss - dst[1]) / sd)


def install_drift_guard(arm, cam: dict, table_z: float, fix_px: float = 3.0, max_deg: float = 3.0, hsv=(124, 12, 45, 90),
                        static_mask: str = "", v2c=None, c2v=None):
    """The phone that is the top camera can settle on its mount (2026-09-26: a 16 px shift between calibration and the next
    command made every grasp miss by ~2 cm). After each move the fingertip tape is usually in view: compare where the arm's
    kinematics put it with where the camera sees it. Up to max_deg of camera rotation is corrected in place (the agent reads
    the same dict); more stops the run: recalibrate (real/selfcal_real.py)."""
    import numpy as np
    from PIL import Image
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    import run8.env.world as RW
    from run3.phase2_photoreal import mjexport
    from real.selfcal_real import detect_color, jaw_tip_local
    RW.MARKER_BODY = "moving_jaw_so101_v1"; RW.MARKER_LOCAL[:] = jaw_tip_local(arm, RW.MARKER_BODY)   # the paint is on the moving jaw
    R0 = np.asarray(cam["xmat"], float).reshape(3, 3).copy(); state = {"rot": np.zeros(2), "log": []}
    mask = np.load(static_mask) if static_mask else None

    def check(where: str):
        arm._sync(); p3 = np.asarray(RW.marker_world(arm.kin)) - [0, 0, table_z]
        top = arm.frames(("top",))["top"]
        if mask is not None:   # background spots of the marker colour (hand_fit.py writes the mask)
            top = top.copy(); top[mask] = 0
        pred = mjexport.project(cam, p3)
        # only a blob near where the tip should be counts (a steep camera often cannot see the tip of a gripper pointing
        # down, and other marker-coloured specks then fooled the check, 2026-09-27)
        v2c_, c2v_ = v2c or (lambda u, v: (u, v)), c2v or (lambda u, v: (u, v))
        seen = None if pred is None else detect_color(Image.fromarray(top), tuple(hsv), predict=c2v_(*pred), window=30.0, min_px=12)
        seen = None if seen is None else v2c_(*seen)   # compare in the calibrated crop
        if seen is None or pred is None:
            return
        err = float(np.hypot(pred[0] - seen[0], pred[1] - seen[1])); rec = {"where": where, "err_px": round(err, 1)}
        if err > fix_px:   # small rotation of the camera about its own x / y axes that moves the tape onto what is seen
            def res(r):
                R = R0 @ Rotation.from_rotvec([r[0], r[1], 0.0]).as_matrix()
                return np.asarray(mjexport.project({**cam, "xmat": R.ravel().tolist()}, p3)) - np.asarray(seen)
            r = least_squares(res, state["rot"]).x
            if np.degrees(np.linalg.norm(r)) > max_deg:
                state["log"].append({**rec, "stop": True}); print("[drift]", json.dumps(state["log"][-1]), flush=True)
                raise RuntimeError(f"top camera moved by more than {max_deg} deg since calibration: recalibrate")
            state["rot"] = r; cam["xmat"] = (R0 @ Rotation.from_rotvec([r[0], r[1], 0.0]).as_matrix()).ravel().tolist()
            rec["corrected_deg"] = round(float(np.degrees(np.linalg.norm(r))), 2)
        state["log"].append(rec); print("[drift]", json.dumps(rec), flush=True)

    move_xy = arm.move_xy

    def move_xy_checked(xy):
        fb = move_xy(xy)
        try:
            check("after move")
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001  (a missing frame must not end the run)
            print("[drift] check skipped:", repr(exc)[:120], flush=True)
        return fb
    arm.move_xy = move_xy_checked
    check("start")
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--backend", choices=["socket", "sim"], default="sim")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sim-task", default="place_in"); ap.add_argument("--sim-seed", type=int, default=60000)
    ap.add_argument("--base", default=None, help="model server URL (RunPod proxy)")
    ap.add_argument("--oracle", action="store_true", help="sim dry runs only: decisions and pointing from the simulator (no model)")
    ap.add_argument("--camera-json", default=None, help="the calibrated top camera (real/selfcal_real.py)")
    ap.add_argument("--cams", default="top=2,side=0,wrist=1"); ap.add_argument("--undistort", default="", help="view=file.npz,...")
    ap.add_argument("--cam-server", default="http://127.0.0.1:8766", help="real/cam_server.py URL ('' = open the cameras here)")
    ap.add_argument("--rotate", default="", help="view=deg clockwise,... (e.g. top=90)")
    ap.add_argument("--mirror", default="", help="views to mirror left-right, e.g. side (a side camera on the robot's left)")
    ap.add_argument("--crop", default="", help="view=x0:y0:x1:y1,... square crop box after rotation (default: centre square); top: the calibrated crop")
    ap.add_argument("--view-crop", default="", help="top=x0:y0:x1:y1: a wider crop for what the agent sees; pixels map back to the calibrated crop (same phone position)")
    ap.add_argument("--expect", default="", help="view=WxH raw size the calibration used (frames of another size are refused)")
    ap.add_argument("--carry-z", type=float, default=None, help="lift height (m); above ~0.09 the gripper tilts (tall containers)")
    ap.add_argument("--grasp-to-table", action="store_true", help="grasp depth from contact: lower until the table stops the fingers")
    ap.add_argument("--grasp-clear", type=float, default=None, help="grasp depth from the calibrated heights: fixed-jaw tip this far above the table (m)")
    ap.add_argument("--place-clear", type=float, default=None, help="release height from the calibrated heights: fixed-jaw tip this far above the table (m)")
    ap.add_argument("--jpeg", action="store_true", help="send the model JPEG images (smaller requests through the proxy)")
    ap.add_argument("--record", action=argparse.BooleanOptionalAction, default=True, help="episode video + ledger line (real arm)")
    ap.add_argument("--start-from-rest", action="store_true", help="fold to the owner's parked pose, then unfold, before the agent starts")
    ap.add_argument("--ledger", default="real/runs/episodes.jsonl", help="append-only log of every real episode")
    ap.add_argument("--marker-hsv", default="124,12,45,90", help="fingertip marker colour (PIL HSV H,dH,minS,minV)")
    ap.add_argument("--ref-frame", default="", help="top view (calibrated crop) from the calibration: the static background is matched at start; a moved phone is corrected (or the run stops if the view no longer matches)")
    ap.add_argument("--static-mask", default="", help=".npy mask of marker-coloured background spots (hand_fit.py)")
    ap.add_argument("--drift-guard", action=argparse.BooleanOptionalAction, default=True,
                    help="check the top camera against the fingertip tape after every move; correct small drift, stop on large")
    ap.add_argument("--object-z", type=float, default=None,
                    help="height (m) of a pointed object's centre; the agent assumes the simulator's 0.022 (real 7 cm ball: 0.036)")
    ap.add_argument("--table-z", type=float, default=0.0); ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--max-steps", type=int, default=30); ap.add_argument("--fast", action="store_true", help="sim: sim-speed caps")
    ap.add_argument("--profile", choices=list(PROFILES), default="commissioning", help="speed preset (arm_server must match)")
    ap.add_argument("--wrist-refine", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    flags = [f for f in Path("run8/pods/flags_V4f.txt").read_text().split()]
    if "--image" in flags:
        i = flags.index("--image"); del flags[i:i + 2]
    if not a.wrist_refine and "--wrist-refine" in flags:
        flags.remove("--wrist-refine")
    argv = ["_"] + flags + ["--image", "real", "--base", a.base or "http://localhost:18000"]
    if a.oracle:
        if a.backend != "sim":
            raise SystemExit("--oracle needs the simulator")
        argv = [x for x in argv if x != "--enumerate"]  # no model server in oracle dry runs
        if "--locator-model" in argv:
            i = argv.index("--locator-model"); del argv[i:i + 2]
        argv += ["--planner-mode", "oracle", "--check-mode", "oracle", "--parse-mode", "oracle", "--sim-pointer", "0.006"]
    agent = build_agent(make_parser().parse_args(argv))
    agent.cfg.update(record_gold=bool(a.oracle), record_facts=False, record_snapshots=False, selfcal=False)

    if a.jpeg:
        os.environ["VLM_IMAGE_FORMAT"] = "jpeg"
    cfg = ArmConfig(confirm=a.confirm, table_z=a.table_z, **PROFILES[a.profile], **({"carry_z": a.carry_z} if a.carry_z else {}), grasp_to_table=a.grasp_to_table, grasp_clear_m=a.grasp_clear, place_clear_m=a.place_clear)
    if a.fast:
        cfg = ArmConfig(speed_deg_s=60, near_speed_deg_s=18, gripper_pct_s=72, descend_mm_s=48, grasp_min_close_s=.7,
                        confirm=a.confirm, table_z=a.table_z)
    if a.backend == "sim":
        from real.backend import SimBackend
        from run7.env.world import sample_scene
        scene = sample_scene(a.sim_seed, a.sim_task)
        if a.command:
            scene["instruction"] = a.command
        backend = SimBackend(scene); arm = RealArm(backend, cfg, scene=scene, mirror=True)
    else:
        from real.backend import SocketBackend
        from real.cameras import RealCameras
        cams = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in a.cams.split(","))
        und = dict(kv.split("=") for kv in a.undistort.split(",") if kv)
        scene = {"seed": 0, "task": "real", "instruction": a.command, "objects": [], "containers": [], "subgoals": [],
                 "fault": None, "followup": None}
        rot = {k: int(v) for k, v in (kv.split("=") for kv in a.rotate.split(",") if kv)}
        crops = {k: tuple(int(x) for x in v.split(":")) for k, v in (kv.split("=") for kv in a.crop.split(",") if kv)}
        cal_crop = crops.get("top")
        if a.view_crop:
            crops["top"] = tuple(int(x) for x in a.view_crop.split("=")[1].split(":"))
        v2c, c2v = crop_map(crops.get("top"), cal_crop), crop_map(cal_crop, crops.get("top"))
        expect = {k: tuple(int(x) for x in v.split("x")) for k, v in (kv.split("=") for kv in a.expect.split(",") if kv)}
        rc = RealCameras(cams, und, crops=crops, server=a.cam_server or None, rot=rot, mirror=tuple(v for v in a.mirror.split(",") if v), expect=expect)   # cameras first: fail before the arm connects
        backend = SocketBackend(port=a.port); arm = RealArm(backend, cfg, scene=scene, cameras=rc)
    if a.camera_json:
        cam = json.loads(Path(a.camera_json).read_text()); cam = cam.get("camera", cam)
        agent._cam_belief = lambda env: dict(cam)
        if a.backend == "socket" and a.drift_guard:
            install_drift_guard(arm, cam, a.table_z, hsv=tuple(int(v) for v in a.marker_hsv.split(",")),
                                static_mask=a.static_mask if not a.view_crop else "", v2c=v2c, c2v=c2v)

    Hinv = None
    if a.backend == "socket" and a.ref_frame:   # did the phone move since the calibration? (static background, not the arm)
        import cv2
        H, info = background_shift(a.ref_frame, a.cam_server, cams["top"], int(dict(kv.split("=") for kv in a.rotate.split(",") if kv).get("top", 0)), cal_crop)
        print("[camera]", json.dumps(info), flush=True)
        if H is None:   # weak match: usually the arm / objects cover the textured table (2026-09-27: two false alarms while
            # the phone had not moved; the white walls carry few features): warn only, never stop or correct on it
            print("[camera] WARNING: reference match too weak to check the phone position (scene changed?)", flush=True)
        elif info["inliers"] >= 40 and (np.hypot(*H[:, 2]) > 4.0 or abs(info["rot_deg"]) > .6):
            Hinv = cv2.invertAffineTransform(H); print("[camera] correcting a moved phone", flush=True)
    import run5.agent.agent as A5
    _p2t = A.pixel_to_table
    _v2c = v2c if a.backend == "socket" else (lambda u, v: (u, v))
    obj_z = a.object_z

    def p2t(cam, u, v, z):   # the agent's pixels (view crop) -> calibrated crop (-> reference view); objects at their real centre height
        uc, vc = _v2c(u, v)
        if Hinv is not None:
            uc, vc = Hinv[:, :2] @ [uc, vc] + Hinv[:, 2]
        return _p2t(cam, uc, vc, obj_z if (obj_z and abs(z - .022) < 1e-9) else z)
    A.pixel_to_table = p2t; A5.pixel_to_table = p2t
    if a.backend == "socket":
        install_real_reach(arm, agent)
    A.World = lambda _scene, sensing="motor": arm          # the agent's World is the arm
    A.flat_views = lambda env, views: env.frames(views)   # its images are the cameras
    A.max_steps = lambda _scene: a.max_steps
    out = Path(a.out or f"real/runs/{time.strftime('%Y%m%d-%H%M%S')}"); out.mkdir(parents=True, exist_ok=True)
    print(f"[run] command: {scene['instruction']!r}; backend {a.backend}; logs -> {out}", flush=True)
    t0 = time.time()
    rec = None
    if a.backend == "socket" and a.record:
        crop = crops.get("top")   # the agent's (possibly wider) view
        rec = Recorder(out, a.cam_server, cams, int(dict(kv.split("=") for kv in a.rotate.split(",") if kv).get("top", 0)), crop, a.command)
    s = {"termination": "crashed", "steps": None, "model_calls": None, "model_s": 0.0, "success": None}
    try:
        if a.backend == "socket" and a.start_from_rest:   # every episode from the same base pose (recorded in the video)
            arm._sync()
            if np.max(np.abs(np.asarray(arm.q[:5]) - np.asarray(arm.REST_POSE))) > 8.0:   # not parked: up and in, then fold
                arm.retreat()
            arm.fold(); print("[run] at the base pose", flush=True); arm.unfold()
        try:
            s = agent.run(scene, out)
        finally:
            if a.backend == "socket" and a.start_from_rest:   # end at the base too, never stretched over the table (2026-09-27)
                try:
                    try:
                        arm.retreat()
                    except Exception as exc:  # noqa: BLE001  (best effort: fold anyway, the fold is guarded)
                        print("[run] retreat skipped:", repr(exc)[:160], flush=True)
                    if arm.observe()["holding"] is None:
                        arm.fold(); print("[run] back at the base pose", flush=True)
                    else:
                        print("[run] still holding an object: raised, not folded (real/put_down.py sets it down)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("[run] could not return to the base pose:", repr(exc)[:200], flush=True)
    finally:
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass
        if rec is not None:
            rec.close()
        if a.backend == "socket" and a.record:   # every attempt is logged, whatever happened (verdicts filled in afterwards)
            with open(a.ledger, "a") as f:
                f.write(json.dumps({"episode": out.name, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                                    "command": a.command, "profile": a.profile, "termination": s["termination"], "steps": s["steps"],
                                    "model_s": round(s["model_s"] or 0, 1), "arm_motion_s": round(arm.motion_s, 1),
                                    "wall_s": round(time.time() - t0, 1), "video": str(out / "episode.mp4"),
                                    "verdict_camera": None, "verdict_owner": None}) + "\n")
    wall = time.time() - t0
    info = {"termination": s["termination"], "steps": s["steps"], "model_calls": s["model_calls"], "model_s": round(s["model_s"], 1),
            "arm_motion_s": round(arm.motion_s, 1), "wall_s": round(wall, 1),
            "sim_success": s["success"] if a.backend == "sim" else "judge by eye"}
    (out / "run_info.json").write_text(json.dumps({**info, "arm_log": cfg.log}, indent=1, default=str))
    print("[run]", json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
