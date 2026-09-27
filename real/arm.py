"""RealArm: the Run 8 agent's arm interface (the parts of run7.env.world.World it uses) on a real or simulated backend.

The agent keeps deciding exactly as in simulation (readouts, checks, pointing, self-calibration maths); this class turns its
skill calls into bounded joint commands:
  move_xy        straight line at the current height (IK waypoints every 1.5 cm, the wrist turned to a feasible yaw)
  grasp          open if needed, lower to grasp height (stops early on touch), close, judge holding from the jaw reading
  lift           straight up to carrying height
  place          lower until the touch rule fires (real rule: lag rise > touch_mm after >= 3 mm of descent), open
  rotate / open_gripper / park / observe / ee / gripper_yaw / jaw_reading / sensed_holding / solve / feasible_yaw / _move
A kinematic twin (the same MuJoCo SO-101 model as the simulator, joints mirrored from the readings) gives FK, IK and the
camera/marker geometry the agent reads through env.model / env.data.
Heights: the agent works in table coordinates (table at z = 0, as in sim); table_z is the table's height in the robot base
frame (measured on arm day). Safety: speed caps, workspace clip, joint bounds (in the driver), stall stop, the jev_arm
camera-wall exclusion, optional confirmation before every skill. The stop key lives in real/arm_server.py.
"""
from __future__ import annotations

import copy
import json
import os
import math
from pathlib import Path
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

import run8.env.world  # noqa: F401
from real.joints import GRIP_CLOSED_PCT, GRIP_OPEN_PCT, HOLD_MIN_PCT, body_model_rad, body_real_deg, grip_pct_to_rad, real_to_sim, sim_to_real
from run7.env.world import CARRY_Z, GRASP_Z, WORK_MAX, WORK_MIN, World, wrap

ROBOT_ONLY = {"seed": 0, "task": "real", "objects": [], "containers": [], "subgoals": [], "instruction": "", "fault": None}


def real_body_bounds(twin_config: str = os.environ.get("SO101_CALIBRATION_JSON", "")):
    """The follower's calibrated body-joint range in degrees (the driver's degree mode: 0 deg at mid-range), or None.
    2026-09-26: shoulder_pan is only +-55 deg; a calibration pose beyond it was clipped by the driver and timed out."""
    try:
        cal = json.loads(Path(twin_config).read_text())["calibration"]
    except (OSError, KeyError, ValueError):
        return None
    out = []
    for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]; mid = (lo + hi) / 2
        out.append(((lo - mid) * 360 / 4095, (hi - mid) * 360 / 4095))
    out = np.asarray(out)
    out[0, 0] = max(out[0, 0], PAN_MIN_DEG); out[0, 1] = min(out[0, 1], PAN_MAX_DEG); out[3, 1] = min(out[3, 1], WRIST_MAX_DEG)
    out[4] = np.clip(out[4], -ROLL_MAX_DEG, ROLL_MAX_DEG)
    return out


PAN_MAX_DEG = 53.0    # right side: the owner's teleop reached +54 without trouble
PAN_MIN_DEG = -40.0   # left-wall side: the gripper camera scraped the wall at pan ~-50 (2026-09-27, calibration and hand touches);
# (the old note below is kept for the record)
_OLD_PAN_MIN_DEG = -53.0   # 2026-09-26 rig: walls on both sides; the owner's teleop reached pan -54.2..54.1 (twin/captures/hardware-
                      # 20260906-142415, 21,661 poses): the recorded range minus 1 deg (planning keeps 2 deg more inside)
REAL_WORK_MAX = np.array([0.46, WORK_MAX[1], 0.20])   # the simulator's box stops at x 0.36 / z 0.10; the owner's arm reached
                                                     # 0.456 m (hand session 2026-09-27) and carries go to 0.15 (tilting IK)
ROLL_MAX_DEG = 52.0   # 2026-09-27: at wrist_roll +70 the gripper camera mount rubbed on the upper arm and its USB dropped
                      # (planning keeps 2 deg inside: |roll| <= 50)
WRIST_MAX_DEG = 96.0  # planning keeps 2 deg inside this; 2026-09-26: a pose needing wrist_flex 95 deg (the model's limit) never settled on the real arm


@dataclass
class ArmConfig:
    hz: float = 30.0
    speed_deg_s: float = 10.0          # jev_arm body speed cap (config speed_deg_s)
    near_speed_deg_s: float = 3.0      # jev_arm near-target speed
    descend_mm_s: float = 8.0          # vertical approach speed for grasp / place descents
    gripper_pct_s: float = 12.0        # jev_arm gripper speed
    tol_deg: float = 1.2               # a waypoint counts as reached within this joint error
    path_tol_deg: float = 6.0          # ... an intermediate path point within this one (2026-09-26: the real elbow sagged
                                       # 3.2 deg behind a path point in the folded pose and a 3 deg tolerance read it as a stall)
    touch_mm: float = 5.0              # REAL_SENSING: lag rise > 5 mm ...
    touch_after_mm: float = 3.0        # ... after at least 3 mm of real descent
    grasp_stop_mm: float = 15.0        # grasp descents go to grasp height (jaws around the object's middle, as in sim);
                                       # the lag rule only stops them if the gripper is pressing hard on something
    grasp_min_close_s: float = 4.0     # jev_arm grasp_min_close_s
    grasp_to_table: bool = False       # grasp depth from contact: lower until the table stops the fingers, back off, close
    grasp_backoff_m: float = .005
    fast_travel: bool = False          # travel moves finish at travel_tol_deg (the grasp/place descents steer onto the intended
                                       # spot and settle fully at the bottom; 2026-09-27 fastest mode). Not settle=False: without
                                       # the gravity integration the stretched arm never got within 3 deg and every carry stalled
    travel_tol_deg: float = 2.5
    grasp_open_pct: float | None = None   # open only this far before a grasp and to release (owner, 2026-09-27: at 95 % the
                                          # ball ends up at the jaws' tips and sticks there; 70 % holds it deeper)
    drop_release: bool = False         # release = open at the carry height over the target (owner: "just drop them"), only
                                       # when the carry arrived (a stuck carry dropped a ball off the table at 18:22)
    place_clear_m: float | None = None # calibrated release: lower the held object until the fixed jaw's tip is this far above
                                       # the table, then open (no slow touch detection)
    grasp_clear_m: float | None = None # calibrated heights (2026-09-27 17:30): lower until the fixed jaw's tip is this far above
                                       # the table, never touching it (pushing on the table slid the fingertips 1-3 cm outward)
    table_z: float = 0.0               # table height in the robot base frame (m); measure on arm day
    step_m: float = .015               # Cartesian waypoint spacing
    stall_s: float = 2.5               # no joint progress for this long while far from the goal -> stop
    move_timeout_s: float = 30.0
    carry_z: float = CARRY_Z           # lift height; above the top-down reach (~9 cm) the gripper tilts (2026-09-26: the
    max_tilt_deg: float = 35.0         # real container is ~10 cm tall, so carries go to 15 cm with the fingers <= 35 deg off down)
    low_tilt_deg: float = 25.0         # below 6 cm (grasps): a tilt reaches further than straight down (2026-09-27: a ball at
                                       # 35 cm needed 3 deg, one in the container's far corner at 39 cm 21 deg; a sphere grasps
                                       # the same tilted)
    wall_pan_below_deg: float | None = -25.0   # jev_arm camera-wall exclusion: pan below this ...
    wall_min_radius_m: float | None = .18      # ... and the tool closer to the base than this -> refused
    confirm: bool = False              # ask the operator before every skill
    log: list = field(default_factory=list)


# Speed presets (simulated put-in arm time, 7 cube/cylinder scenes, all succeeded: 42 / 17.5 / 14 s). Raise them step by step
# on the real arm: "commissioning" for the first supervised moves, then "normal", then "fast" once moves look clean.
# The arm server must allow the same speeds: arm_server.py --body-deg-s / --grip-pct-s (it sets the servos' velocity too).
PROFILES = {
    "commissioning": dict(),  # jev_arm's caps: 10 deg/s, 3 deg/s near targets, gripper 12 %/s, 8 mm/s descents, 4 s squeeze
    "normal": dict(speed_deg_s=30, near_speed_deg_s=6, gripper_pct_s=50, descend_mm_s=20, grasp_min_close_s=1.0),
    "fast": dict(speed_deg_s=60, near_speed_deg_s=10, gripper_pct_s=50, descend_mm_s=20, grasp_min_close_s=1.0),
    "fastest": dict(speed_deg_s=60, near_speed_deg_s=30, gripper_pct_s=50, descend_mm_s=100, grasp_min_close_s=0.4, fast_travel=True,
                    grasp_open_pct=70.0, drop_release=True),
}
SERVER_FLAGS = {"commissioning": "--body-deg-s 10 --grip-pct-s 12", "normal": "--body-deg-s 30 --grip-pct-s 50",
                "fast": "--body-deg-s 60 --grip-pct-s 50", "fastest": "--body-deg-s 60 --grip-pct-s 50"}


class MotionError(RuntimeError):
    pass


class RealArm:
    sensing = "motor"

    def __init__(self, backend, cfg: ArmConfig | None = None, scene: dict | None = None, mirror: bool = False, cameras=None):
        self.b, self.cfg = backend, cfg or ArmConfig()
        self.bounds = None if mirror else real_body_bounds()
        self.mirror = mirror  # dry run: the twin mirrors the whole simulated scene (objects too) for oracle checks
        self.scene = copy.deepcopy(scene) if scene else dict(ROBOT_ONLY)
        self.kin = World(self.scene if mirror else ROBOT_ONLY); self.kin.reset()
        self.model, self.data = self.kin.model, self.kin.data
        self.site, self.gripper_body = self.kin.site, self.kin.gripper_body
        self.cameras = cameras
        self.hold_hint, self.touch_log, self.step_count, self.last_action, self.last_feedback = None, [], 0, None, {}
        self.fault = {"type": None, "fired": False, "event": None}
        self.last_cmd, self.armed, self.grip_closed_cmd = None, False, False
        self.motion_s = 0.0  # arm time spent moving (dry runs: simulated seconds at the real speed caps)
        self._sync()
        self.yaw = self.gripper_yaw()
        self.grip_target = 1.0 if self.q[5] > 50 else -.12

    # ------------------------------------------------------------------ state
    def _sync(self) -> np.ndarray:
        self.q = np.asarray(self.b.read(), float)
        if self.mirror:
            self.kin.data.qpos[:] = self.b.full_qpos()
        else:
            self.kin.data.qpos[:6] = real_to_sim(self.q)
        mujoco.mj_forward(self.kin.model, self.kin.data)
        return self.q

    def _tip(self, q_real) -> np.ndarray:
        d = self.kin._scratch; d.qpos[:] = self.kin.data.qpos; d.qpos[:6] = real_to_sim(q_real)
        mujoco.mj_kinematics(self.kin.model, d)
        return d.site_xpos[self.site].copy()

    ARM_LINKS = ("upper_arm", "lower_arm", "wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")

    def _low_points(self, q_real) -> dict:
        """Lowest point (m above the table) of each moving link's visual mesh at joints q_real (real units)."""
        m = self.kin.model
        if not hasattr(self, "_link_meshes"):
            self._link_meshes = []
            for g in range(m.ngeom):
                name = m.body(m.geom_bodyid[g]).name
                if name in self.ARM_LINKS and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
                    a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                    self._link_meshes.append((g, name, m.mesh_vert[a:a + n].copy()))
        d = self.kin._scratch; d.qpos[:] = self.kin.data.qpos; d.qpos[:6] = real_to_sim(q_real)
        mujoco.mj_kinematics(m, d)
        out = {}
        for g, name, V in self._link_meshes:
            z = float((d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()) - self.cfg.table_z
            out[name] = min(out.get(name, z), z)
        return out

    def unfold(self, target=None, steps: int = 40) -> dict:
        """First move of a session, from the folded rest pose (the fingertip then hangs 1-2 cm above the table): joints
        interpolated to a top-down pose high above the work area. Refused before any motion unless every moving link stays
        at or above the lower of its start and end heights (3 mm slack) and outside the camera-wall exclusion at every one of
        the steps. Relative, not absolute: on 2026-09-26 the model put the real rest pose's gripper 1.8 cm below the table
        while it hung 1-2 cm above it (the real table is lower relative to the base than the model's)."""
        self._sync()
        if target is None:   # ready at the carry height (2026-09-27 18:32: from the sim's 8.5 cm ready pose the first move
            # swept through a 10 cm container and pushed the balls); nearer spots cannot reach 18 cm within the tilt limit
            target = (.25, 0.0, self.cfg.carry_z)
            q_t, e_t, t_t = self._ik_free(list(target), self.yaw, self.kin.data.qpos[:5])
            if e_t > 3.0 or t_t > self.tilt_limit(target[2]):
                target = (.20, 0.0, CARRY_Z)
        y = self.feasible_yaw(list(target))
        if y is None:   # any gripper yaw, nearest the current one (the wrist-roll limit rules out some, 2026-09-27)
            cands = sorted(np.radians(np.arange(-180, 180, 15)), key=lambda c: abs(wrap(c - self.yaw, 2 * math.pi)))
            y = next((float(c) for c in cands if self.solve(list(target), float(c))["success"]), None)
        q0 = np.asarray(self.q, float)
        if y is not None:
            q1 = np.r_[body_real_deg(self.solve(target, y)["joint_pos"][:5]), q0[5]]
        else:   # 2026-09-27 17:20 full calibration: the corrected wrist/elbow cannot point exactly straight down there; the
            # tilting solution (fingers a few degrees off vertical, within tilt_limit) as every other move uses
            qf, err_mm, tilt = self._ik_free(list(target), self.yaw, self.kin.data.qpos[:5])
            if err_mm > 3.0 or tilt > self.tilt_limit(target[2]):
                raise MotionError(f"unfold refused: no solution at the ready pose (tilting: {err_mm:.1f} mm, {tilt:.0f} deg)")
            q1 = np.r_[body_real_deg(qf), q0[5]]
        lo0, lo1 = self._low_points(q0), self._low_points(q1); low = dict(lo0)
        wps = []
        for k in range(1, steps + 1):
            qk = q0 + (q1 - q0) * k / steps
            for name, z in self._low_points(qk).items():
                floor = min(lo0[name], lo1[name]) - .003
                if z < floor:
                    raise MotionError(f"unfold refused: the {name} would dip to {z * 100:.1f} cm at step {k}/{steps}")
                low[name] = min(low[name], z)
            self._check_wall(qk)
            wps.append(np.r_[qk[:5], np.nan])
        self._confirm(f"unfold from rest to the ready pose ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) m, "
                      f"lowest link point on the way {min(low.values()) * 100:.1f} cm")
        self._track(wps)
        if y is not None:
            self.yaw = y
        return {"ok": True, "lowest_cm": {k: round(v * 100, 1) for k, v in low.items()}}

    REST_POSE = [3.956, -102.0, 95.604, 56.923, -2.418]   # the owner's parked pose (twin config, "owner visually matched"),
    # shoulder -102 instead of -104.3: not against the calibrated stop (-104.8); the arm rested at -102.2 on 2026-09-27

    def fold(self, steps: int = 40) -> dict:
        """Return to the owner's parked (base) pose by joint interpolation, refused before any motion unless every link
        stays at or above the lower of its start and end heights (3 mm slack) at every step (as unfold)."""
        self._sync()
        q0 = np.asarray(self.q, float); q1 = np.r_[self.REST_POSE, q0[5]]
        lo0, lo1 = self._low_points(q0), self._low_points(q1); wps = []
        for k in range(1, steps + 1):
            qk = q0 + (q1 - q0) * k / steps
            for name, z in self._low_points(qk).items():
                if z < min(lo0[name], lo1[name]) - .003:
                    raise MotionError(f"fold refused: the {name} would dip to {z * 100:.1f} cm at step {k}/{steps}")
            wps.append(np.r_[qk[:5], np.nan])
        self._track(wps, settle=False)
        self._sync(); self.yaw = self.gripper_yaw()
        return {"ok": True}

    def ee(self) -> np.ndarray:
        self._sync(); p = self.kin.data.site_xpos[self.site].copy(); p[2] -= self.cfg.table_z
        return p

    def gripper_yaw(self) -> float:
        R = self.kin.data.xmat[self.gripper_body].reshape(3, 3)
        return float(math.atan2(R[1, 0], R[0, 0]))

    def jaw_reading(self) -> float:
        self._sync(); return float(grip_pct_to_rad(self.q[5]))

    def sensed_holding(self) -> bool:
        self._sync(); return bool(self.grip_closed_cmd and self.q[5] >= HOLD_MIN_PCT)

    def observe(self) -> dict:
        holding = "sensed" if self.sensed_holding() else None
        state = "holding" if holding else ("open" if not self.grip_closed_cmd else "closed")
        return {"ee_pos": self.ee().tolist(), "gripper_yaw": self.gripper_yaw(), "gripper_state": state, "holding": holding,
                "goal": self.scene.get("instruction", ""), "joint_pos": real_to_sim(self.q).tolist()}

    # ------------------------------------------------------------------ kinematics (table coordinates in, base frame inside)
    def solve(self, target, yaw: float) -> dict:
        t = np.asarray(target, float).copy(); t[2] += self.cfg.table_z
        sol = self.kin.solve(t, yaw)
        sol["achieved_pos"] = (np.asarray(sol["achieved_pos"]) - [0, 0, self.cfg.table_z]).tolist()
        if self.bounds is not None and sol["success"]:   # the real arm's calibrated range (the driver clips beyond it)
            q = body_real_deg(sol["joint_pos"][:5])
            if np.any(q < self.bounds[:, 0] + 2.0) or np.any(q > self.bounds[:, 1] - 2.0):
                sol["success"] = False; sol["note"] = "outside the real joint range"
        return sol

    def ik(self, target):
        return self.solve(target, self.yaw)

    def symmetry(self) -> float:
        shape = self.hold_hint if self.sensed_holding() else None
        if shape:
            return {"cube": math.pi / 2, "ball": math.pi / 6, "cylinder": math.pi / 6}.get(shape, math.pi)
        return math.pi

    def feasible_yaw(self, target) -> float | None:
        per = self.symmetry()
        cands = sorted({round(wrap(self.yaw + j * per, 2 * math.pi), 9) for j in range(-6, 7)}, key=lambda y: abs(wrap(y - self.yaw, 2 * math.pi)))
        for y in cands:
            if self.solve(target, y)["success"]:
                return y
        return None

    def _path(self, a, b, yaw: float, seed=None) -> list[np.ndarray]:
        """Joint targets (real units) along the straight segment a -> b (table coordinates), each IK seeded by the last.
        Points outside the top-down reach use the tilting IK (fingers as close to down as possible, at most tilt_limit(z))."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / self.cfg.step_m)))
        saved = self.kin.data.qpos.copy(); out = []; prev = saved[:5].copy()
        try:
            if seed is not None:   # continue a path planned in pieces from its last joints, not from the arm's pose
                prev = np.asarray(seed, float).copy(); self.kin.data.qpos[:5] = prev
            for k in range(1, n + 1):
                pt = a + (b - a) * k / n
                sol = self.solve(pt, yaw)
                if sol["success"]:
                    q = np.asarray(sol["joint_pos"][:5], float)
                else:
                    q, err_mm, tilt = self._ik_free(pt, yaw, prev)
                    if err_mm > 3.0 or tilt > self.tilt_limit(pt[2]):
                        raise MotionError(f"no IK solution at {np.round(pt, 3).tolist()} (tilting: error {err_mm:.1f} mm, tilt {tilt:.0f} deg)")
                if np.max(np.abs(np.degrees(q - prev))) > 25.0:
                    raise MotionError(f"joint jump of {np.max(np.abs(np.degrees(q - prev))):.0f} deg at {np.round(pt, 3).tolist()}")
                self.kin.data.qpos[:5] = q; prev = q
                out.append(np.r_[body_real_deg(q), np.nan])
        finally:
            self.kin.data.qpos[:] = saved; mujoco.mj_forward(self.kin.model, self.kin.data)
        return out

    def _site_z_for_clearance(self, p, clear: float) -> float:
        """Tool-site height above (x, y) of p at which the fixed jaw's tip is `clear` above the table, solved at that bottom pose
        (the finger hangs differently at the tilted carry pose, 2026-09-27: using the start pose's offset stopped 1.5-2 cm high)."""
        z = float(p[2]) - (self._low_points(self.q)["gripper"] - clear)
        for _ in range(5):
            pt = [float(p[0]), float(p[1]), z]; sol = self.solve(pt, self.yaw)
            if sol["success"]:
                q = np.asarray(sol["joint_pos"][:5], float)
            else:
                q, err, tilt = self._ik_free(pt, self.yaw, self.kin.data.qpos[:5])
                if err > 3.0 or tilt > self.tilt_limit(z):
                    break
            dz = self._low_points(np.r_[body_real_deg(q), self.q[5]])["gripper"] - clear
            z -= dz
            if abs(dz) < .0005:
                break
        return z

    def tilt_limit(self, z: float) -> float:
        """How far the fingers may tilt from pointing down at height z (table coordinates): carries above 6 cm up to
        max_tilt_deg, grasps and placements below it up to low_tilt_deg."""
        return self.cfg.max_tilt_deg if z > .06 else self.cfg.low_tilt_deg

    def _ik_free(self, target, yaw: float, seed) -> tuple[np.ndarray, float, float]:
        """Position IK with the gripper allowed to tilt, for carries above the top-down reach: joints (sim radians) that put
        the tool at `target` (table coordinates) with the fingers as close to pointing down, and the yaw as close to `yaw`,
        as the reach allows. Returns (q, position error mm, tilt from vertical deg)."""
        from scipy.optimize import least_squares
        m, d = self.kin.model, self.kin._scratch
        if self.bounds is not None:
            lo, hi = body_model_rad(self.bounds[:, 0] + 2.0), body_model_rad(self.bounds[:, 1] - 2.0)
        else:
            lo, hi = m.jnt_range[:5, 0] + .03, m.jnt_range[:5, 1] - .03
        t = np.asarray(target, float) + [0, 0, self.cfg.table_z]

        def fk(q):
            d.qpos[:] = self.kin.data.qpos; d.qpos[:5] = q; mujoco.mj_kinematics(m, d)
            return d.site_xpos[self.site].copy(), d.xmat[self.gripper_body].reshape(3, 3).copy()

        def res(q):
            p, R = fk(q); down = -R[:, 2]   # the fingers point along the gripper body's -z
            dy = wrap(math.atan2(R[1, 0], R[0, 0]) - yaw, 2 * math.pi)
            return np.r_[(p - t) * 100.0, .3 * (down - [0, 0, -1]), .05 * math.sin(dy), .05 * (1 - math.cos(dy))]

        r = least_squares(res, np.clip(np.asarray(seed, float), lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
        p, R = fk(r.x)
        return r.x, float(np.linalg.norm(p - t) * 1000), float(math.degrees(math.acos(float(np.clip(-(-R[:, 2])[2], -1, 1)))))

    WALLS = Path(__file__).parent / "cal" / "walls.json"

    def _wall_clearance(self, q_real) -> tuple[float, str]:
        """Smallest (n.p - d + margin) over the corner walls and every moving link's mesh points (< 0: past a wall)."""
        if not hasattr(self, "_walls"):
            try:
                w = json.loads(self.WALLS.read_text()); self._walls = ([(np.asarray(p["n"], float), float(p["d"])) for p in w["planes"]], float(w["margin_m"]))
            except (OSError, KeyError, ValueError):
                self._walls = ([], 0.0)
        planes, margin = self._walls
        if not planes:
            return 1.0, ""
        self._low_points(q_real)   # builds the link mesh list and runs kinematics on the scratch data
        d_ = self.kin._scratch; best = (1.0, "")
        for g, name, V in self._link_meshes:
            P = (d_.geom_xpos[g] + V[::15] @ d_.geom_xmat[g].reshape(3, 3).T)[:, :2]
            for n, dd in planes:
                c = float((P @ n).min() - dd + margin)
                if c < best[0]:
                    best = (c, name)
        return best

    def _check_wall(self, q_real):
        c, name = self._wall_clearance(q_real)
        if c < 0:   # 2026-09-27: the gripper camera mount scraped the wall during calibration
            raise MotionError(f"wall: the {name} would go {-c * 100:.1f} cm past the corner wall limit")
        c = self.cfg
        if c.wall_pan_below_deg is None or q_real[0] >= c.wall_pan_below_deg:
            return
        if np.linalg.norm(self._tip(q_real)[:2]) < c.wall_min_radius_m:
            raise MotionError("camera-wall exclusion: that pose brings the tool too close to the base at this pan angle")

    # ------------------------------------------------------------------ motion core
    def _hold(self):
        """Make sure a hold is active before every motion: the arm server's watchdog ends holds while the agent is thinking
        between skills. A new hold primes the current position as the goal (nothing jumps), which for the gripper means no
        squeeze: when holding an object, the close command is restored at once so it cannot slip out."""
        q, new = self.b.begin_hold()
        if new or self.last_cmd is None:
            self.last_cmd = np.asarray(q, float)
            if self.grip_closed_cmd:
                self.last_cmd[5] = GRIP_CLOSED_PCT
                self.last_cmd = np.asarray(self.b.send(self.last_cmd), float)
        self.armed = True

    def _confirm(self, what: str):
        self.cfg.log.append({"t": time.time(), "skill": what, "motion_s": self.motion_s})
        if self.cfg.confirm:
            ans = input(f"\n[arm] next: {what}. Enter = do it, q = stop: ").strip().lower()
            if ans == "q":
                raise KeyboardInterrupt("operator stop")

    def _track(self, waypoints: list[np.ndarray], grip_pct: float | None = None, speed: float | None = None, touch=None,
               settle: bool = True, lead: bool = True, tol_deg: float | None = None) -> dict:
        """Follow joint waypoints (real units; nan gripper = keep) at a capped joint speed. touch(q, cmd) -> True stops the
        motion early. Returns what happened.
        The command glides along the joint-space polyline through all waypoints at one steady speed (max-norm), ramping
        up over 0.3 s and slowing near the end; it pauses only while the arm lags more than path_tol_deg behind. (Until
        2026-09-26 it stopped at every 1.5 cm waypoint until the arm caught up: the real arm moved in small jerks.) At the
        end, the static (gravity) error is integrated out as in jev_arm, never on pan."""
        self._hold()
        c = self.cfg; dt = 1.0 / c.hz; speed = speed or c.speed_deg_s
        cmd = np.asarray(self.last_cmd, float).copy()
        # holding: keep driving the jaw to the closed command (squeeze), never just "keep where it is" (2026-09-26: after a
        # hold renewal the jaw goal sat at the ball's width with no force and the ball slipped out during the lift)
        g_goal = (GRIP_CLOSED_PCT if self.grip_closed_cmd else cmd[5]) if grip_pct is None else float(grip_pct)
        goals = []
        for wp in waypoints:
            g = np.asarray(wp, float).copy(); g[5] = g_goal; self._check_wall(g); goals.append(g)
        goal = goals[-1]
        pts = [cmd[:5].copy()] + [g[:5] for g in goals]
        seg = np.asarray([float(np.max(np.abs(b - a))) for a, b in zip(pts[:-1], pts[1:])]); total = float(seg.sum())

        def point_at(sp: float) -> np.ndarray:
            for a, b, L in zip(pts[:-1], pts[1:], seg):
                if sp <= L:
                    return a + (b - a) * (sp / L if L > 1e-12 else 1.0)
                sp -= L
            return pts[-1].copy()

        tol = (tol_deg or c.tol_deg) if settle else 3.0
        hist, t, sp, v, bias, at_goal_t = [], 0.0, 0.0, 0.0, np.zeros(5), None
        while True:
            q = np.asarray(self.b.read(), float)
            err = goal - q
            at_goal = sp >= total - 1e-9
            if at_goal and at_goal_t is None:
                at_goal_t = t
            if at_goal and (np.max(np.abs(err[:5])) < tol or (np.max(np.abs(err[:5])) < max(3.0, tol) and t - at_goal_t > 3.0)):
                break
            base = point_at(sp)
            lag = float(np.max(np.abs(base - q[:5])))
            paused = (not at_goal) and lag > c.path_tol_deg
            if paused:
                v = 0.0
            elif not at_goal:
                v = min(v + speed / 0.3 * dt, float(np.clip((total - sp) * 2.0, c.near_speed_deg_s, speed)))
                sp = min(total, sp + v * dt); base = point_at(sp)
            if at_goal and settle:  # jev_arm: correct static (gravity) error near the target only; never pan
                bias = np.where(np.abs(err[:5]) < 14.0, np.clip(bias + .35 * err[:5] * dt, -12.0, 12.0), bias); bias[0] = 0.0
            elif not at_goal and lead:   # gravity-loaded shoulder / elbow during the move: slowly lead the command by the lag, so a
                # stretched-out arm can be lifted (2026-09-27: the lag pause capped their effort at 6 deg and the arm stalled)
                lagv = (base - q[:5])[1:3]
                bias[1:3] = np.clip(bias[1:3] + np.where(np.abs(lagv) > 1.0, 1.0 * lagv * dt, 0.0), -12.0, 12.0)
            cmd[:5] = base + bias
            cmd[5] = cmd[5] + float(np.clip(g_goal - cmd[5], -c.gripper_pct_s * dt, c.gripper_pct_s * dt))
            cmd = np.asarray(self.b.send(cmd), float); self.last_cmd = cmd
            self.b.wait(dt); t += dt; self.motion_s += dt
            if touch is not None and touch(q, cmd):
                return {"reached": False, "touched": True, "t": t}
            hist.append((t, q[:5].copy()))
            hist = [h for h in hist if t - h[0] <= c.stall_s]
            if (t > c.stall_s and hist and np.max(np.abs(hist[-1][1] - hist[0][1])) < .3 and (paused or at_goal)
                    and np.max(np.abs((base if paused else goal[:5]) - q[:5])) > max(3.0, tol) and not np.any(bias)):
                self.b.stop_hold(); self.armed = False
                raise MotionError(f"stalled: no joint progress for {c.stall_s} s, joint error {np.round(err[:5], 1).tolist()} deg")
            if t > c.move_timeout_s:
                self.b.stop_hold(); self.armed = False
                raise MotionError(f"move timed out: joint error {np.round(err[:5], 1).tolist()} deg, "
                                  f"sent {np.round(cmd[:5], 1).tolist()}, goal {np.round(goal[:5], 1).tolist()}")
        return {"reached": True, "touched": False, "t": t}

    def _grip(self, pct: float, max_s: float, min_s: float = 0.0) -> float:
        """Drive the gripper to pct; stop waiting once the reading is still (an object stops the jaw) or at max_s."""
        self._hold()
        c = self.cfg; dt = 1.0 / c.hz; cmd = np.asarray(self.last_cmd, float).copy(); t, still = 0.0, 0.0
        prev = self.b.read()[5]; closing = pct < prev
        while t < max_s:
            cmd[5] = cmd[5] + float(np.clip(pct - cmd[5], -c.gripper_pct_s * dt, c.gripper_pct_s * dt))
            cmd = np.asarray(self.b.send(cmd), float); self.last_cmd = cmd
            self.b.wait(dt); t += dt; self.motion_s += dt
            g = self.b.read()[5]
            still = still + dt if abs(g - prev) < .05 else 0.0; prev = g
            reached = abs(g - pct) < 3.0
            settled = abs(cmd[5] - pct) < 1e-6 and still > .5
            blocked = closing and still > .5 and g - cmd[5] > 1.0  # the jaw stopped short of the command: an object
            if t >= min_s and (reached or settled or blocked):
                break
        self._sync()
        return float(self.q[5])

    def _move(self, target, mid_event=None, settle: bool = True, tol_deg: float | None = None) -> dict:
        """Straight-line move of the tool to `target` (table coordinates) with the current yaw (used by self-calibration)."""
        target = np.clip(np.asarray(target, float), WORK_MIN, REAL_WORK_MAX)
        self._sync(); start = self.ee()
        try:
            wps = self._path(start, target, self.yaw)
        except MotionError:
            # the straight line leaves the top-down reach: interpolate joints instead (as the simulator's own moves do),
            # checking that the tool never dips more than 1 cm below the lower end of the move
            sol = self.solve(target, self.yaw)
            if sol["success"]:
                q1 = body_real_deg(sol["joint_pos"])
            else:   # outside the top-down reach: the tilting solution (2026-09-27: straight-down -> tilted
                qf, err_mm, tilt = self._ik_free(target, self.yaw, self.kin.data.qpos[:5])   # switch jumped 47 deg mid-path)
                if err_mm > 3.0 or tilt > self.tilt_limit(target[2]):
                    raise
                q1 = body_real_deg(qf)
            q0 = self.q[:5].copy(); floor = min(start[2], target[2]) - .01
            wps = []
            for k in range(1, 21):
                qk = q0 + (q1 - q0) * k / 20
                if self._tip(np.r_[qk, self.q[5]])[2] - self.cfg.table_z < floor:
                    raise MotionError("joint-space fallback would dip towards the table")
                wps.append(np.r_[qk, np.nan])
        self._track(wps, settle=settle, tol_deg=tol_deg)
        return self.solve(target, self.yaw)

    # ------------------------------------------------------------------ skills (World-compatible feedback dicts)
    def move_xy(self, xy) -> dict:
        self.last_move_arrived = False   # set True only by a completed move (a refused carry must never allow a drop)
        before = self.ee(); req = np.asarray(xy, float)
        target = np.clip(req, WORK_MIN[:2], REAL_WORK_MAX[:2])
        holding = bool(self.grip_closed_cmd)
        floor = self.cfg.carry_z - .025 if holding else min(.13, self.cfg.carry_z)   # a held ball hangs ~6 cm below the tool; an empty gripper at 13 cm
        # (never above the configured carry height: the simulator dry runs carry at 8.5 cm and every move was refused)
        # clears a 10 cm container wall (2026-09-27 20:12: a ball 36 cm out was unreachable at 18 cm and every approach was refused)
        if float(np.hypot(*(target - before[:2]))) > .01:
            # travel height: the highest in [floor, carry_z] reachable both here and at the target; rise (or lower) to it first,
            # then travel (owner, 2026-09-27: travel high, lower only right above the target; 18:32 a move from the 8.5 cm
            # ready pose swept through the container wall and pushed the balls)
            def ok(xy, zz):
                if self.solve([float(xy[0]), float(xy[1]), zz], self.yaw)["success"]:
                    return True
                _q, e_, t_ = self._ik_free([float(xy[0]), float(xy[1]), zz], self.yaw, self.kin.data.qpos[:5])
                return e_ <= 3.0 and t_ <= self.tilt_limit(zz)
            zt = next((float(zz) for zz in np.arange(self.cfg.carry_z, floor - 1e-9, -.01) if ok(before[:2], zz) and ok(target, zz)), None)
            if zt is None and holding:   # no one safe height from here (a far pick): go in at the current height to the first point
                # on the line where a safe carry height is reachable, rise there, then cross to the target high (the target end is
                # where the container is; 2026-09-27: a ball 36 cm out could only be lifted to 14 cm)
                mid = next((before[:2] + (target - before[:2]) * f for f in np.linspace(.05, 1, 20) if ok(before[:2] + (target - before[:2]) * f, floor)), None)
                if mid is not None and float(np.hypot(*(mid - target))) > .03:
                    try:
                        self._move([float(mid[0]), float(mid[1]), float(before[2])], tol_deg=self.cfg.travel_tol_deg if self.cfg.fast_travel else None)
                        before = self.ee()
                        zt = next((float(zz) for zz in np.arange(self.cfg.carry_z, floor - 1e-9, -.01) if ok(before[:2], zz) and ok(target, zz)), None)
                    except MotionError:
                        pass
            if zt is not None and abs(zt - before[2]) > .005:
                try:
                    self._move([before[0], before[1], zt], tol_deg=self.cfg.travel_tol_deg if self.cfg.fast_travel else None)
                except MotionError:
                    pass
                before = self.ee()
        z = before[2]
        if z < floor - .005 and float(np.hypot(*(target - before[:2]))) > .01:
            self.step_count += 1; self.last_action = "move"
            self.cfg.log.append({"error": f"refused: travel at {z * 100:.1f} cm is below the carry height"})
            fb = {"clamped": False, "stalled": True, "holding": self.observe()["holding"]}; self.last_feedback = fb
            return fb
        y = self.feasible_yaw([*target, z])
        self._confirm(f"move to ({target[0]:.3f}, {target[1]:.3f}) m at height {z * 100:.1f} cm")
        if y is not None:
            self.yaw = y
        stalled = False
        self.travel_xy = np.asarray(target, float).copy()
        try:   # _move falls back to a joint-space move when the straight line is refused (2026-09-27: carries from a far
            self._move([*target, z], tol_deg=self.cfg.travel_tol_deg if self.cfg.fast_travel else None)   # grasp to a container at the pan limit flipped the roll 38 deg)
        except MotionError as e:
            stalled = True; self.cfg.log.append({"error": str(e)})
        self.step_count += 1; self.last_action = "move"
        self.last_move_arrived = not stalled and float(np.linalg.norm(self.ee()[:2] - target)) <= .02
        fb = {"clamped": bool(np.any(target != req)), "stalled": bool(stalled or np.linalg.norm(self.ee()[:2] - target) > .006),
              "holding": self.observe()["holding"]}
        self.last_feedback = fb
        return fb

    def _descend(self, z_min: float, detect: bool, thr_mm: float | None = None, lead: bool = True, xy=None) -> dict:
        """Lower towards z_min (table coordinates); with detect, stop at the touch rule. With xy, the line ends above xy
        instead of straight below the start (fast travel stops within ~3 deg; the descent corrects the rest)."""
        c = self.cfg; p0 = self.ee(); steps = []
        xy1 = np.asarray(p0[:2] if xy is None else xy, float)
        at = lambda z: (p0[:2] + (xy1 - p0[:2]) * ((p0[2] - z) / max(p0[2] - z_min, 1e-6))).tolist()  # noqa: E731
        for z in np.arange(p0[2] - c.step_m, z_min - 1e-6, -c.step_m).tolist() + [z_min]:
            try:   # each piece seeded by the last one (2026-09-27: seeded from the top pose, the jump guard compared deep
                # pieces with the start and stopped far/tilted descents 4-6 cm above the ball)
                steps += self._path(p0 if not steps else [*at(prev_z), prev_z], [*at(z), z], self.yaw,
                                    seed=body_model_rad(steps[-1][:5]) if steps else None)
            except MotionError:
                break  # the lowest reachable height
            prev_z = z
        if not steps:
            return {"reached": False, "touched": False, "t": 0.0, "z": float(p0[2])}
        z_cmd0 = self._tip(self.last_cmd if self.last_cmd is not None else self.q)[2]
        base = []

        def touch(q, cmd):
            if not detect:
                return False
            zc, zm = self._tip(cmd)[2], self._tip(q)[2]
            lag = (zm - zc) * 1000
            if z_cmd0 - zc < c.touch_after_mm / 1000:
                base.append(lag); return False
            return lag - (float(np.median(base)) if base else 0.0) > (thr_mm or c.touch_mm)

        span = float(np.max(np.abs(steps[-1][:5] - self.q[:5])))  # joint degrees for this descent
        mm_per_deg = max((p0[2] - self._tip(np.r_[steps[-1][:5], self.q[5]])[2] + c.table_z) * 1000, 1.0) / max(span, .1)
        try:
            r = self._track(steps, speed=float(np.clip(c.descend_mm_s / mm_per_deg, .5, c.near_speed_deg_s * 2)), touch=touch,
                            lead=lead)
        except MotionError as e:  # blocked on the way down = contact: the arm already holds position; report it as a touch
            self.cfg.log.append({"descend_stall": str(e)}); r = {"reached": False, "touched": True, "t": 0.0, "stall": True}
        z = float(self.ee()[2])
        self.touch_log.append({"sensed_z": z if r["touched"] else None, "lag_baseline_mm": float(np.median(base)) if base else None})
        return {**r, "z": z}

    def grasp(self) -> dict:
        before = self.ee()
        self._confirm(f"grasp at ({before[0]:.3f}, {before[1]:.3f}): lower to {GRASP_Z * 100:.1f} cm and close")
        g_open = self.cfg.grasp_open_pct or GRIP_OPEN_PCT
        if abs(self.q[5] - g_open) > 5 if self.cfg.grasp_open_pct else self.q[5] < GRIP_OPEN_PCT - 5:
            self._grip(g_open, max_s=8.0)
        if self.cfg.grasp_clear_m is not None:   # heights now measured to ~2 mm (full-territory calibration + table touches,
            # 2026-09-27 17:30): lower the fixed jaw's tip (the lowest part with the gripper open; the calibrated geometry) to
            # grasp_clear_m above the table with the gravity lead, so the descent stays vertical; contact-based descents pushed
            # on the table and the fingertips slid 1-3 cm outward, landing a finger on the ball (ep_20260927-174258)
            self._sync(); p = self.ee(); xy = getattr(self, "travel_xy", None) if self.cfg.fast_travel else None
            q = np.r_[p[:2] if xy is None else xy, p[2]]
            d = self._descend(self._site_z_for_clearance(q, self.cfg.grasp_clear_m), detect=False, lead=True, xy=xy)
        elif self.cfg.grasp_to_table:   # the model's heights are cm off across the table (2026-09-27): lower until the fingers
            # are stopped by the table (a stall counts as contact; no gravity push, so the touch is gentle), back off, close
            d = self._descend(-.005, detect=False, lead=False)
            p = self.ee(); self._move([p[0], p[1], p[2] + self.cfg.grasp_backoff_m])
        else:
            d = self._descend(GRASP_Z, detect=True, thr_mm=self.cfg.grasp_stop_mm)
        self.grip_closed_cmd = True; self.grip_target = -.12
        g = self._grip(GRIP_CLOSED_PCT, max_s=self.cfg.grasp_min_close_s + 4.0, min_s=self.cfg.grasp_min_close_s)
        self.step_count += 1; self.last_action = "grasp"
        obs = self.observe()
        fb = {"actual_displacement": (self.ee() - before).tolist(), "contact": bool(d["touched"]), "stalled": False,
              "gripper_state": obs["gripper_state"], "holding": obs["holding"], "ik_error": None, "gripper_pct": g,
              "touched_early": bool(d["touched"])}
        self.last_feedback = fb
        return fb

    def retreat(self, back_m: float = .06) -> None:
        """Clear the work area before folding: straight up at the current spot to (at least) the carry height, out of any
        container (2026-09-27: after a give-up the arm stayed stretched with its fingers at a container's rim), then in
        towards the base as far as the reach allows (back_m, else less; 18:20: pulling in 6 cm at 18 cm above the container
        needed a 37 deg tilt and the whole retreat was refused, leaving the arm over the container)."""
        self._sync(); p = self.ee()
        for z in np.arange(self.cfg.carry_z, p[2] + .015, -.02):   # as high as the reach allows here (never refuse the return:
            try:                                                  # 18:21, rising to 18 cm at the ready pose needed 37 deg)
                self._move([p[0], p[1], float(z)]); break
            except MotionError:
                continue
        p = self.ee(); r = float(np.hypot(p[0], p[1]))
        for b in (back_m, .04, .02):
            k = max(r - b, .15) / max(r, 1e-6)
            try:
                self._move([p[0] * k, p[1] * k, p[2]]); return
            except MotionError:
                continue

    def lift(self) -> dict:
        before = self.ee()
        z = self.cfg.carry_z
        if self.cfg.fast_travel:   # as high as the reach allows here, up to the carry height (far balls cannot reach 18 cm)
            def ok(zz):
                if self.solve([float(before[0]), float(before[1]), zz], self.yaw)["success"]:
                    return True
                _q, e_, t_ = self._ik_free([float(before[0]), float(before[1]), zz], self.yaw, self.kin.data.qpos[:5])
                return e_ <= 3.0 and t_ <= self.tilt_limit(zz)
            z = next((float(zz) for zz in np.arange(self.cfg.carry_z, before[2] + .02, -.01) if ok(zz)), self.cfg.carry_z)
        self._confirm(f"lift to {z * 100:.1f} cm")
        stalled = False
        try:
            self._move([before[0], before[1], z], tol_deg=self.cfg.travel_tol_deg if self.cfg.fast_travel else None)
        except MotionError as e:
            stalled = True; self.cfg.log.append({"error": str(e)})
        self.step_count += 1; self.last_action = "lift"
        obs = self.observe()
        fb = {"actual_displacement": (self.ee() - before).tolist(), "contact": False, "stalled": stalled,
              "gripper_state": obs["gripper_state"], "holding": obs["holding"], "ik_error": None}
        self.last_feedback = fb
        return fb

    def _turn_in_place(self, yaw: float):
        """Turn the wrist to `yaw` without moving the tool (tilting IK when above the top-down reach)."""
        p = self.ee(); sol = self.solve(p, yaw)
        q = np.asarray(sol["joint_pos"][:5], float) if sol["success"] else self._ik_free(p, yaw, self.kin.data.qpos[:5])[0]
        self._track([np.r_[body_real_deg(q), np.nan]]); self.yaw = yaw

    def place(self) -> dict:
        p = self.ee()
        if self.cfg.drop_release:   # open over the target at the carry height, only if the carry arrived
            self._confirm(f"drop at ({p[0]:.3f}, {p[1]:.3f}) from {p[2] * 100:.0f} cm")
            self.step_count += 1; self.last_action = "place"
            if not getattr(self, "last_move_arrived", False):
                fb = {"touched": False, "stalled": True, "release_z": float(p[2]), "holding": self.observe()["holding"]}
                self.cfg.log.append({"drop_refused": "the carry did not arrive"}); self.last_feedback = fb
                return fb
            self.grip_closed_cmd = False; self.grip_target = 1.0
            self._grip(self.cfg.grasp_open_pct or GRIP_OPEN_PCT, max_s=4.0)
            fb = {"touched": True, "release_z": float(p[2]), "holding": self.observe()["holding"]}; self.last_feedback = fb
            return fb
        self._confirm(f"put down at ({p[0]:.3f}, {p[1]:.3f}): lower until touch, then open")
        # the descent is top-down: if this yaw cannot go straight down here, turn the wrist first, while still high
        # (2026-09-26: spots beside the container were reachable only at some yaws; a round object allows any)
        low_ok = lambda yy: all(self.solve([p[0], p[1], z], yy)["success"] for z in (GRASP_Z + .02, .06))  # noqa: E731
        if not low_ok(self.yaw):
            per = self.symmetry()
            for y in sorted({round(wrap(self.yaw + j * per, 2 * math.pi), 9) for j in range(-12, 13)},
                            key=lambda y: abs(wrap(y - self.yaw, 2 * math.pi))):
                if low_ok(y):
                    self._turn_in_place(y); break
        if self.cfg.place_clear_m is not None:   # heights calibrated (17:30): lower the held object fast to a known release height
            xy = getattr(self, "travel_xy", None) if self.cfg.fast_travel else None
            q = np.r_[p[:2] if xy is None else xy, p[2]]
            d = self._descend(self._site_z_for_clearance(q, self.cfg.place_clear_m), detect=False, lead=True, xy=xy)
            if not d.get("touched") and p[2] - d["z"] >= .01:
                d = {**d, "touched": True}   # arrived at the release height: the release counts as placed
        else:
            d = self._descend(GRASP_Z, detect=True)
        if d.get("touched") and p[2] - d["z"] < .01:   # "touched" without descending 1 cm: the arm is stuck, not on a surface.
            # Never open there (2026-09-27 18:23: a carry to an unreachable spot stalled, the descent stalled at once, counted as a
            # touch, and the ball was dropped from 18 cm onto the container's rim and bounced off the table)
            self.step_count += 1; self.last_action = "place"
            fb = {"touched": False, "stalled": True, "release_z": float(d["z"]), "holding": self.observe()["holding"]}
            self.cfg.log.append({"place_refused": "no descent before the touch", "z": float(d["z"])}); self.last_feedback = fb
            return fb
        self.grip_closed_cmd = False; self.grip_target = 1.0
        self._grip(GRIP_OPEN_PCT, max_s=9.0)
        self.step_count += 1; self.last_action = "place"
        fb = {"touched": bool(d["touched"]), "release_z": float(d["z"]), "holding": self.observe()["holding"]}
        self.last_feedback = fb
        return fb

    def open_gripper(self) -> dict:
        self._confirm("open the gripper")
        self.grip_closed_cmd = False; self.grip_target = 1.0
        self._grip(self.cfg.grasp_open_pct or GRIP_OPEN_PCT, max_s=9.0)
        self.step_count += 1; self.last_action = "open"
        obs = self.observe()
        return {"actual_displacement": [0, 0, 0], "contact": False, "stalled": False, "gripper_state": obs["gripper_state"],
                "holding": obs["holding"], "ik_error": None}

    def rotate(self, yaw: float) -> dict:
        self._confirm(f"turn the wrist to {math.degrees(yaw):.0f} deg")
        self.yaw = wrap(float(yaw), 2 * math.pi)
        p = self.ee(); sol = self.solve(p, self.yaw); stalled = not sol["success"]
        if sol["success"]:
            try:
                self._track([np.r_[body_real_deg(sol["joint_pos"]), np.nan]])
            except MotionError as e:
                stalled = True; self.cfg.log.append({"error": str(e)})
        self.step_count += 1; self.last_action = "rotate"
        self.last_feedback = {"stalled": stalled, "yaw_error_deg": math.degrees(sol["yaw_error"]), "holding": self.observe()["holding"]}
        return self.last_feedback

    def park(self) -> dict:
        self._confirm("park near the base, out of the top camera's view")
        if self.ee()[2] < CARRY_Z - .012:
            self.lift()
        y = self.feasible_yaw([.10, 0.0, CARRY_Z])
        if y is not None:
            self.yaw = y
        stalled = False
        try:
            self._move([.10, 0.0, CARRY_Z])
        except MotionError as e:
            stalled = True; self.cfg.log.append({"error": str(e)})
        self.step_count += 1; self.last_action = "park"
        return {"stalled": stalled, "holding": self.observe()["holding"]}

    def reset(self, seed=None, task=None) -> dict:
        """The agent calls reset() on a fresh World; the real arm stays where it is."""
        self._sync(); self.yaw = self.gripper_yaw()
        return self.observe()

    # ------------------------------------------------------------------ what only a simulator knows
    def held(self):
        return self.kin.held() if self.mirror else None

    def evaluate(self) -> dict:
        if self.mirror:
            return self.kin.evaluate()
        return {"success": False, "subgoals": [], "seq_progress": None, "note": "real arm: success is judged by the operator"}

    def frames(self, views) -> dict:
        if self.cameras is not None:
            return self.cameras.frames(views)
        from run5.env.render import flat_views
        w = self.b.world if hasattr(self.b, "world") else self.kin
        return flat_views(w, views)

    def close(self):
        try:
            if self.armed:
                self.b.stop_hold(); self.armed = False
        finally:
            self.kin.close()

    def __getattr__(self, name):  # dry runs: ground-truth helpers (satisfied, dest_frame, ...) from the mirrored twin
        kin = self.__dict__.get("kin")
        if kin is not None and self.__dict__.get("mirror"):
            return getattr(kin, name)
        raise AttributeError(name)
