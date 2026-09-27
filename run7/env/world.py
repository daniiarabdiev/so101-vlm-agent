"""Run 7 world = the Run 5 world (run5/env/world.py, frozen) plus motor-feedback sensing and three new tasks.

Run 7 changes (run7/BRIEF.md):
- sensing="motor" (default): everything the agent or the controller reads comes from what SO-101 servos report.
    holding   the gripper servo stopped short of its closed target (jaw reading > target + HOLD_Q_MARGIN), with reading
              noise; it cannot tell an object from a pinched rim, just like the real arm
    touched   during the release descent, the end-effector height from joint readings lags the commanded height by more
              than CONTACT_MM over the free-air tracking error (position-error / stall signal), with reading noise
  The simulator's contact-based `held()` stays for faults, gold labels and scoring only. sensing="privileged" is Run 5/6.
- new tasks, appended to TASKS so every earlier scene seed stays identical:
    take_out     take an object that starts inside a container out onto the table        (tuning task)
    next_to      put an object on the table next to another object                        (HELD OUT)
    in_then_out  put it in, then take it out again: one compound sentence (odd seeds) or a
                 follow-up command given after the first is done (even seeds)             (HELD OUT)
  relations "out" (dest = the container it must leave) and "next_to" (dest = the reference object); sequential
  scenes are scored in order (every sub-goal reached in turn, the last one holding at the end).

Run 5 description:
Run 5 multi-task world for the SO-101 (MuJoCo physics; Run 4 cameras and harness controller).

Tasks (instruction templates in `TASKS`):
  place_in     put one object in a container                         (tuning task)
  stack        put one object on top of a block                        (tuning task)
  sort         two objects into two containers                         (tuning task)
  bar_in_tray  a long bar into a narrow tray, both at random angles    (tuning task; needs rotate)
  gather       all objects of one colour into a container              (HELD OUT of every kind of tuning)

Changes vs the Run 4 environment:
- scenes are built here (same arm asset, gripper pads, table, top/side/wrist cameras) with any number of free
  objects `object_i` and trays `container_j` (rotatable);
- the gripper has a yaw: IK targets a top-down gripper rotated by `yaw` about the vertical (jaw closing axis
  points along `yaw`); `rotate` re-solves IK in place;
- `place` (the release step) lowers the held object until it touches something, then opens: the load/stall
  signal a real servo arm has; it works for containers, block tops and trays alike;
- faults (BRIEF.md F1-F7) are injected by the environment itself at well-defined events, so every agent faces
  the same fault in the same scene;
- scoring covers in / on relations for every sub-task.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from run4.env.scene import COLORS, REACH_BOX, SIDE_CAMERA, grasp_map, reach_map, reachable
from so101_vlm.embodiment import Embodiment
from so101_vlm.scene import ASSET, DEFAULTS, configured_side_camera

TASKS = {
    "place_in": "Put the {o0} in the {c0}.",
    "stack": "Put the {o0} on top of the {o1}.",
    "sort": "Put the {o0} in the {c0} and the {o1} in the {c1}.",
    "bar_in_tray": "Put the {o0} in the {c0}.",
    "gather": "Put all the {color} objects in the {c0}.",
    # Run 7 (appended: list(TASKS).index(task) seeds every scene)
    "take_out": "Take the {o0} out of the {c0}.",
    "next_to": "Put the {o0} next to the {o1}.",
    "in_then_out": "Put the {o0} in the {c0}, then take it out again.",
}
FOLLOWUP = {"in_then_out": ("Put the {o0} in the {c0}.", "Now take it out.")}  # even seeds: two commands
TUNING_TASKS = ("place_in", "stack", "sort", "bar_in_tray", "take_out")
HELD_OUT_TASKS = ("gather", "next_to", "in_then_out")
OBJECT_COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "pink"]
CONTAINER_COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "white", "black", "brown"]
GRASP_Z, CARRY_Z, RELEASE_Z = .028, .085, .065
# The controller's clamp box is the arm's IK envelope; scenes are still sampled inside REACH_BOX, and IK decides reachability.
WORK_MIN = np.array([.06, -.28, .02])
WORK_MAX = np.array([.36, .28, .10])
FLOOR, RIM = .004, .016  # container floor thickness and rim top height (harness values)
# Grasp tolerances measured in validate.py (2026-09-24): cubes grasp 100% up to 30 deg off-face (93% at 35, ~68% at 40-45);
# bars 100% up to 50 deg off-perpendicular (35% at 60, 0% beyond 75). A 11-13 cm bar fits a 5.6-6.2 cm tray within ~11 deg.
CUBE_YAW_TOL = math.radians(30)
BAR_YAW_TOL = math.radians(45)
TRAY_YAW_TOL = math.radians(8)
FAULTS = ("F1_missed_grasp", "F2_slip", "F3_object_moved", "F4_bad_release", "F5_knocked_out", "F6_dest_moved", "F7_out_of_reach")
RECOVERABLE = FAULTS[:6]
# faults that mean something for each new task (F3/F7 would carry a take_out object out of its container: already solved)
TASK_FAULTS = {t: FAULTS for t in ("place_in", "stack", "sort", "bar_in_tray", "gather")}
TASK_FAULTS.update(take_out=("F1_missed_grasp", "F2_slip"),
                   next_to=("F1_missed_grasp", "F2_slip", "F3_object_moved", "F7_out_of_reach"),
                   in_then_out=("F1_missed_grasp", "F2_slip", "F3_object_moved", "F7_out_of_reach"))
# motor sensing (calibrated in sim, run7/diag/sensing.py; re-measure on the real arm: ARM_DAY.md)
HOLD_Q_MARGIN = .08     # rad: the gripper servo stopped this far short of its closed target -> something is in the jaws
JAW_NOISE = .004        # rad, jaw reading noise (STS3215 resolution is 0.0015 rad)
FREE_AIR_MM = .25       # steady ee-height tracking error while lowering in free air (sim: 0.1-0.23 mm)
CONTACT_MM = 1.2        # extra lag over free air that means the held object (or the gripper) touched something
Z_NOISE_MM = .15        # ee-height noise from joint readings
TABLE_OK = .008         # an object "on the table" has its bottom below this height
NEXT_GAP = .05          # next_to: at most this gap between the two footprints (inscribed half-widths)


def wrap(a: float, period: float) -> float:
    """Angle difference wrapped into [-period/2, period/2)."""
    return (a + period / 2) % period - period / 2


def Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


# ---------------------------------------------------------------------------------------------- scene specs
def _object(rng, shape: str, color: str, role: str) -> dict:
    """Graspable objects are 4.2-4.5 cm tall: the harness grasp height cannot hold lower balls, and anything taller
    leaves under 1 cm between a carried object and one already standing in a container (carry height 8.5 cm)."""
    if shape == "cube":
        e = float(rng.uniform(.042, .045)); size = [e, e, e]
        yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
    elif shape == "ball":
        r = float(rng.uniform(.0215, .0225)); size = [2 * r] * 3; yaw = 0.0
    elif shape == "cylinder":
        r, h = float(rng.uniform(.018, .022)), float(rng.uniform(.042, .045)); size = [2 * r, 2 * r, h]; yaw = 0.0
    elif shape == "bar":
        size = [float(rng.uniform(.11, .13)), .03, .045]; yaw = float(rng.uniform(0, math.pi))
    elif shape == "block":  # stacking base: wide and low, not meant to be carried
        w = float(rng.uniform(.058, .066)); size = [w, w, float(rng.uniform(.03, .04))]
        yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
    else:
        raise ValueError(shape)
    noun = {"cube": "cube", "ball": "ball", "cylinder": "cylinder", "bar": "bar", "block": "block"}[shape]
    return {"shape": shape, "size": size, "yaw": yaw, "color": color, "role": role, "noun": f"{color} {noun}",
            "half_z": size[2] / 2, "radius": .5 * math.hypot(size[0], size[1])}


def _container(rng, color: str, kind: str = "container") -> dict:
    if kind == "tray":  # narrow tray for the bar
        inner = [float(rng.uniform(.15, .165)), float(rng.uniform(.056, .062))]
        yaw = float(rng.uniform(0, math.pi))
    else:
        s = float(rng.uniform(.10, .14)); inner = [s, s]; yaw = 0.0
    return {"inner": inner, "yaw": yaw, "color": color, "kind": kind, "noun": f"{color} {kind}",
            "radius": .5 * math.hypot(inner[0] + .016, inner[1] + .016)}


_IK = None


def _ik_engine():
    global _IK
    if _IK is None:
        _IK = World({"seed": 0, "task": "none", "objects": [], "containers": [], "subgoals": [], "instruction": ""})
        _IK.reset()
    return _IK


def yaw_for_object(spec: dict) -> float | None:
    """Gripper yaw that grasps this object (None: any yaw works)."""
    if spec["shape"] == "bar":
        return wrap(spec["yaw"] + math.pi / 2, math.pi)
    if spec["shape"] in ("cube", "block"):
        return wrap(spec["yaw"], math.pi / 2)
    return None


def reach_ok(xy, zs, yaw: float | None = 0.0, tol: float = 0.0) -> bool:
    """IK reaches xy at every height in zs with this gripper yaw, or any yaw within +-tol (yaw None: any yaw)."""
    e = _ik_engine()
    if yaw is None:
        yaws = [j * math.pi / 4 for j in range(4)]
    else:
        yaws = [yaw + d for d in sorted(np.linspace(-tol, tol, 5) if tol else [0.0], key=abs)]
    return any(all(e.solve([xy[0], xy[1], z], y)["success"] for z in zs) for y in yaws)


def sample_scene(seed: int, task: str, fault: str | None = None, max_distractors: int = 3) -> dict:
    """Seeded scene for one task (seed ranges are split into tuning / screen / test sets by the caller)."""
    rm, gm = reach_map(), grasp_map()
    rng = np.random.default_rng([91_000, int(seed), list(TASKS).index(task)])
    lo, hi = np.asarray(REACH_BOX["min"][:2]), np.asarray(REACH_BOX["max"][:2])
    for _ in range(2000):
        objs, conts, subgoals = [], [], []
        colors = list(rng.permutation(OBJECT_COLORS))
        ccolors = [c for c in rng.permutation(CONTAINER_COLORS) if c not in colors[:2]]  # never the colour of a task object
        if task == "place_in":
            objs.append(_object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], colors[0], "target"))
            conts.append(_container(rng, ccolors[0])); subgoals = [{"object": 0, "dest": ("container", 0), "relation": "in"}]
        elif task == "stack":
            objs.append(_object(rng, ["cube", "cylinder"][rng.integers(2)], colors[0], "target"))
            objs.append(_object(rng, "block", colors[1], "base")); subgoals = [{"object": 0, "dest": ("object", 1), "relation": "on"}]
        elif task == "sort":
            for k in range(2):
                objs.append(_object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], colors[k], "target"))
                conts.append(_container(rng, ccolors[k]))
                conts[-1]["inner"] = [min(conts[-1]["inner"][0], .12)] * 2
                conts[-1]["radius"] = .5 * math.hypot(conts[-1]["inner"][0] + .016, conts[-1]["inner"][1] + .016)
                subgoals.append({"object": k, "dest": ("container", k), "relation": "in"})
        elif task == "bar_in_tray":
            objs.append(_object(rng, "bar", colors[0], "target"))
            conts.append(_container(rng, ccolors[0], "tray")); subgoals = [{"object": 0, "dest": ("container", 0), "relation": "in"}]
        elif task == "gather":
            n = int(rng.integers(2, 4)); shapes = list(rng.permutation(["cube", "ball", "cylinder"]))[:n]
            for s in shapes:
                objs.append(_object(rng, s, colors[0], "target"))
            conts.append(_container(rng, ccolors[0])); conts[0]["inner"] = [float(rng.uniform(.15, .17))] * 2  # room for 2-3 objects
            conts[0]["radius"] = .5 * math.hypot(conts[0]["inner"][0] + .016, conts[0]["inner"][1] + .016)
            subgoals = [{"object": k, "dest": ("container", 0), "relation": "in"} for k in range(n)]
        elif task == "take_out":  # the object starts inside the container (placed below, after the container)
            objs.append(_object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], colors[0], "target"))
            conts.append(_container(rng, ccolors[0])); conts[0]["inner"] = [float(rng.uniform(.12, .15))] * 2
            conts[0]["radius"] = .5 * math.hypot(conts[0]["inner"][0] + .016, conts[0]["inner"][1] + .016)
            objs[0]["start_in"] = 0
            subgoals = [{"object": 0, "dest": ("container", 0), "relation": "out"}]
        elif task == "next_to":
            objs.append(_object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], colors[0], "target"))
            objs.append(_object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], colors[1], "ref"))
            if rng.random() < .5:  # a container that is not part of the task (keeps "next to" from meaning "in")
                conts.append(_container(rng, ccolors[0]))
            subgoals = [{"object": 0, "dest": ("object", 1), "relation": "next_to"}]
        elif task == "in_then_out":
            objs.append(_object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], colors[0], "target"))
            conts.append(_container(rng, ccolors[0]))
            subgoals = [{"object": 0, "dest": ("container", 0), "relation": "in"},
                        {"object": 0, "dest": ("container", 0), "relation": "out"}]
        else:
            raise ValueError(task)
        placed: list[tuple[np.ndarray, float]] = []

        def free(xy, r, gap=.012):
            return all(np.hypot(*(xy - p)) >= r + q + gap for p, q in placed)

        ok = True
        for c in conts:  # trays: the usable interior must be reachable at release and carry height
            for _t in range(200):
                xy = lo + rng.random(2) * (hi - lo)
                margin = max(min(c["inner"]) / 2 - .03, 0)
                if not reachable(rm, xy, margin) or not free(xy, c["radius"]):
                    continue
                if task == "gather" and not reachable(gm, xy, min(c["inner"]) / 2 - .01):
                    continue  # several objects go in: anything landing in it or on its rim must stay graspable
                if task in ("take_out", "in_then_out") and not reachable(gm, xy, min(c["inner"]) / 2 - .03):
                    continue  # the object is grasped inside it
                if c["kind"] == "tray" and not reach_ok(xy, (CARRY_Z, RELEASE_Z), wrap(c["yaw"] + math.pi / 2, math.pi)):
                    continue
                c["xy"] = xy.tolist(); placed.append((xy, c["radius"])); break
            else:
                ok = False
        for o in objs if ok else []:
            if "start_in" in o:  # inside its container, clear of the walls so the open jaws fit around it
                c = conts[o["start_in"]]
                room = np.asarray(c["inner"]) / 2 - np.max(o["size"][:2]) / 2 - .024
                for _t in range(100):
                    xy = np.asarray(c["xy"]) + (rng.random(2) * 2 - 1) * np.maximum(room, 0)
                    if reachable(gm, xy, .01) and (o["shape"] != "cube" or reach_ok(xy, (GRASP_Z, CARRY_Z), yaw_for_object(o))):
                        o["xy"] = xy.tolist(); break
                else:
                    ok = False
                continue
            for _t in range(300):
                xy = lo + rng.random(2) * (hi - lo)
                need = gm if o["role"] in ("target", "ref") else rm
                if not reachable(need, xy, .015 if o["role"] == "target" else 0) or not free(xy, o["radius"] + (.05 if o["role"] == "ref" else 0)):
                    continue  # (a next_to reference keeps room around it)
                y = yaw_for_object(o)
                if o["role"] == "target" and o["shape"] in ("bar", "cube") and not reach_ok(xy, (GRASP_Z, CARRY_Z), y):
                    continue
                if o["role"] == "base" and not reach_ok(xy, (CARRY_Z, .07), 0.0):
                    continue
                o["xy"] = xy.tolist(); placed.append((xy, o["radius"])); break
            else:
                ok = False
        if not ok:
            continue
        taken = {(o["color"], o["shape"]) for o in objs}
        n_d = int(rng.integers(max_distractors + 1))
        if task == "sort":
            n_d = min(n_d, 2)
        dist = []
        for _t in range(300):
            if len(dist) == n_d:
                break
            dcol = OBJECT_COLORS[rng.integers(len(OBJECT_COLORS))]
            if task == "gather" and dcol == objs[0]["color"]:
                continue
            d = _object(rng, ["cube", "ball", "cylinder"][rng.integers(3)], dcol, "distractor")
            if (d["color"], d["shape"]) in taken:
                continue
            xy = lo + rng.random(2) * (hi - lo)
            if not free(xy, d["radius"], .02):
                continue
            d["xy"] = xy.tolist(); placed.append((xy, d["radius"])); dist.append(d); taken.add((d["color"], d["shape"]))
        if len(dist) < n_d:
            continue
        objs += dist
        names = {f"o{k}": o["noun"] for k, o in enumerate(objs)}
        names.update({f"c{k}": c["noun"] for k, c in enumerate(conts)})
        instruction = TASKS[task].format(color=objs[0]["color"], **names)
        out = {"seed": int(seed), "task": task, "instruction": instruction, "objects": objs, "containers": conts,
               "subgoals": subgoals, "fault": fault, "fault_seed": int(rng.integers(1 << 30))}
        if task == "in_then_out":
            out["sequential"] = True
            if seed % 2 == 0:
                first, then = FOLLOWUP[task]
                out["instruction"], out["followup"] = first.format(**names), then
        return out
    raise RuntimeError(f"could not sample {task} scene {seed}")


# ---------------------------------------------------------------------------------------------- MJCF
def _rgba(color):
    return " ".join(map(str, COLORS[color]))


def build_xml(scene: dict, width: int = 448, height: int = 448) -> str:
    cfg = {**DEFAULTS, "width": width, "height": height, "side_camera": {**SIDE_CAMERA, "width": width, "height": height}}
    root = ET.parse(ASSET / "so101.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSET / "assets"))
    opt = root.find("option"); opt.set("timestep", str(cfg["timestep"])); opt.set("iterations", "50")
    root.find("visual").append(ET.Element("global", offwidth=str(width), offheight=str(height)))
    for g in root.findall(".//default[@class='collision_gripper']/geom") + root.findall(".//default[@class='collision_gripper_mesh']/geom"):
        g.set("friction", "2.5 .02 .002"); g.set("solref", ".006 1")
    root.find(".//site[@name='gripperframe']").set("pos", "0.027 0 -.095")
    w = root.find("worldbody")
    ET.SubElement(w, "geom", name="table", type="plane", size="1 1 .02", rgba=".72 .76 .79 1", friction="1 .01 .001")
    ET.SubElement(w, "light", pos=".1 -.2 .8", dir="0 0 -1", diffuse=".85 .85 .85")
    ET.SubElement(w, "camera", name="overhead", pos=".20 0 .68", xyaxes="1 0 0 0 1 0", fovy="48")
    side = configured_side_camera(cfg)
    ET.SubElement(w, "camera", name=side["name"], pos=" ".join(f"{v:.12g}" for v in side["position"]),
                  xyaxes=" ".join(f"{v:.12g}" for v in side["xyaxes"]), fovy=f"{side['fovy']:.12g}")
    for j, c in enumerate(scene["containers"]):
        b = ET.SubElement(w, "body", name=f"container_{j}", pos=f"{c['xy'][0]} {c['xy'][1]} 0",
                          quat=f"{math.cos(c['yaw'] / 2)} 0 0 {math.sin(c['yaw'] / 2)}")
        ix, iy = c["inner"][0] / 2, c["inner"][1] / 2
        ET.SubElement(b, "geom", name=f"container_{j}_floor", type="box", size=f"{ix + .004} {iy + .004} {FLOOR / 2}",
                      pos=f"0 0 {FLOOR / 2}", rgba=_rgba(c["color"]))
        for k, (sx, sy, px, py) in enumerate([(.004, iy + .008, -ix - .004, 0), (.004, iy + .008, ix + .004, 0),
                                              (ix, .004, 0, -iy - .004), (ix, .004, 0, iy + .004)]):
            ET.SubElement(b, "geom", name=f"container_{j}_rim{k}", type="box", size=f"{sx} {sy} {RIM / 2}", pos=f"{px} {py} {RIM / 2}",
                          rgba=_rgba(c["color"]))
    for i, o in enumerate(scene["objects"]):
        z0 = o["half_z"] + .001 + (FLOOR if "start_in" in o else 0.0)
        b = ET.SubElement(w, "body", name=f"object_{i}", pos=f"{o['xy'][0]} {o['xy'][1]} {z0}",
                          quat=f"{math.cos(o['yaw'] / 2)} 0 0 {math.sin(o['yaw'] / 2)}")
        ET.SubElement(b, "freejoint", name=f"object_{i}_free")
        s = o["size"]
        kw = dict(name=f"object_{i}", friction="2.5 .02 .002", condim="6", solref=".006 1", rgba=_rgba(o["color"]),
                  mass={"bar": ".06", "block": ".08"}.get(o["shape"], ".035"))
        if o["shape"] in ("cube", "bar", "block"):
            ET.SubElement(b, "geom", type="box", size=f"{s[0] / 2} {s[1] / 2} {s[2] / 2}", **kw)
        elif o["shape"] == "ball":
            ET.SubElement(b, "geom", type="sphere", size=f"{s[0] / 2}", **kw)
        else:
            ET.SubElement(b, "geom", type="cylinder", size=f"{s[0] / 2} {s[2] / 2}", **kw)
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------------------------- embodiment
class World(Embodiment):
    """Harness controller on a Run 5 scene. Actions: move_xy, rotate, grasp, lift, place, open (+ the harness steps)."""

    def __init__(self, scene: dict, render_size: int = 448, sensing: str = "motor"):
        self.scene = copy.deepcopy(scene)
        super().__init__({"width": render_size, "height": render_size, "side_camera": {**SIDE_CAMERA, "width": render_size,
                                                                                         "height": render_size}})
        self.yaw = 0.0
        assert sensing in ("motor", "privileged"), sensing
        self.sensing = sensing

    # --- setup
    def reset(self, seed=None, task="A"):
        self.close()
        self.seed, self.task, self.step_count, self.last_action, self.last_feedback = self.scene["seed"], "R5", 0, None, {}
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.scene, self.config["width"], self.config["height"]))
        self.data = mujoco.MjData(self.model); self._scratch = mujoco.MjData(self.model)
        self.site = self.model.site("gripperframe").id; self.gripper_body = self.model.body("gripper").id
        self.jaw_body = self.model.body("moving_jaw_so101_v1").id
        self.workspace = {"min": WORK_MIN.tolist(), "max": WORK_MAX.tolist()}
        self.grip_target, self.yaw = 1.0, 0.0
        sol = self.solve([.23, 0, CARRY_Z], 0.0)
        self.data.qpos[:5] = sol["joint_pos"]; self.data.qpos[5] = self.grip_target
        self.data.ctrl[:] = self.data.qpos[:6]; mujoco.mj_forward(self.model, self.data); self._advance(.5)
        self.fault = {"type": self.scene.get("fault"), "fired": False, "event": None,
                      "rng": np.random.default_rng(self.scene.get("fault_seed", 0))}
        self.subgoal_done_once = False
        self.contacts_ok = True
        self.sense_rng = np.random.default_rng([7, self.scene["seed"]])
        self.hold_hint = None  # shape word the agent believes it grasped (controller symmetry); never read by scoring
        self.seq_progress = 0  # sequential scenes: sub-goals reached in order so far
        self.touch_log = []
        return self.observe()

    def solve(self, target, yaw: float) -> dict:
        """Top-down gripper at `target` with jaw axis along `yaw` (tries yaw and yaw+pi; the jaws are symmetric)."""
        target = np.asarray(target, float); d, m = self._scratch, self.model
        roll_now = float(self.data.qpos[4])
        cands = []
        for y in (yaw, wrap(yaw + math.pi, 2 * math.pi)):
            R = Rz(y)
            d.qpos[:] = self.data.qpos

            def fun(q):
                d.qpos[:5] = q; mujoco.mj_kinematics(m, d)
                return np.r_[d.site_xpos[self.site] - target, .08 * (d.xmat[self.gripper_body].reshape(3, 3) - R).ravel()]

            b = (m.jnt_range[:5, 0] + 1e-5, m.jnt_range[:5, 1] - 1e-5)
            for x0 in (np.clip(self.data.qpos[:5], *b), np.clip([0, -.2, .5, 1.3, y], *b)):
                r = least_squares(fun, x0, bounds=b, max_nfev=80, ftol=1e-8, gtol=1e-8)
                d.qpos[:5] = r.x; mujoco.mj_kinematics(m, d)
                err = float(np.linalg.norm(d.site_xpos[self.site] - target))
                yerr = abs(wrap(math.atan2(d.xmat[self.gripper_body][3], d.xmat[self.gripper_body][0]) - y, 2 * math.pi))
                cands.append((err < .003 and yerr < math.radians(5), err, yerr, r.x.copy(), d.site_xpos[self.site].tolist()))
        # the jaws are symmetric, so yaw and yaw+pi both grasp: prefer a solution that keeps the wrist roll continuous
        # (switching branches mid-motion spins the wrist half a turn and sweeps the jaws through the object)
        good = [c for c in cands if c[0]]
        pick = min(good, key=lambda c: abs(c[3][4] - roll_now)) if good else min(cands, key=lambda c: c[1] + .02 * c[2])
        ok, err, yerr, q, achieved = pick
        return dict(joint_pos=q.tolist(), achieved_pos=achieved, error=err, yaw_error=yerr, success=bool(ok))

    def ik(self, target):
        return self.solve(target, self.yaw)

    def _move(self, target, mid_event=None):
        target = np.clip(target, self.workspace["min"], self.workspace["max"]); sol = self.ik(target)
        start = self.data.ctrl[:5].copy(); end = np.array(sol["joint_pos"]); n = round(self.config["action_seconds"] / self.model.opt.timestep)
        for k in range(n):
            t = (k + 1) / n; t = t * t * (3 - 2 * t)
            self.data.ctrl[:5] = start + (end - start) * t; self.data.ctrl[5] = self.grip_target; mujoco.mj_step(self.model, self.data)
            if mid_event is not None and k == n // 2:
                mid_event()
        self._advance(self.config["settle_seconds"]); return sol

    # --- scene queries
    def obj_names(self):
        return [f"object_{i}" for i in range(len(self.scene["objects"]))]

    def held(self) -> str | None:
        """Ground truth (labels, faults, scoring). Run 5 rule (opposing jaw forces), plus Run 7: an object carried off the
        table touching only the gripper (wedged against one jaw and the palm; run7 DECISIONS #3)."""
        for name in self.obj_names():
            if self._contacts(name):
                return name
        for name in self.obj_names():
            if self._carried(name):
                return name
        return None

    def _carried(self, name: str) -> bool:
        i = int(name.split("_")[1]); bid = self.model.body(name).id
        if self.data.xpos[bid][2] - self.scene["objects"][i]["half_z"] < .005:
            return False
        grip, touch_grip = {self.gripper_body, self.jaw_body}, False
        gid = self.model.geom(name).id
        for c in self.data.contact[:self.data.ncon]:
            if gid not in (c.geom1, c.geom2):
                continue
            other = self.model.geom_bodyid[c.geom2 if c.geom1 == gid else c.geom1]
            if other not in grip:
                return False
            touch_grip = True
        return touch_grip

    def jaw_reading(self) -> float:
        """Gripper servo position as the arm reports it (with reading noise)."""
        return float(self.data.qpos[5] + self.sense_rng.normal(0, JAW_NOISE))

    def sensed_holding(self) -> bool:
        """What the arm can know: the gripper was commanded closed and stopped well short of its target."""
        if self.sensing == "privileged":
            return self.held() is not None
        return bool(self.grip_target < .5 and self.jaw_reading() > self.grip_target + HOLD_Q_MARGIN)

    def ee(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def body_xy_yaw(self, name: str) -> tuple[np.ndarray, float]:
        bid = self.model.body(name).id
        R = self.data.xmat[bid].reshape(3, 3)
        return self.data.xpos[bid][:2].copy(), math.atan2(R[1, 0], R[0, 0])

    def gripper_yaw(self) -> float:
        R = self.data.xmat[self.gripper_body].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def _touching_other(self, name: str) -> bool:
        gid = self.model.geom(name).id
        grip = {self.gripper_body, self.jaw_body}
        for c in self.data.contact[:self.data.ncon]:
            if gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == gid else c.geom1
                if self.model.geom_bodyid[other] not in grip:
                    return True
        return False

    # --- teleports used by fault injection (someone bumps / moves things)
    def set_object_pose(self, name: str, xy, z: float | None = None, yaw: float | None = None):
        j = self.model.joint(f"{name}_free").id; a = self.model.jnt_qposadr[j]; va = self.model.jnt_dofadr[j]
        i = int(name.split("_")[1]); spec = self.scene["objects"][i]
        cur_xy, cur_yaw = self.body_xy_yaw(name)
        yaw = cur_yaw if yaw is None else yaw
        self.data.qpos[a:a + 3] = [xy[0], xy[1], spec["half_z"] + .002 if z is None else z]
        self.data.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[va:va + 6] = 0
        mujoco.mj_forward(self.model, self.data); self._advance(.3)

    def set_container_xy(self, j: int, xy):
        bid = self.model.body(f"container_{j}").id
        self.model.body_pos[bid][:2] = xy
        self.scene["containers"][j]["xy"] = list(map(float, xy))
        mujoco.mj_forward(self.model, self.data); self._advance(.2)

    def free_spot(self, radius: float, rng, reach="grasp", avoid=(), inside_reach=True, exclude: str | None = None) -> np.ndarray | None:
        """A random table spot clear of everything except `exclude` (graspable when reach='grasp')."""
        gm = grasp_map() if reach == "grasp" else reach_map()
        lo, hi = np.asarray(REACH_BOX["min"][:2]), np.asarray(REACH_BOX["max"][:2])
        things = [(self.body_xy_yaw(n)[0], self.scene["objects"][k]["radius"]) for k, n in enumerate(self.obj_names()) if n != exclude]
        things += [(np.asarray(c["xy"]), c["radius"]) for j, c in enumerate(self.scene["containers"]) if f"container_{j}" != exclude]
        things += [(np.asarray(a), r) for a, r in avoid]
        for _ in range(400):
            xy = lo + rng.random(2) * (hi - lo)
            if inside_reach and not reachable(gm, xy, .015):
                continue
            if all(np.hypot(*(xy - p)) > radius + q + .015 for p, q in things):
                return xy
        return None

    # --- evaluation (ground truth)
    def dest_frame(self, dest) -> tuple[np.ndarray, float, np.ndarray, float]:
        """(xy, yaw, half inner size, support z) of a destination."""
        kind, j = dest
        if kind == "container":
            c = self.scene["containers"][j]
            xy, yaw = self.body_xy_yaw(f"container_{j}")
            return xy, yaw, np.asarray(c["inner"]) / 2, FLOOR
        o = self.scene["objects"][j]
        bid = self.model.body(f"object_{j}").id
        xy, yaw = self.body_xy_yaw(f"object_{j}")
        return xy, yaw, np.asarray(o["size"][:2]) / 2, float(self.data.xpos[bid][2] + o["half_z"])

    def half_extents(self, i: int) -> np.ndarray:
        o = self.scene["objects"][i]
        R = self.data.xmat[self.model.body(f"object_{i}").id].reshape(3, 3)
        return np.abs(R) @ (np.asarray(o["size"]) / 2) if o["shape"] != "ball" else np.asarray(o["size"]) / 2

    def satisfied(self, sg: dict, held: str | None = None) -> bool:
        i = sg["object"]; name = f"object_{i}"
        held = self.held() if held is None else held
        if held == name:
            return False
        bid = self.model.body(name).id
        v = np.zeros(6); mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
        if np.linalg.norm(v[3:]) > .03:
            return False
        pos = self.data.xpos[bid]
        xy, yaw, half, support = self.dest_frame(sg["dest"])
        local = Rz(-yaw)[:2, :2] @ (pos[:2] - xy)
        R = Rz(-yaw) @ self.data.xmat[bid].reshape(3, 3)
        o = self.scene["objects"][i]
        he = np.abs(R) @ (np.asarray(o["size"]) / 2) if o["shape"] != "ball" else np.asarray(o["size"]) / 2
        bottom = pos[2] - he[2]
        if sg["relation"] in ("out", "next_to"):
            return bool(bottom < TABLE_OK and self.release_ok(i, pos[:2], sg))
        if sg["relation"] == "in":
            return bool(np.all(np.abs(local) + he[:2] <= half + .001) and bottom < RIM + .01)
        # on: centre over the block top, resting on it (and the block itself still on the table)
        base = self.scene["objects"][sg["dest"][1]]
        base_ok = abs(self.data.xpos[self.model.body(f"object_{sg['dest'][1]}").id][2] - base["half_z"]) < .006
        return bool(base_ok and np.all(np.abs(local) <= half - .004) and abs(bottom - support) < .008)

    def footprint(self, i: int) -> float:
        """Inscribed half-width of an object's footprint (a cube's half edge, a ball's radius)."""
        return float(np.max(self.scene["objects"][i]["size"][:2]) / 2)

    def release_ok(self, i: int, xy, sg: dict, margin: float = 0.0) -> bool:
        """Table relations: would object i resting at table xy satisfy sub-goal sg (clear by `margin` m)?"""
        xy = np.asarray(xy, float); r = self.footprint(i)
        kind, j = sg["dest"]
        if sg["relation"] == "out":
            c = self.scene["containers"][j]; cxy, cyaw = self.body_xy_yaw(f"container_{j}")
            local = Rz(-cyaw)[:2, :2] @ (xy - cxy)
            outer = np.asarray(c["inner"]) / 2 + .008
            return bool(np.any(np.abs(local) - r >= outer + margin))
        # next_to: footprints at most NEXT_GAP apart, the reference itself still on the table
        rid = self.model.body(f"object_{j}").id
        rxy = self.data.xpos[rid][:2]
        ref_on_table = self.data.xpos[rid][2] - self.scene["objects"][j]["half_z"] < TABLE_OK
        gap = float(np.hypot(*(xy - rxy)) - r - self.footprint(j))
        return bool(ref_on_table and gap <= NEXT_GAP - margin)

    def _track(self):
        """Sequential scenes: advance the in-order progress while the next sub-goal is met (the last one is scored live)."""
        sgs = self.scene["subgoals"]
        if not self.scene.get("sequential"):
            return
        held = self.held()
        while self.seq_progress < len(sgs) - 1 and self.satisfied(sgs[self.seq_progress], held):
            self.seq_progress += 1

    def evaluate(self) -> dict:
        held = self.held()
        done = [self.satisfied(sg, held) for sg in self.scene["subgoals"]]
        if self.scene.get("sequential"):
            self._track()
            ok = self.seq_progress >= len(done) - 1 and done[-1] and held is None
            return {"success": bool(ok), "subgoals": done, "held": held, "seq_progress": self.seq_progress}
        return {"success": bool(all(done) and held is None), "subgoals": done, "held": held}

    # --- observation
    def observe(self):
        if self.sensing == "privileged":
            held = self.held()
        else:
            held = "sensed" if self.sensed_holding() else None
        if held:
            state = "holding"
        else:
            state = "open" if self.grip_target > .5 else "closed"
        return {"ee_pos": self.ee().tolist(), "gripper_yaw": self.gripper_yaw(), "gripper_state": state, "holding": held,
                "goal": self.scene["instruction"], "joint_pos": self.data.qpos[:6].tolist()}

    # --- actions
    def move_xy(self, xy) -> dict:
        """Move the gripper to table xy at the current height (the agent's move_to_object / move_to_place)."""
        before = self.ee(); req = np.asarray(xy, float)
        target = np.clip(req, WORK_MIN[:2], WORK_MAX[:2])
        held = self.held()
        mid = None
        if self._arm_fault("F2_slip") and held == self._sg_object_name() and reachable(grasp_map(), (before[:2] + target) / 2, .02):
            mid = self._slip  # a recoverable slip: the object drops where the arm can still grasp it
        y = self.feasible_yaw([*target, before[2]])
        if y is not None and abs(wrap(y - self.yaw, 2 * math.pi)) > 1e-6:
            self.yaw = y  # the wrist turns to an equivalent angle the arm can reach there (symmetry of the jaws / held object)
        sol = self._move([*target, before[2]], mid_event=mid)
        self.step_count += 1; self.last_action = "move"
        fb = {"clamped": bool(np.any(target != req)), "stalled": bool(not sol["success"] or np.linalg.norm(np.asarray(sol["achieved_pos"]) - self.ee()) > .004),
              "holding": self.observe()["holding"]}
        if not held:
            self._after_approach(target)
        self._track()
        self.last_feedback = fb
        return fb

    def symmetry(self) -> float:
        """Yaw period that leaves the grasp unchanged: jaws pi; a held cube pi/2; a held ball or cylinder: any.
        Motor sensing: the controller only knows it holds *something* and what the agent says it grasped (hold_hint)."""
        if self.sensing == "privileged":
            held = self.held()
            shape = self.scene["objects"][int(held.split("_")[1])]["shape"] if held else None
        else:
            shape = self.hold_hint if self.sensed_holding() else None
        if shape:
            return {"cube": math.pi / 2, "ball": math.pi / 6, "cylinder": math.pi / 6}.get(shape, math.pi)
        return math.pi

    def feasible_yaw(self, target) -> float | None:
        """The current yaw if the arm reaches `target` with it, else the nearest equivalent yaw that works."""
        per = self.symmetry()
        cands = sorted({round(wrap(self.yaw + j * per, 2 * math.pi), 9) for j in range(-6, 7)}, key=lambda y: abs(wrap(y - self.yaw, 2 * math.pi)))
        for y in cands:
            if self.solve(target, y)["success"]:
                return y
        return None

    def park(self) -> dict:
        """Move the gripper out of the top camera's view of the work area (near the base, carrying height)."""
        if self.ee()[2] < CARRY_Z - .012:
            Embodiment.step(self, "lift")
        y = self.feasible_yaw([.10, 0.0, CARRY_Z])
        if y is not None:
            self.yaw = y
        sol = self._move([.10, 0.0, CARRY_Z])
        self.step_count += 1; self.last_action = "park"
        self._track()
        return {"stalled": not sol["success"], "holding": self.observe()["holding"]}

    def rotate(self, yaw: float) -> dict:
        self.yaw = wrap(float(yaw), 2 * math.pi)
        sol = self._move(self.ee())
        self.step_count += 1; self.last_action = "rotate"
        self._track()
        self.last_feedback = {"stalled": not sol["success"], "yaw_error_deg": math.degrees(sol["yaw_error"]), "holding": self.observe()["holding"]}
        return self.last_feedback

    def grasp(self) -> dict:
        if self._arm_fault("F1_missed_grasp"):
            a = self.fault["rng"].uniform(0, 2 * math.pi); r = self.fault["rng"].uniform(.03, .035)
            self._fire("F1_missed_grasp", {"offset": [r * math.cos(a), r * math.sin(a)]})
            p = self.ee(); self._move([p[0] + r * math.cos(a), p[1] + r * math.sin(a), p[2]])
        return self._harness("grasp")

    def lift(self) -> dict:
        fb = self._harness("lift")
        if self._arm_fault("F6_dest_moved") and self.held() == self._sg_object_name():
            self._move_destination()
        return fb

    def place(self) -> dict:
        """Lower until the held object (or the gripper) touches something, then open. Motor sensing: "touched" is the
        servo tracking error (the arm lags its commanded height), not a simulator contact."""
        held = self.held()
        if self._arm_fault("F4_bad_release") and held == self._sg_object_name():
            self._bad_release_offset()
        p0 = self.ee(); z, touched = p0[2], False
        lags, truth_at = [], None
        while z > GRASP_Z + 1e-4:
            z = max(z - .004, GRASP_Z)
            sol = self.ik([p0[0], p0[1], z])
            start = self.data.ctrl[:5].copy(); end = np.asarray(sol["joint_pos"])
            for k in range(20):
                self.data.ctrl[:5] = start + (end - start) * (k + 1) / 20; self.data.ctrl[5] = self.grip_target
                mujoco.mj_step(self.model, self.data)
            name = held or self.held()
            contact = bool((name and self._touching_other(name)) or self._gripper_touching())
            if contact and truth_at is None:
                truth_at = float(z)
            if self.sensing == "privileged":
                if contact:
                    touched = True; break
                continue
            lag = (float(self.ee()[2]) - z) * 1000 + float(self.sense_rng.normal(0, Z_NOISE_MM))
            lags.append(round(lag, 2))
            if lag > FREE_AIR_MM + CONTACT_MM:
                touched = True; break
        self.touch_log.append({"sensed_z": float(z) if touched else None, "contact_z": truth_at, "lags_mm": lags[-4:]})
        self.grip_target = 1.0; self.data.ctrl[5] = 1.0; self._advance(.8)
        self.step_count += 1; self.last_action = "place"
        self.last_feedback = {"touched": touched, "release_z": float(self.ee()[2]), "holding": self.observe()["holding"]}
        self._after_step()
        self._track()
        return self.last_feedback

    def open_gripper(self) -> dict:
        return self._harness("open")

    def _harness(self, action) -> dict:
        fb = Embodiment.step(self, action)
        self._after_step()
        self._track()
        return fb

    def _gripper_touching(self) -> bool:
        """The gripper itself touches the table, a tray, or an object it is not holding."""
        grip = {self.gripper_body, self.jaw_body}
        held = self.held()
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in grip) == (b2 in grip):
                continue
            other = c.geom2 if b1 in grip else c.geom1
            gname = self.model.geom(other).name
            body = self.model.body(self.model.geom_bodyid[other]).name
            if gname == "table" or body.startswith("container_") or (body.startswith("object_") and body != held):
                return True
        return False

    # --- fault machinery
    def _sg_object_name(self) -> str:
        return f"object_{self.scene['subgoals'][0]['object']}"

    def _arm_fault(self, kind: str) -> bool:
        return self.fault["type"] == kind and not self.fault["fired"]

    def _fire(self, kind: str, info: dict):
        self.fault.update(fired=True, event={"type": kind, "step": self.step_count, **info})

    def _slip(self):
        self.grip_target = 1.0; self.data.ctrl[5] = 1.0; self._advance(.35)
        self.grip_target = -.12; self.data.ctrl[5] = -.12
        self._fire("F2_slip", {"at_xy": self.ee()[:2].tolist()})

    def _after_approach(self, target_xy):
        """F3 / F7 fire right after the first approach that ends above the sub-task object."""
        if self.fault["type"] not in ("F3_object_moved", "F7_out_of_reach") or self.fault["fired"]:
            return
        name = self._sg_object_name()
        xy, _ = self.body_xy_yaw(name)
        if np.max(np.abs(self.ee()[:2] - xy)) > .02:
            return
        rng = self.fault["rng"]; spec = self.scene["objects"][int(name.split("_")[1])]
        if self.fault["type"] == "F3_object_moved":
            for _ in range(200):
                new = self.free_spot(spec["radius"], rng, exclude=name)
                if new is not None and 0.05 <= np.hypot(*(new - xy)) <= .12 and (spec["shape"] not in ("bar", "cube") or reach_ok(new, (GRASP_Z, CARRY_Z), yaw_for_object(spec))):
                    self.set_object_pose(name, new); self._fire("F3_object_moved", {"from": xy.tolist(), "to": new.tolist()}); return
        else:
            for _ in range(200):
                new = np.array([rng.uniform(.38, .44), rng.uniform(-.16, .16)])
                if all(np.hypot(*(new - self.body_xy_yaw(n)[0])) > .08 for n in self.obj_names() if n != name):
                    self.set_object_pose(name, new); self._fire("F7_out_of_reach", {"from": xy.tolist(), "to": new.tolist()}); return

    def _bad_release_offset(self):
        sg = self.scene["subgoals"][0]
        dxy, dyaw, half, _ = self.dest_frame(sg["dest"])
        ee = self.ee()[:2]
        for _ in range(100):
            a = self.fault["rng"].uniform(0, 2 * math.pi)
            d = np.array([math.cos(a), math.sin(a)])
            far = dxy + d * (np.max(half) + .045)
            if reachable(reach_map(), far, 0) and self.solve([far[0], far[1], self.ee()[2]], self.yaw)["success"]:
                self._move([far[0], far[1], self.ee()[2]])
                self._fire("F4_bad_release", {"intended": ee.tolist(), "actual": far.tolist()}); return

    def _move_destination(self):
        sg = self.scene["subgoals"][0]; kind, j = sg["dest"]
        rng = self.fault["rng"]
        spec = self.scene["containers"][j] if kind == "container" else self.scene["objects"][j]
        old = np.asarray(self.dest_frame(sg["dest"])[0])
        for _ in range(600):
            new = self.free_spot(spec["radius"], rng, reach="reach", avoid=[(self.ee()[:2], .05)],
                                 exclude=f"container_{j}" if kind == "container" else f"object_{j}")
            if new is None or not (.05 <= np.hypot(*(new - old)) <= .12):
                continue
            if not reachable(reach_map(), new, max(min(spec["inner"] if "inner" in spec else spec["size"][:2]) / 2 - .04, 0)):
                continue
            if kind == "container":
                self.set_container_xy(j, new)
            else:
                self.set_object_pose(f"object_{j}", new)
            self._fire("F6_dest_moved", {"from": old.tolist(), "to": new.tolist()}); return

    def _after_step(self):
        """F5: the first time sub-task 0 is satisfied, the object is put back on the table."""
        if not self._arm_fault("F5_knocked_out"):
            return
        sg = self.scene["subgoals"][0]
        self._advance(.3)
        if self.satisfied(sg):
            name = f"object_{sg['object']}"; spec = self.scene["objects"][sg["object"]]
            for _ in range(50):
                new = self.free_spot(spec["radius"], self.fault["rng"], avoid=[(self.ee()[:2], .06)], exclude=name)
                if new is not None and (spec["shape"] not in ("bar", "cube") or reach_ok(new, (GRASP_Z, CARRY_Z), yaw_for_object(spec))):
                    self.set_object_pose(name, new); self._fire("F5_knocked_out", {"to": new.tolist()}); return


# ---------------------------------------------------------------------------------------------- snapshots
def snapshot(env: World) -> dict:
    """Everything needed to rebuild this exact state elsewhere (moved trays live in env.scene)."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    st = np.empty(mujoco.mj_stateSize(env.model, spec)); mujoco.mj_getState(env.model, env.data, st, spec)
    return {"scene": copy.deepcopy(env.scene), "integration": st.tolist(), "grip_target": float(env.grip_target),
            "yaw": float(env.yaw), "step_count": int(env.step_count), "sensing": env.sensing, "hold_hint": env.hold_hint,
            "seq_progress": int(env.seq_progress), "ctrl": env.data.ctrl.tolist()}


def restore(snap: dict, render_size: int = 448) -> World:
    scene = copy.deepcopy(snap["scene"]); scene["fault"] = None  # a restored state never injects faults
    env = World(scene, render_size, snap.get("sensing", "motor")); env.reset()
    mujoco.mj_setState(env.model, env.data, np.asarray(snap["integration"]), mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(env.model, env.data)
    env.grip_target, env.yaw, env.step_count = snap["grip_target"], snap["yaw"], snap["step_count"]
    env.hold_hint, env.seq_progress = snap.get("hold_hint"), snap.get("seq_progress", 0)
    if snap.get("ctrl") is not None:
        env.data.ctrl[:] = snap["ctrl"]
    env.data.ctrl[5] = env.grip_target
    return env
