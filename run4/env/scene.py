"""Run 4 honest environment (wraps the unchanged harness Embodiment).

Changes vs the harness/Run 3 scene, all by MJCF post-processing and observation wrapping:
- placement anywhere the arm can grasp, carry and release (IK-verified reach map), not ±8 mm;
- object shape (cube / ball / cylinder), size and colour; cube yaw; container size and colour;
- 0-3 physical distractor objects (visible, collidable, never the target's shape+colour);
- three cameras: the harness overhead camera (centred over the reachable area), a fixed side camera
  at the middle of the robot's right side (robot base at the left image edge), and the wrist camera;
- motion workspace widened to the reach box (the harness clamps to a 9.5 x 28 cm box).
Physics, IK, macros (grasp/lift/release/open), contact-based holding and success scoring are the harness's.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import so101_vlm.embodiment as emb
from so101_vlm.embodiment import Embodiment
from so101_vlm.tasks import score

REACH_BOX = {"min": [0.10, -0.24, 0.027], "max": [0.32, 0.24, 0.09]}
SIDE_CAMERA = dict(name="side", position=(0.20, -0.60, 0.28), target=(0.20, 0.0, 0.03), fovy=50., width=448, height=448)
COLORS = {"red": [.85, .08, .07, 1], "blue": [.08, .27, .88, 1], "green": [.1, .65, .22, 1], "purple": [.58, .18, .72, 1],
          "orange": [.95, .45, .06, 1], "yellow": [.95, .80, .06, 1], "pink": [.95, .42, .62, 1], "white": [.9, .9, .88, 1],
          "black": [.06, .06, .06, 1], "brown": [.45, .28, .12, 1]}
OBJECT_COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "pink"]
CONTAINER_COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "white", "black", "brown"]
SHAPES = ("cube", "ball", "cylinder")
NOUN = {"cube": "cube", "ball": "ball", "cylinder": "cylinder"}
GRASP_Z, RELEASE_Z, CARRY_Z = .028, .065, .085
REACH_MAP = Path(__file__).parent / "reach_map.npz"
GRASP_MAP = Path(__file__).parent / "grasp_map.npz"  # built by grasp_map.py (empirical grasp-and-lift test)
TARGET_MARGIN = .015  # the target and its +-1.5 cm neighbourhood must be graspable


def reach_map(step: float = .01) -> dict:
    """Cached IK reachability over REACH_BOX at grasp, release and carry heights (gripper pointing down)."""
    if REACH_MAP.exists():
        z = np.load(REACH_MAP)
        return {k: z[k] for k in z.files}
    e = Embodiment(); e.reset(0, "A")
    xs = np.arange(REACH_BOX["min"][0], REACH_BOX["max"][0] + 1e-9, step)
    ys = np.arange(REACH_BOX["min"][1], REACH_BOX["max"][1] + 1e-9, step)
    ok = np.ones((len(xs), len(ys)), bool)
    for z in (GRASP_Z, RELEASE_Z, CARRY_Z):
        ok &= np.array([[e.ik([x, y, z])["success"] for y in ys] for x in xs])
    e.close()
    np.savez(REACH_MAP, xs=xs, ys=ys, ok=ok)
    return {"xs": xs, "ys": ys, "ok": ok}


def grasp_map() -> dict:
    z = np.load(GRASP_MAP)
    return {k: z[k] for k in z.files}


def reachable(rm: dict, xy, margin: float = 0.0) -> bool:
    """True if xy and the four points at +-margin are all reachable."""
    for dx, dy in ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)):
        x, y = xy[0] + dx, xy[1] + dy
        i = int(round((x - rm["xs"][0]) / (rm["xs"][1] - rm["xs"][0])))
        j = int(round((y - rm["ys"][0]) / (rm["ys"][1] - rm["ys"][0])))
        if not (0 <= i < len(rm["xs"]) and 0 <= j < len(rm["ys"]) and rm["ok"][i, j]):
            return False
    return True


def _object_spec(rng, shape: str) -> dict:
    if shape == "cube":
        edge = float(rng.uniform(.04, .05))
        return {"shape": "cube", "size": [edge] * 3, "half_z": edge / 2, "footprint": edge * 0.75}
    if shape == "ball":
        r = float(rng.uniform(.0215, .025))  # harness grasp height cannot hold balls under ~21 mm radius
        return {"shape": "ball", "size": [2 * r] * 3, "half_z": r, "footprint": r}
    r, hh = float(rng.uniform(.018, .022)), float(rng.uniform(.02, .025))
    return {"shape": "cylinder", "size": [2 * r, 2 * r, 2 * hh], "half_z": hh, "footprint": r}


def sample_scene(seed: int, max_distractors: int = 3) -> dict:
    """Seeded scene: target object, container and distractors with positions, all reachable where needed."""
    rm, gm = reach_map(), grasp_map()
    rng = np.random.default_rng(40_000 + int(seed))
    for _ in range(500):
        tgt = _object_spec(rng, SHAPES[rng.integers(3)])
        tgt["color"] = OBJECT_COLORS[rng.integers(len(OBJECT_COLORS))]
        tgt["yaw"] = float(rng.uniform(-np.pi / 9, np.pi / 9)) if tgt["shape"] == "cube" else 0.0
        inner = float(rng.uniform(.10, .15))
        cont = {"inner": inner, "color": [c for c in CONTAINER_COLORS if c != tgt["color"]][rng.integers(len(CONTAINER_COLORS) - 1)]}
        lo, hi = np.asarray(REACH_BOX["min"][:2]), np.asarray(REACH_BOX["max"][:2])
        cont["xy"] = (lo + rng.random(2) * (hi - lo)).tolist()
        usable = inner / 2 - tgt["footprint"] - .005
        if not reachable(rm, cont["xy"], margin=max(usable, 0)):
            continue
        tgt["xy"] = (lo + rng.random(2) * (hi - lo)).tolist()
        if not reachable(gm, tgt["xy"], margin=TARGET_MARGIN):
            continue
        if np.max(np.abs(np.subtract(tgt["xy"], cont["xy"]))) < inner / 2 + .008 + tgt["footprint"] + .02:
            continue
        distractors, n = [], int(rng.integers(max_distractors + 1))
        for _k in range(200):
            if len(distractors) == n:
                break
            d = _object_spec(rng, SHAPES[rng.integers(3)])
            d["color"] = OBJECT_COLORS[rng.integers(len(OBJECT_COLORS))]
            if (d["shape"], d["color"]) == (tgt["shape"], tgt["color"]):
                continue
            d["yaw"] = float(rng.uniform(0, np.pi / 2)) if d["shape"] == "cube" else 0.0
            d["xy"] = (lo + np.array([0.0, 0.0]) + rng.random(2) * (hi - lo)).tolist()
            others = [tgt] + distractors
            if any(np.hypot(*np.subtract(d["xy"], o["xy"])) < .075 for o in others):
                continue
            if np.max(np.abs(np.subtract(d["xy"], cont["xy"]))) < inner / 2 + .008 + d["footprint"] + .01:
                continue
            distractors.append(d)
        if len(distractors) < n:
            continue
        return {"seed": int(seed), "target": tgt, "container": cont, "distractors": distractors}
    raise RuntimeError(f"could not sample a scene for seed {seed}")


def _set_geom(geom, spec):
    s = spec["size"]
    if spec["shape"] == "cube":
        geom.set("type", "box"); geom.set("size", f"{s[0] / 2} {s[1] / 2} {s[2] / 2}")
    elif spec["shape"] == "ball":
        geom.set("type", "sphere"); geom.set("size", f"{s[0] / 2}")
    else:
        geom.set("type", "cylinder"); geom.set("size", f"{s[0] / 2} {s[2] / 2}")
    geom.set("rgba", " ".join(map(str, COLORS[spec["color"]])))


class WideEmbodiment(Embodiment):
    """Harness Embodiment with the Run 4 scene; `scene` comes from sample_scene()."""

    def __init__(self, scene: dict, render_size: int = 448):
        self.scene = copy.deepcopy(scene)
        cfg = {"container_size": scene["container"]["inner"], "randomize_positions": False, "randomize_colors": False,
               "side_camera": dict(SIDE_CAMERA), "width": render_size, "height": render_size}
        cfg["side_camera"]["width"] = cfg["side_camera"]["height"] = render_size
        super().__init__(cfg)

    def _patch(self, xml, containers, cube_color):
        sc = self.scene
        root = ET.fromstring(xml)
        world = root.find("worldbody")
        body = root.find(".//body[@name='cube']")
        tgt = sc["target"]
        _set_geom(body.find("geom"), tgt)
        body.set("pos", f"{tgt['xy'][0]} {tgt['xy'][1]} {tgt['half_z'] + .001}")
        body.attrib.pop("quat", None)
        if tgt["yaw"]:
            body.set("quat", f"{np.cos(tgt['yaw'] / 2)} 0 0 {np.sin(tgt['yaw'] / 2)}")
        cb = root.find(".//body[@name='container_2']")
        cx, cy = sc["container"]["xy"]
        cb.set("pos", f"{cx} {cy} 0")
        for g in cb.findall("geom"):
            g.set("rgba", " ".join(map(str, COLORS[sc["container"]["color"]])))
        containers = copy.deepcopy(containers)
        containers[-1]["pos"] = [cx, cy, 0.]
        containers[-1]["color"] = sc["container"]["color"]
        for k, d in enumerate(sc["distractors"]):
            db = copy.deepcopy(body)
            db.set("name", f"distractor_{k}")
            db.find("freejoint").set("name", f"distractor_{k}_free")
            g = db.find("geom"); g.set("name", f"distractor_{k}")
            _set_geom(g, d)
            db.attrib.pop("quat", None)
            db.set("pos", f"{d['xy'][0]} {d['xy'][1]} {d['half_z'] + .001}")
            if d["yaw"]:
                db.set("quat", f"{np.cos(d['yaw'] / 2)} 0 0 {np.sin(d['yaw'] / 2)}")
            world.append(db)
        return ET.tostring(root, encoding="unicode"), containers, tgt["color"]

    def reset(self, seed=None, task="A"):
        original = emb.build_scene
        emb.build_scene = lambda config, s, t: self._patch(*original(config, s, t))
        try:
            super().reset(self.scene["seed"] if seed is None else seed, task)
        finally:
            emb.build_scene = original
        self.workspace = copy.deepcopy(REACH_BOX)
        return self.observe()

    def _obj(self, name: str, spec: dict) -> dict:
        d, m = self.data, self.model
        bid = m.body(name).id
        v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
        rot = np.eye(3) if spec["shape"] == "ball" else d.xmat[bid].reshape(3, 3)
        return dict(name=name, shape=spec["shape"], color=spec["color"], noun=f"{spec['color']} {NOUN[spec['shape']]}",
                    pos=d.xpos[bid].tolist(), size=list(spec["size"]), velocity=v[3:].tolist(),
                    angular_velocity=v[:3].tolist(), rotation_matrix=rot.tolist())

    def observe(self):
        obs = super().observe()
        sc = self.scene
        obs["objects"] = [self._obj("cube", sc["target"])]
        obs["distractors"] = [self._obj(f"distractor_{k}", d) for k, d in enumerate(sc["distractors"])]
        held = None
        if self._contacts("cube"):
            held = "cube"
        else:
            for k in range(len(sc["distractors"])):
                if self._contacts(f"distractor_{k}"):
                    held = f"distractor_{k}"; break
        obs["holding"] = held
        if held:
            obs["gripper_state"] = "holding"
        elif obs.get("gripper_state") == "holding":
            obs["gripper_state"] = "closed"
        obs["goal"] = f"Put the {obs['objects'][0]['noun']} into the {sc['container']['color']} container."
        return obs


def point_oracle(env: WideEmbodiment, attempts: int = 3) -> dict:
    """Privileged exact-coordinate controller: the upper bound for pointing executors."""
    log = []
    for _ in range(attempts):
        obs = env.observe()
        if score(obs)["success"]:
            break
        obj = np.asarray(obs["objects"][0]["pos"][:2])
        env.step_target_xy(obj); env.step("grasp"); log.append("grasp")
        if env.observe()["holding"] != "cube":
            env.step("open"); env.step("lift"); log.append("miss"); continue
        env.step("lift")
        obs = env.observe()
        offset = np.asarray(obs["ee_pos"][:2]) - np.asarray(obs["objects"][0]["pos"][:2])
        env.step_target_xy(np.asarray(obs["containers"][-1]["pos"][:2]) + offset)
        env.step("release"); log.append("release")
        if score(env.observe())["success"]:
            break
        env.step("lift")
    final = score(env.observe())
    return {"success": bool(final["success"]), "reason": final["reason"], "log": log}
