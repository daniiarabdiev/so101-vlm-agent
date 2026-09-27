"""Run 7 copy of run5/env/policy.py: the same vocabulary and gold rules, plus the table relations "out" / "next_to"
(facts, oracle release spots), sequential scenes, and the grasp hint for motor sensing.

Ground-truth facts, the correct next step (gold labels), and a privileged oracle for the Run 5 world.

The step vocabulary is shared by the oracle, the screen, the LoRA data and the model agent:
  move_to_object  move the open gripper above the current sub-task's object
  grasp           lower and close the gripper
  lift            raise the gripper to carrying height
  move_to_place   move the held object above the destination (a free spot in the container / the block top)
  rotate          turn the gripper to line up (with the object when empty-handed, with the destination when holding)
  release         lower until the object touches, then open
  open            open the gripper (it closed on nothing)
  done            the current sub-task is complete
  give_up         the current sub-task cannot be completed (e.g. the object is out of reach)
"""
from __future__ import annotations

import math

import numpy as np

from run4.env.scene import REACH_BOX, grasp_map, reachable
from run7.env.world import (BAR_YAW_TOL, CARRY_Z, CUBE_YAW_TOL, GRASP_Z, TRAY_YAW_TOL, WORK_MAX, WORK_MIN, World, Rz, reach_ok, wrap,
                             yaw_for_object)

STEPS = ["move_to_object", "grasp", "lift", "move_to_place", "rotate", "release", "open", "done", "give_up"]


def current_subgoal(env: World, pointer: int | None = None) -> int | None:
    """The sub-task the agent works on: the one whose object is held, else `pointer` (or the first unsatisfied)."""
    held = env.held()
    sgs = env.scene["subgoals"]
    if env.scene.get("sequential"):  # in order: the first sub-goal not yet reached
        env._track()
        p = env.seq_progress
        return None if (p == len(sgs) - 1 and env.satisfied(sgs[p], held)) else p
    if held:
        for k, sg in enumerate(sgs):
            if f"object_{sg['object']}" == held and not env.satisfied(sg, held):
                return k
    if pointer is not None:
        return pointer if pointer < len(sgs) else None
    for k, sg in enumerate(sgs):
        if not env.satisfied(sg, held):
            return k
    return None


def facts(env: World, k: int) -> dict:
    sg = env.scene["subgoals"][k]
    i = sg["object"]; name = f"object_{i}"; spec = env.scene["objects"][i]
    held = env.held(); ee = env.ee(); g = env.gripper_yaw()
    oxy, oyaw = env.body_xy_yaw(name)
    f = {"subgoal": k, "holding": held, "holding_target": held == name, "holding_other": bool(held and held != name),
         "closed_empty": held is None and env.grip_target < .5, "lifted": bool(ee[2] >= CARRY_Z - .012),
         "satisfied": env.satisfied(sg, held)}
    f["aligned_xy"] = bool(held is None and np.max(np.abs(ee[:2] - oxy)) <= .012)
    if spec["shape"] == "cube":
        f["yaw_ok_object"] = abs(wrap(g - oyaw, math.pi / 2)) <= CUBE_YAW_TOL
    elif spec["shape"] == "bar":
        f["yaw_ok_object"] = abs(wrap(g - (oyaw + math.pi / 2), math.pi)) <= BAR_YAW_TOL
    else:
        f["yaw_ok_object"] = True
    # the arm can reach the object at grasp and carrying height (any jaw yaw for round objects, the needed yaw otherwise)
    in_box = bool(np.all(oxy >= WORK_MIN[:2]) and np.all(oxy <= WORK_MAX[:2]))  # the controller's workspace clamp
    tol = {"cube": CUBE_YAW_TOL, "bar": BAR_YAW_TOL}.get(spec["shape"], 0.0)
    f["reachable"] = held == name or (in_box and bool(reach_ok(oxy, (GRASP_Z, CARRY_Z), yaw_for_object({**spec, "yaw": oyaw}), tol)))
    dxy, dyaw, half, _support = env.dest_frame(sg["dest"])
    local = Rz(-dyaw)[:2, :2] @ (oxy - dxy)
    if sg["dest"][0] == "container" and env.scene["containers"][sg["dest"][1]]["kind"] == "tray":
        f["yaw_ok_dest"] = abs(wrap(oyaw - dyaw, math.pi)) <= TRAY_YAW_TOL
    else:
        f["yaw_ok_dest"] = True
    R = Rz(-dyaw) @ env.data.xmat[env.model.body(name).id].reshape(3, 3)
    he = np.abs(R) @ (np.asarray(spec["size"]) / 2) if spec["shape"] != "ball" else np.asarray(spec["size"]) / 2
    if sg["relation"] in ("out", "next_to"):
        over = table_spot_ok(env, i, oxy, sg, .004)
    elif sg["relation"] == "in":
        over = bool(np.all(np.abs(local) + he[:2] <= half - .002))
    else:
        over = bool(np.all(np.abs(local) <= half - .008))
    f["over_dest"] = bool(f["holding_target"] and over)
    return f


def table_spot_ok(env: World, i: int, xy, sg: dict, margin: float) -> bool:
    """Releasing object i at table xy satisfies a table relation and lands clear of every other object and container."""
    if not env.release_ok(i, xy, sg, margin):
        return False
    if sg["relation"] == "next_to":  # not inside or on the rim of a container that is not part of the task
        for j, c in enumerate(env.scene["containers"]):
            cxy, cyaw = env.body_xy_yaw(f"container_{j}")
            if np.all(np.abs(Rz(-cyaw)[:2, :2] @ (np.asarray(xy) - cxy)) - env.footprint(i) < np.asarray(c["inner"]) / 2 + .008):
                return False
    r = env.footprint(i)
    for n in env.obj_names():
        k = int(n.split("_")[1])
        if k != i and np.hypot(*(np.asarray(xy) - env.body_xy_yaw(n)[0])) < r + env.footprint(k) + .004:
            return False
    return True


def gold_step(f: dict) -> str:
    if f["holding_other"]:
        return "release"
    if f["holding_target"]:
        if not f["lifted"]:
            return "lift"
        if not f["yaw_ok_dest"]:
            return "rotate"
        return "release" if f["over_dest"] else "move_to_place"
    if f["closed_empty"]:
        return "open"
    if f["satisfied"]:
        return "done"
    if not f["reachable"]:
        return "give_up"
    if not f["aligned_xy"]:
        return "move_to_object" if f["lifted"] else "lift"
    return "grasp" if f["yaw_ok_object"] else "rotate"


def place_spot(env: World, k: int) -> np.ndarray:
    """Oracle destination for the object centre: the freest spot inside a container, or the block centre."""
    sg = env.scene["subgoals"][k]
    if sg["relation"] in ("out", "next_to"):
        return table_spot(env, k)
    dxy, dyaw, half, _ = env.dest_frame(sg["dest"])
    if sg["dest"][0] != "container" or env.scene["containers"][sg["dest"][1]]["kind"] == "tray":
        return dxy
    me = f"object_{sg['object']}"
    inside = []  # objects already in this container are the only obstacles for the drop spot
    for n in env.obj_names():
        if n != me:
            q = env.body_xy_yaw(n)[0]
            if np.all(np.abs(Rz(-dyaw)[:2, :2] @ (q - dxy)) <= half + .01):
                inside.append(q)
    he = env.half_extents(sg["object"])[:2]
    offset = env.ee()[:2] - env.body_xy_yaw(me)[0]
    cands = []
    for u in np.linspace(-1, 1, 7):
        for v in np.linspace(-1, 1, 7):
            local = np.array([u, v]) * np.maximum(half - np.max(he) - .01, 0)
            p = dxy + Rz(dyaw)[:2, :2] @ local
            cands.append((min([np.hypot(*(p - o)) for o in inside] + [.06]) - .01 * np.hypot(u, v), p))
    for _s, p in sorted(cands, key=lambda c: -c[0]):  # freest spot the arm can actually reach at carrying height
        t = p + offset
        if env.feasible_yaw([t[0], t[1], env.ee()[2]]) is not None:
            return p
    return dxy


def table_spot(env: World, k: int) -> np.ndarray:
    """Oracle table spot for "out" / "next_to": clear, graspable afterwards, reachable with the held object, and close to
    the destination (a 1.5-3 cm gap), then close to the gripper."""
    sg = env.scene["subgoals"][k]; i = sg["object"]; me = f"object_{i}"
    kind, j = sg["dest"]
    dxy = env.body_xy_yaw(f"container_{j}" if kind == "container" else f"object_{j}")[0]
    lo, hi = np.asarray(REACH_BOX["min"][:2]), np.asarray(REACH_BOX["max"][:2])
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], .01), np.arange(lo[1], hi[1], .01))
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[np.hypot(*(pts - dxy).T) < .20]
    offset = env.ee()[:2] - env.body_xy_yaw(me)[0] if env.held() == me else np.zeros(2)
    cands = []
    for p in pts:
        if not table_spot_ok(env, i, p, sg, .012) or not reachable(grasp_map(), p, .01):
            continue
        # preferred: "out" at least 3 cm clear of the container; "next_to" a gap of at most 2.5 cm
        preferred = table_spot_ok(env, i, p, sg, .03) if sg["relation"] == "out" else env.release_ok(i, p, sg, .025)
        cands.append((not preferred, float(np.hypot(*(p - env.ee()[:2]))), p))
    for _worse, _d, p in sorted(cands, key=lambda c: (c[0], c[1])):
        t = p + offset
        if env.feasible_yaw([t[0], t[1], max(float(env.ee()[2]), CARRY_Z - .012)]) is not None:
            return p
    return dxy


def execute(env: World, step: str, k: int, xy=None, yaw=None, hint: str | None = None) -> dict:
    """Execute one vocabulary step. xy / yaw are the agent's (or oracle's) targets for moves and rotations; `hint` is the
    shape word the agent believes it is grasping (motor sensing: the controller learns nothing else about the held object)."""
    if step in ("move_to_object", "move_to_place"):
        return env.move_xy(xy)
    if step == "rotate":
        return env.rotate(yaw)
    if step == "grasp":
        env.hold_hint = hint or env.scene["objects"][env.scene["subgoals"][k]["object"]]["shape"]
        return env.grasp()
    if step == "lift":
        return env.lift()
    if step == "release":
        return env.place()
    if step == "open":
        return env.open_gripper()
    return {}


def oracle_targets(env: World, step: str, k: int) -> tuple[np.ndarray | None, float | None]:
    sg = env.scene["subgoals"][k]; name = f"object_{sg['object']}"
    if step == "move_to_object":
        return env.body_xy_yaw(name)[0], None
    if step == "move_to_place":
        oxy = env.body_xy_yaw(name)[0]
        return place_spot(env, k) + (env.ee()[:2] - oxy), None
    if step == "rotate":
        spec = env.scene["objects"][sg["object"]]
        oxy, oyaw = env.body_xy_yaw(name)
        if env.held() == name:  # line the held bar up with the tray
            _dxy, dyaw, _h, _s = env.dest_frame(sg["dest"])
            return None, env.gripper_yaw() + wrap(dyaw - oyaw, math.pi)
        return None, yaw_for_object({**spec, "yaw": oyaw})
    return None, None


def run_oracle(scene: dict, max_steps: int = 80, sensing: str = "motor") -> dict:
    """Privileged agent: gold next step + exact targets, with the same sub-task pointer logic as the model agent."""
    env = World(scene, sensing=sensing); env.reset()
    log, pointer, outcome = [], 0, "horizon"
    try:
        for _ in range(max_steps):
            k = current_subgoal(env, pointer)
            if k is None:
                outcome = "done"; break
            f = facts(env, k)
            step = gold_step(f)
            log.append(step)
            if step == "done":
                pointer = k + 1
                if scene.get("sequential"):
                    if pointer >= len(scene["subgoals"]):
                        outcome = "done"; break
                    continue
                while pointer < len(scene["subgoals"]) and env.satisfied(scene["subgoals"][pointer]):
                    pointer += 1
                if pointer >= len(scene["subgoals"]):
                    if all(env.satisfied(sg) for sg in scene["subgoals"]):
                        outcome = "done"; break
                    pointer = next(j for j, sg in enumerate(scene["subgoals"]) if not env.satisfied(sg))
                continue
            if step == "give_up":
                outcome = "give_up"; break
            xy, yaw = oracle_targets(env, step, k)
            execute(env, step, k, xy, yaw)
        ev = env.evaluate()
        return {"success": ev["success"], "outcome": outcome, "steps": len(log), "log": log, "fault": env.fault["event"],
                "fault_fired": env.fault["fired"], "touch": env.touch_log}
    finally:
        env.close()
