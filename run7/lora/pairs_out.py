"""Near-miss twins for take_out (Run 7 brief D; the Run 6 twins, run6/nearmiss/pairs.py, cover the old tuning tasks).

Per clean take_out scene the oracle walks to the release (object held above a free table spot outside the container);
from that one snapshot, twins that share the scene and the photoreal look and differ only in the outcome:
  pre_ok     held above a free spot outside                                     over_dest = yes, gold release
  pre_near   held 0.3-1.2 cm too far towards the container (over its rim)       over_dest = no,  gold move_to_place
  post_ok    released outside                                                   done = yes
  post_near  released 0-1.5 cm too far towards the container (rim / inside)     done = no (kept only if the simulator says so)
Rows use the Run 5 screen-state format plus "pair" / "variant" (render: run7.lora.render_states).
Splits as Run 6: train 80000+, tune 82000+, eval 85000+ (take_out scenes; its task index keeps them apart from Run 6's).
Usage: python -m run7.lora.pairs_out {train|tune|eval} N OUT.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from run7.agent import prompts as P
from run7.agent.agent import agent_gold, env_subgoal, max_steps
from run7.env.policy import execute, oracle_targets, table_spot_ok
from run7.env.world import World, restore, sample_scene, snapshot

BASE = {"train": 80_000, "tune": 82_000, "eval": 85_000}


def targets(env: World, subs, k) -> dict:
    j = env_subgoal(env, subs, k)
    if j is None:
        return {}
    sg = env.scene["subgoals"][j]; i = sg["object"]; spec = env.scene["objects"][i]
    oxy, oyaw = env.body_xy_yaw(f"object_{i}")
    dxy, dyaw, half, _support = env.dest_frame(sg["dest"])
    return {"object_xy": oxy.tolist(), "object_yaw": oyaw, "object_shape": spec["shape"], "object_radius": spec["radius"],
            "dest_xy": dxy.tolist(), "dest_yaw": dyaw, "dest_half": np.asarray(half).tolist(), "relation": sg["relation"],
            "dest_is_tray": False, "held_half": env.half_extents(i)[:2].tolist()}


def row(env, subs, k, task, seed, last, pair, variant) -> dict:
    gold, f = agent_gold(env, subs, k)
    obs = env.observe()
    grip = P.gripper_text(obs["holding"] is not None, obs["gripper_state"] == "closed")
    r = {"task": task, "seed": seed, "fault": None, "t": 0, "instruction": env.scene["instruction"], "subs": subs, "k": k,
         "last": last, "gripper": grip, "gold": gold, "facts": f, "targets": targets(env, subs, k), "fault_fired": False,
         "pair": pair, "variant": variant, "snapshot": snapshot(env)}
    if f is not None:
        r["check_over_dest"] = bool(f["over_dest"]) if f["holding_target"] else None
        r["check_done"] = bool(f["satisfied"]) if not obs["holding"] else None
    r["sid"] = f"nm7-{pair}-{variant}"
    return r


def edge_shift(env: World, j: int, u) -> float | None:
    """Smallest shift of the held object along u after which releasing there no longer counts as 'over the destination'."""
    sg = env.scene["subgoals"][j]; i = sg["object"]
    oxy = env.body_xy_yaw(f"object_{i}")[0]
    for s in np.arange(0, .15, .001):
        if not table_spot_ok(env, i, oxy + s * u, sg, .004):
            return float(s)
    return None


def scene_rows(seed: int, rng, task: str = "take_out") -> list[dict]:
    scene = sample_scene(seed, task)
    subs = P.subtasks_from_scene(scene)
    env = World(scene); env.reset(); envs = [env]
    rows, k = [], 0
    try:
        steps = 0
        while steps < max_steps(scene):
            gold, _f = agent_gold(env, subs, k)
            if gold == "release":
                break
            if gold in ("done", "give_up"):
                return []
            xy, yaw = oracle_targets(env, gold, k) if gold in ("move_to_object", "move_to_place", "rotate") else (None, None)
            execute(env, gold, k, xy, yaw); steps += 1
        else:
            return []
        base = snapshot(env)
        pair = f"{task}-{seed}"
        last_pre, last_post = P.last_text("move_to_place", subs[k]), P.last_text("release", subs[k])
        rows.append(row(env, subs, k, task, seed, last_pre, pair, "pre_ok"))
        cxy = env.body_xy_yaw("container_0")[0]; oxy = env.body_xy_yaw("object_0")[0]
        u = (cxy - oxy) / max(np.hypot(*(cxy - oxy)), 1e-9)  # towards the container
        s_edge = edge_shift(env, k, u)
        if s_edge is not None:
            env = restore(base); envs.append(env); env.move_xy(env.ee()[:2] + (s_edge + rng.uniform(.003, .012)) * u)
            r = row(env, subs, k, task, seed, last_pre, pair, "pre_near")
            if r.get("check_over_dest") is False:
                rows.append(r)
        env = restore(base); envs.append(env); env.place()
        r = row(env, subs, k, task, seed, last_post, pair, "post_ok")
        if r.get("check_done") is True:
            rows.append(r)
        if s_edge is not None:
            env = restore(base); envs.append(env); env.move_xy(env.ee()[:2] + (s_edge + rng.uniform(0, .015)) * u); env.place()
            r = row(env, subs, k, task, seed, last_post, pair, "post_near")
            if r.get("check_done") is False:
                rows.append(r)
    finally:
        for e in envs:
            e.close()
    return rows


def main(split: str, n: int, out: str) -> int:
    rng = np.random.default_rng(BASE[split] + 7)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out, "w") as fh:
        for s in range(n):
            for r in scene_rows(BASE[split] + s, rng):
                fh.write(json.dumps(r, default=float) + "\n")
                counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    print(out, counts, sum(counts.values()), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
