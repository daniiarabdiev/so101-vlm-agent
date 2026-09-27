"""Tier 2 (run7/BRIEF.md D): step-level outcome labels from simulator branching. CPU MuJoCo only; no model.

For every saved state (a snapshot; the agent's own DAgger states or replay/twin states):
  over_outcome   holding the sub-task object: restore, release right here, let it settle -> does the sub-goal hold?
                 ("would releasing now succeed", instead of the rule's geometric margin)
  q              DAgger states only: for each candidate step (gold, the agent's choice, and the plausible alternatives),
                 restore, execute it (move / rotate targets with the pointing noise the agent has: 6 mm, 3 deg; two draws),
                 continue with the privileged oracle for up to 25 steps, score = success - 0.02 * steps. The label is the
                 best-scoring step (ties -> gold). "done" is a valid candidate only when the sub-task is satisfied (the agent's
                 check rejects a false "done"; unscored, it would beat give-up and tie real work: run7 DECISIONS #13).
Restored states never inject faults. Output: one JSON line per state id (sid).
Usage: python -m run7.lora.outcome states STATES.jsonl OUT.jsonl [--procs N]
       python -m run7.lora.outcome dagger CELL_DIR OUT.jsonl [--procs N]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import zlib
from pathlib import Path

import numpy as np

from run7.env.policy import STEPS, current_subgoal, execute, facts, gold_step, oracle_targets
from run7.env.world import restore

NOISE_M, NOISE_DEG, STEP_COST, BUDGET, DRAWS = .006, 3.0, .02, 25, 2


def release_outcome(snap: dict, j: int) -> bool | None:
    env = restore(snap)
    try:
        sg = env.scene["subgoals"][j]
        if env.held() != f"object_{sg['object']}":
            return None
        env.place(); env._advance(.5)
        return bool(env.satisfied(sg))
    finally:
        env.close()


def oracle_rest(env, pointer: int, budget: int) -> tuple[bool, int]:
    """The run_oracle loop on an existing env, from sub-goal pointer; returns (success, steps used)."""
    sgs = env.scene["subgoals"]; steps = 0
    for _ in range(budget):
        k = current_subgoal(env, pointer)
        if k is None:
            break
        f = facts(env, k); step = gold_step(f)
        if step == "done":
            pointer = k + 1
            if env.scene.get("sequential"):
                if pointer >= len(sgs):
                    break
                continue
            while pointer < len(sgs) and env.satisfied(sgs[pointer]):
                pointer += 1
            if pointer >= len(sgs):
                if all(env.satisfied(sg) for sg in sgs):
                    break
                pointer = next(j for j, sg in enumerate(sgs) if not env.satisfied(sg))
            continue
        if step == "give_up":
            return False, steps
        xy, yaw = oracle_targets(env, step, k)
        execute(env, step, k, xy, yaw); steps += 1
    return bool(env.evaluate()["success"]), steps


def q_value(snap: dict, j: int, step: str, budget: int, rng) -> float:
    n_sub = len(snap["scene"]["subgoals"])
    if step == "give_up":
        return -STEP_COST
    scores = []
    draws = DRAWS if step in ("move_to_object", "move_to_place", "rotate") else 1
    for _ in range(draws):
        env = restore(snap)
        try:
            if step == "done":
                if j + 1 >= n_sub:
                    ok, n = bool(env.evaluate()["success"]), 0
                else:
                    ok, n = oracle_rest(env, j + 1, budget)
                scores.append(ok - STEP_COST * n); continue
            xy, yaw = oracle_targets(env, step, j)
            if xy is not None:
                xy = np.asarray(xy) + rng.normal(0, NOISE_M, 2)
            if yaw is not None:
                yaw = yaw + math.radians(rng.normal(0, NOISE_DEG))
            if step in ("move_to_object", "move_to_place") and xy is None:
                scores.append(-1.0); continue
            execute(env, step, j, xy, yaw)
            ok, n = oracle_rest(env, j, budget - 1)
            scores.append(ok - STEP_COST * (n + 1))
        finally:
            env.close()
    return float(np.mean(scores))


def q_label(q: dict, gold: str | None, satisfied: bool) -> str:
    """Best-scoring step; ties go to gold. "done" only counts when the sub-task is satisfied and "give_up" only when the rule
    says the object is out of reach (gold give_up): the agent verifies both, and an unverifiable end would win on step cost."""
    q = {c: v for c, v in q.items() if (c != "done" or satisfied) and (c != "give_up" or gold == "give_up")}
    best = max(q.values())
    return gold if gold in q and q[gold] >= best - 1e-9 else max(q, key=q.get)


def label_state(item: dict) -> dict:
    sid, snap, j = item["sid"], item["snapshot"], item["j"]
    out = {"sid": sid}
    try:
        if item.get("holding_target"):
            out["over_outcome"] = release_outcome(snap, j)
        if item.get("candidates"):
            rng = np.random.default_rng(zlib.crc32(sid.encode()))
            q = {c: q_value(snap, j, c, item.get("budget", BUDGET), rng) for c in item["candidates"]}
            out["q"] = q
            out["q_label"] = q_label(q, item.get("gold"), item.get("satisfied", True))
    except Exception as exc:  # noqa: BLE001  (kept as missing; the builder falls back to the rule label)
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def dagger_items(cell: Path) -> list[dict]:
    items = []
    for summ in sorted(cell.glob("*/*/seed_*/summary.json")):
        s = json.loads(summ.read_text())
        if not s.get("subtasks_correct") or s["task"] == "gather":
            continue
        d = summ.parent
        snaps = {}
        if (d / "snapshots.jsonl").exists():
            for line in (d / "snapshots.jsonl").read_text().splitlines():
                r = json.loads(line); snaps[r["step"]] = r["snapshot"]
        n_max = 20 + 15 * len(s["subtasks"])
        for line in (d / "episode.jsonl").read_text().splitlines():
            rec = json.loads(line)
            f = rec.get("facts")
            if rec["step"] not in snaps or f is None or rec.get("gold") is None:
                continue
            plan = rec.get("plan") or {}
            planned = "done" if plan.get("mode") == "auto_verify" else ("release" if plan.get("mode") == "place_check_first" else
                                                                        (STEPS[ord(plan["label"]) - 65] if plan.get("label") else None))
            cands = {rec["gold"]}
            if planned:
                cands.add(planned)
            if rec.get("executed") in STEPS:
                cands.add(rec["executed"])
            if f["holding_target"]:
                cands |= {"release", "move_to_place"}
            elif not f["holding"]:
                cands.add("done")
                if f["closed_empty"]:
                    cands.add("open")
            sid = f"dagger7|{s['task']}|{s.get('fault') or 'clean'}|{s['seed']}|{rec['step']}"
            items.append({"sid": sid, "snapshot": snaps[rec["step"]], "j": f["subgoal"], "gold": rec["gold"], "satisfied": bool(f["satisfied"]),
                          "holding_target": f["holding_target"], "candidates": sorted(cands),
                          "budget": max(3, min(BUDGET, n_max - rec["step"] - 1))})
    return items


def state_items(path: Path) -> list[dict]:
    items = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        f = r.get("facts")
        if f is None or not f.get("holding_target"):
            continue
        items.append({"sid": r["sid"], "snapshot": r["snapshot"], "j": f["subgoal"], "holding_target": True})
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["states", "dagger"]); ap.add_argument("src", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() // 2)); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = dagger_items(a.src) if a.mode == "dagger" else state_items(a.src)
    if a.limit:
        items = items[:a.limit]
    done = set()
    if a.out.exists():
        done = {json.loads(l)["sid"] for l in a.out.read_text().splitlines() if l.strip()}
    items = [it for it in items if it["sid"] not in done]
    print(json.dumps({"todo": len(items), "already": len(done), "procs": a.procs}), flush=True)
    with mp.Pool(a.procs) as pool, a.out.open("a") as fh:
        for n, r in enumerate(pool.imap_unordered(label_state, items, chunksize=2), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if n % 100 == 0:
                print(json.dumps({"done": n, "of": len(items)}), flush=True)
    print("OUTCOME_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
