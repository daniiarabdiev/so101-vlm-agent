"""Run 5 closed-loop agent: parse -> (plan -> verify -> act -> observe)* with recovery, loop guard and give-up.

Per episode:
  parse      the instruction -> sub-tasks {object, dest, relation, repeat} (one constrained-JSON call, or oracle)
Per step (images: top / side / wrist):
  plan       one readout over 9 steps (policy.STEPS), calibrated; or oracle / random / free generation (reference)
  guard      the same (sub-task, last step, gripper reading, choice) a third time without progress: that choice is
             banned in that situation and the calibrated runner-up is taken
  verify     release <- "held object over the destination?"; done <- "resting inside / on top?"; give up <- the agent
             points at the object and the reach check must confirm it is out of reach
  act        move_to_object / move_to_place: point on the top view -> table xy (reach-checked before moving);
             rotate: point at two ends / corners -> angle; grasp / lift / release / open: controller macros
Ends when every sub-task is confirmed done (earlier ones are re-checked), give-up is confirmed, or the step limit.
Success is the simulator's ground truth at the end. Episodes are never stopped early by the evaluator.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from run3.phase2_photoreal import mjexport
from run3.phase3_closed_loop.runner import pixel_to_table
from run4.screen.clients import parse_answer
from run5.agent import prompts as P
from run5.env.policy import STEPS, current_subgoal, execute, facts, gold_step, oracle_targets
from run5.env.render import PhotoRenderer, flat_views, photo_config
from run5.env.world import CARRY_Z, GRASP_Z, World, reach_ok

VIEW_ORDER = ("top", "side", "wrist")


def max_steps(scene: dict) -> int:
    return 20 + 15 * len(scene["subgoals"])


def env_subgoal(env: World, subs: list[dict], k: int) -> int | None:
    """Ground-truth sub-goal behind the agent's sub-task k (for oracle components and gold labels)."""
    sgs = env.scene["subgoals"]
    if subs[k].get("repeat"):
        held = env.held()
        for j, sg in enumerate(sgs):
            if held == f"object_{sg['object']}" and not env.satisfied(sg, held):
                return j
        open_ = [j for j, sg in enumerate(sgs) if not env.satisfied(sg, held)]
        return open_[0] if open_ else None
    return k if k < len(sgs) else None


def agent_gold(env: World, subs: list[dict], k: int) -> tuple[str, dict | None]:
    """Correct next step for the agent's sub-task k (and the facts behind it)."""
    j = env_subgoal(env, subs, k)
    if j is None:
        return "done", None
    f = facts(env, j)
    if subs[k].get("repeat") and f["satisfied"]:
        f = dict(f, satisfied=all(env.satisfied(sg) for sg in env.scene["subgoals"]))
    return gold_step(f), f


def render_retry(renderer, model, data, pcfg, cameras, seed: int, tries: int = 5) -> dict:
    """A render server can fail transiently (GPU memory pressure; the keeper restarts dead servers): retry with backoff.
    Only infrastructure is retried; the simulation state is untouched, so the episode continues identically."""
    for attempt in range(tries):
        try:
            arr, _ = renderer.views(model, data, pcfg, cameras=cameras, samples=32, seed=seed)
            return arr
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(15 * (attempt + 1))
            try:
                renderer.set_scene(model, pcfg, force=True)
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError("unreachable")


class Agent:
    def __init__(self, cfg: dict, planner=None, locator=None, checker=None, parser=None, shots: dict | None = None):
        self.cfg, self.planner, self.locator, self.checker, self.parser = cfg, planner, locator, checker, parser
        self.shots = shots or {}
        self.cal = json.loads(Path(cfg["calibration"]).read_text()) if cfg.get("calibration") else {}
        self.views = P.VIEWS[cfg.get("views", "top+side+wrist")]

    # ------------------------------------------------------------------------------------------ helpers
    def _cal(self, key: str, scores: dict) -> dict:
        c = self.cal.get(key)
        if not c:
            return dict(scores)
        off = dict(zip(c["labels"], c["offsets"]))
        return {l: v + off.get(l, 0.0) for l, v in scores.items()}

    def _readout(self, kind: str, text: str, ims, options: dict, question: str, client) -> dict:
        r = client.readout(text, ims, options, question, shots=self.shots.get(kind))
        cal = self._cal(f"{kind}|{self.cfg.get('views', 'top+side+wrist')}", r["scores"])
        return {**r, "calibrated": cal, "cal_label": max(cal, key=cal.get)}

    # ------------------------------------------------------------------------------------------ components
    def parse(self, scene: dict) -> tuple[list[dict] | None, dict]:
        if self.cfg.get("parse_mode", "model") == "oracle":
            return P.subtasks_from_scene(scene), {"mode": "oracle"}
        r = self.parser.generate(P.PARSE_PROMPT.format(instruction=scene["instruction"]), [], max_tokens=300, schema=P.PARSE_SCHEMA,
                                 reasoning=False)  # parsing never uses reasoning (for Qwen and the hosted reference alike)
        info = {"mode": "model", "text": r["text"][-800:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        try:
            body = r["text"][r["text"].index("{"): r["text"].rindex("}") + 1]
            subs = json.loads(body)["subtasks"]
            clean = []
            for s in subs:
                obj, dest = str(s["object"]).strip().lower(), str(s["dest"]).strip().lower()
                for art in ("the ", "a ", "an ", "all the ", "every "):
                    obj = obj[len(art):] if obj.startswith(art) else obj
                    dest = dest[len(art):] if dest.startswith(art) else dest
                if s.get("repeat") and obj.endswith("objects"):
                    obj = obj[:-1]
                clean.append({"object": obj, "dest": dest, "relation": s["relation"] if s["relation"] in ("in", "on") else "in",
                              "repeat": bool(s.get("repeat"))})
            return (clean or None), info
        except (ValueError, KeyError, TypeError):
            return None, {**info, "error": "unparseable"}

    def plan(self, env, subs, k, text, ims, banned: set) -> tuple[str | None, dict]:
        mode = self.cfg["planner_mode"]
        if mode == "oracle":
            return agent_gold(env, subs, k)[0], {"mode": "oracle", "latency_s": 0.0}
        if mode == "random":
            choice = [s for s in STEPS if s not in banned]
            return choice[self._rng.integers(len(choice))], {"mode": "random", "latency_s": 0.0}
        opts = P.next_step_options(subs[k])
        if mode == "readout":
            r = self._readout("next", text, ims, opts, P.NEXT_Q, self.planner)
            allowed = {l: v for l, v in r["calibrated"].items() if STEPS[ord(l) - 65] not in banned}
            lab = max(allowed, key=allowed.get)
            return STEPS[ord(lab) - 65], {"mode": mode, "raw_label": r["label"], "label": lab, "scores": r["scores"],
                                          "calibrated_top": r["cal_label"], "latency_s": r["latency_s"],
                                          "prompt_tokens": r.get("prompt_tokens"), "cached_tokens": r.get("cached_tokens")}
        # gen: the general-model reference answers in free text with reasoning; options banned by the loop guard are not offered
        opts = {k_: v for k_, v in opts.items() if STEPS[ord(k_) - 65] not in banned}
        lines = "\n".join(f"{k_} = {v}" for k_, v in opts.items())
        r = self.planner.generate(f"{text}\n\n{P.NEXT_Q}\nOptions:\n{lines}\nThink briefly about what the images show, then end with "
                                  "one final line 'ANSWER: <letter>'.", ims, max_tokens=2048, reasoning=True)
        a = parse_answer(r["text"], list(opts))
        step = STEPS[ord(a) - 65] if a else None
        if step in banned:
            step = None
        return step, {"mode": mode, "answer": a, "text": r["text"][-500:], "latency_s": r["latency_s"], "usd": r.get("usd")}

    def check(self, env, subs, k, kind: str, text, ims) -> tuple[bool | None, dict]:
        mode = self.cfg.get("check_mode", "readout")
        q = P.check_question(kind, subs[k])
        if mode == "oracle":
            _g, f = agent_gold(env, subs, k)
            truth = (f is None) or (f["over_dest"] if kind == "over_dest" else f["satisfied"])
            return bool(truth), {"mode": mode, "q": q}
        if mode == "readout":
            r = self._readout(f"check_{kind}", text, ims, P.YES_NO, q, self.checker)
            return r["cal_label"] == "A", {"mode": mode, "q": q, "raw_label": r["label"], "label": r["cal_label"], "scores": r["scores"],
                                           "latency_s": r["latency_s"]}
        r = self.checker.generate(f"{text}\n\n{q}\nOptions:\nA = yes\nB = no\nThink briefly, then end with one final line 'ANSWER: <letter>'.",
                                  ims, max_tokens=1024, reasoning=True)
        a = parse_answer(r["text"], ["A", "B"])
        return (None if a is None else a == "A"), {"mode": mode, "q": q, "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}

    def _cam(self, env):
        return mjexport.camera_pose(env.model, env.data, "overhead")

    def locate(self, env, subs, k, what: str, top: Image.Image) -> tuple[np.ndarray | None, dict]:
        """what: 'object' | 'place'. Returns table xy of the pointed spot."""
        sub = subs[k]
        if self.cfg.get("locate_mode", "point") == "oracle":
            j = env_subgoal(env, subs, k)
            xy, _ = oracle_targets(env, "move_to_object" if what == "object" else "move_to_place", j)
            return np.asarray(xy), {"mode": "oracle"}
        noun = P.target_noun(sub) if what == "object" else P.dest_phrase(sub)
        r = self.locator.point(noun, top, 1)
        info = {"mode": "point", "noun": noun, "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        if not r["points"]:
            return None, {**info, "px": None}
        u, v = r["points"][0]
        z = .022 if what == "object" else (.01 if sub["relation"] == "in" else .035)
        xy = pixel_to_table(self._cam(env), u, v, z)
        return (None if xy is None else np.asarray(xy)), {**info, "px": [u, v], "xy": None if xy is None else list(map(float, xy))}

    def angle(self, env, subs, k, top: Image.Image) -> tuple[float | None, dict]:
        """Target gripper yaw for a rotate step (None: nothing to line up)."""
        sub = subs[k]
        j = env_subgoal(env, subs, k)
        holding = env.held() is not None
        if self.cfg.get("angle_mode", "twopoint") == "oracle":
            if j is None:
                return None, {"mode": "oracle"}
            _xy, yaw = oracle_targets(env, "rotate", j)
            return yaw, {"mode": "oracle"}
        shape_word = sub["object"].split()[-1]
        what = P.two_point_target(sub, shape_word, holding)
        if what is None:
            return None, {"mode": "twopoint", "skipped": "nothing to line up"}
        r = self.locator.point(what, top, 2)
        info = {"mode": "twopoint", "what": what, "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        if not r["points"]:
            return None, {**info, "error": "no points"}
        z = .01 if holding else .045
        xy = [pixel_to_table(self._cam(env), u, v, z) for u, v in r["points"]]
        if any(p is None for p in xy):
            return None, {**info, "error": "no table intersection"}
        xy = np.asarray(xy)
        if len(xy) >= 4:  # tray: principal axis of the four inner corners
            c = xy - xy.mean(0); _w, V = np.linalg.eigh(c.T @ c); ax = V[:, -1]
            ang = math.atan2(ax[1], ax[0])
        else:
            a, b = xy[0], xy[1]
            if np.hypot(*(b - a)) < .01:
                return None, {**info, "error": "degenerate points"}
            ang = math.atan2(b[1] - a[1], b[0] - a[0])
        if holding or shape_word == "bar":  # jaws across the long side (a held bar is assumed square in the jaws)
            yaw = ang + math.pi / 2
        else:  # cube: jaws along an edge direction
            yaw = ang
        return yaw, {**info, "axis_deg": math.degrees(ang), "yaw_deg": math.degrees(yaw)}

    def align(self, env, subs, k, top: Image.Image, holding: bool) -> dict:
        """Skill-level alignment (DECISIONS #19-20): measure the angle by two-point pointing and rotate only when the
        gripper is outside the measured grasp / fit tolerance. Uses the wrist angle the arm itself reports."""
        sub = subs[k]
        word = sub["object"].split()[-1]
        if holding and not sub["dest"].endswith("tray"):
            return {"skipped": "no tray"}
        if not holding and word not in ("bar", "cube"):
            return {"skipped": "round object"}
        yaw, info = self.angle(env, subs, k, top)
        if yaw is None:
            return {**info, "rotated": False}
        if holding and self.cfg.get("measure_held"):
            # post hoc only (DECISIONS #35): measure the held bar's own axis instead of assuming it sits square in the jaws
            r = self.locator.point(f"the two ends of the {sub['object']}", top, 2)
            info["held_points"] = r.get("points")
            if r.get("points"):
                a_, b_ = (pixel_to_table(self._cam(env), u, v, .085 - .006) for u, v in r["points"])
                if a_ is not None and b_ is not None and np.hypot(*(np.asarray(b_) - a_)) > .02:
                    bar_axis = math.atan2(b_[1] - a_[1], b_[0] - a_[0])
                    tray_axis = yaw - math.pi / 2  # angle() returned tray axis + 90 deg
                    turn = (tray_axis - bar_axis + math.pi / 2) % math.pi - math.pi / 2
                    info.update(bar_axis_deg=math.degrees(bar_axis), misalignment_deg=math.degrees(turn))
                    if abs(turn) > math.radians(6):
                        info.update(rotated=True, rotate_outcome=env.rotate(env.gripper_yaw() + turn))
                    else:
                        info["rotated"] = False
                    return info
        period = math.pi / 2 if (word == "cube" and not holding) else math.pi
        tol = math.radians(6 if holding else (25 if word == "cube" else 40))
        diff = (yaw - env.gripper_yaw() + period / 2) % period - period / 2
        info.update(misalignment_deg=math.degrees(diff))
        if abs(diff) > tol:
            fb = env.rotate(env.gripper_yaw() + diff)
            info.update(rotated=True, rotate_outcome=fb)
        else:
            info["rotated"] = False
        return info

    def look_again(self, env, subs, k, what: str, step_i: int) -> tuple[np.ndarray | None, dict]:
        """DECISIONS #25: before concluding "out of reach", park the arm out of the top camera's view, re-render the top
        view and point again (the gripper hides objects it is hovering over)."""
        fb = env.park()
        if self._pcfg:
            arr = render_retry(self._renderer, env.model, env.data, self._pcfg, ("top",), self._seed * 1000 + 500 + step_i)
        else:
            arr = flat_views(env, ("top",))
        top = Image.fromarray(arr["top"])
        top.save(self._out / f"{step_i:03d}_top_lookagain.jpg", quality=85)
        xy, info = self.locate(env, subs, k, what, top)
        return xy, {**info, "park": fb}

    # ------------------------------------------------------------------------------------------ episode
    def run(self, scene: dict, out_dir: Path, renderer: PhotoRenderer | None = None) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(80_000 + scene["seed"])
        env = World(scene); env.reset()
        photo = self.cfg.get("image_source", "photoreal") == "photoreal"
        pcfg = photo_config(scene) if photo else None
        if photo:
            for attempt in range(5):
                try:
                    renderer.set_scene(env.model, pcfg, force=attempt > 0); break
                except Exception:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    time.sleep(15 * (attempt + 1))
        self._renderer, self._pcfg, self._seed, self._out = renderer, pcfg, scene["seed"], out_dir
        times = {"model_s": 0.0, "render_s": 0.0, "physics_s": 0.0}
        t0 = time.time(); records = []; termination = "horizon"
        subs, pinfo = self.parse(scene)
        times["model_s"] += pinfo.get("latency_s", 0.0)
        k, last, seen, bans, confirmed = 0, P.START, Counter(), {}, []
        prev_executed, stuck, prev_holding = None, Counter(), False
        out_of_reach_seen = False
        try:
            if subs is None:
                termination = "parse_error"
            for step_i in range(max_steps(scene) if subs else 0):
                if self.cfg.get("retreat_verify") and prev_executed == "release" and env.ee()[2] < CARRY_Z - .012:
                    env.lift()  # post-hoc variant (DECISIONS #29): retreat before looking, so the gripper hides nothing
                obs = env.observe()
                tr = time.time()
                if photo:
                    arr = render_retry(renderer, env.model, env.data, pcfg, VIEW_ORDER, scene["seed"] * 1000 + step_i)
                else:
                    arr = flat_views(env, VIEW_ORDER)
                times["render_s"] += time.time() - tr
                imgs = {v: Image.fromarray(arr[v]) for v in VIEW_ORDER}
                for v, im in imgs.items():
                    im.save(out_dir / f"{step_i:03d}_{v}.jpg", quality=85)
                ims = [imgs[v] for v in self.views]
                grip = P.gripper_text(obs["holding"] is not None, obs["gripper_state"] == "closed")
                text = P.context(scene["instruction"], subs, k, last, grip, self.views)
                rec = {"step": step_i, "subtask": k, "last": last, "gripper": grip, "holding": obs["holding"],
                       "gold": agent_gold(env, subs, k)[0] if self.cfg.get("record_gold", True) else None}
                sig = (k, last, grip)
                holding_now = obs["holding"] is not None
                if holding_now != prev_holding:  # progress: something was picked up or dropped
                    stuck.clear(); prev_holding = holding_now
                banned_now = set(bans.get(sig, set())) | {st for st, n in stuck.items() if n >= 4 and self.cfg.get("guard")}
                tm = time.time()
                auto_done = None
                if self.cfg.get("auto_verify") and prev_executed == "release" and obs["holding"] is None:
                    ok_v, vinfo = self.check(env, subs, k, "done", text, ims)  # DECISIONS #22: verify right after a release
                    rec["auto_verify"] = {"ok": ok_v, **vinfo}
                    if ok_v is True:
                        auto_done = True
                try:
                    if auto_done:
                        step, info = "done", {"mode": "auto_verify", "latency_s": 0.0}
                    else:
                        step, info = self.plan(env, subs, k, text, ims, banned_now)
                except Exception as exc:
                    step, info = None, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                rec["plan"] = info
                if banned_now - set(bans.get(sig, set())):
                    rec["stuck_ban"] = sorted(banned_now - set(bans.get(sig, set())))
                if step is None:
                    rec["outcome"] = "invalid plan output"; last = P.INVALID
                    times["model_s"] += time.time() - tm; records.append(rec); self._log(out_dir, rec); continue
                # loop guard: the same choice in the same situation a third time -> banned there, runner-up instead
                if self.cfg.get("guard") and seen[(sig, step)] >= self.cfg.get("max_repeat", 2):
                    bans.setdefault(sig, set()).add(step)
                    try:
                        step2, info2 = self.plan(env, subs, k, text, ims, bans[sig] | banned_now)
                    except Exception as exc:
                        step2, info2 = None, {"error": str(exc)[:200]}
                    rec["guard"] = {"banned": step, "replacement": step2}
                    rec["plan_after_guard"] = info2
                    step = step2
                    if step is None:
                        rec["outcome"] = "invalid plan output"; last = P.INVALID
                        times["model_s"] += time.time() - tm; records.append(rec); self._log(out_dir, rec); continue
                seen[(sig, step)] += 1
                stuck[step] += 1
                executed, xy, yaw, finished = step, None, None, False
                sub = subs[k]
                if step == "release" and "release" in self.cfg.get("verify_steps", ["release", "done"]):
                    ok, cinfo = self.check(env, subs, k, "over_dest", text, ims); rec["check"] = cinfo
                    if ok is not True:
                        executed = "move_to_place"
                if step == "done" and auto_done:
                    executed = "done"
                elif step == "done":
                    ok, cinfo = (True, {"skipped": True}) if "done" not in self.cfg.get("verify_steps", ["release", "done"]) else \
                        self.check(env, subs, k, "done", text, ims)
                    rec["check"] = cinfo
                    executed = "done" if ok is True else None
                if step == "give_up":  # verify: park, look again, point at the object; the reach check must say out of reach
                    xy, linfo = self.look_again(env, subs, k, "object", step_i) if self.cfg.get("look_again") else \
                        self.locate(env, subs, k, "object", imgs["top"])
                    rec["locate"] = linfo
                    far = xy is None or not reach_ok(xy, (GRASP_Z, CARRY_Z), None)
                    rec["give_up_check"] = {"out_of_reach": bool(far), "xy": None if xy is None else list(map(float, xy))}
                    executed = "give_up" if (far or (out_of_reach_seen and not self.cfg.get("look_again"))) else None
                if executed in ("move_to_object", "move_to_place"):
                    xy, linfo = self.locate(env, subs, k, "object" if executed == "move_to_object" else "place", imgs["top"])
                    rec["locate"] = linfo
                if executed == "rotate":
                    yaw, ainfo = self.angle(env, subs, k, imgs["top"]); rec["angle"] = ainfo
                times["model_s"] += time.time() - tm
                rec["executed"] = executed
                tp = time.time()
                if executed is None and step == "done":
                    last = P.CHECK_FAILED_DONE; rec["outcome"] = "done rejected by the check"
                elif executed is None and step == "give_up":
                    last = "considered giving up, but the object looks reachable"; rec["outcome"] = "give-up rejected"
                elif executed == "give_up":
                    termination = "give_up"; rec["outcome"] = "gave up"; finished = True
                elif executed == "done":
                    confirmed.append(k); k += 1
                    rec["outcome"] = f"sub-task {k} confirmed"
                    if k >= len(subs):  # final pass: re-check the earlier sub-tasks
                        redo = None
                        for kk in range(len(subs) - 1):
                            ok2, c2 = self.check(env, subs, kk, "done", P.context(scene["instruction"], subs, kk, last, grip, self.views), ims)
                            rec.setdefault("final_checks", []).append({"subtask": kk, "ok": ok2, **c2})
                            if ok2 is not True:
                                redo = kk; break
                        if redo is None:
                            termination = "done"; finished = True
                        else:
                            k = redo; last = P.CHECK_FAILED_DONE
                    else:
                        last = f"completed the sub-task: {P.subtask_text(subs[k - 1])}"
                        bans.clear(); stuck.clear()
                elif executed in ("move_to_object", "move_to_place"):
                    if (executed == "move_to_object" and self.cfg.get("look_again") and xy is not None
                            and not reach_ok(xy, (GRASP_Z, CARRY_Z), None)):
                        # DECISIONS #25: an "out of reach" object may just be hidden under the gripper: park, look again, re-point
                        xy2, linfo2 = self.look_again(env, subs, k, "object", step_i)
                        rec["look_again"] = linfo2
                        if xy2 is not None:
                            xy = xy2
                    if xy is None:
                        last = "tried to move, but the location answer could not be used"; rec["outcome"] = "invalid location"
                    elif not (reach_ok(xy, (GRASP_Z, CARRY_Z) if executed == "move_to_object" else (float(env.ee()[2]),), None)
                              or (self.cfg.get("fix_reach_check") and executed == "move_to_place"
                                  and env.feasible_yaw([xy[0], xy[1], float(env.ee()[2])]) is not None)):
                        # (fix_reach_check, post hoc only - DECISIONS #34: also accept the wrist angle the arm is holding)
                        out_of_reach_seen = executed == "move_to_object"
                        last = P.OUT_OF_REACH.format(o=P.target_noun(sub) if executed == "move_to_object" else P.dest_phrase(sub))
                        rec["outcome"] = "out of reach (not moved)"
                    elif (self.cfg.get("noop_move_grasps") and executed == "move_to_object" and obs["holding"] is None
                          and obs["gripper_state"] == "open" and np.max(np.abs(np.asarray(xy) - env.ee()[:2])) <= .012):
                        # DECISIONS #23: the pointer says the gripper is already above the object -> grasp (pointing beats the
                        # planner's alignment judgment)
                        executed = rec["executed"] = "grasp"; rec["noop_move"] = True
                        if self.cfg.get("auto_align"):
                            rec["align"] = self.align(env, subs, k, imgs["top"], holding=False)
                        rec["outcome"] = execute(env, "grasp", k); last = P.last_text("grasp", sub)
                    else:
                        if self.cfg.get("safe_transit") and env.ee()[2] < CARRY_Z - .012:
                            rec["transit_lift"] = env.lift()  # DECISIONS #21: never drag the gripper at table height
                        fb = execute(env, executed, k, xy)
                        rec["outcome"] = fb
                        last = P.last_text(executed, sub) if not (step == "release" and executed == "move_to_place") else \
                            P.CHECK_FAILED_RELEASE.format(d=P.dest_phrase(sub))
                elif executed == "rotate":
                    if yaw is None:
                        rec["outcome"] = "rotation skipped"; last = P.last_text("rotate", sub)
                    else:
                        rec["outcome"] = execute(env, "rotate", k, None, yaw); last = P.last_text("rotate", sub)
                else:
                    if self.cfg.get("auto_align") and executed in ("grasp", "release"):
                        tq = time.time()
                        rec["align"] = self.align(env, subs, k, imgs["top"], holding=obs["holding"] is not None)
                        times["model_s"] += time.time() - tq
                    rec["outcome"] = execute(env, executed, k)
                    last = P.last_text(executed, sub)
                times["physics_s"] += time.time() - tp
                rec["fault_fired"] = bool(env.fault["fired"])
                prev_executed = rec.get("executed")
                records.append(rec); self._log(out_dir, rec)
                if finished:
                    break
            ev = env.evaluate()
        finally:
            env.close()
        n_calls = sum(("plan" in r and r["plan"].get("mode") not in ("oracle", "random")) + ("check" in r and "mode" in r["check"] and r["check"]["mode"] != "oracle")
                      + ("locate" in r and r["locate"].get("mode") == "point") + ("angle" in r and r["angle"].get("mode") == "twopoint")
                      + ("plan_after_guard" in r) + len(r.get("final_checks", [])) + ("auto_verify" in r)
                      + ("align" in r and r["align"].get("mode") == "twopoint") for r in records)
        summary = {"seed": scene["seed"], "task": scene["task"], "fault": scene.get("fault"), "config": self.cfg,
                   "instruction": scene["instruction"], "subtasks": subs, "parse": pinfo,
                   "subtasks_correct": subs == P.subtasks_from_scene(scene) if subs else False,
                   "success": bool(ev["success"]), "subgoals_satisfied": ev["subgoals"], "termination": termination,
                   "steps": len(records), "confirmed": confirmed, "fault_fired": env.fault["fired"], "fault_event": env.fault["event"],
                   "guard_triggers": sum("guard" in r for r in records), "invalid": sum(r.get("outcome") in ("invalid plan output", "invalid location") for r in records),
                   "gold_agreement": float(np.mean([r["executed"] == r["gold"] for r in records if r.get("gold") and r.get("executed")])) if records else None,
                   "model_calls": n_calls, "wall_s": time.time() - t0, **times,
                   "look": pcfg["factors"] if pcfg else None,
                   "usd": sum(float((r.get(x) or {}).get("usd") or 0) for r in records for x in ("plan", "check", "locate", "angle")) + float(pinfo.get("usd") or 0)}
        (out_dir / "episode.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in records))
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        return summary

    @staticmethod
    def _log(out_dir: Path, rec: dict):
        with (out_dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
