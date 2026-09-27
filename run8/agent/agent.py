"""Run 8 agent = the Run 7 agent (run7/agent/agent.py, frozen) on the Run 8 world (calibration marker), plus flags:
  render_samples   Cycles samples per view (Run 8: 16 with OptiX; Runs 5-7: 32)
  selfcal          top-camera self-calibration at episode start: the arm visits K poses, the cyan wrist marker is found in
                   the top image by colour (selfcal_detector="marker"; "sim" = its true projection + noise, CPU only) and
                   a 6-DOF fit corrects the agent's camera pose (the injected cam_err stands for a badly mounted camera)
                   selfcal_focal: 7-DOF, the focal length too (a new camera's field of view is only nominally known);
                   selfcal_grid "wide": poses at 5-10 cm height instead of 6.5/8.5 cm (better-conditioned depth);
                   cam_true_fovy (sensitivity): the true top camera's fovy (the model's camera, so photoreal renders too)
                   differs from the agent's nominal one
                   selfcal_robust: size-aware detection (split marker parts merged; blobs under 15% of the marker's expected
                   pixel area ignored: scene speckles near the prediction were taken as the marker in C3) and iterative
                   outlier rejection; info["accepted"] = >= 9 inliers (>= 6 with 12 poses), RMS <= 2.5 px, correction
                   <= 8 deg / 100 mm (arm day: stop and recalibrate when not accepted)
  wrist_refine     before each grasp: point at the object in the wrist image, map through the wrist camera (known mount,
                   optional wrist_err_deg), and correct the gripper by that relative offset (top-camera errors cancel)
  enumerate        "all the X" sub-tasks: a planner lists the specific objects from the top image (repeat sub-task ->
                   one ordinary sub-task per object), and re-lists once at the end to catch any left outside
Run 8 details: run8/BRIEF.md, run8/DECISIONS.md.

Run 7 agent = the Run 6 agent (run6/agent/agent.py, frozen) on the Run 7 world, plus (run7/BRIEF.md):
  sensing          "motor" (default): the agent reads only what SO-101 servos report (jaw reading -> holding; the
                   release descent stops on the tracking-error signal); "privileged" = Run 6
  follow-ups       a scene's "followup" command is parsed after the first command is done (PARSE_FOLLOWUP), with the
                   completed command as context; pronouns left in a parse ("it") resolve to the previous object
  final re-check   skips earlier sub-tasks whose object a later sub-task moves again (put in, then take out)
  ban_fix          tier 0: the loop guard can never ban every step (Run 6 F6: 102 steps of "invalid plan output")
  noop_move_releases  tier 0: holding, and the pointed release spot is within 1.2 cm of the gripper -> release there
                   (pointing beats the check's over-caution; the mirror of noop_move_grasps)
  grasp_lift       speed: a grasp the jaw reading confirms is followed by the lift in the same step
  place_check_first  speed: right after a carry, ask "over the destination?" first; yes -> release without planning
  cam_err_deg / cam_err_mm / jaw_err_mm   sensitivity tests: a fixed camera-calibration / jaw-offset error per episode
  record_snapshots  per-step simulator snapshots (DAgger re-rendering, tier-2 branching)

Run 6 closed-loop agent = the frozen Run 5 agent (run5/agent/agent.py) plus skill fixes behind flags (run6/BRIEF.md):
  jaw_comp       place the held object, not the gripper site, at the pointed spot: subtract the fixed-jaw offset
                 (a function of the jaw reading; SO-101 has one fixed and one moving jaw)
  bar_two_end    bars: point at both ends, move to their midpoint, rotate to the measured axis (tolerance bar_tol_deg)
  tray_corners   trays: point at the four inner corners, rotate to their axis, move to their centre
  done_margin    confirm "done" only when the calibrated yes-score leads the no-score by this margin
  dest_point     how to point at a block to stack on: top_of (Run 5) | noun | centre_top_face | four_corners |
                 planner (the Run 5 phrase, answered by the planner model, i.e. the LoRA, instead of the locator)
  record_facts   log the simulator's facts behind each state (DAgger labels; never used for decisions)
With every flag off it behaves exactly like the Run 5 agent.

Run 5 closed-loop agent: parse -> (plan -> verify -> act -> observe)* with recovery, loop guard and give-up.

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

import mujoco
import numpy as np
from PIL import Image

from run3.phase2_photoreal import mjexport
from run3.phase3_closed_loop.runner import pixel_to_table
from run4.screen.clients import parse_answer
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciRot

import run8.env.world  # noqa: F401  (calibration marker in every World built by this process)
from run8.env.world import MARKER_RADIUS, marker_world
from run7.agent import prompts as P
from run7.env.policy import STEPS, current_subgoal, execute, facts, gold_step, oracle_targets
from run5.env.render import PhotoRenderer, flat_views, photo_config
from run7.env.world import CARRY_Z, GRASP_Z, World, reach_ok, snapshot, wrap

VIEW_ORDER = ("top", "side", "wrist")


def max_steps(scene: dict) -> int:
    return 20 + 15 * len(scene["subgoals"])


PRONOUNS = {"it", "it again", "them", "that", "this", "the object", "object", "the same object"}


def rotation(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


ENUM_PROMPT = ('A robot arm must carry out this instruction: "{instruction}". Image 1 is the top camera (looking straight down; '
               'the robot base is at the left edge), image 2 the side camera. List every {what} that is not yet inside the {dest}. '
               'Name each one by its colour and shape so that every name is unique (for example "red cube", "red ball"). '
               'If there are none, return an empty list. Answer only with JSON.')
ENUM_SCHEMA = {"type": "object", "properties": {"objects": {"type": "array", "items": {"type": "string"}, "maxItems": 6}},
               "required": ["objects"]}


def apply_fix(cam: dict, x) -> dict:
    """A camera pose corrected by a self-calibration result x = (rotation vector, translation[, log focal scale])."""
    R = SciRot.from_rotvec(np.asarray(x[:3])).as_matrix() @ np.asarray(cam["xmat"]).reshape(3, 3)
    out = {**cam, "xmat": R.ravel().tolist(), "pos": (np.asarray(cam["pos"]) + np.asarray(x[3:6])).tolist()}
    if len(x) > 6:  # focal length f' = f * exp(x6), i.e. tan(fovy'/2) = tan(fovy/2) / exp(x6)
        out["fovy_deg"] = float(np.degrees(2 * np.arctan(np.tan(np.radians(cam["fovy_deg"]) / 2) / np.exp(x[6]))))
    return out


def detect_marker(img: Image.Image, predict=None, window: float = 140.0, min_px: int = 6, merge_px: float = 0.0):
    """Centroid (u, v) of the cyan calibration marker in an RGB image, or None. Marker-coloured pixels (thresholds fitted
    on photoreal renders: hue about 124/255, pale under studio light) are grouped into connected blobs; the blob nearest
    the predicted position (the robot knows roughly where its marker should be) within `window` px is used.
    Robust mode (selfcal_robust): blobs within merge_px of each other are one marker split by an occluding edge, and blobs
    smaller than min_px are speckles."""
    from scipy import ndimage
    hsv = np.asarray(img.convert("HSV")).astype(np.int32)
    mask = (np.abs(hsv[..., 0] - 124) <= 12) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 90)
    lab, n = ndimage.label(mask)
    blobs = []
    for c in range(1, n + 1):
        ys, xs = np.nonzero(lab == c)
        blobs.append([float(xs.mean() + .5), float(ys.mean() + .5), len(xs), float(max(xs.std(), ys.std()))])
    if merge_px > 0:
        merged = []
        for b in sorted(blobs, key=lambda b: -b[2]):
            for m in merged:
                if np.hypot(b[0] - m[0], b[1] - m[1]) <= merge_px:
                    k = m[2] + b[2]; m[0] = (m[0] * m[2] + b[0] * b[2]) / k; m[1] = (m[1] * m[2] + b[1] * b[2]) / k
                    m[2] = k; m[3] = max(m[3], b[3]); break
            else:
                merged.append(list(b))
        blobs = merged
    best = None
    for u, v, npx, spread in blobs:
        if npx < max(6, min_px) or spread > 9:
            continue
        d = 0.0 if predict is None else float(np.hypot(u - predict[0], v - predict[1]))
        if d <= window and (best is None or d < best[0]):
            best = (d, u, v)
    return None if best is None else (best[1], best[2])


def fit_camera_robust(cam0: dict, pts3d, pix, focal: bool = False) -> tuple[np.ndarray, list[int], float]:
    """fit_camera with iterative outlier rejection: drop residuals > max(4 px, 3 x median) and refit (>= 6 kept)."""
    idx = list(range(len(pts3d)))
    x, _ = fit_camera(cam0, pts3d, pix, focal)
    for _ in range(4):
        c = apply_fix(cam0, x)
        r = [float(np.hypot(*(np.asarray(mjexport.project(c, pts3d[i])) - np.asarray(pix[i])))) for i in idx]
        keep = [i for i, ri in zip(idx, r) if ri <= max(4.0, 3 * float(np.median(r)))]
        if len(keep) == len(idx) or len(keep) < 6:
            break
        idx = keep; x, _ = fit_camera(cam0, [pts3d[i] for i in idx], [pix[i] for i in idx], focal)
    c = apply_fix(cam0, x)
    r = [float(np.hypot(*(np.asarray(mjexport.project(c, pts3d[i])) - np.asarray(pix[i])))) for i in idx]
    return x, idx, float(np.sqrt(np.mean(np.square(r))))


def fit_camera(cam0: dict, pts3d, pix, focal: bool = False) -> tuple[np.ndarray, float]:
    """6-DOF (7 with focal) correction of cam0 so the known 3D marker positions reproject onto the detected pixels."""
    def res(x):
        c = apply_fix(cam0, x)
        return np.concatenate([np.asarray(mjexport.project(c, P3)) - np.asarray(uv) for P3, uv in zip(pts3d, pix)])
    n = 7 if focal else 6
    sol = least_squares(res, np.zeros(n), x_scale=[.01] * 6 + [.05] * (n - 6), loss="soft_l1", f_scale=3.0)
    return sol.x, float(np.sqrt(np.mean(sol.fun ** 2)))


SELFCAL_GRID = [(x, y, z) for x in (.14, .19, .24, .29) for y in (-.14, -.05, .05, .14) for z in (.085, .065)]
SELFCAL_GRID_WIDE = [(x, y, z) for x in (.14, .19, .24, .29) for y in (-.14, -.05, .05, .14) for z in (.10, .085, .065, .05)]


def jaw_offset(q: float) -> float:
    """Held-object centre minus gripper site along the jaw axis (m), from the jaw joint reading q. Robot calibration fitted
    on dev seeds 60000-60024 (run6/diag/bar_geometry.py): bar -1.86 cm at q 0.193, cube -1.21 cm at q 0.372."""
    return -.0186 + .0363 * (q - .193)


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


RENDER_SAMPLES = [32]  # set per agent from cfg["render_samples"] (Run 8: 16)


def render_retry(renderer, model, data, pcfg, cameras, seed: int, tries: int = 5) -> dict:
    """A render server can fail transiently (GPU memory pressure; the keeper restarts dead servers): retry with backoff.
    Only infrastructure is retried; the simulation state is untouched, so the episode continues identically."""
    for attempt in range(tries):
        try:
            arr, _ = renderer.views(model, data, pcfg, cameras=cameras, samples=RENDER_SAMPLES[0], seed=seed)
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
        self._env, self._pre_aligned, self._tray_aligned = None, None, None
        self.enum_client, self._point_camera, self._cam_fix, self.verifier = None, "overhead", None, None
        RENDER_SAMPLES[0] = int(cfg.get("render_samples") or 32)

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
    def parse(self, scene: dict, followup: bool = False, previous: list[dict] | None = None) -> tuple[list[dict] | None, dict]:
        if self.cfg.get("parse_mode", "model") == "oracle":
            truth = P.subtasks_from_scene(scene)
            if scene.get("followup"):  # the first command's sub-tasks, then the follow-up's
                n1 = len(truth) - 1
                return (truth[n1:] if followup else truth[:n1]), {"mode": "oracle"}
            return truth, {"mode": "oracle"}
        prompt = P.PARSE_FOLLOWUP.format(previous=scene["instruction"], instruction=scene["followup"]) if followup else \
            P.PARSE_PROMPT.format(instruction=scene["instruction"])
        r = self.parser.generate(prompt, [], max_tokens=300, schema=P.PARSE_SCHEMA,
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
                if obj in PRONOUNS:  # "then take it out": the object of the previous sub-task (or command)
                    prev = clean[-1] if clean else (previous[-1] if previous else None)
                    if prev is not None:
                        info.setdefault("pronoun_resolved", []).append(obj); obj = prev["object"]
                clean.append({"object": obj, "dest": dest, "relation": s["relation"] if s["relation"] in P.RELATIONS else "in",
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
            margin = float(self.cfg.get("done_margin") or 0.0) if kind == "done" else 0.0
            lead = r["calibrated"]["A"] - r["calibrated"]["B"]
            ok = (lead >= margin) if margin else (r["cal_label"] == "A")  # asymmetric: a false "done" ends the sub-task
            info = {"mode": mode, "q": q, "raw_label": r["label"], "label": r["cal_label"], "scores": r["scores"],
                    "lead": lead, "margin": margin, "latency_s": r["latency_s"]}
            if kind == "done" and ok and self.verifier is not None:  # Run 8: a hosted VLM may veto "done" (run8 #11)
                try:
                    v = self.verifier.generate(f"{text}\n\n{q}\nOptions:\nA = yes\nB = no\nThink briefly, then end with one final "
                                               "line 'ANSWER: <letter>'.", ims, max_tokens=1500, reasoning=True)
                except Exception as e:  # noqa: BLE001  API failure or spend cap: keep our checker's answer (logged, counted)
                    v = {"text": "", "latency_s": 0.0, "usd": 0.0, "error": repr(e)[:200]}
                a = parse_answer(v["text"], ["A", "B"])
                info["verifier"] = {"answer": a, "latency_s": v["latency_s"], "usd": v.get("usd"), "model": getattr(self.verifier, "model", None),
                                    "error": v.get("error")}
                info["latency_s"] = float(info["latency_s"]) + float(v["latency_s"])
                if a == "B":
                    ok = False
            return ok, info
        r = self.checker.generate(f"{text}\n\n{q}\nOptions:\nA = yes\nB = no\nThink briefly, then end with one final line 'ANSWER: <letter>'.",
                                  ims, max_tokens=1024, reasoning=True)
        a = parse_answer(r["text"], ["A", "B"])
        return (None if a is None else a == "A"), {"mode": mode, "q": q, "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}

    def _true_top(self, env) -> dict:
        """The true top camera (with cam_true_fovy, run() has set the model's camera to it, so renders use it too)."""
        return mjexport.camera_pose(env.model, env.data, "overhead")

    def _cam_belief(self, env):
        """The camera pose the agent was given (a badly mounted camera = the true pose plus cam_err; nominal fovy)."""
        cam = mjexport.camera_pose(env.model, env.data, "overhead")
        if getattr(self, "_nominal_fovy", None) is not None:
            cam = {**cam, "fovy_deg": self._nominal_fovy}
        if self._cam_err is not None:  # sensitivity test: the agent's camera calibration is off by a fixed error
            R, t = self._cam_err
            cam = {**cam, "xmat": (R @ np.asarray(cam["xmat"]).reshape(3, 3)).ravel().tolist(),
                   "pos": (np.asarray(cam["pos"]) + t).tolist()}
        return cam

    def _cam(self, env):
        cam = self._cam_belief(env)
        if getattr(self, "_cam_fix", None) is not None:  # Run 8 self-calibration result
            cam = apply_fix(cam, self._cam_fix)
        return cam

    def _holding(self, env) -> bool:
        return env.sensed_holding()

    def locate(self, env, subs, k, what: str, top: Image.Image) -> tuple[np.ndarray | None, dict]:
        """what: 'object' | 'place'. Returns the table xy the gripper should move to (Run 6 skills behind flags)."""
        sub = subs[k]
        if self.cfg.get("locate_mode", "point") == "oracle":  # oracle targets already account for the grasp offset
            return self._locate_point(env, subs, k, what, top)
        holding = self._holding(env)
        word = sub["object"].split()[-1]
        if what == "object" and not holding and word == "bar" and self.cfg.get("bar_two_end"):
            return self._locate_bar(env, subs, k, top)
        if what == "place" and holding and sub["dest"].endswith("tray") and self.cfg.get("tray_corners"):
            xy, info = self._locate_tray(env, subs, k, top)
        else:
            xy, info = self._locate_point(env, subs, k, what, top)
        if what == "place" and holding and xy is not None and self.cfg.get("jaw_comp"):
            xy, info = self._jaw_comp(env, xy, info)
        if info.get("mode") == "tray_corners" and xy is not None and info.get("tray_yaw_deg") is not None:
            self._tray_aligned = (np.asarray(xy), math.radians(info["tray_yaw_deg"]))
        return xy, info

    def _jaw_comp(self, env, spot, info: dict) -> tuple[np.ndarray, dict]:
        """Move the gripper so that the held object's centre, not the gripper site, lands on `spot`. The wrist may turn to an
        equivalent yaw at the target (World.move_xy), which moves the offset with it, so the yaw the move will use is solved."""
        q = env.jaw_reading(); off = jaw_offset(q) + float(self.cfg.get("jaw_err_mm") or 0) / 1000; z = float(env.ee()[2])
        g0, cmd = env.gripper_yaw(), env.yaw
        g = g0; spot = np.asarray(spot, float); tgt = spot
        for _ in range(3):
            tgt = spot - off * np.array([math.cos(g), math.sin(g)])
            y = env.feasible_yaw([float(tgt[0]), float(tgt[1]), z])
            g_next = g0 if y is None else g0 + wrap(y - cmd, 2 * math.pi)
            if abs(wrap(g_next - g, 2 * math.pi)) < 1e-6:
                break
            g = g_next
        return tgt, {**info, "jaw_comp": {"q": q, "offset_m": off, "yaw_deg": math.degrees(g), "spot": spot.tolist(),
                                          "target": tgt.tolist()}}

    def _lift_if_low(self, env) -> dict | None:
        return env.lift() if env.ee()[2] < CARRY_Z - .012 else None  # never rotate at table height (DECISIONS #21 spirit)

    def _locate_bar(self, env, subs, k, top) -> tuple[np.ndarray | None, dict]:
        """Bars: point at both ends; go to their midpoint and turn the jaws across the measured axis."""
        sub = subs[k]
        what = f"the two ends of the {sub['object']}"
        r = self.locator.point(what, top, 2)
        info = {"mode": "bar_two_end", "noun": what, "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        pts = [pixel_to_table(self._cam(env), u, v, .045) for u, v in (r["points"] or [])]
        if len(pts) < 2 or any(p is None for p in pts) or np.hypot(*(np.asarray(pts[1]) - pts[0])) < .03:
            xy, i2 = self._locate_point(env, subs, k, "object", top)  # unusable ends: fall back to the single point
            return xy, {**info, "fallback": i2, "error": "ends unusable"}
        a, b = np.asarray(pts[0]), np.asarray(pts[1])
        mid = (a + b) / 2; axis = math.atan2(b[1] - a[1], b[0] - a[0]); yaw = axis + math.pi / 2
        diff = wrap(yaw - env.gripper_yaw(), math.pi)
        info.update(px=r["points"], xy=mid.tolist(), axis_deg=math.degrees(axis), misalignment_deg=math.degrees(diff), rotated=False)
        if abs(diff) > math.radians(float(self.cfg.get("bar_tol_deg", 5))):
            info["lift"] = self._lift_if_low(env)
            info.update(rotated=True, rotate_outcome=env.rotate(env.gripper_yaw() + diff))
        self._pre_aligned = (mid, env.gripper_yaw())
        return mid, info

    def _locate_tray(self, env, subs, k, top) -> tuple[np.ndarray | None, dict]:
        """Trays: point at the four inner corners; turn the held bar to their principal axis; aim at their centre."""
        sub = subs[k]
        r = self.locator.point(f"FOUR:{sub['dest']}", top, 4)
        info = {"mode": "tray_corners", "noun": sub["dest"], "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        pts = [pixel_to_table(self._cam(env), u, v, .01) for u, v in (r["points"] or [])]
        if len(pts) < 4 or any(p is None for p in pts):
            xy, i2 = self._locate_point(env, subs, k, "place", top)
            return xy, {**info, "fallback": i2, "error": "corners unusable"}
        c = np.asarray(pts[:4]); ctr = c.mean(0); cc = c - ctr
        if np.max(np.linalg.norm(cc, axis=1)) < .03:
            xy, i2 = self._locate_point(env, subs, k, "place", top)
            return xy, {**info, "fallback": i2, "error": "corners degenerate"}
        _w, V = np.linalg.eigh(cc.T @ cc); ax = V[:, -1]; axis = math.atan2(ax[1], ax[0])
        yaw = axis + math.pi / 2  # jaws across the bar, bar along the tray (the jaws square a held bar: run6 diagnosis)
        diff = wrap(yaw - env.gripper_yaw(), math.pi)
        info.update(px=r["points"], centre=ctr.tolist(), axis_deg=math.degrees(axis), misalignment_deg=math.degrees(diff), rotated=False)
        if abs(diff) > math.radians(3):
            info["lift"] = self._lift_if_low(env)
            info.update(rotated=True, rotate_outcome=env.rotate(env.gripper_yaw() + diff))
        info["tray_yaw_deg"] = math.degrees(env.gripper_yaw())
        return ctr, info

    def _near(self, env, mark, xy_tol: float) -> bool:
        if mark is None:
            return False
        xy, yaw = mark
        return bool(np.max(np.abs(env.ee()[:2] - xy)) <= xy_tol and abs(wrap(env.gripper_yaw() - yaw, math.pi)) <= math.radians(3))

    def _locate_point(self, env, subs, k, what: str, top: Image.Image) -> tuple[np.ndarray | None, dict]:
        """Run 5 locate: one pointed spot (or the oracle target)."""
        sub = subs[k]
        if self.cfg.get("locate_mode", "point") == "oracle":
            j = env_subgoal(env, subs, k)
            xy, _ = oracle_targets(env, "move_to_object" if what == "object" else "move_to_place", j)
            return np.asarray(xy), {"mode": "oracle"}
        dp = self.cfg.get("dest_point") or "top_of"
        if what == "place" and sub["relation"] == "on" and dp == "four_corners":
            return self._locate_block_corners(env, subs, k, top)
        noun = P.target_noun(sub) if what == "object" else self._dest_noun(sub)
        client = self.planner if (what == "place" and sub["relation"] == "on" and dp == "planner"
                                  and hasattr(self.planner, "point")) else self.locator  # run6 #11: LoRA points at block tops
        r = client.point(noun, top, 1)
        info = {"mode": "point", "noun": noun, "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        if not r["points"]:
            return None, {**info, "px": None}
        u, v = r["points"][0]
        z = .022 if what == "object" else {"in": .01, "on": .035}.get(sub["relation"], 0.0)  # out / next_to: the table
        xy = pixel_to_table(self._cam(env), u, v, z)
        return (None if xy is None else np.asarray(xy)), {**info, "px": [u, v], "xy": None if xy is None else list(map(float, xy))}

    BLOCK_CORNERS = ('Locate the four corners of the top face of the {b} in the image. Output the four points as JSON: '
                     '[{{"point_2d": [x, y], "label": "1"}}, {{"point_2d": [x, y], "label": "2"}}, {{"point_2d": [x, y], "label": "3"}}, '
                     '{{"point_2d": [x, y], "label": "4"}}], coordinates normalized to 0-1000.')

    def _dest_noun(self, sub: dict) -> str:
        """Destination phrase for single-point pointing (Run 6 dest_point option; Run 5 phrase by default)."""
        dp = self.cfg.get("dest_point") or "top_of"
        if sub["relation"] != "on" or dp in ("top_of", "planner"):
            return P.dest_phrase(sub)
        return {"noun": f"the {sub['dest']}", "centre_top_face": f"the centre of the top face of the {sub['dest']}"}[dp]

    def _locate_block_corners(self, env, subs, k, top) -> tuple[np.ndarray | None, dict]:
        """Blocks (stack destination): point at the four corners of the top face; aim at their centre."""
        from run5.agent.models import parse_points
        sub = subs[k]
        r = self.locator.generate(self.BLOCK_CORNERS.format(b=sub["dest"]), [top], max_tokens=300)
        pts_px = parse_points(r["text"], *top.size)
        info = {"mode": "block_corners", "noun": sub["dest"], "text": r["text"][-300:], "latency_s": r["latency_s"], "usd": r.get("usd")}
        pts = [pixel_to_table(self._cam(env), u, v, .045) for u, v in pts_px[:4]]
        if len(pts) < 4 or any(p is None for p in pts):
            noun = P.dest_phrase(sub); r1 = self.locator.point(noun, top, 1)
            if not r1["points"]:
                return None, {**info, "error": "corners unusable", "px": None}
            xy = pixel_to_table(self._cam(env), *r1["points"][0], .035)
            return (None if xy is None else np.asarray(xy)), {**info, "error": "corners unusable; single point", "xy": None if xy is None else list(map(float, xy))}
        ctr = np.mean(np.asarray(pts), 0)
        return ctr, {**info, "px": pts_px[:4], "xy": ctr.tolist()}

    def angle(self, env, subs, k, top: Image.Image) -> tuple[float | None, dict]:
        """Target gripper yaw for a rotate step (None: nothing to line up)."""
        sub = subs[k]
        j = env_subgoal(env, subs, k)
        holding = self._holding(env)
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
        if holding and self.cfg.get("tray_corners") and self._near(env, self._tray_aligned, .015):
            return {"skipped": "aligned by the four-corner move", "rotated": False}  # re-turning here would swing the offset bar
        if not holding and word == "bar" and self.cfg.get("bar_two_end") and self._near(env, self._pre_aligned, .02):
            return {"skipped": "aligned by the two-end move", "rotated": False}
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
        bar_tol = float(self.cfg.get("bar_tol_deg", 5)) if self.cfg.get("bar_two_end") else 40.0
        tol = math.radians(6 if holding else (25 if word == "cube" else bar_tol))
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
        elif self.cfg.get("image_source") == "none":
            arr = {"top": np.zeros((448, 448, 3), np.uint8)}
        else:
            arr = flat_views(env, ("top",))
        top = Image.fromarray(arr["top"])
        top.save(self._out / f"{step_i:03d}_top_lookagain.jpg", quality=85)
        xy, info = self.locate(env, subs, k, what, top)
        return xy, {**info, "park": fb}

    # ------------------------------------------------------------------------------------------ Run 8 skills
    def _selfcal(self, env, seed: int) -> dict:
        """Top-camera self-calibration from the arm's own motion (see module docstring)."""
        K = int(self.cfg.get("selfcal_poses") or 12)
        cam0 = self._cam_belief(env)
        rng = np.random.default_rng([17, seed])
        pts, pix, info = [], [], {"poses": 0, "found": 0}
        t0 = time.time()
        grid = SELFCAL_GRID_WIDE if self.cfg.get("selfcal_grid") == "wide" else SELFCAL_GRID
        robust = bool(self.cfg.get("selfcal_robust"))
        for idx in rng.permutation(len(grid)):
            if len(pts) >= K or info["poses"] >= K + 6:
                break
            x, y, z = grid[idx]
            if not env.solve([x, y, z], env.yaw)["success"]:
                continue
            env._move([x, y, z]); info["poses"] += 1
            P3 = marker_world(env)
            if self._pcfg:
                arr = render_retry(self._renderer, env.model, env.data, self._pcfg, ("top",), seed * 1000 + 700 + info["poses"])
                kw = {}
                if robust:  # the marker's expected pixel radius from the belief camera: r = R f / distance
                    f = cam0["height"] / (2 * np.tan(np.radians(cam0["fovy_deg"]) / 2))
                    r_px = MARKER_RADIUS * f / float(np.linalg.norm(np.asarray(cam0["pos"]) - P3))
                    kw = {"min_px": int(0.15 * np.pi * r_px ** 2), "merge_px": max(6.0, 1.3 * r_px)}
                uv = detect_marker(Image.fromarray(arr["top"]), predict=mjexport.project(cam0, P3), **kw)
            else:  # CPU development: the marker's true projection plus pixel noise
                uv = np.asarray(mjexport.project(self._true_top(env), P3))
                uv = tuple(uv + rng.normal(0, float(self.cfg.get("selfcal_px_noise") or 1.0), 2))
            if uv is None:
                continue
            pts.append(P3); pix.append(uv); info["found"] += 1
        if len(pts) >= 6:
            focal = bool(self.cfg.get("selfcal_focal"))
            if robust:
                x, inl, rms = fit_camera_robust(cam0, pts, pix, focal)
                rot, tr = float(np.degrees(np.linalg.norm(x[:3]))), float(1000 * np.linalg.norm(x[3:6]))
                info.update(inliers=len(inl), accepted=bool(len(inl) >= (9 if K >= 18 else 6) and rms <= 2.5 and rot <= 8 and tr <= 100))
            else:
                x, rms = fit_camera(cam0, pts, pix, focal=focal)
            self._cam_fix = x  # sim: used even when not accepted (arm day: stop instead); counted in the report
            info["fix"] = [float(v) for v in x]  # the fitted correction (offline: table-plane error against the true camera)
            info.update(rot_deg=float(np.degrees(np.linalg.norm(x[:3]))), trans_mm=float(1000 * np.linalg.norm(x[3:6])), rms_px=rms)
            if len(x) > 6:
                info["fovy_deg"] = apply_fix(cam0, x)["fovy_deg"]
        env._move([.23, 0.0, CARRY_Z])
        info["seconds"] = time.time() - t0
        return info

    def _wrist_refine(self, env, sub: dict, wrist_img: Image.Image) -> dict:
        """Before a grasp: find the object in the wrist image and move the gripper by the relative offset."""
        self._point_camera = "wrist_cam"
        try:
            r = self.locator.point(P.target_noun(sub), wrist_img, 1)
        finally:
            self._point_camera = "overhead"
        info = {"latency_s": r.get("latency_s"), "text": (r.get("text") or "")[-80:]}
        if not r.get("points"):
            return {**info, "refined": False, "reason": "no point"}
        wcam = mjexport.camera_pose(env.model, env.data, "wrist_cam")
        if self.cfg.get("wrist_err_deg"):  # an imperfectly known wrist-camera mount
            er = np.random.default_rng([19, self._seed]); ax = er.normal(size=3)
            wcam = {**wcam, "xmat": (rotation(ax, float(self.cfg["wrist_err_deg"])) @ np.asarray(wcam["xmat"]).reshape(3, 3)).ravel().tolist()}
        xy = pixel_to_table(wcam, *r["points"][0], .022)
        if xy is None:
            return {**info, "refined": False, "reason": "no table intersection"}
        d = np.asarray(xy) - env.ee()[:2]
        info.update(offset_mm=[float(1000 * v) for v in d])
        if not (.004 < float(np.hypot(*d)) < .06):
            return {**info, "refined": False, "reason": "offset outside 4-60 mm"}
        info["move"] = env.move_xy(env.ee()[:2] + d)
        return {**info, "refined": True}

    def _enumerate(self, scene: dict, sub: dict, imgs: dict) -> tuple[list[dict] | None, dict]:
        """"All the X": list the specific objects (top + side views); one ordinary sub-task per object."""
        client = self.enum_client or self.parser
        prompt = ENUM_PROMPT.format(instruction=scene["instruction"], what=sub["object"], dest=sub["dest"])
        r = client.generate(prompt, [imgs["top"], imgs["side"]], max_tokens=int(self.cfg.get("enum_max_tokens") or 3000),
                            schema=ENUM_SCHEMA, reasoning=bool(self.cfg.get("enum_reasoning", True)))
        info = {"model": getattr(client, "model", None), "text": (r.get("text") or "")[-300:], "latency_s": r.get("latency_s"), "usd": r.get("usd")}
        try:
            body = r["text"][r["text"].index("{"): r["text"].rindex("}") + 1]
            names = [str(n).strip().lower() for n in json.loads(body)["objects"]]
        except (ValueError, KeyError, TypeError):
            return None, {**info, "error": "unparseable"}
        names = [n[4:] if n.startswith("the ") else n for n in names]
        names = list(dict.fromkeys(n for n in names if n))
        return [{"object": n, "dest": sub["dest"], "relation": "in", "repeat": False, "enumerated": True} for n in names], info

    # ------------------------------------------------------------------------------------------ episode
    def run(self, scene: dict, out_dir: Path, renderer: PhotoRenderer | None = None) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(80_000 + scene["seed"])
        env = World(scene, sensing=self.cfg.get("sensing") or "motor"); env.reset()
        self._env, self._pre_aligned, self._tray_aligned = env, None, None
        self._nominal_fovy = None
        if self.cfg.get("cam_true_fovy"):  # sensitivity: the real top camera's field of view is not the nominal one
            cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
            self._nominal_fovy = float(env.model.cam_fovy[cid]); env.model.cam_fovy[cid] = float(self.cfg["cam_true_fovy"])
        self._cam_err = None
        if self.cfg.get("cam_err_deg") or self.cfg.get("cam_err_mm"):  # fixed magnitude, random direction per episode
            er = np.random.default_rng([13, scene["seed"], len(scene["task"])])
            ax = er.normal(size=3); d = er.normal(size=3); d /= np.linalg.norm(d)
            self._cam_err = (rotation(ax, float(self.cfg.get("cam_err_deg") or 0)), d * float(self.cfg.get("cam_err_mm") or 0) / 1000)
        photo = self.cfg.get("image_source", "photoreal") == "photoreal"
        pcfg = photo_config(scene, allow_heldout=not self.cfg.get("train_looks_only")) if photo else None  # DAgger: training looks only
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
        self._cam_fix, selfcal_info, enum_state = None, None, {"used": False, "orig": None, "relisted": False, "log": []}
        if self.cfg.get("selfcal"):  # Run 8: calibrate the top camera from the arm's own motion before anything else
            selfcal_info = self._selfcal(env, scene["seed"])
        subs, pinfo = self.parse(scene)
        times["model_s"] += pinfo.get("latency_s", 0.0)
        k, last, seen, bans, confirmed = 0, P.START, Counter(), {}, []
        prev_executed, stuck, prev_holding = None, Counter(), False
        out_of_reach_seen = False
        instr, followup_pending, finfo = scene["instruction"], bool(scene.get("followup")), None
        pre_check = None  # place_check_first: the check answer for this step's images, reused if the planner asks again
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
                elif self.cfg.get("image_source") == "none":  # CPU development with simulated components only
                    arr = {v: np.zeros((448, 448, 3), np.uint8) for v in VIEW_ORDER}
                else:
                    arr = flat_views(env, VIEW_ORDER)
                times["render_s"] += time.time() - tr
                imgs = {v: Image.fromarray(arr[v]) for v in VIEW_ORDER}
                for v, im in imgs.items():
                    im.save(out_dir / f"{step_i:03d}_{v}.jpg", quality=85)
                    if self.cfg.get("save_png"):  # DAgger training images, lossless like the Run 5 training renders
                        im.save(out_dir / f"{step_i:03d}_{v}.png")
                ims = [imgs[v] for v in self.views]
                if self.cfg.get("enumerate") and subs and k < len(subs) and subs[k].get("repeat"):
                    tq = time.time()
                    new, einfo = self._enumerate(scene, subs[k], imgs)  # Run 8: "all the X" -> one sub-task per object
                    times["model_s"] += time.time() - tq
                    enum_state["log"].append({"step": step_i, "objects": [n["object"] for n in (new or [])], **einfo})
                    if new:
                        enum_state.update(used=True, orig=subs[k]); subs = subs[:k] + new + subs[k + 1:]
                grip = P.gripper_text(obs["holding"] is not None, obs["gripper_state"] == "closed")
                text = P.context(instr, subs, k, last, grip, self.views)
                rec = {"step": step_i, "subtask": k, "last": last, "gripper": grip, "holding": obs["holding"],
                       "held_true": env.held(),  # log only (scoring / DAgger analysis); decisions use the sensed reading
                       "gold": agent_gold(env, subs, k)[0] if self.cfg.get("record_gold", True) else None}
                if self.cfg.get("record_snapshots"):
                    with (out_dir / "snapshots.jsonl").open("a") as fsn:
                        fsn.write(json.dumps({"step": step_i, "snapshot": snapshot(env)}, default=float) + "\n")
                if self.cfg.get("record_facts"):  # DAgger labels (simulator truth for this state); never used for decisions
                    _gold, f = agent_gold(env, subs, k)
                    rec["facts"] = f; rec["text"] = text
                    rec["proprio"] = {"q": float(env.data.qpos[5]), "gripper_yaw": env.gripper_yaw(), "ee": env.ee().tolist()}
                sig = (k, last, grip)
                holding_now = obs["holding"] is not None
                if holding_now != prev_holding:  # progress: something was picked up or dropped
                    stuck.clear(); prev_holding = holding_now
                banned_now = set(bans.get(sig, set())) | {st for st, n in stuck.items() if n >= 4 and self.cfg.get("guard")}
                if self.cfg.get("ban_fix") and not set(STEPS) - banned_now:  # tier 0: never ban every step; start afresh
                    stuck.clear(); bans.pop(sig, None); banned_now = set(); rec["ban_reset"] = True
                tm = time.time()
                auto_done = None
                pre_check, pre_release = None, False
                if (self.cfg.get("place_check_first") and prev_executed == "move_to_place" and obs["holding"] is not None
                        and "release" not in banned_now):  # speed: straight to the release check after a carry
                    ok_p, pinfo_ = self.check(env, subs, k, "over_dest", text, ims)
                    pre_check = (ok_p, pinfo_); rec["pre_check"] = {"ok": ok_p, **pinfo_}
                    pre_release = ok_p is True
                if self.cfg.get("auto_verify") and prev_executed == "release" and obs["holding"] is None:
                    ok_v, vinfo = self.check(env, subs, k, "done", text, ims)  # DECISIONS #22: verify right after a release
                    rec["auto_verify"] = {"ok": ok_v, **vinfo}
                    if ok_v is True:
                        auto_done = True
                try:
                    if auto_done:
                        step, info = "done", {"mode": "auto_verify", "latency_s": 0.0}
                    elif pre_release:
                        step, info = "release", {"mode": "place_check_first", "latency_s": 0.0}
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
                    ok, cinfo = (pre_check[0], {**pre_check[1], "reused": True}) if pre_check is not None else \
                        self.check(env, subs, k, "over_dest", text, ims)
                    rec["check"] = cinfo
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
                            if any(s2["object"] == subs[kk]["object"] for s2 in subs[kk + 1:]):
                                continue  # a later sub-task moves this object again (put in, then take out)
                            ok2, c2 = self.check(env, subs, kk, "done", P.context(instr, subs, kk, last, grip, self.views), ims)
                            rec.setdefault("final_checks", []).append({"subtask": kk, "ok": ok2, **c2})
                            if ok2 is not True:
                                redo = kk; break
                        if redo is None and followup_pending:  # the user's next command, with the completed one as context
                            followup_pending = False
                            new, finfo = self.parse(scene, followup=True, previous=subs)
                            times["model_s"] += finfo.get("latency_s", 0.0)
                            rec["followup_parse"] = {**finfo, "subtasks": new}
                            if new is None:
                                termination = "parse_error"; finished = True
                            else:
                                subs = subs + new; instr = f"{scene['instruction']} {scene['followup']}"
                                last = f"completed the sub-task: {P.subtask_text(subs[k - 1])}"
                                bans.clear(); stuck.clear()
                        elif redo is None and enum_state["used"] and not enum_state["relisted"] and self.cfg.get("enum_relist"):
                            enum_state["relisted"] = True  # Run 8: list once more; anything still outside becomes a sub-task
                            new, einfo = self._enumerate(scene, enum_state["orig"], imgs)
                            enum_state["log"].append({"step": step_i, "final": True, "objects": [n["object"] for n in (new or [])], **einfo})
                            if new:
                                subs = subs + new; last = f"completed the sub-task: {P.subtask_text(subs[k - 1])}"
                                bans.clear(); stuck.clear()
                            else:
                                termination = "done"; finished = True
                        elif redo is None:
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
                    if (xy is not None and executed == "move_to_place" and sub["relation"] in ("out", "next_to")
                            and not reach_ok(xy, (float(env.ee()[2]),), None)):
                        # table placements (Run 7): a free spot beyond the arm's reach -> the nearest reachable spot on the way
                        # back towards the workspace centre (the release check still verifies it before letting go)
                        xy0 = np.asarray(xy, float); c = np.array([.21, 0.0]); u = (c - xy0) / max(np.hypot(*(c - xy0)), 1e-9)
                        for s_ in np.arange(.01, .16, .01):
                            if reach_ok(xy0 + s_ * u, (float(env.ee()[2]),), None):
                                xy = xy0 + s_ * u; rec["reach_pull"] = {"from": xy0.tolist(), "to": xy.tolist(), "m": float(s_)}; break
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
                        if self.cfg.get("wrist_refine"):
                            rec["wrist_refine"] = self._wrist_refine(env, sub, imgs["wrist"])
                        if self.cfg.get("auto_align"):
                            rec["align"] = self.align(env, subs, k, imgs["top"], holding=False)
                        rec["outcome"] = execute(env, "grasp", k, hint=sub["object"].split()[-1]); last = P.last_text("grasp", sub)
                        last = self._grasp_lift(env, rec, sub, last)
                    elif (self.cfg.get("noop_move_releases") and executed == "move_to_place" and obs["holding"] is not None
                          and np.max(np.abs(np.asarray(xy) - env.ee()[:2])) <= .012):
                        # tier 0 (run7 BRIEF): the pointed release spot is where the gripper already is -> release there
                        executed = rec["executed"] = "release"; rec["noop_release"] = True
                        if self.cfg.get("auto_align"):
                            rec["align"] = self.align(env, subs, k, imgs["top"], holding=True)
                        rec["outcome"] = execute(env, "release", k); last = P.last_text("release", sub)
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
                    if self.cfg.get("wrist_refine") and executed == "grasp" and obs["holding"] is None:
                        rec["wrist_refine"] = self._wrist_refine(env, sub, imgs["wrist"])
                    if self.cfg.get("auto_align") and executed in ("grasp", "release"):
                        tq = time.time()
                        rec["align"] = self.align(env, subs, k, imgs["top"], holding=obs["holding"] is not None)
                        times["model_s"] += time.time() - tq
                    rec["outcome"] = execute(env, executed, k, hint=sub["object"].split()[-1])
                    last = P.last_text(executed, sub)
                    if executed == "grasp":
                        last = self._grasp_lift(env, rec, sub, last)
                times["physics_s"] += time.time() - tp
                rec["fault_fired"] = bool(env.fault["fired"])
                prev_executed = rec.get("executed")
                records.append(rec); self._log(out_dir, rec)
                if finished:
                    break
            ev = env.evaluate()
        finally:
            env.close()
        n_calls = sum(("plan" in r and r["plan"].get("mode") not in ("oracle", "random", "auto_verify", "place_check_first"))
                      + ("check" in r and "mode" in r["check"] and r["check"]["mode"] != "oracle" and not r["check"].get("reused"))
                      + ("locate" in r and r["locate"].get("mode") == "point") + ("angle" in r and r["angle"].get("mode") == "twopoint")
                      + ("plan_after_guard" in r) + len(r.get("final_checks", [])) + ("auto_verify" in r) + ("pre_check" in r)
                      + ("followup_parse" in r and r["followup_parse"].get("mode") == "model")
                      + ("align" in r and r["align"].get("mode") == "twopoint") for r in records)
        summary = {"seed": scene["seed"], "task": scene["task"], "fault": scene.get("fault"), "config": self.cfg,
                   "instruction": scene["instruction"], "followup": scene.get("followup"), "followup_parse": finfo,
                   "subtasks": subs, "parse": pinfo, "sensing": env.sensing, "touch_log": env.touch_log,
                   "seq_progress": ev.get("seq_progress"), "cam_err": self.cfg.get("cam_err_deg") or self.cfg.get("cam_err_mm"),
                   "selfcal": selfcal_info, "enumerate": enum_state["log"] or None,
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

    def _grasp_lift(self, env, rec: dict, sub: dict, last: str) -> str:
        """Speed (grasp_lift): the jaw reading confirms a grasp -> lift in the same step (the only correct next step)."""
        if self.cfg.get("grasp_lift") and env.sensed_holding():
            rec["auto_lift"] = env.lift()
            return P.last_text("lift", sub)
        return last

    @staticmethod
    def _log(out_dir: Path, rec: dict):
        with (out_dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
