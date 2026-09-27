"""Phase 3 closed-loop episodes (runs on a GPU Pod next to the Blender server and vLLM).

Physics, IK, macros, oracle and success scoring are the unchanged harness
(so101_vlm Embodiment/GridAdapter/GridOraclePolicy/tasks.score), wrapped only by
VariantEmbodiment for object shape/colour. The decision-frame image source is either the
Blender Cycles render of the exact state (photoreal) or MuJoCo's own render (flat).

Executors
  grid     Run 2 policy: harness build_grid_input + workspace-crop transform, coarse/fine/macro
           questions answered by categorical readout with a calibration (Run 2 request format).
  point    Split: native pointing ("Point to the {target}.") on the clean overhead image; the
           pixel is intersected with the plane z = POINT_PLANE_Z through the camera model and the
           gripper moves there (same step_target_xy as the grid fine stage); macros are the Run 2
           macro question answered by readout. The target is the goal's object while the robot's
           holding reading is false, else the goal's container (plan step selection from the
           proprioceptive reading shown to the grid policy too).
  oracle / random   harness controls (images rendered but unused).
Invalid model output (unparseable point, missing label scores) ends the episode as a failure.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import random as _random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from run2.experiments.workspace_crop import WorkspaceCropVisualTransform
from run2.grid_calibration import apply_question_calibration, question_stage_for
from run3.phase2_photoreal import mjexport, scene_pool
from run3.phase2_photoreal.render_client import Renderer
from run3.phase2_photoreal.variants import VariantEmbodiment, sample_variant
from run3.pointing import parse_points
from run3.readout import VllmClient
from so101_vlm.grid_actions import GridAdapter
from so101_vlm.grid_oracle import GridOraclePolicy
from so101_vlm.grid_pipeline import build_grid_input
from so101_vlm.tasks import score

POINT_PLANE_Z = 0.02  # fixed nominal grasp-plane height used to back-project pointed pixels
SAMPLES = 64


class InvalidOutput(Exception):
    pass


def pixel_to_table(cam: dict, u: float, v: float, z: float = POINT_PLANE_Z) -> np.ndarray | None:
    R = np.asarray(cam["xmat"], float).reshape(3, 3)
    f = cam["height"] / (2 * math.tan(math.radians(cam["fovy_deg"]) / 2))
    d_cam = np.array([(u - cam["width"] / 2) / f, -(v - cam["height"] / 2) / f, -1.0])
    d = R @ d_cam
    o = np.asarray(cam["pos"], float)
    if abs(d[2]) < 1e-9:
        return None
    t = (z - o[2]) / d[2]
    return (o + t * d)[:2] if t > 0 else None


class Env:
    def __init__(self, seed: int, task: str, image_source: str, renderer: Renderer | None, variant: dict | None = None):
        self.seed, self.task, self.image_source = int(seed), task, image_source
        self.variant = variant or sample_variant(seed)
        self.sim = VariantEmbodiment(variant=self.variant)
        self.sim.reset(self.seed, task)
        obs = self.sim.observe()
        self.scene = scene_pool.sample(self.seed, obs["objects"][0]["color"], obs["containers"][-1]["color"])
        self.scene["factors"].update(shape=self.variant["shape"], object_color=self.variant["object_color"],
                                     container_color=self.variant["container_color"])
        self.renderer = renderer
        if image_source == "photoreal":
            self.renderer.set_scene(self.sim.model, self.scene)

    def overhead(self, decision: int) -> tuple[np.ndarray, float]:
        t0 = time.time()
        if self.image_source == "photoreal":
            img, _, _ = self.renderer.render(self.sim.model, self.sim.data, "photo", SAMPLES, seed=self.seed * 1000 + decision)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            img = scene_pool.camera_realism(buf.getvalue(), self.scene["noise_sigma"], self.scene["jpeg_quality"], self.seed * 1000 + decision)
            arr = np.asarray(img)
        else:
            arr = self.sim.render()["overhead"]
        return arr, time.time() - t0

    def close(self):
        self.sim.close()


def readout_questions(client: VllmClient, calibration: dict | None, prompt: str, images, questions) -> tuple[list, list, float]:
    answers, raw, t = [], [], 0.0
    for q in questions:
        res = client.readout(prompt, images, q)
        t += res["latency_s"]
        scores = res["label_scores"]
        label = res["label"]
        if calibration is not None:
            stage = question_stage_for(q, questions)
            try:
                _, label, _ = apply_question_calibration(calibration, stage, q, scores)
            except ValueError:
                pass  # no calibration key for this question: raw answer
        answers.append(q["options"][label])
        raw.append({"kind": q["kind"], "raw_label": res["label"], "label": label, "scores": scores, "latency_s": res["latency_s"]})
    return answers, raw, t


def run_episode(executor: str, seed: int, task: str, image_source: str, out_dir: Path, *, renderer=None,
                grid_client=None, grid_cal=None, point_client=None, macro_client=None, macro_cal=None,
                max_decisions: int = 30, save_images: bool = True, variant: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Env(seed, task, image_source, renderer, variant)
    adapter = GridAdapter()
    oracle = GridOraclePolicy(adapter) if executor == "oracle" else None
    rng = _random.Random(seed)
    transform = WorkspaceCropVisualTransform()
    records, termination, invalid = [], "horizon", None
    point_stage = executor == "point"
    t_start = time.time()
    try:
        for decision in range(max_decisions):
            obs = env.sim.observe()
            if score(obs)["success"]:
                termination = "state_success"; break
            img, render_s = env.overhead(decision)
            if save_images:
                Image.fromarray(img).save(out_dir / f"{decision:03d}.jpg", quality=88)
            rec = {"decision": decision, "holding": obs.get("holding"), "gripper_state": obs.get("gripper_state"),
                   "render_s": render_s, "model_s": 0.0}
            try:
                if executor in ("oracle", "random"):
                    qs = adapter.questions(obs)
                    if executor == "oracle":
                        choice = oracle.choose(obs)
                    else:
                        picks = [rng.choice(list(q.options.values())) for q in qs]
                        choice = tuple(picks) if len(picks) > 1 else picks[0]
                    rec["stage"] = adapter.stage(obs)
                elif executor == "grid":
                    built = build_grid_input(env.sim, obs, {"overhead": img}, adapter, visual_transform=transform)
                    qs = built["questions"]
                    answers, raw, model_s = readout_questions(grid_client, grid_cal, built["prompt"], built["images"], qs)
                    choice = tuple(answers) if len(answers) > 1 else answers[0]
                    rec.update(stage=built["stage"], answers=raw, model_s=model_s)
                elif executor == "point":
                    if point_stage:
                        target = obs["containers"][-1] if obs.get("holding") else None
                        noun = obs["goal"].split(" into the ")[1].rstrip(".").split(".")[0] if obs.get("holding") else \
                            obs["goal"].split("Put the ")[1].split(" into the ")[0]
                        res = point_client.generate(f"Point to the {noun}.", [Image.fromarray(img)], max_tokens=48)
                        rec.update(stage="point", target_name=noun, point_text=res["text"], model_s=res["latency_s"])
                        pts = parse_points(res["text"], img.shape[1], img.shape[0])
                        if not pts:
                            raise InvalidOutput(f"unparseable point: {res['text'][:80]!r}")
                        cam = mjexport.camera_pose(env.sim.model, env.sim.data)
                        xy = pixel_to_table(cam, *pts[0])
                        if xy is None:
                            raise InvalidOutput("point does not intersect the table plane")
                        rec.update(point_px=list(pts[0]), target_xy=xy.tolist())
                        t0 = time.time()
                        fb = env.sim.step_target_xy(xy)
                        rec.update(physics_s=time.time() - t0, feedback={k: fb[k] for k in ("clamped", "stalled", "holding") if k in fb})
                        adapter._stage = "macro"  # pointing replaces coarse+fine; macros are the harness's
                        point_stage = False
                        records.append(rec); continue
                    built = build_grid_input(env.sim, obs, {"overhead": img}, adapter, visual_transform=transform)
                    qs = built["questions"]
                    answers, raw, model_s = readout_questions(macro_client, macro_cal, built["prompt"], built["images"], qs)
                    choice = answers[0]
                    rec.update(stage="macro", answers=raw, model_s=model_s)
                else:
                    raise ValueError(executor)
            except InvalidOutput as exc:
                invalid = str(exc); termination = "invalid_output"; rec["invalid"] = invalid
                records.append(rec); break
            except (ValueError, KeyError) as exc:
                invalid = f"{type(exc).__name__}: {exc}"; termination = "invalid_output"; rec["invalid"] = invalid
                records.append(rec); break
            rec["choice"] = list(choice) if isinstance(choice, tuple) else choice
            t0 = time.time()
            fb = adapter.apply(env.sim, choice, obs)
            rec["physics_s"] = time.time() - t0
            rec["feedback"] = {k: fb.get(k) for k in ("physics_step", "stalled", "clamped", "holding", "grid_target_xy")}
            if executor == "point" and adapter.stage(obs) == "coarse":
                point_stage = True  # lift / reselect return to the pointing stage
            records.append(rec)
            after = env.sim.observe()
            if score(after)["success"]:
                termination = "state_success"; break
            if choice == "done":
                termination = "done"; break
        final_obs = env.sim.observe()
        final = score(final_obs)
    finally:
        env.close()
    summary = {"executor": executor, "seed": seed, "task": task, "image_source": image_source,
               "success": bool(final["success"]), "reason": final["reason"], "termination": termination,
               "invalid": invalid, "decisions": len(records), "wall_s": time.time() - t_start,
               "model_s": sum(r.get("model_s", 0) for r in records), "render_s": sum(r.get("render_s", 0) for r in records),
               "physics_s": sum(r.get("physics_s", 0) for r in records), "variant": env.variant,
               "scene_factors": env.scene["factors"], "goal": final_obs.get("goal"),
               "ever_held": any(r.get("holding") for r in records) or bool(final_obs.get("holding"))}
    (out_dir / "episode.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in records))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
