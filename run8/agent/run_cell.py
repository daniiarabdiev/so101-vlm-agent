"""Run 8 closed-loop cell: the Run 7 runner with the Run 8 agent and flags (render samples, JSON pointing, self-calibration,
wrist refinement, "all the X" enumeration with a local or hosted planner).

Run 7 closed-loop cell: the Run 6 runner with the Run 7 agent and world (new tasks, motor sensing, tier-0 / speed flags).
Each task is tested on clean scenes and on the faults that apply to it (world.TASK_FAULTS).

Run 6 closed-loop cell: the Run 5 runner (run5/agent/run_cell.py) with the Run 6 agent and its skill flags.

Test set (fixed before any model result): per task, 30 clean scenes (seeds 40000-40029) and 8 scenes per fault type
(seeds 41000 + 100 * fault_index + 0..7). Episode folders: OUT/<task>/<clean|fault>/seed_<seed>/.
Parallel processes take disjoint shards (--shard i/n) of the same deterministic job list; finished episodes are skipped.
Every attempted episode is kept: a harness exception writes error.json and counts as a failure.

Usage: python -m run5.agent.run_cell OUT --planner-mode readout --shard 0/4 [--tasks place_in,stack] [--conditions clean,F1_missed_grasp]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from run7.env.world import FAULTS, TASK_FAULTS, TASKS, sample_scene

CLEAN_SEEDS = range(40_000, 40_030)


def fault_seeds(fault: str, n: int = 8):
    return range(41_000 + 100 * FAULTS.index(fault), 41_000 + 100 * FAULTS.index(fault) + n)


def jobs(tasks, conditions, n_clean=30, n_fault=8) -> list[tuple[str, str, int]]:
    out = []
    for t in tasks:
        for c in conditions:
            if c != "clean" and c not in TASK_FAULTS[t]:
                continue
            seeds = list(CLEAN_SEEDS)[:n_clean] if c == "clean" else list(fault_seeds(c, n_fault))
            out += [(t, c, s) for s in seeds]
    return out


def build_agent(a):
    from run8.agent.agent import Agent
    from run5.agent.models import HOSTED, Hosted
    from run8.agent.models import QwenFast as Qwen
    cfg = {k: getattr(a, k) for k in ("planner_mode", "locate_mode", "check_mode", "parse_mode", "angle_mode", "views", "calibration",
                                      "guard", "max_repeat", "image_source", "model", "fewshot", "auto_align", "auto_verify",
                                      "safe_transit", "noop_move_grasps", "look_again", "retreat_verify", "fix_reach_check",
                                      "measure_held", "jaw_comp", "bar_two_end", "tray_corners", "done_margin", "bar_tol_deg",
                                      "record_facts", "train_looks_only", "save_png", "dest_point", "sensing", "ban_fix",
                                      "noop_move_releases", "grasp_lift", "place_check_first", "cam_err_deg", "cam_err_mm",
                                      "jaw_err_mm", "record_snapshots", "render_samples", "point_json", "selfcal", "selfcal_poses",
                                      "selfcal_px_noise", "selfcal_focal", "selfcal_grid", "selfcal_robust", "cam_true_fovy", "wrist_refine", "wrist_err_deg", "enumerate", "enum_model", "enum_reasoning",
                                      "enum_relist", "verifier_model")}
    cfg["verify_steps"] = a.verify_steps.split(",") if a.verify_steps else []
    need_model = any(getattr(a, k) in ("readout", "gen", "point", "twopoint", "model")
                     for k in ("planner_mode", "check_mode", "locate_mode", "angle_mode", "parse_mode"))
    if a.sim_pointer is not None and not any(getattr(a, k) in ("readout", "gen", "model")
                                             for k in ("planner_mode", "check_mode", "parse_mode")):
        need_model = False  # CPU development: every model component is simulated or oracle
    m = None
    if need_model:
        m = Hosted(a.model) if a.model == "gemini38flash" else Qwen(a.model, a.base)
    loc = Qwen(a.locator_model, a.base, point_json=a.point_json) if a.locator_model else m  # hybrid (#38); Run 8: JSON points
    shots = None
    if a.fewshot:
        from run5.fewshot.library import load_shots
        shots = load_shots(Path(a.fewshot), a.views)
    cfg["locator_model"] = a.locator_model
    agent = Agent(cfg, planner=m, locator=loc, checker=m, parser=m, shots=shots)
    if a.verifier_model:  # Run 8: a hosted VLM that may veto "done" (the check must also say yes)
        HOSTED[a.verifier_model] = a.verifier_model; agent.verifier = Hosted(a.verifier_model)
    if a.enumerate and a.enum_model:  # Run 8: the "all the X" planner (local base model with reasoning, or a hosted model)
        if a.enum_model == "local":
            agent.enum_client = Qwen(a.locator_model or a.model, a.base)
        else:
            HOSTED[a.enum_model] = a.enum_model; agent.enum_client = Hosted(a.enum_model)
    if a.sim_pointer is not None:  # dev seeds only: pointing answered from simulator truth + noise
        from run8.agent.simpointer import SimPointer
        agent.locator = SimPointer(agent, a.sim_pointer)
    return agent


def make_parser() -> argparse.ArgumentParser:
    """The run_cell command line (also used by real/run_real.py to build the same agent)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--conditions", default="clean," + ",".join(FAULTS))
    ap.add_argument("--n-clean", type=int, default=30); ap.add_argument("--n-fault", type=int, default=8)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--planner-mode", dest="planner_mode", default="readout")
    ap.add_argument("--locate-mode", dest="locate_mode", default="point")
    ap.add_argument("--check-mode", dest="check_mode", default="readout")
    ap.add_argument("--parse-mode", dest="parse_mode", default="model")
    ap.add_argument("--angle-mode", dest="angle_mode", default="twopoint")
    ap.add_argument("--views", default="top+side+wrist")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--guard", action="store_true"); ap.add_argument("--max-repeat", dest="max_repeat", type=int, default=2)
    ap.add_argument("--auto-align", dest="auto_align", action="store_true")
    ap.add_argument("--auto-verify", dest="auto_verify", action="store_true")
    ap.add_argument("--safe-transit", dest="safe_transit", action="store_true")
    ap.add_argument("--noop-move-grasps", dest="noop_move_grasps", action="store_true")
    ap.add_argument("--look-again", dest="look_again", action="store_true")
    ap.add_argument("--retreat-verify", dest="retreat_verify", action="store_true")
    ap.add_argument("--fix-reach-check", dest="fix_reach_check", action="store_true")
    ap.add_argument("--measure-held", dest="measure_held", action="store_true")
    ap.add_argument("--jaw-comp", dest="jaw_comp", action="store_true", help="Run 6: fixed-jaw offset compensation")
    ap.add_argument("--bar-two-end", dest="bar_two_end", action="store_true", help="Run 6: two-end bar grasp")
    ap.add_argument("--tray-corners", dest="tray_corners", action="store_true", help="Run 6: four-corner tray placement")
    ap.add_argument("--bar-tol-deg", dest="bar_tol_deg", type=float, default=5.0)
    ap.add_argument("--done-margin", dest="done_margin", type=float, default=0.0)
    ap.add_argument("--dest-point", dest="dest_point", default="top_of", choices=["top_of", "noun", "centre_top_face", "four_corners", "planner"])
    ap.add_argument("--record-facts", dest="record_facts", action="store_true", help="log simulator facts (DAgger labels)")
    ap.add_argument("--train-looks-only", dest="train_looks_only", action="store_true", help="DAgger: no held-out looks")
    ap.add_argument("--save-png", dest="save_png", action="store_true", help="DAgger: also save lossless PNG views")
    ap.add_argument("--sim-pointer", dest="sim_pointer", type=float, default=None, help="dev only: simulated pointing sd (m)")
    ap.add_argument("--sensing", default="motor", choices=["motor", "privileged"], help="Run 7: what the agent can read")
    ap.add_argument("--ban-fix", dest="ban_fix", action="store_true", help="Run 7 tier 0: never ban every step")
    ap.add_argument("--noop-move-releases", dest="noop_move_releases", action="store_true", help="Run 7 tier 0")
    ap.add_argument("--grasp-lift", dest="grasp_lift", action="store_true", help="Run 7 speed: lift right after a sensed grasp")
    ap.add_argument("--place-check-first", dest="place_check_first", action="store_true", help="Run 7 speed")
    ap.add_argument("--cam-err-deg", dest="cam_err_deg", type=float, default=0.0)
    ap.add_argument("--cam-err-mm", dest="cam_err_mm", type=float, default=0.0)
    ap.add_argument("--jaw-err-mm", dest="jaw_err_mm", type=float, default=0.0)
    ap.add_argument("--record-snapshots", dest="record_snapshots", action="store_true", help="per-step snapshots (DAgger/tier 2)")
    ap.add_argument("--render-samples", dest="render_samples", type=int, default=32)
    ap.add_argument("--point-json", dest="point_json", action="store_true", help="Run 8: JSON-constrained single points")
    ap.add_argument("--selfcal", action="store_true", help="Run 8: top-camera self-calibration from the wrist marker")
    ap.add_argument("--selfcal-poses", dest="selfcal_poses", type=int, default=12)
    ap.add_argument("--selfcal-px-noise", dest="selfcal_px_noise", type=float, default=1.0, help="CPU only")
    ap.add_argument("--selfcal-focal", dest="selfcal_focal", action="store_true", help="7-DOF self-calibration (+ focal length)")
    ap.add_argument("--selfcal-grid", dest="selfcal_grid", default="standard", choices=["standard", "wide"])
    ap.add_argument("--selfcal-robust", dest="selfcal_robust", action="store_true", help="size-aware marker detection + outlier rejection")
    ap.add_argument("--cam-true-fovy", dest="cam_true_fovy", type=float, default=None, help="CPU sensitivity only: true top-camera fovy")
    ap.add_argument("--wrist-refine", dest="wrist_refine", action="store_true", help="Run 8: wrist-camera correction before grasps")
    ap.add_argument("--wrist-err-deg", dest="wrist_err_deg", type=float, default=0.0)
    ap.add_argument("--enumerate", action="store_true", help="Run 8: list the objects for 'all the X' sub-tasks")
    ap.add_argument("--enum-model", dest="enum_model", default="local", help="local | an OpenRouter model id")
    ap.add_argument("--enum-reasoning", dest="enum_reasoning", type=int, default=1)
    ap.add_argument("--enum-relist", dest="enum_relist", action="store_true", help="final re-listing (off: harmful in C1, run8 #11)")
    ap.add_argument("--verifier-model", dest="verifier_model", default=None, help="OpenRouter id of a hosted 'done' veto")
    ap.add_argument("--locator-model", dest="locator_model", default=None, help="separate model for pointing / angles")
    ap.add_argument("--rerun-errors", default=None, help="only re-run episodes that ended in error.json; their folders move here first")
    ap.add_argument("--resume-all", default=None, help="run every job not finished; error folders move here first (#28)")
    ap.add_argument("--exclude-list", default=None, help="file of task/cond/seed_N lines never to run here (handled on another Pod)")
    ap.add_argument("--seed-offset", type=int, default=0, help="post-hoc sets: shift every test seed (e.g. 2000 -> 42000+)")
    ap.add_argument("--verify-steps", dest="verify_steps", default="release,done")
    ap.add_argument("--image", dest="image_source", default="photoreal")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--fewshot", default=None)
    ap.add_argument("--render-url", default="http://localhost:8002")
    ap.add_argument("--base", default=None, help="vLLM base URL")
    return ap


def main() -> int:
    ap = make_parser()
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    todo = [(t, c, s + a.seed_offset) for t, c, s in jobs(a.tasks.split(","), a.conditions.split(","), a.n_clean, a.n_fault)][i::n]
    agent = build_agent(a)
    renderer = None
    if a.image_source == "photoreal":
        from run5.env.render import PhotoRenderer
        renderer = PhotoRenderer(a.render_url)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"cell_shard{i}.json").write_text(json.dumps({**vars(a), "out": str(a.out), "started": time.time(), "jobs": len(todo)}, indent=1))
    excluded = set(Path(a.exclude_list).read_text().split()) if a.exclude_list else set()
    for task, cond, seed in todo:
        ep = a.out / task / cond / f"seed_{seed}"
        if f"{task}/{cond}/seed_{seed}" in excluded:
            continue
        if a.resume_all:
            if (ep / "summary.json").exists():
                continue
            if (ep / "error.json").exists():
                import shutil
                keep = Path(a.resume_all) / f"{task}_{cond}_seed_{seed}"
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep), str(keep))
        elif a.rerun_errors:  # infrastructure re-run (DECISIONS #28): only episodes with an error record; keep the record
            if not (ep / "error.json").exists() or (ep / "summary.json").exists():
                continue
            import shutil
            keep = Path(a.rerun_errors) / f"{task}_{cond}_seed_{seed}"
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep), str(keep))
        elif (ep / "summary.json").exists() or (ep / "error.json").exists():
            continue
        scene = sample_scene(seed, task, None if cond == "clean" else cond)
        try:
            s = agent.run(scene, ep, renderer)
        except Exception as exc:  # kept in the denominator
            ep.mkdir(parents=True, exist_ok=True)
            s = {"seed": seed, "task": task, "fault": None if cond == "clean" else cond, "success": False, "termination": "harness_error",
                 "error": f"{type(exc).__name__}: {str(exc)[:400]}", "trace": traceback.format_exc()[-2000:]}
            (ep / "error.json").write_text(json.dumps(s, indent=1))
        print(json.dumps({k: s.get(k) for k in ("task", "fault", "seed", "success", "termination", "steps", "fault_fired", "wall_s", "error")},
                         default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
