"""Frozen, no-inference grid screening corpus collection."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from so101_vlm.embodiment import Embodiment
from so101_vlm.grid_actions import GridAdapter, GridSpec
from so101_vlm.grid_assessment import CRITERION_VERSION, assess_grid_macro, assessment_config_hash
from so101_vlm.grid_oracle import GridOraclePolicy
from so101_vlm.grid_pipeline import build_grid_input
from so101_vlm.tasks import score


CORPUS_SPLITS = {
    "fit": tuple(range(1000, 1010)),
    "selection": tuple(range(2000, 2010)),
    "evaluation": tuple(range(3000, 3020)),
}
PHASE_SCHEDULE = (
    "approach_coarse", "approach_fine", "grasp_alignment", "held_lift",
    "carry_coarse", "carry_fine", "release", "completed",
    "controlled_empty_gripper_recovery", "controlled_misaligned_recovery",
)
SOURCE_FILES = (
    "so101_vlm/embodiment.py", "so101_vlm/grid_actions.py", "so101_vlm/grid_assessment.py", "so101_vlm/grid_oracle.py",
    "so101_vlm/grid_pipeline.py", "so101_vlm/grid_runner.py", "so101_vlm/pipelines.py",
    "so101_vlm/scene.py", "so101_vlm/tasks.py", "run2/grid_benchmark.py",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _hash_value(value):
    return _hash_bytes(_canonical(value).encode())


def acceptable_grid_labels(spec: GridSpec, target_xy, tolerance=.01):
    target = np.asarray(target_xy, dtype=float)
    fine_by_coarse = {}
    joint = []
    for column in spec.column_labels:
        for row in spec.row_labels:
            cell = column + row
            fine = [label for label in spec.fine_labels
                    if np.max(np.abs(spec.target_xy(column, row, label)-target)) <= tolerance]
            fine_by_coarse[cell] = fine
            if fine:
                joint.append(cell)
    return {
        "tolerance_m": float(tolerance),
        "coarse_joint": joint,
        "coarse_columns": sorted({cell[0] for cell in joint}),
        "coarse_rows": sorted({cell[1:] for cell in joint}, key=int),
        "fine_by_coarse": fine_by_coarse,
    }


def _advance_to_phase(sim, adapter, oracle, phase):
    def oracle_step():
        observation = sim.observe()
        choice = oracle.choose(observation)
        adapter.apply(sim, choice, observation)

    natural_steps = {
        "approach_coarse": 0, "approach_fine": 1, "grasp_alignment": 2,
        "held_lift": 3, "carry_coarse": 4, "carry_fine": 5,
        "release": 6, "completed": 7,
    }
    if phase in natural_steps:
        for _ in range(natural_steps[phase]):
            oracle_step()
        return "oracle_trajectory"
    if phase == "controlled_empty_gripper_recovery":
        oracle_step(); oracle_step()
        sim.step("close")
        return "controlled_perturbation_closed_empty_after_alignment"
    if phase == "controlled_misaligned_recovery":
        observation = sim.observe()
        true = oracle.choose(observation)
        wrong_column = adapter.spec.column_labels[-1] if true[0] != adapter.spec.column_labels[-1] else adapter.spec.column_labels[0]
        wrong_row = adapter.spec.row_labels[0] if true[1] != adapter.spec.row_labels[0] else adapter.spec.row_labels[-1]
        adapter.apply(sim, (wrong_column, wrong_row), observation)
        adapter.apply(sim, "5", sim.observe())
        return "controlled_perturbation_wrong_coarse_center"
    raise ValueError(f"unknown phase {phase}")


def _question_labels(inputs, phase, observation, adapter, spatial):
    macro_assessment=assess_grid_macro(observation,adapter)
    acceptable = {}
    for question in inputs["questions"]:
        kind, options = question["kind"], question["options"]
        if kind == "coarse_column":
            semantics = spatial["coarse_columns"]
        elif kind == "coarse_row":
            semantics = spatial["coarse_rows"]
        elif kind == "fine_cell":
            semantics = spatial["fine_by_coarse"]["".join(adapter.coarse)]
        else:
            semantics = macro_assessment["acceptable_actions"]
        labels = [label for label, semantic in options.items() if semantic in semantics]
        acceptable[kind] = {"semantics": semantics, "labels": labels}
        if kind=="macro":acceptable[kind]["assessment"]=macro_assessment
    holding = "A" if observation.get("holding") else "B"
    completed = "A" if score(observation)["success"] else "B"
    anomaly = "B" if phase.startswith("controlled_") else "A"
    acceptable.update(verified_holding={"semantics": [bool(observation.get("holding"))], "labels": [holding]},
                      completion={"semantics": [bool(score(observation)["success"])], "labels": [completed]},
                      anomaly={"semantics": ["recovery_required" if anomaly == "B" else "normal"], "labels": [anomaly]})
    return acceptable


def _judgment_questions():
    return [
        {"kind": "verified_holding", "group": "judgment",
         "prompt_suffix": "Is an object verifiably held now? Answer A=yes or B=no.",
         "options": {"A": "yes", "B": "no"}},
        {"kind": "completion", "group": "judgment",
         "prompt_suffix": "Is the stated task visibly complete and settled? Answer A=yes or B=no.",
         "options": {"A": "yes", "B": "no"}},
        {"kind": "anomaly", "group": "judgment",
         "prompt_suffix": "Does this state require recovery before normal progress? Answer A=normal or B=recovery_required.",
         "options": {"A": "normal", "B": "recovery_required"}},
    ]


def provider_request(record, root):
    """Strict allowlist: phase/seed/privileged labels cannot enter a request."""
    return {
        "prompt": record["request"]["prompt"],
        "image_paths": [str(Path(root) / path) for path in record["request"]["image_paths"]],
        "questions": record["request"]["questions"],
    }


def collect_grid_benchmark(output_dir, *, splits=CORPUS_SPLITS, phase_schedule=PHASE_SCHEDULE,
                           frames_per_seed=1):
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("benchmark output directory must be empty")
    root.mkdir(parents=True, exist_ok=True); (root / "images").mkdir()
    all_seeds = [int(seed) for values in splits.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("benchmark partitions contain duplicate seeds")
    if frames_per_seed < 1 or frames_per_seed > len(phase_schedule):
        raise ValueError("frames_per_seed outside phase schedule")
    project_root = Path(__file__).resolve().parents[1]
    source_hashes = {name: _hash_bytes((project_root/name).read_bytes()) for name in SOURCE_FILES}
    trajectory = frames_per_seed > 1
    manifest = {"schema_version": 1, "status": "collecting",
                "kind": "calibration_trajectories" if trajectory else "small_balanced_screening",
                "comprehensive_calibration": bool(frames_per_seed == len(phase_schedule)), "splits": {k: list(v) for k, v in splits.items()},
                "phase_schedule": list(phase_schedule), "frames_per_seed": frames_per_seed,
                "criterion_version": CRITERION_VERSION, "assessment_config_hash":assessment_config_hash(), "source_hashes": source_hashes}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(root / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.write(project_root/name, name)
    manifest["source_snapshot_sha256"] = _hash_bytes((root/"source_snapshot.zip").read_bytes())
    fingerprints = set(); phase_counts = Counter(); question_counts = Counter(); records = 0
    with (root/"inputs.jsonl").open("x") as inputs_file, (root/"labels.jsonl").open("x") as labels_file:
        for partition, seeds in splits.items():
            for seed_index, seed in enumerate(seeds):
                for offset in range(frames_per_seed):
                    phase = phase_schedule[(seed_index*frames_per_seed+offset) % len(phase_schedule)]
                    sim = Embodiment(); adapter = GridAdapter(); oracle = GridOraclePolicy(adapter)
                    try:
                        observation = sim.reset(int(seed), "A")
                        provenance = _advance_to_phase(sim, adapter, oracle, phase)
                        observation = sim.observe(); observation["goal"] = observation["goal"]
                        raw = sim.render()["overhead"]
                        built = build_grid_input(sim, observation, {"overhead": raw}, adapter)
                        processed = np.asarray(built["images"][-1])
                        frame_id = f"{partition}-{int(seed)}-{offset:02d}-{phase}"
                        raw_path = Path("images")/(frame_id+"_raw_overhead.png")
                        processed_path = Path("images")/(frame_id+"_processed.png")
                        Image.fromarray(raw).save(root/raw_path); Image.fromarray(processed).save(root/processed_path)
                        base_questions = [{**question, "group": "grounding" if question["kind"].startswith(("coarse_", "fine_")) else "judgment"}
                                          for question in built["questions"]]
                        questions = base_questions + _judgment_questions()
                        input_record = {
                            "schema_version": 1, "frame_id": frame_id,
                            "metadata": {"partition": partition, "seed": int(seed), "phase": phase,
                                         "phase_provenance": provenance},
                            "request": {"prompt": built["prompt"],
                                        "image_paths": ([str(raw_path), str(processed_path)] if built["stage"] == "fine" else [str(processed_path)]),
                                        "image_names": built["image_names"], "questions": questions},
                            "images": {
                                "raw_overhead": {"path": str(raw_path), "pixel_sha256": _hash_bytes(raw.tobytes()),
                                                 "file_sha256": _hash_bytes((root/raw_path).read_bytes())},
                                "processed": {"path": str(processed_path), "pixel_sha256": _hash_bytes(processed.tobytes()),
                                              "file_sha256": _hash_bytes((root/processed_path).read_bytes())},
                            },
                            "robot_state": built["robot_state"], "adapter_state": adapter.get_state(),
                            "adapter_state_sha256": _hash_value(adapter.get_state()),
                            "source_signature": _hash_value(source_hashes),
                        }
                        fingerprint = _hash_value({"prompt": built["prompt"], "pixels": input_record["images"]["processed"]["pixel_sha256"]})
                        if fingerprint in fingerprints:
                            raise ValueError("duplicate prompt+pixel fingerprint across corpus")
                        fingerprints.add(fingerprint); input_record["prompt_pixel_fingerprint"] = fingerprint
                        target = GridOraclePolicy._target_xy(observation)
                        spatial = acceptable_grid_labels(adapter.spec, target)
                        label_record = {"schema_version": 1, "frame_id": frame_id,
                                        "criterion_version": CRITERION_VERSION,
                                        "acceptable": _question_labels(built, phase, observation, adapter, spatial),
                                        "spatial": spatial, "oracle_target_xy": target.tolist(),
                                        "source": "evaluation_only_privileged_simulator_state"}
                        inputs_file.write(_canonical(input_record)+"\n"); labels_file.write(_canonical(label_record)+"\n")
                        records += 1; phase_counts[phase] += 1
                        question_counts.update(question["kind"] for question in questions)
                    finally:
                        sim.close()
    manifest.update(status="complete", frames=records, phase_counts=dict(sorted(phase_counts.items())),
                    question_counts=dict(sorted(question_counts.items())),
                    inputs_sha256=_hash_bytes((root/"inputs.jsonl").read_bytes()),
                    labels_sha256=_hash_bytes((root/"labels.jsonl").read_bytes()),
                    duplicate_prompt_pixel_fingerprints=0,
                    limitations=(["Calibration-input corpus only; it contains no unbiased evaluation outcome estimate.",
                                  f"Each seed contributes {frames_per_seed} explicitly scheduled phase states, including labelled controlled perturbations.",
                                  "Question-key sample sizes still constrain affine fitting and must be reported before selection."]
                                 if trajectory else
                                 ["Small balanced screening corpus, not a comprehensive calibration set.",
                                  "Fit and selection contain one frame per phase; only two evaluation frames per phase.",
                                  "Use frames_per_seed up to 10 for a later per-seed trajectory corpus; it was not launched here."] ))
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
