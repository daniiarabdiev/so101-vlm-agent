"""Typed, task-independent optional side-view questions for model runners."""
from __future__ import annotations

import numpy as np

from .tasks import _half_extents


SIDE_HEIGHT_OPTIONS={
    "A":"task_cube_bottom_below_destination_rim",
    "B":"task_cube_bottom_at_or_above_destination_rim",
}


def side_height_question(camera_name="fixed_side_v1"):
    if not isinstance(camera_name,str) or not camera_name:raise ValueError("side camera name is required")
    return {"kind":"side_height_relation",
        "prompt_suffix":("Use the final fixed side-camera image. Image up corresponds to increasing world height "
            "for points at the same planar location. Is the BOTTOM of the task cube below the top "
            "rim of the destination container, or at/above that rim? Answer one label."),
        "options":dict(SIDE_HEIGHT_OPTIONS),"source_camera":camera_name,
        "image_orientation":{"horizontal":"image x increases right",
            "vertical":"image y increases down; greater world z projects toward image top at fixed XY"}}


def format_side_height_prompt(goal,question):
    if not isinstance(goal,str) or not goal.strip():raise ValueError("height question requires a goal")
    if question!=side_height_question(question.get("source_camera") if isinstance(question,dict) else None):
        raise ValueError("height question protocol mismatch")
    return ("Answer one visual height question from the supplied current images.\nGoal: "+goal.strip()+"\n"+
        question["prompt_suffix"]+"\nAllowed options in order:\n"+
        "\n".join(f"{label} = {meaning}" for label,meaning in question["options"].items())+"\nANSWER:\n")


def validate_side_height_response(response,question):
    if not isinstance(response,dict) or response.get("label") not in question["options"]:
        raise ValueError("height response lacks an allowed label")
    return response["label"]


def assess_side_height_evaluation_only(observation):
    objects=observation.get("objects") or [];containers=observation.get("containers") or []
    task=observation.get("task","A")
    if task not in ("A","B"):raise ValueError(f"height assessment does not support task {task!r}")
    destination_index=0 if task=="A" else 1
    if len(objects)!=1 or len(containers)<=destination_index:
        raise ValueError("height assessment requires one task cube and the task destination container")
    cube=objects[0];destination=containers[destination_index]
    center=cube.get("pos");rim=destination.get("rim_z")
    if (not isinstance(center,(list,tuple)) or len(center)!=3 or not all(np.isfinite(center))
            or not isinstance(rim,(int,float)) or not np.isfinite(rim)):
        raise ValueError("height assessment requires finite cube and rim geometry")
    try: vertical_half_extent=float(_half_extents(cube)[2])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError("height assessment requires finite cube size and rotation") from exc
    if not np.isfinite(vertical_half_extent) or vertical_half_extent<=0:
        raise ValueError("height assessment requires finite cube size and rotation")
    label="B" if float(center[2])-vertical_half_extent>=float(rim)-1e-12 else "A"
    return {"kind":"side_height_relation","labels":[label],"semantics":[SIDE_HEIGHT_OPTIONS[label]],
            "evaluation_only":True}
