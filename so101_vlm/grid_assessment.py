"""Evaluation-only acceptable macro sets for the categorical grid policy."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .tasks import _half_extents, score


CRITERION_VERSION="grid-acceptable-v2-20260920"


def assess_grid_macro(observation,adapter):
    if score(observation)["success"]:
        actions=["done"];phase="complete"
    elif observation.get("holding"):
        if adapter.location_context!="carry":actions=["lift"];phase="held_approach"
        else:
            obj=observation["objects"][0]
            destination=observation["containers"][1 if observation.get("task")=="B" else 0]
            usable=np.asarray(destination["inner_size"][:2],dtype=float)/2-_half_extents(obj)[:2]-.001
            inside=np.all(np.abs(np.asarray(obj["pos"][:2])-np.asarray(destination["pos"][:2]))<=usable)
            actions=["release"] if inside else ["reselect"]
            phase="held_carry"
    elif observation.get("gripper_state")!="open":actions=["open"];phase="closed_empty"
    elif adapter.last_choice=="open" and float(observation["ee_pos"][2])<float(observation.get("carry_height",.085))-.012:
        actions=["lift"];phase="open_low_recovery"
    elif adapter.location_context=="approach" and adapter.fine is not None:
        aligned=np.max(np.abs(np.asarray(observation["ee_pos"][:2])-np.asarray(observation["objects"][0]["pos"][:2])))<=.01
        actions=["grasp"] if aligned else ["reselect"]
        phase="approach_alignment"
    else:actions=["reselect"];phase="reselection"
    return {"acceptable_actions":actions,"acceptable_phase":phase,"criterion":CRITERION_VERSION,
            "supported":True,"exclusions":[],"reason":None}


def assessment_config_hash():
    config={"criterion":CRITERION_VERSION,"alignment_linf_m":.01,"container_clearance_m":.001,
            "principle":"actions must make local progress; reselect is acceptable when alignment or footprint fails"}
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
