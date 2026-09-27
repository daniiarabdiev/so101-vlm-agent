"""Discrete, task-independent gripper-tip controls (metres)."""
import numpy as np
STEP_SIZES = {"small": 0.01, "large": 0.04}
DIRECTIONS = {"left": (-1, 0, 0), "right": (1, 0, 0), "forward": (0, 1, 0), "back": (0, -1, 0), "up": (0, 0, 1), "down": (0, 0, -1)}
ACTIONS = tuple(f"{direction}_{size}" for direction in DIRECTIONS for size in STEP_SIZES) + ("open", "close", "done")
MACRO_ACTIONS = ("grasp", "lift", "release")
EXTENDED_ACTIONS = ACTIONS + MACRO_ACTIONS
JOINT_ACTIONS = tuple(f"joint_{joint}_{direction}" for joint in range(6) for direction in ("minus", "plus")) + ("done",)

def action_components(action):
    if action not in EXTENDED_ACTIONS and action not in JOINT_ACTIONS:
        raise ValueError(f"Unknown discrete action: {action}")
    result = {"direction": None, "step_size": None, "gripper": None, "done": action == "done"}
    if action.startswith("joint_"):
        result.update(direction=action, step_size="small")
    elif action in ("open", "close"):
        result["gripper"] = action
    elif action in MACRO_ACTIONS:
        result.update(direction=action, step_size="macro", gripper={"grasp": "close", "release": "open"}.get(action))
    elif action != "done":
        result["direction"], result["step_size"] = action.rsplit("_", 1)
    return result

def action_delta(action):
    c = action_components(action)
    if c["direction"] not in DIRECTIONS:
        return None
    return np.asarray(DIRECTIONS[c["direction"]], dtype=float) * STEP_SIZES[c["step_size"]]
