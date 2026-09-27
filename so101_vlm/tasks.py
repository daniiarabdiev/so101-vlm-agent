"""Ground-truth scoring requires enclosed footprint, floor support and rest."""
import numpy as np

def goal_for(task, observation=None):
    if observation and observation.get("goal") and task in ("A", "B"): return observation["goal"]
    if task == "A": return "Put the cube inside the container, release it, and let it rest."
    if task == "B": return "Move the cube from container 1 into container 2, release it, and let it rest."
    if task == "C":
        order = (observation or {}).get("color_order", ["red", "blue"])
        return f"Arrange the two cubes left to right in this colour order: {', '.join(order)}. Release both cubes."
    raise ValueError(f"Unknown task: {task}")

def _half_extents(obj):
    size = np.asarray(obj["size"], dtype=float)
    if size.ndim == 0: size = np.repeat(size, 3)
    rotation = np.asarray(obj.get("rotation_matrix", np.eye(3)), dtype=float).reshape(3, 3)
    return np.abs(rotation) @ (size / 2)

def _resting(obj, floor_z, holding):
    if holding == obj["name"]: return False, "cube still held"
    if np.linalg.norm(np.asarray(obj.get("velocity", [0, 0, 0]))[:3]) > 0.025: return False, "cube moving or airborne"
    if np.linalg.norm(obj.get("angular_velocity", [0, 0, 0])) > 0.25: return False, "cube rotating"
    bottom = float(obj["pos"][2]) - _half_extents(obj)[2]
    if abs(bottom - floor_z) > 0.004: return False, "cube not resting on destination floor"
    return True, "resting"

def score(observation):
    task = observation.get("task", "A")
    objects = observation.get("objects", [])
    if not objects: return {"success": False, "reason": "missing cube"}
    if task == "C":
        if len(objects) < 2: return {"success": False, "reason": "missing second cube"}
        order = observation.get("color_order", ["red", "blue"])
        by_color = {str(obj["color"]): obj for obj in objects}
        if any(str(color) not in by_color for color in order): return {"success": False, "reason": "requested colours missing"}
        ordered = [by_color[str(color)] for color in order]
        for obj in ordered:
            ok, reason = _resting(obj, observation.get("table_z", 0.0), observation.get("holding"))
            if not ok: return {"success": False, "reason": reason}
        margin = observation.get("order_margin", 0.02)
        ok = all(a["pos"][0] + margin <= b["pos"][0] for a, b in zip(ordered, ordered[1:]))
        return {"success": bool(ok), "reason": "colour order satisfied" if ok else "incorrect colour order"}
    if task not in ("A", "B"): raise ValueError(f"Unknown task: {task}")
    containers = observation.get("containers", [])
    index = 1 if task == "B" else 0
    if len(containers) <= index: return {"success": False, "reason": "missing destination"}
    destination, obj = containers[index], objects[0]
    offset = np.abs(np.asarray(obj["pos"][:2]) - np.asarray(destination["pos"][:2]))
    interior = np.asarray(destination["inner_size"][:2]) / 2
    if np.any(offset + _half_extents(obj)[:2] > interior - 0.001): return {"success": False, "reason": "cube outside interior or on rim"}
    ok, reason = _resting(obj, destination["floor_z"], observation.get("holding"))
    return {"success": bool(ok), "reason": "cube resting inside destination" if ok else reason}
