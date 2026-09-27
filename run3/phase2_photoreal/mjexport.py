"""Export the exact MuJoCo visual state for an external path tracer.

Only what MuJoCo's default renderer shows (geom groups 0-2) is exported. Static data
(meshes, primitive sizes, categories) is sent once per scene; each decision frame sends
world poses of the visual geoms plus the camera pose/intrinsics. Physics is untouched.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json

import mujoco
import numpy as np

GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
              int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
              int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
              int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}


def visual_geoms(model) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_group[g]) <= 2 and model.geom_rgba[g][3] > 0]


def category(model, g: int) -> str:
    body = model.body(model.geom_bodyid[g]).name
    name = model.geom(g).name
    mat = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, model.geom_matid[g]) if model.geom_matid[g] >= 0 else ""
    if name == "table":
        return "table"
    if body.startswith("container"):
        return "container"
    if body.startswith("cube") or body.startswith("object"):
        return "object"
    if mat and mat.startswith("sts3215"):
        return "arm_motor"
    return "arm_printed"


def _npy_b64(array: np.ndarray) -> str:
    buf = io.BytesIO(); np.save(buf, np.ascontiguousarray(array)); return base64.b64encode(buf.getvalue()).decode()


def static_scene(model) -> dict:
    geoms, meshes = [], {}
    for g in visual_geoms(model):
        kind = GEOM_TYPES.get(int(model.geom_type[g]), "unsupported")
        entry = {"id": g, "name": model.geom(g).name, "body": model.body(model.geom_bodyid[g]).name,
                 "type": kind, "size": model.geom_size[g].tolist(), "rgba": model.geom_rgba[g].tolist(),
                 "category": category(model, g)}
        if kind == "mesh":
            mid = int(model.geom_dataid[g])
            entry["mesh"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            if entry["mesh"] not in meshes:
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                meshes[entry["mesh"]] = {"vertices": _npy_b64(model.mesh_vert[va:va + vn].astype(np.float32)),
                                         "faces": _npy_b64(model.mesh_face[fa:fa + fn].astype(np.int32))}
        geoms.append(entry)
    payload = {"geoms": geoms, "meshes": meshes}
    payload["static_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def camera_pose(model, data, camera: str = "overhead", width: int = 448, height: int = 448) -> dict:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return {"name": camera, "pos": data.cam_xpos[cid].tolist(), "xmat": data.cam_xmat[cid].tolist(),
            "fovy_deg": float(model.cam_fovy[cid]), "width": width, "height": height}


def frame_state(model, data, camera: str = "overhead", width: int = 448, height: int = 448) -> dict:
    poses = {str(g): [*data.geom_xpos[g].tolist(), *data.geom_xmat[g].tolist()] for g in visual_geoms(model)}
    return {"poses": poses, "camera": camera_pose(model, data, camera, width, height)}


def project(cam: dict, point) -> tuple[float, float] | None:
    """Pinhole projection matching MuJoCo's camera (looks along local -Z, v down)."""
    R = np.asarray(cam["xmat"], float).reshape(3, 3)
    local = R.T @ (np.asarray(point, float) - np.asarray(cam["pos"], float))
    if local[2] >= -1e-9:
        return None
    f = cam["height"] / (2 * np.tan(np.radians(cam["fovy_deg"]) / 2))
    u = cam["width"] / 2 + f * local[0] / -local[2]
    v = cam["height"] / 2 - f * local[1] / -local[2]
    return float(u), float(v)
