"""Photoreal looks for Run 5 scenes (Run 3/4 scene pool; every object and tray gets its own material).

Held-out looks: some HDRIs, table surfaces and arm colours are never used for LoRA training data
(`photo_config(..., allow_heldout=False)`). Test scenes use the natural pool, and results are split by `is_heldout_look`.
The Run 4 Blender server and PhotoRenderer are reused unchanged (body_materials per body).
"""
from __future__ import annotations

import numpy as np

from run3.phase2_photoreal import scene_pool
from run4.env.render import PhotoRenderer, flat_views  # noqa: F401  (re-exported)

HELDOUT = {"hdri": set(scene_pool.HDRIS[::4]), "table": {"marble_01", "brushed_concrete", "green cutting mat"},
           "arm_color": {"orange", "grey"}}
_RGB = scene_pool.COLOR_RGB | {"brown": (0.40, 0.24, 0.10)}


def is_heldout_look(cfg: dict) -> bool:
    f = cfg["factors"]
    return f["hdri"] in HELDOUT["hdri"] or f["table"] in HELDOUT["table"] or f["arm_color"] in HELDOUT["arm_color"]


def _jit(rgb, rng, a=.05):
    return tuple(float(np.clip(c * (1 + rng.normal(0, a)), 0, 1)) for c in rgb)


def photo_config(scene: dict, allow_heldout: bool = True) -> dict:
    base_seed = 1_000_000 + 97 * scene["seed"] + len(scene["task"])
    for bump in range(200):
        cfg = scene_pool.sample(base_seed + 7919 * bump, "red", "blue", distractors=False)
        if allow_heldout or not is_heldout_look(cfg):
            break
    cfg["scene_key"] = f"run5-{scene['task']}-{scene['seed']}-{scene.get('fault')}"
    rng = np.random.default_rng(base_seed)
    finish = cfg["factors"]["object_finish"]; kind = cfg["factors"]["container_kind"]
    bm = {}
    for i, o in enumerate(scene["objects"]):
        rgb = _jit(_RGB[o["color"]], rng)
        if finish == "painted wood":
            bm[f"object_{i}"] = {"kind": "textured", "asset": "stained_pine", "tint": rgb, "scale": 10.0, "name": f"object_{i}"}
        else:
            bm[f"object_{i}"] = {"kind": "plastic", "color": rgb, "roughness": .2 if finish == "glossy plastic" else .6,
                                 "coat": .3 if finish == "glossy plastic" else 0.0, "name": f"object_{i}"}
    for j, c in enumerate(scene["containers"]):
        rgb = _jit(_RGB[c["color"]], rng)
        if kind == "plastic bin":
            bm[f"container_{j}"] = {"kind": "plastic", "color": rgb, "roughness": float(rng.uniform(.25, .5)), "name": f"container_{j}"}
        else:
            asset = {"painted wooden tray": "wood_table_001", "plywood tray": "plywood", "fabric-lined box": "denim_fabric"}[kind]
            bm[f"container_{j}"] = {"kind": "textured", "asset": asset, "tint": rgb, "scale": 6.0, "name": f"container_{j}"}
    cfg["body_materials"] = bm
    cfg["factors"].update(task=scene["task"], n_objects=len(scene["objects"]), heldout_look=is_heldout_look(cfg))
    return cfg
