"""Run 4 image sources for the three cameras: flat MuJoCo renders or photoreal Cycles renders.

Photoreal reuses Run 3's scene pool (HDRIs, tables, arm colours, container kinds, object finishes,
lamps, exposure, sensor noise, JPEG) with its Blender-only distractors turned off; Run 4's physical
distractors get their own materials through the server's `body_materials` override.
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from run3.phase2_photoreal import mjexport, scene_pool
from run3.phase2_photoreal.render_client import Renderer
from run4.env.scene import COLORS

CAMERAS = {"top": "overhead", "side": "side", "wrist": "wrist_cam"}
_RGB = {k: tuple(v[:3]) for k, v in COLORS.items()}


def photo_config(scene: dict) -> dict:
    """Seeded photoreal look for a Run 4 scene; object/container/distractor colours follow the scene."""
    sc = scene
    cfg = scene_pool.sample(sc["seed"], "red", "blue", distractors=False)
    cfg["scene_key"] = f"run4-{sc['seed']}"
    rng = np.random.default_rng(50_000 + sc["seed"])
    mats = cfg["materials"]
    for name, color in (("object", sc["target"]["color"]), ("container", sc["container"]["color"])):
        spec = mats[name]
        rgb = tuple(float(np.clip(c * (1 + rng.normal(0, .05)), 0, 1)) for c in _RGB[color])
        spec["tint" if spec["kind"] == "textured" else "color"] = rgb
    finish = cfg["factors"]["object_finish"]
    cfg["body_materials"] = {}
    for k, d in enumerate(sc["distractors"]):
        rgb = tuple(float(np.clip(c * (1 + rng.normal(0, .05)), 0, 1)) for c in _RGB[d["color"]])
        cfg["body_materials"][f"distractor_{k}"] = {"kind": "plastic", "color": rgb, "name": f"distractor_{k}",
                                                    "roughness": .2 if finish == "glossy plastic" else .6}
    cfg["factors"].update(target=f"{sc['target']['color']} {sc['target']['shape']}", distractors=len(sc["distractors"]),
                          container=f"{sc['container']['color']} {sc['container']['inner']:.3f}")
    return cfg


def _static_with_distractors(model) -> dict:
    static = mjexport.static_scene(model)
    for g in static["geoms"]:
        if g["body"].startswith("distractor_"):
            g["category"] = "object"
    return static


class PhotoRenderer(Renderer):
    """Run 3 render client with the distractor category fix and multi-camera rendering."""

    def set_scene(self, model, config: dict, force: bool = False) -> float:
        static = _static_with_distractors(model)
        key = (config["scene_key"], static["static_sha256"])
        if key == self.scene_key and not force:
            return 0.0
        r = self.client.post(self.base + "/scene", json={"static": static, "config": config})
        r.raise_for_status()
        self.scene_key, self._model, self._config = key, model, config
        return float(r.json()["build_s"])

    def views(self, model, data, config: dict, cameras=("top", "side", "wrist"), samples: int = 64, seed: int = 0,
              size: int = 448) -> tuple[dict, float]:
        out, t0 = {}, time.time()
        for i, name in enumerate(cameras):
            state = mjexport.frame_state(model, data, CAMERAS[name], size, size)
            body = {**state, "mode": "photo", "samples": samples, "seed": seed * 10 + i, "scene_key": self.scene_key[0]}
            r = self.client.post(self.base + "/render", json=body)
            if r.status_code == 409:
                self.set_scene(self._model, self._config, force=True)
                r = self.client.post(self.base + "/render", json=body)
            if r.status_code != 200:
                raise RuntimeError(r.text[:2000])
            img = scene_pool.camera_realism(r.content, config["noise_sigma"], config["jpeg_quality"], seed * 10 + i)
            out[name] = np.asarray(img)
        return out, time.time() - t0


def flat_views(env, cameras=("top", "side", "wrist")) -> dict:
    v = env.render()
    return {name: v[{"top": "overhead", "side": "side", "wrist": "wrist"}[name]] for name in cameras}


def to_png(arr) -> bytes:
    buf = io.BytesIO(); Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return buf.getvalue()
