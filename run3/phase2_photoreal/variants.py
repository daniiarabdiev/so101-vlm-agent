"""Scene variants for Phases 2-3: object shape/colour and container colour.

The harness (so101_vlm) is not edited. `VariantEmbodiment` wraps the unchanged Embodiment:
the generated MJCF is post-processed only for the manipulated object's geom type/size/colour and
the container colour; timestep, solver, friction, masses, controllers, IK, macros, oracle and the
success rule are the harness's. Observations report the object's true size so the unchanged
scoring/oracle geometry stays correct for non-cube shapes (a sphere reports identity rotation,
since a rolling ball's orientation says nothing about its footprint).
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

import so101_vlm.embodiment as emb
from so101_vlm.embodiment import Embodiment

RGBA = {"red": [.85, .08, .07, 1], "blue": [.08, .27, .88, 1], "green": [.1, .65, .22, 1], "purple": [.58, .18, .72, 1],
        "orange": [.95, .45, .06, 1], "yellow": [.95, .80, .06, 1], "pink": [.95, .42, .62, 1], "white": [.9, .9, .88, 1]}
OBJECT_COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "pink"]
CONTAINER_COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "white"]
SHAPES = {"cube": {"noun": "cube"}, "ball": {"noun": "ball", "radius": .025}, "cylinder": {"noun": "cylinder", "radius": .022, "half_height": .025}}


def sample_variant(seed: int, shapes=("cube", "ball", "cylinder")) -> dict:
    rng = np.random.default_rng(20_000 + int(seed))
    shape = shapes[rng.integers(len(shapes))]
    obj = OBJECT_COLORS[rng.integers(len(OBJECT_COLORS))]
    cont = [c for c in CONTAINER_COLORS if c != obj][rng.integers(len(CONTAINER_COLORS) - 1)]
    return {"shape": shape, "object_color": obj, "container_color": cont}


class VariantEmbodiment(Embodiment):
    def __init__(self, config=None, variant: dict | None = None):
        super().__init__(config)
        self.variant = dict(variant or {"shape": "cube"})

    def _patch(self, xml, containers, cube_color):
        v = self.variant
        root = ET.fromstring(xml)
        geom = root.find(".//geom[@name='cube']")
        shape = v.get("shape", "cube")
        if shape == "ball":
            geom.set("type", "sphere"); geom.set("size", f"{SHAPES['ball']['radius']}")
        elif shape == "cylinder":
            geom.set("type", "cylinder"); geom.set("size", f"{SHAPES['cylinder']['radius']} {SHAPES['cylinder']['half_height']}")
        body = root.find(".//body[@name='cube']")
        if shape != "cube":
            half_h = SHAPES["ball"]["radius"] if shape == "ball" else SHAPES["cylinder"]["half_height"]
            x, y, _ = map(float, body.get("pos").split())
            body.set("pos", f"{x} {y} {half_h + .001}")
            body.attrib.pop("quat", None)
        if v.get("object_color"):
            geom.set("rgba", " ".join(map(str, RGBA[v["object_color"]]))); cube_color = v["object_color"]
        if v.get("container_color"):
            for g in root.findall(".//body[@name='container_2']/geom"):
                g.set("rgba", " ".join(map(str, RGBA[v["container_color"]])))
            containers = copy.deepcopy(containers)
            containers[-1]["color"] = v["container_color"]
        return ET.tostring(root, encoding="unicode"), containers, cube_color

    def reset(self, seed, task="A"):
        original = emb.build_scene
        emb.build_scene = lambda config, s, t: self._patch(*original(config, s, t))
        try:
            return super().reset(seed, task)
        finally:
            emb.build_scene = original

    def observe(self):
        obs = super().observe()
        shape = self.variant.get("shape", "cube")
        if shape != "cube" and obs.get("objects"):
            o = obs["objects"][0]
            if shape == "ball":
                r = SHAPES["ball"]["radius"]; o["size"] = [2 * r] * 3; o["rotation_matrix"] = np.eye(3).tolist()
            else:
                c = SHAPES["cylinder"]; o["size"] = [2 * c["radius"], 2 * c["radius"], 2 * c["half_height"]]
            o["shape"] = shape
            obs["goal"] = obs["goal"].replace(f"{self.cube_color} cube", f"{self.cube_color} {SHAPES[shape]['noun']}")
        return obs
