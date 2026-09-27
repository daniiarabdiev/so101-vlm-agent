"""Seeded pool of photoreal tabletop looks (visual only; physics stays MuJoCo's).

Every factor is recorded in the returned config so results can be broken down by it.
All HDRIs/textures are Poly Haven CC0 (assets_manifest.json); plain materials are procedural.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MANIFEST = json.loads((Path(__file__).parent / "assets_manifest.json").read_text())
HDRIS = sorted(MANIFEST["hdri"])
TABLE_TEXTURES = ["wood_table_001", "wood_table_worn", "plywood", "laminate_floor_02", "american_walnut_veneer",
                  "stained_pine", "brushed_concrete", "marble_01", "oriented_strand_board"]
PLAIN_TABLES = {"white desk": (0.86, 0.86, 0.84), "light grey desk": (0.62, 0.63, 0.64),
                "dark grey desk": (0.22, 0.22, 0.23), "black mat": (0.04, 0.04, 0.045),
                "green cutting mat": (0.10, 0.30, 0.18)}
PLA = {"yellow": (0.95, 0.72, 0.05), "white": (0.85, 0.85, 0.83), "black": (0.03, 0.03, 0.03),
       "orange": (0.95, 0.35, 0.04), "red": (0.70, 0.04, 0.04), "blue": (0.05, 0.20, 0.65),
       "grey": (0.35, 0.36, 0.37), "green": (0.10, 0.50, 0.15)}
# Object/container colours follow the MuJoCo scene's colour names so the goal text stays true.
COLOR_RGB = {"red": (0.75, 0.05, 0.04), "blue": (0.05, 0.18, 0.75), "green": (0.06, 0.50, 0.14),
             "purple": (0.40, 0.10, 0.55), "orange": (0.95, 0.38, 0.03), "yellow": (0.95, 0.78, 0.05),
             "pink": (0.95, 0.40, 0.60), "white": (0.88, 0.88, 0.86), "black": (0.05, 0.05, 0.05)}
CONTAINER_KINDS = ["plastic bin", "painted wooden tray", "plywood tray", "fabric-lined box"]


def _jitter(rgb, rng, amount=0.06):
    return tuple(float(np.clip(c * (1 + rng.normal(0, amount)), 0, 1)) for c in rgb)


def sample(seed: int, object_color: str, container_color: str, distractors: bool | None = None) -> dict:
    rng = np.random.default_rng(10_000 + int(seed))
    factors = {}
    hdri = HDRIS[rng.integers(len(HDRIS))]
    factors["hdri"] = hdri
    if rng.random() < 0.6:
        tex = TABLE_TEXTURES[rng.integers(len(TABLE_TEXTURES))]
        table_mat = {"kind": "textured", "asset": tex, "scale": float(rng.uniform(1.5, 4.0)), "name": "table"}
        factors["table"] = tex
    else:
        name = list(PLAIN_TABLES)[rng.integers(len(PLAIN_TABLES))]
        table_mat = {"kind": "plastic", "color": _jitter(PLAIN_TABLES[name], rng, 0.04), "roughness": float(rng.uniform(0.55, 0.9)),
                     "specular": 0.3, "name": "table"}
        factors["table"] = name
    pla = list(PLA)[rng.integers(len(PLA))]
    factors["arm_color"] = pla
    kind = CONTAINER_KINDS[rng.integers(len(CONTAINER_KINDS))]
    factors["container_kind"] = kind
    crgb = COLOR_RGB[container_color]
    if kind == "plastic bin":
        cont = {"kind": "plastic", "color": _jitter(crgb, rng), "roughness": float(rng.uniform(0.25, 0.5)), "name": "container"}
    elif kind == "painted wooden tray":
        cont = {"kind": "textured", "asset": "wood_table_001", "tint": _jitter(crgb, rng), "scale": 6.0, "name": "container"}
    elif kind == "plywood tray":  # container colour name still applies via a painted tint
        cont = {"kind": "textured", "asset": "plywood", "tint": _jitter(crgb, rng, 0.03), "scale": 6.0, "name": "container"}
    else:
        cont = {"kind": "textured", "asset": "denim_fabric", "tint": _jitter(crgb, rng), "scale": 8.0, "name": "container"}
    obj_finish = ["glossy plastic", "matte plastic", "painted wood"][rng.integers(3)]
    factors["object_finish"] = obj_finish
    orgb = COLOR_RGB[object_color]
    if obj_finish == "painted wood":
        obj = {"kind": "textured", "asset": "stained_pine", "tint": _jitter(orgb, rng), "scale": 10.0, "name": "object"}
    else:
        obj = {"kind": "plastic", "color": _jitter(orgb, rng), "roughness": 0.2 if obj_finish == "glossy plastic" else 0.65,
               "coat": 0.3 if obj_finish == "glossy plastic" else 0.0, "name": "object"}
    lights = []
    if rng.random() < 0.6:
        temp = [(1.0, 0.85, 0.7), (1.0, 1.0, 1.0), (0.85, 0.92, 1.0)][rng.integers(3)]
        lights.append({"type": "AREA", "energy": float(rng.uniform(15, 90)), "size": float(rng.uniform(0.3, 1.0)),
                       "color": temp, "location": [float(rng.uniform(-0.3, 0.6)), float(rng.uniform(-0.6, 0.6)), float(rng.uniform(1.1, 1.8))],
                       "target": [0.23, 0.0, 0.0]})
    factors["lamp"] = bool(lights)
    use_d = bool(rng.random() < 0.5) if distractors is None else distractors
    dist = []
    if use_d:
        for i in range(int(rng.integers(1, 4))):
            side = rng.choice([-1, 1])
            loc = [float(rng.uniform(0.05, 0.42)), float(side * rng.uniform(0.21, 0.30)), 0.0]
            kind_d = ["box", "cylinder", "sphere"][rng.integers(3)]
            size = [float(rng.uniform(0.012, 0.03)), float(rng.uniform(0.012, 0.03)), float(rng.uniform(0.01, 0.03))]
            if kind_d == "sphere":
                size = [size[0]]
            loc[2] = size[-1] if kind_d != "sphere" else size[0]
            if kind_d == "cylinder":
                size = [size[0], size[2]]; loc[2] = size[1]
            col = list(COLOR_RGB)[rng.integers(len(COLOR_RGB))]
            dist.append({"type": kind_d, "size": size, "location": loc, "yaw": float(rng.uniform(0, 3.14)),
                         "material": {"kind": "plastic", "color": _jitter(COLOR_RGB[col], rng), "roughness": 0.5}})
    factors["distractors"] = len(dist)
    table = {"center_x": float(rng.uniform(0.15, 0.3)), "center_y": float(rng.uniform(-0.1, 0.1)),
             "half_x": float(rng.uniform(0.33, 0.6)), "half_y": float(rng.uniform(0.35, 0.7)), "height": 0.74}
    materials = {"table": table_mat, "printed": {"kind": "printed", "color": _jitter(PLA[pla], rng, 0.03)},
                 "motor": {"kind": "plastic", "color": (0.02, 0.02, 0.02), "roughness": 0.35},
                 "object": obj, "container": cont,
                 "floor": {"kind": "textured", "asset": ["laminate_floor_03", "brushed_concrete", "cotton_jersey"][rng.integers(3)], "scale": 1.0}}
    config = {"seed": int(seed), "scene_key": f"pool-{seed}", "hdri": f"{hdri}.hdr",
              "hdri_rotation_deg": float(rng.uniform(0, 360)), "hdri_strength": float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))),
              "lights": lights, "materials": materials, "table": table, "distractors": dist,
              "exposure": float(rng.normal(0, 0.25)), "view_transform": "AgX",
              "noise_sigma": float(rng.uniform(1.0, 4.0)), "jpeg_quality": int(rng.integers(80, 96)),
              "factors": factors}
    return config


def camera_realism(png_bytes: bytes, sigma: float, quality: int, seed: int) -> "Image.Image":
    """Mild sensor noise + JPEG compression, applied after the path-traced render."""
    import io
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(seed)
    img = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
