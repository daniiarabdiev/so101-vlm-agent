"""Run 8 benchmark copy of run6/env/blender_server.py. Defaults are unchanged; added knobs for the render-speed benchmark:
  env RENDER_ENGINE=EEVEE  -> EEVEE (Next) instead of Cycles for this server process (samples = TAA samples)
  env CYCLES_BACKEND=OPTIX -> the OptiX backend (already supported by the Run 4 server)
  request fields: "bounces" (Cycles max bounces, default 6), "denoiser" ("OPENIMAGEDENOISE" | "OPTIX"),
                  "adaptive_threshold" (Cycles adaptive-sampling noise threshold)
  POST /render_multi {"poses", "cameras": [camera, ...], ...} -> JSON {"pngs": [base64, ...], "seconds": [...]} (one call, all views)

Run 6 copy of run4/env/blender_server.py; the only change is the optional RENDER_PERSISTENT=1 cache.

Run 4 copy of run3/phase2_photoreal/blender_server.py; only change: config["body_materials"]
(per-body material overrides).

Blender Cycles render server for exact MuJoCo states (runs inside `blender -b --python`).

POST /scene   {"static": <mjexport.static_scene>, "config": {...}}      -> {"scene_key": ...}
POST /render  {"poses": {...}, "camera": {...}, "mode": "photo"|"id", "samples": N} -> PNG bytes
GET  /health

Geometry and poses come from MuJoCo only; this server never simulates anything.
Asset files (HDRIs, textures) are read from ASSET_DIR (downloaded from Poly Haven, CC0).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import bpy
import numpy as np
from mathutils import Matrix

ASSET_DIR = os.environ.get("ASSET_DIR", "/workspace/assets")
TOKEN = open(os.environ["RENDER_TOKEN_FILE"]).read().strip() if os.environ.get("RENDER_TOKEN_FILE") else None
STATE = {"scene_key": None, "objects": {}, "config": None}
ID_COLORS = {"object": (1, 0, 0), "container": (0, 1, 0), "gripper": (0, 0, 1), "arm": (1, 1, 0),
             "table": (0, 0, 0), "distractor": (1, 0, 1), "floor": (0, 0, 0)}


# ----------------------------------------------------------------------------------- setup
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if os.environ.get("RENDER_ENGINE") == "EEVEE":
        items = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in items else "BLENDER_EEVEE"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    # CUDA by default: A100/H100 have no RT cores, and OptiX kernel compilation hung for 20+ min on one Pod
    for backend in (os.environ.get("CYCLES_BACKEND", "CUDA"), "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
            devices = [d for d in prefs.devices if d.type == backend]
            if devices:
                for d in prefs.devices:
                    d.use = d.type == backend
                break
        except Exception:
            continue
    scene.cycles.device = "GPU"
    # Run 6: RENDER_PERSISTENT=1 keeps Cycles render data between renders (a cache; the images are unchanged, checked in
    # run6/diag/render_bench.py). Without it every render re-syncs the scene and re-uploads textures and the HDRI.
    scene.render.use_persistent_data = os.environ.get("RENDER_PERSISTENT") == "1"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    return scene


def _npy(b64):
    return np.load(io.BytesIO(base64.b64decode(b64)))


def _mesh_object(name, verts, faces, smooth_angle=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    if smooth_angle is not None:
        try:
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(smooth_angle))
        except Exception:
            pass
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _box(name, hx, hy, hz):
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    f = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    # small bevel so cube edges catch light like a real object
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _primitive(name, kind, size):
    if kind == "box":
        obj = _box(name, *size[:3])
        if min(size[:3]) > 0.004:
            mod = obj.modifiers.new("bevel", "BEVEL"); mod.width = min(0.0015, min(size[:3]) * 0.3); mod.segments = 2
        return obj
    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size[0], segments=48, ring_count=24)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=64)
    elif kind == "capsule":
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=48)
    else:
        raise ValueError(kind)
    obj = bpy.context.active_object
    obj.name = name
    try:
        obj.data.shade_smooth()
        obj.data.set_sharp_from_angle(angle=math.radians(40))
    except Exception:
        bpy.ops.object.shade_smooth()
    return obj


# ------------------------------------------------------------------------------- materials
def _principled(name, color, roughness=0.5, specular=0.5, coat=0.0, transmission=0.0, ior=1.45):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1)
    bsdf.inputs["Roughness"].default_value = roughness
    for key in ("Specular IOR Level", "Specular"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = specular
            break
    if coat and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if transmission:
        key = "Transmission Weight" if "Transmission Weight" in bsdf.inputs else "Transmission"
        bsdf.inputs[key].default_value = transmission
        bsdf.inputs["IOR"].default_value = ior
    return mat


def _printed_plastic(name, color, layer_mm=0.2, rng=None):
    """FDM PLA: slightly rough plastic with fine horizontal layer lines (bump)."""
    mat = _principled(name, color, roughness=0.42 + 0.1 * (rng.random() if rng is not None else 0.5), specular=0.45)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"; wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 1.0 / (layer_mm / 1000.0) / 40  # object-space metres
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(tex.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _textured(name, folder, scale=1.0, roughness_default=0.5, tint=None):
    """Poly Haven PBR set (diffuse / rough / nor_gl jpgs) with object-space box mapping."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    files = {f.lower(): os.path.join(folder, f) for f in os.listdir(folder)}

    def find(*keys):
        for key, path in files.items():
            if any(k in key for k in keys):
                return path
        return None

    diff = find("diff", "col", "albedo")
    if diff:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(diff); t.projection = "BOX"; t.projection_blend = 0.2
        nt.links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        if tint is not None:
            # paint: the named colour dominates, the texture only modulates it (factor 0.35)
            mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 0.35
            mix.inputs[6].default_value = (*tint[:3], 1); nt.links.new(t.outputs["Color"], mix.inputs[7])
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    rough = find("rough")
    if rough:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(rough); t.image.colorspace_settings.name = "Non-Color"
        t.projection = "BOX"; nt.links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = roughness_default
    nor = find("nor_gl", "normal")
    if nor:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(nor); t.image.colorspace_settings.name = "Non-Color"
        t.projection = "BOX"; nt.links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.6
        nt.links.new(t.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _emission(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*color, 1); em.inputs["Strength"].default_value = 1
    out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def material_for(spec, rng):
    kind = spec["kind"]
    if kind == "printed":
        return _printed_plastic(spec.get("name", "printed"), spec["color"], rng=rng)
    if kind == "plastic":
        return _principled(spec.get("name", "plastic"), spec["color"], roughness=spec.get("roughness", 0.35),
                           specular=spec.get("specular", 0.5), coat=spec.get("coat", 0.0))
    if kind == "textured":
        return _textured(spec.get("name", "tex"), os.path.join(ASSET_DIR, "textures", spec["asset"]),
                         scale=spec.get("scale", 1.0), tint=spec.get("tint"))
    if kind == "glass":
        return _principled(spec.get("name", "glass"), spec.get("color", (0.9, 0.95, 1)), roughness=0.05,
                           transmission=spec.get("transmission", 0.9), ior=1.5)
    raise ValueError(kind)


# ---------------------------------------------------------------------------------- world
def set_world(config):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("world"); scene.world = world; world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    hdri = config.get("hdri")
    if hdri:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(os.path.join(ASSET_DIR, "hdri", hdri))
        coord = nt.nodes.new("ShaderNodeTexCoord"); mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0, 0, math.radians(config.get("hdri_rotation_deg", 0)))
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = config.get("hdri_strength", 1.0)
    for i, lamp in enumerate(config.get("lights", [])):
        data = bpy.data.lights.new(f"lamp{i}", type=lamp.get("type", "AREA"))
        data.energy = lamp["energy"]
        if data.type == "AREA":
            data.size = lamp.get("size", 0.5)
        data.color = lamp.get("color", (1, 1, 1))
        obj = bpy.data.objects.new(f"lamp{i}", data); scene.collection.objects.link(obj)
        obj.location = lamp["location"]
        direction = np.asarray(lamp.get("target", (0.2, 0, 0))) - np.asarray(lamp["location"])
        obj.rotation_mode = "QUATERNION"
        from mathutils import Vector
        obj.rotation_quaternion = Vector(direction).to_track_quat("-Z", "Y")


def build_scene(static, config):
    scene = reset_scene()
    rng = np.random.default_rng(config.get("seed", 0))
    set_world(config)
    mats = {name: material_for(spec, rng) for name, spec in config["materials"].items()}
    # Run 4: per-body material overrides (e.g. each physical distractor keeps its own colour)
    body_mats = {name: material_for(spec, rng) for name, spec in config.get("body_materials", {}).items()}
    meshes = {name: (_npy(m["vertices"]), _npy(m["faces"])) for name, m in static["meshes"].items()}
    objects = {}
    for g in static["geoms"]:
        cat = g["category"]
        if cat == "table":
            continue  # replaced by the configured table below
        if g["type"] == "mesh":
            v, f = meshes[g["mesh"]]
            obj = _mesh_object(f"g{g['id']}", v, f, smooth_angle=35)
        else:
            obj = _primitive(f"g{g['id']}", g["type"], g["size"])
        key = {"arm_printed": "printed", "arm_motor": "motor", "object": "object", "container": "container"}.get(cat, "printed")
        obj.data.materials.clear(); obj.data.materials.append(body_mats.get(g["body"], mats[key]))
        if g.get("name") == "calib_marker":  # Run 8: matte cyan calibration sticker
            obj.data.materials.clear(); obj.data.materials.append(_principled("calib_marker", (0.0, 0.70, 0.75), roughness=0.8, specular=0.1))
        role = cat
        if g["body"] in ("gripper", "moving_jaw_so101_v1"):
            role = "gripper"
        elif cat.startswith("arm"):
            role = "arm"
        obj["role"] = role
        objects[str(g["id"])] = obj
    # table: finite slab with a textured top (physics plane is unchanged; the arm never reaches the edge)
    tb = config["table"]
    table = _box("table", tb["half_x"], tb["half_y"], 0.02)
    table.location = (tb["center_x"], tb["center_y"], -0.02)
    table.data.materials.append(mats["table"]); table["role"] = "table"
    if "floor" in mats:
        floor = _box("floor", 3, 3, 0.01); floor.location = (0, 0, -tb.get("height", 0.75)); floor.data.materials.append(mats["floor"])
        floor["role"] = "floor"
    for i, d in enumerate(config.get("distractors", [])):
        obj = _primitive(f"distractor{i}", d["type"], d["size"])
        obj.location = d["location"]; obj.rotation_euler = (0, 0, d.get("yaw", 0))
        obj.data.materials.append(material_for(d["material"], rng)); obj["role"] = "distractor"
    cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam); scene.camera = cam
    scene.view_settings.view_transform = config.get("view_transform", "AgX")
    for look in ([config["look"]] if config.get("look") else []) + ["AgX - Punchy", "Punchy"]:
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = config.get("exposure", 0.0)
    STATE.update(scene_key=config.get("scene_key"), objects=objects, config=config)


def apply_frame(poses, camera):
    for gid, pose in poses.items():
        obj = STATE["objects"].get(gid)
        if obj is None:
            continue
        pos, R = pose[:3], np.asarray(pose[3:12]).reshape(3, 3)
        M = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                M[i][j] = float(R[i, j])
            M[i][3] = float(pos[i])
        obj.matrix_world = M
    scene = bpy.context.scene
    cam = scene.camera
    R = np.asarray(camera["xmat"]).reshape(3, 3)
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = float(R[i, j])
        M[i][3] = float(camera["pos"][i])
    cam.matrix_world = M
    cam.data.sensor_fit = "VERTICAL"
    cam.data.angle_y = math.radians(camera["fovy_deg"])  # MuJoCo fovy is vertical
    cam.data.clip_start = 0.01
    scene.render.resolution_x = int(camera["width"])
    scene.render.resolution_y = int(camera["height"])
    scene.render.resolution_percentage = 100


def render(mode="photo", samples=64, seed=0, opts=None):
    opts = opts or {}
    scene = bpy.context.scene
    saved = {}
    if mode != "id" and scene.render.engine != "CYCLES":  # EEVEE benchmark server
        ee = scene.eevee
        ee.taa_render_samples = int(samples)
        for attr, val in (("use_raytracing", True), ("use_shadows", True), ("use_gtao", True)):
            if hasattr(ee, attr):
                setattr(ee, attr, val)
        path = tempfile.mktemp(suffix=".png"); scene.render.filepath = path
        t0 = time.time(); bpy.ops.render.render(write_still=True); dt = time.time() - t0
        data = open(path, "rb").read(); os.remove(path)
        return data, dt
    if mode == "id":
        # flat emission IDs, 1 sample, tiny filter: exact per-pixel ownership for alignment/labels
        id_mats = {role: _emission(f"id_{role}", c) for role, c in ID_COLORS.items()}
        for obj in scene.objects:
            if obj.type == "MESH":
                saved[obj.name] = [m for m in obj.data.materials]
                obj.data.materials.clear(); obj.data.materials.append(id_mats[obj.get("role", "table")])
        world_strength = scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        scene.cycles.samples = 1; scene.cycles.use_denoising = False; scene.cycles.filter_width = 0.01
        vt, exp = scene.view_settings.view_transform, scene.view_settings.exposure
        scene.view_settings.view_transform = "Standard"; scene.view_settings.exposure = 0
    else:
        scene.cycles.samples = int(samples); scene.cycles.use_denoising = True
        scene.cycles.filter_width = 1.5; scene.cycles.seed = int(seed)
        scene.cycles.max_bounces = int(opts.get("bounces") or 6)
        if opts.get("denoiser"):
            scene.cycles.denoiser = opts["denoiser"]
        if opts.get("adaptive_threshold"):
            scene.cycles.use_adaptive_sampling = True; scene.cycles.adaptive_threshold = float(opts["adaptive_threshold"])
    if opts.get("png_compression") is not None:
        scene.render.image_settings.compression = int(opts["png_compression"])
    path = tempfile.mktemp(suffix=".png")
    scene.render.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    if mode == "id":
        for obj in scene.objects:
            if obj.name in saved:
                obj.data.materials.clear()
                for m in saved[obj.name]:
                    obj.data.materials.append(m)
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = world_strength
        scene.view_settings.view_transform = vt; scene.view_settings.exposure = exp
    data = open(path, "rb").read(); os.remove(path)
    return data, dt


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _authorized(self):
        return TOKEN is None or self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_GET(self):
        if not self._authorized():
            return self._send(401, b"{}")
        self._send(200, json.dumps({"ok": True, "scene_key": STATE["scene_key"], "blender": bpy.app.version_string}).encode())

    def do_POST(self):
        if not self._authorized():
            return self._send(401, b"{}")
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/scene":
                t0 = time.time(); build_scene(body["static"], body["config"])
                self._send(200, json.dumps({"scene_key": STATE["scene_key"], "build_s": time.time() - t0}).encode())
            elif self.path == "/render":
                if body.get("scene_key") is not None and body["scene_key"] != STATE["scene_key"]:
                    return self._send(409, json.dumps({"error": "scene_key mismatch", "server": STATE["scene_key"]}).encode())
                apply_frame(body["poses"], body["camera"])
                png, dt = render(body.get("mode", "photo"), body.get("samples", 64), body.get("seed", 0), body)
                self._send(200, png, "image/png", {"X-Render-Seconds": f"{dt:.4f}"})
            elif self.path == "/render_multi":
                pngs, secs = [], []
                for cam in body["cameras"]:
                    apply_frame(body["poses"], cam)
                    png, dt = render(body.get("mode", "photo"), body.get("samples", 64), body.get("seed", 0), body)
                    pngs.append(base64.b64encode(png).decode()); secs.append(dt)
                self._send(200, json.dumps({"pngs": pngs, "seconds": secs}).encode())
            else:
                self._send(404, b"{}")
        except Exception:
            self._send(500, json.dumps({"error": traceback.format_exc()[-3000:]}).encode())


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    port = int(argv[0]) if argv else 8002
    reset_scene()
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
