"""Task-independent visual inputs for categorical coarse-to-fine control."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw

from .grid_actions import GridAdapter, GridSpec
from .pipelines import project_point, robot_state


def _world_polygon(sim, points, size):
    projected = [project_point(sim, [*point, 0.], "overhead", *size) for point in points]
    return projected if all(point is not None for point in projected) else None


def _cell_polygon(sim, spec, column, row, size):
    low, high = spec.coarse_bounds(column, row)
    return _world_polygon(sim, [low, [high[0], low[1]], high, [low[0], high[1]]], size)


def render_coarse_grid(sim, image, observation, spec: GridSpec, reachable=None):
    """Draw calibrated cells, reach mask, and current gripper footprint."""
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result, "RGBA")
    z = float(observation["ee_pos"][2])
    if reachable is None:
        reachable = lambda xy: bool(sim.ik([*map(float, xy), z])["success"])
    cells = []
    for column in spec.column_labels:
        for row in spec.row_labels:
            low, high = spec.coarse_bounds(column, row)
            center = (low + high) / 2
            is_reachable = bool(reachable(center))
            polygon = _cell_polygon(sim, spec, column, row, result.size)
            cells.append({"column": column, "row": row, "reachable": is_reachable,
                          "center_xy": center.tolist()})
            if not polygon:
                continue
            if not is_reachable:
                draw.polygon(polygon, fill=(105, 105, 105, 125))
            draw.line(polygon + [polygon[0]], fill=(235, 240, 245, 220), width=2)
            pixel_center = tuple(np.mean(np.asarray(polygon), axis=0))
            draw.text((pixel_center[0]-8, pixel_center[1]-7), column+row,
                      fill=(10, 22, 35, 255), stroke_width=2, stroke_fill=(255, 255, 255, 240))

    footprint_radius = max(0., float(observation.get("gripper_opening", 0.))) / 2
    ee_xy = np.asarray(observation["ee_pos"][:2], dtype=float)
    center = project_point(sim, [*ee_xy, 0.], "overhead", *result.size)
    edge_x = project_point(sim, [ee_xy[0]+footprint_radius, ee_xy[1], 0.], "overhead", *result.size)
    edge_y = project_point(sim, [ee_xy[0], ee_xy[1]+footprint_radius, 0.], "overhead", *result.size)
    if center and edge_x and edge_y:
        rx, ry = abs(edge_x[0]-center[0]), abs(edge_y[1]-center[1])
        draw.ellipse((center[0]-rx, center[1]-ry, center[0]+rx, center[1]+ry),
                     outline=(255, 40, 40, 255), width=3)
        draw.line([(center[0]-7, center[1]), (center[0]+7, center[1])], fill=(255, 40, 40, 255), width=2)
        draw.line([(center[0], center[1]-7), (center[0], center[1]+7)], fill=(255, 40, 40, 255), width=2)
    boundary = _world_polygon(sim, [spec.xy_min, [spec.xy_max[0], spec.xy_min[1]],
                                    spec.xy_max, [spec.xy_min[0], spec.xy_max[1]]], result.size)
    if boundary:
        draw.line(boundary + [boundary[0]], fill=(250, 205, 40, 255), width=4)
    return result, {"cells": cells, "footprint_radius_m": footprint_radius,
                    "reach_envelope_xy": [list(spec.xy_min), list(spec.xy_max)]}


def render_fine_grid(sim, image, spec: GridSpec, column: str, row: str):
    """Show one-cell context while projecting the true physical 3x3 grid."""
    selected_polygon = _cell_polygon(sim, spec, column, row, image.size)
    if not selected_polygon:
        raise ValueError("selected coarse cell is outside the overhead camera")
    low, high = spec.coarse_bounds(column, row)
    context_low = np.maximum(np.asarray(spec.xy_min), low-spec.coarse_size)
    context_high = np.minimum(np.asarray(spec.xy_max), high+spec.coarse_size)
    context_polygon = _world_polygon(sim, [context_low, [context_high[0], context_low[1]],
                                             context_high, [context_low[0], context_high[1]]], image.size)
    points = np.asarray(context_polygon)
    left, top = np.floor(points.min(axis=0)).astype(int)
    right, bottom = np.ceil(points.max(axis=0)).astype(int)
    left, top = max(0, left), max(0, top)
    right, bottom = min(image.width, right), min(image.height, bottom)
    if right-left < 3 or bottom-top < 3:
        raise ValueError("selected coarse crop is empty")
    crop = image.convert("RGB").crop((left, top, right, bottom)).resize(image.size, Image.Resampling.BICUBIC)
    width, height = crop.size
    sx, sy = width/(right-left), height/(bottom-top)
    def transform(point):
        return ((point[0]-left)*sx, (point[1]-top)*sy)
    selected_pixels = [transform(point) for point in selected_polygon]
    # Context helps identify objects crossing a coarse boundary, but only the
    # selected model cell remains bright and selectable.
    rgba = crop.convert("RGBA")
    shade = Image.new("RGBA", crop.size, (0, 0, 0, 105))
    ImageDraw.Draw(shade).polygon(selected_pixels, fill=(0, 0, 0, 0))
    crop = Image.alpha_composite(rgba, shade).convert("RGB")
    draw = ImageDraw.Draw(crop, "RGBA")
    draw.line(selected_pixels+[selected_pixels[0]], fill=(255, 210, 40, 255), width=5)
    vertical_lines=[];horizontal_lines=[]
    for index in (1, 2):
        x = low[0]+(high[0]-low[0])*index/3
        source = [project_point(sim, [x, low[1], 0.], "overhead", *image.size),
                  project_point(sim, [x, high[1], 0.], "overhead", *image.size)]
        line = [transform(point) for point in source];vertical_lines.append(line)
        draw.line(line, fill=(250, 250, 250, 245), width=4)
        y = low[1]+(high[1]-low[1])*index/3
        source = [project_point(sim, [low[0], y, 0.], "overhead", *image.size),
                  project_point(sim, [high[0], y, 0.], "overhead", *image.size)]
        line = [transform(point) for point in source];horizontal_lines.append(line)
        draw.line(line, fill=(250, 250, 250, 245), width=4)
    fine_cells=[]
    for index, label in enumerate(spec.fine_labels):
        center_xy=spec.target_xy(column,row,label)
        source=project_point(sim,[*center_xy,0.],"overhead",*image.size)
        x,y=transform(source)
        draw.text((x-5, y-8), label, fill=(8, 20, 34, 255), stroke_width=3,
                  stroke_fill=(255, 255, 255, 255))
        fine_cells.append({"label":label,"center_xy":center_xy.tolist(),"center_pixel":[x,y]})
    return crop, {"selected_coarse": [column, row], "crop_box_pixels": [left, top, right, bottom],
                  "context_padding_cells":1,"selected_polygon_pixels":[list(point) for point in selected_pixels],
                  "vertical_grid_lines":vertical_lines,"horizontal_grid_lines":horizontal_lines,
                  "fine_cells":fine_cells}


def _question_payload(question):
    suffixes = {
        "coarse_column": "Which COLUMN contains the desired next planar target? Answer one column letter.",
        "coarse_row": "Which ROW contains the desired next planar target? Answer one row digit.",
        "fine_cell": "Which numbered 3x3 cell contains the desired next planar target? Answer one digit.",
        "macro": "Which physically appropriate macro should execute next? Answer one option label.",
    }
    return {"kind": question.kind, "prompt_suffix": suffixes[question.kind],
            "options": dict(question.options)}


def _apply_visual_transform(base,transform,*,sim,images,adapter):
    """Apply a hash-pinned representation transform without changing decisions.

    A transform may change only prompt text, images, image names and visual
    metadata.  Its identity is serialized into every transformed input so a
    collector can archive and replay the exact implementation/configuration.
    """
    if not callable(getattr(transform,'identity',None)) or not callable(getattr(transform,'apply',None)):
        raise ValueError('visual transform must expose identity() and apply()')
    identity=transform.identity()
    try:encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as exc:raise ValueError('visual transform identity must be finite JSON') from exc
    if not isinstance(identity,dict) or not identity.get('name') or not identity.get('version') or not identity.get('source_sha256'):
        raise ValueError('visual transform identity requires name, version, and source_sha256')
    before=copy.deepcopy(base)
    candidate=transform.apply(copy.deepcopy(base),sim=sim,images=images,adapter=adapter)
    if not isinstance(candidate,dict):raise ValueError('visual transform must return a grid input mapping')
    protected=('stage','robot_state','questions','passes_required','auxiliary')
    if any(candidate.get(key)!=before.get(key) for key in protected):
        raise ValueError('visual transform changed protected model-visible semantics')
    allowed=set(before)|{'visual_input_transform'}
    if set(candidate)-allowed:raise ValueError('visual transform added unsupported fields')
    if not isinstance(candidate.get('prompt'),str) or not candidate['prompt']:
        raise ValueError('visual transform prompt must be nonempty')
    if not isinstance(candidate.get('images'),list) or not candidate['images']:
        raise ValueError('visual transform images must be a nonempty list')
    if not isinstance(candidate.get('image_names'),list) or len(candidate['images'])!=len(candidate['image_names']):
        raise ValueError('visual transform image names must match images')
    candidate['visual_input_transform']={**json.loads(encoded),'applied':True}
    return candidate


def _apply_auxiliary_transform(base,transform,*,sim,images,adapter):
    """Apply an optional evidence augmentation without changing action semantics."""
    if not callable(getattr(transform,'identity',None)) or not callable(getattr(transform,'apply',None)):
        raise ValueError('auxiliary transform must expose identity() and apply()')
    identity=transform.identity()
    try:encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as exc:raise ValueError('auxiliary transform identity must be finite JSON') from exc
    if not isinstance(identity,dict) or not identity.get('name') or not identity.get('version') or not identity.get('source_sha256'):
        raise ValueError('auxiliary transform identity requires name, version, and source_sha256')
    before=copy.deepcopy(base);candidate=transform.apply(copy.deepcopy(base),sim=sim,images=images,adapter=adapter)
    if not isinstance(candidate,dict):raise ValueError('auxiliary transform must return a grid input mapping')
    protected=('stage','robot_state','questions','passes_required')
    if any(candidate.get(key)!=before.get(key) for key in protected):
        raise ValueError('auxiliary transform changed protected action semantics')
    allowed=set(before)|{'auxiliary_input_transform','sam3_marks'}
    if set(candidate)-allowed:raise ValueError('auxiliary transform added unsupported fields')
    if not isinstance(candidate.get('prompt'),str) or not candidate['prompt']:
        raise ValueError('auxiliary transform prompt must be nonempty')
    if not isinstance(candidate.get('images'),list) or not candidate['images']:
        raise ValueError('auxiliary transform images must be a nonempty list')
    if not isinstance(candidate.get('image_names'),list) or len(candidate['images'])!=len(candidate['image_names']):
        raise ValueError('auxiliary transform image names must match images')
    if not isinstance(candidate.get('auxiliary'),list):raise ValueError('auxiliary questions must be a list')
    candidate['auxiliary_input_transform']={**json.loads(encoded),'applied':True}
    return candidate


def build_grid_input(sim, observation, images, adapter: GridAdapter, visual_transform=None,
                     auxiliary_transform=None):
    """Build model-visible data without object coordinates or oracle state."""
    overhead = Image.fromarray(np.asarray(images["overhead"])).convert("RGB")
    stage = adapter.stage(observation)
    if stage == "fine":
        processed, metadata = render_fine_grid(sim, overhead, adapter.spec, *adapter.coarse)
        image_list = [overhead, processed]
        image_names = ["full overhead context", "selected coarse fine grid"]
        overlay_description = ("The first image is the full overhead scene. In the second image, the yellow outline is the selected coarse cell, "
                               "the numbered 3x3 grid is projected from its physical table coordinates, and grey context is outside that selected cell.")
    else:
        processed, metadata = render_coarse_grid(sim, overhead, observation, adapter.spec)
        image_list = [processed]
        image_names = ["calibrated overhead grid"]
        overlay_description = ("The overlay is calibrated to the table plane. Grey cells are outside the current kinematic reach estimate; "
                               "the red outline is the current gripper footprint and the yellow boundary is the declared reach envelope.")
    if adapter.coarse:
        metadata["selected_coarse"] = list(adapter.coarse)
    questions = [_question_payload(question) for question in adapter.questions(observation)]
    state = robot_state(observation)
    state["grid_stage"] = stage
    state["selected_coarse"] = list(adapter.coarse) if adapter.coarse else None
    goal = str(observation.get("goal", "Complete the manipulation goal shown in the scene."))
    prompt = (f"Goal: {goal}\nUse only visible scene evidence and the robot readings. "
              f"{overlay_description} Choose categories only; code owns "
              "all coordinates, IK, limits, and macro execution. Do not invent coordinates.\n"
              f"Robot readings: {state!r}\nGrid stage: {stage}.")
    result={"stage": stage, "prompt": prompt, "images": image_list, "image_names": image_names, "robot_state": state,
            "questions": questions, "grid_metadata": metadata,
            "passes_required": len(questions), "auxiliary": []}
    if visual_transform is not None:
        result=_apply_visual_transform(result,visual_transform,sim=sim,images=images,adapter=adapter)
    if auxiliary_transform is not None:
        result=_apply_auxiliary_transform(result,auxiliary_transform,sim=sim,images=images,adapter=adapter)
    return result
