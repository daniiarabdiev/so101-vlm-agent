"""Build the prespecified CPU-only workspace-legibility comparison corpus."""
from __future__ import annotations

import hashlib
import json
import shutil
import copy
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run2.grid_benchmark import PHASE_SCHEDULE, _advance_to_phase
from so101_vlm.embodiment import Embodiment
from so101_vlm.grid_actions import GridAdapter
from so101_vlm.grid_oracle import GridOraclePolicy
from so101_vlm.pipelines import project_point


MARGIN_M=0.075
OUTPUT_SIZE=(448,448)
FONT_PATH=Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_SIZE=22
SPATIAL_PHASES=frozenset({"approach_coarse","approach_fine","carry_coarse","carry_fine"})


class WorkspaceCropVisualTransform:
    """Runtime form of the exact saved workspace-crop representation.

    Macro inputs are returned unchanged.  Coarse/fine inputs use the same
    renderer and prompt replacement as the frozen CPU corpus builder.
    """
    def identity(self):
        return {"name":"workspace_crop_axis_labels_and_fine_corner_labels_v1","version":1,
                "source_sha256":sha(Path(__file__)),
                "config":{"output_size":list(OUTPUT_SIZE),"coarse_margin_m":MARGIN_M,
                          "font_path":str(FONT_PATH),"font_sha256":sha(FONT_PATH),
                          "font_size":FONT_SIZE,"macro_mode":"unchanged"}}

    def apply(self,base,*,sim,images,adapter):
        result=copy.deepcopy(base);stage=result["stage"]
        if stage not in ("coarse","fine"):return result
        raw=Image.fromarray(np.asarray(images["overhead"])).convert("RGB")
        if stage=="coarse":
            processed,metadata=render_coarse_variant(sim,raw,result["robot_state"],adapter.spec)
            name="task-independent workspace legibility crop"
        else:
            if not adapter.coarse:raise ValueError("fine workspace transform requires selected coarse cell")
            processed,metadata=render_fine_variant(sim,raw,adapter.spec,*adapter.coarse)
            name="selected coarse fine grid with corner labels"
        result["prompt"]=variant_prompt(result["prompt"],stage)
        result["images"]=[raw,processed]
        result["image_names"]=["full overhead context",name]
        result["grid_metadata"]=metadata
        return result


def _json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f"unsupported JSON value {type(value).__name__}")
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",", ":"),allow_nan=False,default=_json_default)
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def pixel_sha(image):return sha_bytes(np.asarray(image).tobytes())


def _world_polygon(sim,points,size=OUTPUT_SIZE):
    result=[project_point(sim,[*map(float,point),0.],"overhead",*size) for point in points]
    if any(point is None for point in result):raise ValueError("crop geometry is outside overhead projection")
    return result


def _box_from_world(sim,low,high,size=OUTPUT_SIZE):
    points=np.asarray(_world_polygon(sim,[low,[high[0],low[1]],high,[low[0],high[1]]],size))
    left,top=np.floor(points.min(axis=0)).astype(int);right,bottom=np.ceil(points.max(axis=0)).astype(int)
    return tuple(map(int,(max(0,left),max(0,top),min(size[0],right),min(size[1],bottom))))


def _transform_for(box):
    left,top,right,bottom=box;sx=OUTPUT_SIZE[0]/(right-left);sy=OUTPUT_SIZE[1]/(bottom-top)
    return lambda point:((point[0]-left)*sx,(point[1]-top)*sy)


def _text(draw,xy,value,font,anchor="mm"):
    draw.text(xy,value,font=font,anchor=anchor,fill=(8,20,34,255),stroke_width=3,stroke_fill=(255,255,255,255))


def render_coarse_variant(sim,raw,observation,spec):
    """Full declared workspace plus fixed physical margin; labels stay outside cells."""
    low=np.asarray(spec.xy_min)-MARGIN_M;high=np.asarray(spec.xy_max)+MARGIN_M
    box=_box_from_world(sim,low,high,raw.size);transform=_transform_for(box)
    crop=raw.convert("RGB").crop(box).resize(OUTPUT_SIZE,Image.Resampling.BICUBIC)
    draw=ImageDraw.Draw(crop,"RGBA");font=ImageFont.truetype(str(FONT_PATH),FONT_SIZE)
    z=float(observation["ee_pos"][2]);cells=[]
    for column in spec.column_labels:
        for row in spec.row_labels:
            cell_low,cell_high=spec.coarse_bounds(column,row);center=(cell_low+cell_high)/2
            reachable=bool(sim.ik([*map(float,center),z])["success"])
            polygon=[transform(point) for point in _world_polygon(sim,[cell_low,[cell_high[0],cell_low[1]],cell_high,[cell_low[0],cell_high[1]]])]
            if not reachable:draw.polygon(polygon,fill=(105,105,105,125))
            draw.line(polygon+[polygon[0]],fill=(235,240,245,235),width=3)
            cells.append({"column":column,"row":row,"reachable":reachable,"center_xy":center.tolist()})
    boundary=[transform(point) for point in _world_polygon(sim,[spec.xy_min,[spec.xy_max[0],spec.xy_min[1]],spec.xy_max,[spec.xy_min[0],spec.xy_max[1]]])]
    draw.line(boundary+[boundary[0]],fill=(250,205,40,255),width=5)
    xs=[point[0] for point in boundary];ys=[point[1] for point in boundary]
    axis_labels={"columns":{},"rows":{}}
    for column in spec.column_labels:
        cell_low,cell_high=spec.coarse_bounds(column,"1");center=(cell_low+cell_high)/2
        pixel=transform(project_point(sim,[*center,0.],"overhead",*OUTPUT_SIZE))
        position=(pixel[0],min(ys)-20);_text(draw,position,column,font);axis_labels["columns"][column]=list(position)
    for row in spec.row_labels:
        cell_low,cell_high=spec.coarse_bounds("A",row);center=(cell_low+cell_high)/2
        pixel=transform(project_point(sim,[*center,0.],"overhead",*OUTPUT_SIZE))
        position=(min(xs)-20,pixel[1]);_text(draw,position,row,font);axis_labels["rows"][row]=list(position)
    radius=max(0.,float(observation.get("gripper_opening",0.)))/2;ee=np.asarray(observation["ee_pos"][:2])
    center=transform(project_point(sim,[*ee,0.],"overhead",*OUTPUT_SIZE))
    edge_x=transform(project_point(sim,[ee[0]+radius,ee[1],0.],"overhead",*OUTPUT_SIZE))
    edge_y=transform(project_point(sim,[ee[0],ee[1]+radius,0.],"overhead",*OUTPUT_SIZE))
    rx,ry=abs(edge_x[0]-center[0]),abs(edge_y[1]-center[1])
    draw.ellipse((center[0]-rx,center[1]-ry,center[0]+rx,center[1]+ry),outline=(255,40,40,255),width=4)
    draw.line([(center[0]-8,center[1]),(center[0]+8,center[1])],fill=(255,40,40,255),width=3)
    draw.line([(center[0],center[1]-8),(center[0],center[1]+8)],fill=(255,40,40,255),width=3)
    return crop,{"kind":"workspace_crop_v1","crop_box_pixels":list(box),"crop_world_low":low.tolist(),"crop_world_high":high.tolist(),
        "workspace_polygon_pixels":[list(point) for point in boundary],"axis_label_centers":axis_labels,"cells":cells}


def render_fine_variant(sim,raw,spec,column,row):
    """Existing one-cell context geometry with labels moved to cell corners."""
    low,high=spec.coarse_bounds(column,row);context_low=np.maximum(np.asarray(spec.xy_min),low-spec.coarse_size)
    context_high=np.minimum(np.asarray(spec.xy_max),high+spec.coarse_size)
    box=_box_from_world(sim,context_low,context_high,raw.size);transform=_transform_for(box)
    crop=raw.convert("RGB").crop(box).resize(OUTPUT_SIZE,Image.Resampling.BICUBIC)
    selected=[transform(point) for point in _world_polygon(sim,[low,[high[0],low[1]],high,[low[0],high[1]]])]
    rgba=crop.convert("RGBA");shade=Image.new("RGBA",OUTPUT_SIZE,(0,0,0,105));ImageDraw.Draw(shade).polygon(selected,fill=(0,0,0,0))
    crop=Image.alpha_composite(rgba,shade).convert("RGB");draw=ImageDraw.Draw(crop,"RGBA");font=ImageFont.truetype(str(FONT_PATH),FONT_SIZE)
    draw.line(selected+[selected[0]],fill=(255,210,40,255),width=5)
    vertical=[];horizontal=[]
    for index in (1,2):
        x=low[0]+(high[0]-low[0])*index/3
        line=[transform(point) for point in _world_polygon(sim,[[x,low[1]],[x,high[1]]])];vertical.append(line);draw.line(line,fill=(250,250,250,245),width=4)
        y=low[1]+(high[1]-low[1])*index/3
        line=[transform(point) for point in _world_polygon(sim,[[low[0],y],[high[0],y]])];horizontal.append(line);draw.line(line,fill=(250,250,250,245),width=4)
    cells=[];size=(high-low)/3
    for label in spec.fine_labels:
        index=int(label)-1;display_row,x_index=divmod(index,3);y_index=2-display_row
        cell_low=low+size*[x_index,y_index];cell_high=cell_low+size
        polygon=[transform(point) for point in _world_polygon(sim,[cell_low,[cell_high[0],cell_low[1]],cell_high,[cell_low[0],cell_high[1]]])]
        left=min(point[0] for point in polygon);top=min(point[1] for point in polygon)
        _text(draw,(left+8,top+8),label,font,anchor="la")
        center=spec.target_xy(column,row,label);center_pixel=transform(project_point(sim,[*center,0.],"overhead",*OUTPUT_SIZE))
        cells.append({"label":label,"center_xy":center.tolist(),"center_pixel":list(center_pixel),"polygon_pixels":[list(point) for point in polygon]})
    return crop,{"kind":"fine_corner_labels_v1","selected_coarse":[column,row],"crop_box_pixels":list(box),
        "context_padding_cells":1,"selected_polygon_pixels":[list(point) for point in selected],"vertical_grid_lines":vertical,
        "horizontal_grid_lines":horizontal,"fine_cells":cells}


def variant_prompt(original,stage):
    if stage=="coarse":
        old=("The overlay is calibrated to the table plane. Grey cells are outside the current kinematic reach estimate; "
             "the red outline is the current gripper footprint and the yellow boundary is the declared reach envelope.")
        new=("The first image is the full overhead scene. The second is a fixed task-independent crop of the declared workspace plus a 75 mm physical margin. "
             "In the crop, column letters are above the grid and row numbers are left of the grid; grey cells are outside the current kinematic reach estimate, "
             "the red outline is the current gripper footprint, and the yellow boundary is the declared reach envelope.")
    else:
        old=("The first image is the full overhead scene. In the second image, the yellow outline is the selected coarse cell, "
             "the numbered 3x3 grid is projected from its physical table coordinates, and grey context is outside that selected cell.")
        new=("The first image is the full overhead scene. In the second image, the yellow outline is the selected coarse cell, "
             "the 3x3 grid is projected from its physical table coordinates, each number is in its cell's upper-left corner, and grey context is outside that selected cell.")
    if old not in original:raise ValueError("frozen prompt description was not found")
    return original.replace(old,new)


def build_corpus(source,output):
    source=Path(source).resolve();output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("variant output must be empty")
    output.mkdir(parents=True,exist_ok=True);(output/"images").mkdir()
    manifest=read_json(source/"manifest.json")
    project=Path(__file__).resolve().parents[2]
    for name,expected in manifest["source_hashes"].items():
        if sha(project/name)!=expected:raise ValueError(f"frozen corpus source drift: {name}")
    if sha(source/"inputs.jsonl")!=manifest["inputs_sha256"] or sha(source/"labels.jsonl")!=manifest["labels_sha256"]:
        raise ValueError("frozen corpus file hash mismatch")
    inputs=[json.loads(line) for line in (source/"inputs.jsonl").read_text().splitlines()]
    labels={row["frame_id"]:row for row in map(json.loads,(source/"labels.jsonl").read_text().splitlines())}
    selected=[row for row in inputs if row["metadata"]["phase"] in SPATIAL_PHASES]
    input_rows=[];label_rows=[];counts=Counter();image_hashes={};geometry={}
    for record in selected:
        seed=record["metadata"]["seed"];phase=record["metadata"]["phase"]
        sim=Embodiment();adapter=GridAdapter();oracle=GridOraclePolicy(adapter)
        try:
            sim.reset(seed,"A");_advance_to_phase(sim,adapter,oracle,phase);observation=sim.observe();raw=Image.fromarray(sim.render()["overhead"]).convert("RGB")
            if pixel_sha(raw)!=record["images"]["raw_overhead"]["pixel_sha256"]:raise ValueError(f"raw frame reconstruction mismatch: {record['frame_id']}")
            stage=adapter.stage(observation)
            if stage=="coarse":crop,metadata=render_coarse_variant(sim,raw,observation,adapter.spec)
            elif stage=="fine":crop,metadata=render_fine_variant(sim,raw,adapter.spec,*adapter.coarse)
            else:raise ValueError("spatial phase did not reconstruct to coarse/fine stage")
        finally:sim.close()
        frame=record["frame_id"];context_path=Path("images")/(frame+"_full_context.png");crop_path=Path("images")/(frame+"_legibility.png")
        raw.save(output/context_path);crop.save(output/crop_path)
        spatial=[question for question in record["request"]["questions"] if question["kind"].startswith(("coarse_","fine_"))]
        prompt=variant_prompt(record["request"]["prompt"],stage)
        row={"schema_version":1,"frame_id":frame,"metadata":record["metadata"],"adapter_state":record["adapter_state"],
            "robot_state":record["robot_state"],"request":{"prompt":prompt,"image_paths":[str(context_path),str(crop_path)],
                "image_names":["full overhead context","task-independent workspace legibility crop" if stage=="coarse" else "selected coarse fine grid with corner labels"],
                "questions":spatial},"geometry":metadata,
            "images":{"full_context":{"path":str(context_path),"file_sha256":sha(output/context_path),"pixel_sha256":pixel_sha(raw)},
                      "legibility":{"path":str(crop_path),"file_sha256":sha(output/crop_path),"pixel_sha256":pixel_sha(crop)}}}
        row["prompt_pixel_fingerprint"]=digest({"prompt":prompt,"pixels":[row["images"][key]["pixel_sha256"] for key in ("full_context","legibility")]})
        input_rows.append(row);geometry[frame]=metadata;counts.update((record["metadata"]["partition"],q["kind"]) for q in spatial)
        source_label=labels[frame]
        label_rows.append({"schema_version":1,"frame_id":frame,"criterion_version":source_label["criterion_version"],
            "acceptable":{q["kind"]:source_label["acceptable"][q["kind"]] for q in spatial},
            "source":"copied evaluation-only labels from hash-pinned frozen v3 corpus; never model input"})
        image_hashes[str(context_path)]=sha(output/context_path);image_hashes[str(crop_path)]=sha(output/crop_path)
    (output/"inputs.jsonl").write_text("".join(canonical(row)+"\n" for row in input_rows))
    (output/"labels.jsonl").write_text("".join(canonical(row)+"\n" for row in label_rows))
    (output/"geometry.json").write_text(json.dumps(geometry,indent=2,sort_keys=True,default=_json_default)+"\n")
    selection_questions=[{"frame_id":row["frame_id"],"questions":row["request"]["questions"]}
                         for row in input_rows if row["metadata"]["partition"]=="selection"]
    pairing=[{"frame_id":row["frame_id"],"original_prompt":next(source_row["request"]["prompt"] for source_row in selected if source_row["frame_id"]==row["frame_id"]),
              "original_image_paths":next(source_row["request"]["image_paths"] for source_row in selected if source_row["frame_id"]==row["frame_id"]),
              "variant_prompt":row["request"]["prompt"],"variant_image_paths":row["request"]["image_paths"],
              "questions":row["request"]["questions"]}
             for row in input_rows if row["metadata"]["partition"]=="selection"]
    proposal={"schema_version":1,"status":"cpu_prepared_no_inference","variant":"workspace_crop_axis_labels_and_fine_corner_labels_v1",
        "source_corpus":str(source),"source_manifest_sha256":sha(source/"manifest.json"),"source_inputs_sha256":sha(source/"inputs.jsonl"),
        "source_labels_sha256":sha(source/"labels.jsonl"),"partitions":{"fit":list(range(1000,1010)),"selection":list(range(2000,2010))},
        "frames":len(input_rows),"questions":{"fit":sum(v for (p,_),v in counts.items() if p=="fit"),
            "selection":sum(v for (p,_),v in counts.items() if p=="selection"),
            "selection_by_kind":{kind:counts[("selection",kind)] for kind in ("coarse_column","coarse_row","fine_cell")}},
        "paired_hosted_proposal":{"model":"qwen/qwen3.5-9b","provider":"Venice","hosted_exact_weight_revision":None,
            "model_condition":"same corrected-prompt hosted model/provider parameters as the completed qwen35-9b-venice readout screen for both arms",
            "original_calls":60,"variant_calls":60,"total_calls":120,"maximum_additional_usd":1.0,
            "status":"not_launched_requires_parent_review_and_existing_ledger_reservation"},
        "render":{"output_size":list(OUTPUT_SIZE),"coarse_margin_m":MARGIN_M,"interpolation":"Pillow BICUBIC",
            "font_path":str(FONT_PATH),"font_sha256":sha(FONT_PATH),"font_size":FONT_SIZE,
            "coarse_labels":"column letters above grid; row digits left of grid; no in-cell coarse text",
            "fine_labels":"upper-left corner of each projected fine cell","full_context_first":True},
        "limitations":["Single bundled representation variant; any outcome cannot be attributed to crop alone.",
            "Fit frames are retained for geometry/calibration use; the proposed first paid paired comparison scores selection spatial questions only.",
            "No policy seeds 0..19 or ranking seeds 3000..3019 were used."],
        "selection_questions_sha256":digest(selection_questions),"selection_pairing_sha256":digest(pairing),
        "inputs_sha256":sha(output/"inputs.jsonl"),"labels_sha256":sha(output/"labels.jsonl"),"geometry_sha256":sha(output/"geometry.json"),
        "image_hashes_digest":digest(image_hashes),"source_module_sha256":sha(Path(__file__)),"inference_calls_made":0}
    (output/"manifest.json").write_text(json.dumps(proposal,indent=2,sort_keys=True)+"\n")
    return proposal


def read_json(path):return json.loads(Path(path).read_text())
def digest(value):return sha_bytes(canonical(value).encode())
