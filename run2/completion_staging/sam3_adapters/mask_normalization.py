"""Provider-neutral normalization for exact, pixel-registered binary masks."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


MARK_IDS=tuple(str(index) for index in range(1,21))
MARK_ANSWER_LABELS=tuple("ABCDEFGHIJKLMNOPQRST")


def encode_row_major_runs(mask: np.ndarray) -> list[list[int]]:
    flat=np.asarray(mask,dtype=bool).reshape(-1);indices=np.flatnonzero(flat)
    if not len(indices):return []
    runs=[];start=previous=int(indices[0])
    for raw in indices[1:]:
        index=int(raw)
        if index!=previous+1:
            runs.append([start,previous-start+1]);start=index
        previous=index
    runs.append([start,previous-start+1])
    return runs


def decode_row_major_runs(runs: Iterable, size: tuple[int,int]) -> np.ndarray:
    width,height=map(int,size)
    if width<=0 or height<=0:raise ValueError("mask size must be positive")
    flat=np.zeros(width*height,dtype=bool);last_end=0
    for run in runs:
        if (not isinstance(run,(list,tuple)) or len(run)!=2 or
                any(isinstance(value,bool) or not isinstance(value,int) for value in run)):
            raise ValueError("mask runs must be integer [start,length] pairs")
        start,length=run;end=start+length
        if start<last_end or length<=0 or end>len(flat):raise ValueError("mask runs are invalid or overlap")
        flat[start:end]=True;last_end=end
    return flat.reshape(height,width)


def normalize_binary_mask(mask, *, source_size, provider, prompt, prompt_index,
                          provider_mask_index, provider_object_id, bounds, box=None,
                          score=None) -> dict:
    """Normalize an already officially decoded full-source binary raster."""
    width,height=map(int,source_size);array=np.asarray(mask)
    if array.shape!=(height,width):raise ValueError("decoded mask dimensions differ from source image")
    if array.dtype!=np.bool_:
        if not np.issubdtype(array.dtype,np.integer) or not np.isin(array,(0,1)).all():
            raise ValueError("decoded mask must contain only zero and one")
        array=array.astype(bool)
    if not array.any():raise ValueError("individual mask is empty")
    if array.all():raise ValueError("individual mask covers the entire source image")
    packed=np.packbits(array.reshape(-1),bitorder="big").tobytes()
    if score is not None and not isinstance(score,(int,float)):raise ValueError("mask score must be numeric or absent")
    return {"provider":str(provider),"provider_prompt_index":int(prompt_index),
        "provider_mask_index":int(provider_mask_index),"provider_object_id":str(provider_object_id),
        "prompt":str(prompt),"score":float(score) if score is not None else None,
        "size":[width,height],"bounds":[int(value) for value in bounds],
        "box":[float(value) for value in box] if box is not None else None,
        "encoding":"row_major_true_runs_v1","runs":encode_row_major_runs(array),
        "binary_mask_sha256":hashlib.sha256(width.to_bytes(4,"big")+height.to_bytes(4,"big")+packed).hexdigest(),
        "foreground_pixels":int(array.sum())}


def assign_mark_labels(masks: list[dict]) -> list[dict]:
    ordered=sorted(masks,key=lambda item:(item["provider_prompt_index"],item["provider_mask_index"]))
    if len(ordered)>len(MARK_IDS):raise ValueError("mask count exceeds the 20-mark vocabulary")
    return [{**item,"mark_id":MARK_IDS[index],"answer_label":MARK_ANSWER_LABELS[index]}
            for index,item in enumerate(ordered)]


def render_exact_outlines(image, marks: list[dict]) -> Image.Image:
    """Draw deterministic outlines without changing mask/source registration."""
    value=image.copy() if isinstance(image,Image.Image) else Image.fromarray(np.asarray(image))
    value=value.convert("RGB");width,height=value.size
    overlay=np.asarray(value).copy();palette=((235,55,55),(50,155,255),(40,200,110),(245,170,35),(180,80,235))
    label_positions=[]
    for index,mark in enumerate(marks):
        if mark.get("size")!=[width,height]:raise ValueError("mark dimensions differ from image")
        mask=decode_row_major_runs(mark.get("runs"),(width,height))
        interior=mask.copy();interior[0,:]=interior[-1,:]=interior[:,0]=interior[:,-1]=False
        interior[1:-1,1:-1]&=(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
        boundary=mask&~interior;color=palette[index%len(palette)];overlay[boundary]=color
        y,x=np.argwhere(mask)[0];label_positions.append((int(x),int(y),mark["mark_id"],color))
    result=Image.fromarray(overlay);draw=ImageDraw.Draw(result,"RGBA")
    radius=min(9,(width-1)//2,(height-1)//2)
    for x,y,label,color in label_positions:
        x=max(radius,min(width-1-radius,x));y=max(radius,min(height-1-radius,y))
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=(*color,230),outline=(255,255,255,255),width=2)
        box=draw.textbbox((0,0),label);text_width=box[2]-box[0];text_height=box[3]-box[1]
        text_x=max(0,min(width-text_width,round(x-text_width/2)-box[0]))
        text_y=max(0,min(height-text_height,round(y-text_height/2)-box[1]))
        draw.text((text_x,text_y),label,fill=(255,255,255,255))
    return result
