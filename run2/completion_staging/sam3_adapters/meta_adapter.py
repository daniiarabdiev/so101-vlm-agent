"""Isolated official Meta SAM 3.1 adapter with exact parser-based masks."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import httpx
import numpy as np
from PIL import Image
from importlib.metadata import version
from meta_sam_parser import decode_mask_to_raster,image_segmentation_format,parse_responses_stream

from .mask_normalization import normalize_binary_mask


ENDPOINT="https://api.meta.ai/v1/responses"
MODEL_ID="sam-3.1"
MASK_ENCODING="one_bit"
UNIT_COST_USD="0.0025"
PARSER_PACKAGE="meta-sam-parser"
DEFAULT_CREDENTIAL_PATH=Path.home()/".config/meta/so101-sam-api-key"


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def image_data(image):
    value=image.copy() if isinstance(image,Image.Image) else Image.fromarray(np.asarray(image))
    value=value.convert("RGB");array=np.asarray(value);stream=io.BytesIO();value.save(stream,format="PNG",optimize=False)
    return value,stream.getvalue(),hashlib.sha256(array.tobytes()).hexdigest()


def load_credential(path=DEFAULT_CREDENTIAL_PATH):
    value=os.environ.get("META_MODEL_API_KEY")
    if value and value.strip():return value.strip()
    path=Path(path)
    if path.is_file():
        value=path.read_text().strip();return value or None
    return None


def request_body(concept,png):
    if not isinstance(concept,str) or not concept.strip() or "\n" in concept:raise ValueError("concept must be one short nonempty line")
    uri="data:image/png;base64,"+base64.b64encode(png).decode("ascii")
    return {"model":MODEL_ID,"input":[{"type":"message","role":"user","content":[
        {"type":"input_text","text":concept.strip()},
        {"type":"input_image","image_url":uri}]}],"stream":True,
        "metadata":{"mask_encoding":MASK_ENCODING}}


def parse_sse_lines(lines):
    events=[];parts=[]
    def flush():
        if not parts:return
        payload="\n".join(parts);parts.clear()
        if payload=="[DONE]":return
        try:event=json.loads(payload)
        except json.JSONDecodeError as exc:raise ValueError("Meta SSE data is not valid JSON") from exc
        if not isinstance(event,dict) or not isinstance(event.get("type"),str):raise ValueError("Meta SSE event shape is invalid")
        events.append(event)
    for raw in lines:
        line=raw.decode("utf-8") if isinstance(raw,bytes) else str(raw)
        if line=="":flush()
        elif line.startswith("data:"):parts.append(line[5:].lstrip())
        elif line.startswith(("event:","id:","retry:",":")):continue
        else:raise ValueError("Meta response contained a non-SSE line")
    flush();return events


class MetaHttpError(RuntimeError):
    def __init__(self,status_code,safe_headers):
        super().__init__(f"Meta API returned HTTP {status_code}");self.status_code=status_code;self.safe_headers=safe_headers


def post_meta_stream(body,api_key,timeout_s):
    """Send only to the pinned official Meta endpoint and return safe transport evidence."""
    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json","Accept":"text/event-stream"}
    with httpx.stream("POST",ENDPOINT,headers=headers,json=body,timeout=timeout_s) as response:
        safe_headers={key.lower():value for key,value in response.headers.items()
                      if key.lower() in {"x-request-id","request-id","x-meta-request-id","x-model-version"}}
        status=int(response.status_code)
        if status<200 or status>=300:
            response.read();raise MetaHttpError(status,safe_headers)
        events=parse_sse_lines(response.iter_lines())
    return events,{"status_code":status,"headers":safe_headers}


async def _parse_events_async(events):
    async def source():
        for event in events:yield event
    parsed=parse_responses_stream(source(),image_segmentation_format())
    return await parsed.final_result()


def parse_events(events):return asyncio.run(_parse_events_async(events))


def _integer_coordinate(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not float(value).is_integer():
        raise ValueError(f"{name} must be an integer source-pixel coordinate")
    return int(value)


def normalize_parsed_result(result,*,source_size,prompt,prompt_index):
    if result.outcome.status!="completed":raise ValueError("Meta parser outcome is incomplete")
    if result.diagnostics:raise ValueError("Meta parser reported segmentation diagnostics")
    width,height=map(int,source_size)
    boxes={record.object_id:record for record in result.records if record.kind=="box"}
    masks=sorted((record for record in result.records if record.kind=="mask"),key=lambda record:record.order)
    if len({record.object_id for record in masks})!=len(masks):raise ValueError("image response repeats a mask object id")
    normalized=[]
    for provider_index,record in enumerate(masks):
        if record.mask.encoding!=MASK_ENCODING:raise ValueError("Meta mask encoding differs from requested one_bit")
        if record.frame is not None and record.frame.frame_index!=0:raise ValueError("image response mask has a nonzero frame")
        left=_integer_coordinate(record.bounds.left,"mask left");top=_integer_coordinate(record.bounds.top,"mask top")
        right=_integer_coordinate(record.bounds.right,"mask right");bottom=_integer_coordinate(record.bounds.bottom,"mask bottom")
        if not (0<=left<right<=width and 0<=top<bottom<=height):raise ValueError("Meta mask bounds lie outside source image")
        if record.mask.width!=right-left or record.mask.height!=bottom-top:
            raise ValueError("Meta decoded mask dimensions do not match source-pixel bounds")
        raster=np.frombuffer(decode_mask_to_raster(record.mask),dtype=np.uint8)
        if raster.size!=record.mask.width*record.mask.height or not np.isin(raster,(0,1)).all():
            raise ValueError("official parser returned an invalid binary raster")
        full=np.zeros((height,width),dtype=bool);full[top:bottom,left:right]=raster.reshape(record.mask.height,record.mask.width).astype(bool)
        box=boxes.get(record.object_id)
        if box is None:raise ValueError("Meta mask has no paired source-pixel box")
        box_values=[box.left,box.top,box.right,box.bottom]
        if any(not math.isclose(float(actual),float(expected),abs_tol=1e-9,rel_tol=0)
               for actual,expected in zip(box_values,(left,top,right,bottom))):
            raise ValueError("Meta mask bounds and paired box disagree")
        normalized.append(normalize_binary_mask(full,source_size=(width,height),provider="meta",
            prompt=prompt,prompt_index=prompt_index,provider_mask_index=provider_index,
            provider_object_id=record.object_id,bounds=(left,top,right,bottom),box=box_values))
    return normalized


def serving_identity(events,transport):
    completed=next((event for event in reversed(events) if event.get("type")=="response.completed"),{})
    response=completed.get("response") if isinstance(completed.get("response"),dict) else {}
    exposed={key:response.get(key) for key in ("id","model","model_version","revision","created_at","status")
             if response.get(key) is not None}
    header_version=(transport.get("headers") or {}).get("x-model-version")
    version_value=exposed.get("model_version") or exposed.get("revision") or header_version
    return {"requested_model":MODEL_ID,"response_fields":exposed,
        "serving_version":version_value,"serving_version_status":"exposed" if version_value else "unknown_not_exposed"}


class MetaSam31Adapter:
    def __init__(self,cache_root,*,api_key=None,remote_enabled=False,ledger=None,post_stream=None,timeout_s=120):
        self.cache_root=Path(cache_root);self.cache_root.mkdir(parents=True,exist_ok=True)
        self.api_key=api_key;self.remote_enabled=bool(remote_enabled);self.ledger=ledger
        self.post_stream=post_stream or post_meta_stream;self.timeout_s=float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s<=0:raise ValueError("timeout must be finite and positive")

    def _identity(self,pixel_sha,size,concept,prompt_index):
        return {"provider":"meta","endpoint":ENDPOINT,"model":MODEL_ID,"mask_encoding":MASK_ENCODING,
            "parser_package":PARSER_PACKAGE,"parser_version":version(PARSER_PACKAGE),"pixel_sha256":pixel_sha,
            "image_size":list(size),"concept":concept,"prompt_index":int(prompt_index),"unit_cost_usd":UNIT_COST_USD}

    @staticmethod
    def _fallback(identity,reason,*,request_made=False,reservation_id=None,error=None,transport=None,
                  cost_usd=None,client_wall_s=None):
        return {"schema_version":1,"marks_available":False,"masks":[],"fallback":"model_own_grounding_without_marks",
            "reason":reason,"identity":identity,"cache_key":digest(identity),"cache_status":"miss",
            "request_made":request_made,"reservation_id":reservation_id,"cost_usd":cost_usd,
            "client_wall_s":client_wall_s,"error_type":type(error).__name__ if error else None,"transport":transport}

    def segment_one(self,image,concept,*,prompt_index):
        operation_started=time.perf_counter()
        pil,png,pixel_sha=image_data(image);identity=self._identity(pixel_sha,pil.size,concept,prompt_index)
        try:body=request_body(concept,png)
        except (TypeError,ValueError) as exc:return self._fallback(identity,"invalid_request",error=exc)
        cache_path=self.cache_root/f"{digest(identity)}.json"
        if cache_path.is_file():
            try:
                cached=json.loads(cache_path.read_text())
                if cached.get("identity")==identity and cached.get("cache_key")==digest(identity):
                    parsed=parse_events(cached["raw_events"])
                    masks=normalize_parsed_result(parsed,source_size=pil.size,prompt=concept,prompt_index=prompt_index)
                    if masks==cached.get("masks"):
                        return {**cached,"cache_status":"hit","request_made":False,
                            "original_response_cost_usd":cached.get("cost_usd"),"cost_usd":0.0,
                            "original_response_client_wall_s":cached.get("client_wall_s"),
                            "client_wall_s":time.perf_counter()-operation_started}
            except (KeyError,TypeError,ValueError,OSError,json.JSONDecodeError):pass
        if not self.remote_enabled:return self._fallback(identity,"remote_disabled")
        api_key=self.api_key or load_credential()
        if not api_key:return self._fallback(identity,"credential_unavailable")
        if self.ledger is None:return self._fallback(identity,"budget_ledger_unavailable")
        try:
            reservation=self.ledger.reserve(UNIT_COST_USD,{"provider":"meta","model":MODEL_ID,
                "purpose":"sam31_task_independent_scene_marks","cache_key":digest(identity),
                "concept":concept,"prompt_index":prompt_index,"pixel_sha256":pixel_sha})
            if not reservation:raise ValueError("budget ledger returned no reservation identity")
        except Exception as exc:return self._fallback(identity,"budget_reservation_failed",error=exc)
        started=time.perf_counter()
        try:events,transport=self.post_stream(body,api_key,self.timeout_s)
        except Exception as exc:
            details={"error_type":type(exc).__name__}
            if isinstance(exc,MetaHttpError):details.update(status_code=exc.status_code,headers=exc.safe_headers)
            self.ledger.record_pending_note(reservation,reason="meta_sam31_transport_or_provider_charge_unknown",details=details)
            return self._fallback(identity,"provider_call_failed",request_made=True,reservation_id=reservation,error=exc,
                                  transport=details)
        wall=time.perf_counter()-started
        self.ledger.settle(reservation,UNIT_COST_USD,{"provider":"meta","model":MODEL_ID,"image_requests":1})
        try:
            parsed=parse_events(events)
            masks=normalize_parsed_result(parsed,source_size=pil.size,prompt=concept,prompt_index=prompt_index)
        except Exception as exc:
            return self._fallback(identity,"malformed_provider_response",request_made=True,reservation_id=reservation,
                                  error=exc,transport=transport,cost_usd=float(UNIT_COST_USD),client_wall_s=wall)
        record={"schema_version":1,"marks_available":bool(masks),"masks":masks,
            "fallback":None if masks else "model_own_grounding_without_marks",
            "reason":None if masks else "provider_returned_zero_matches","identity":identity,
            "cache_key":digest(identity),"cache_status":"miss_written","request_made":True,
            "reservation_id":reservation,"cost_usd":float(UNIT_COST_USD),"client_wall_s":wall,
            "transport":transport,"serving_identity":serving_identity(events,transport),
            "parser_outcome":asdict(parsed.outcome),"parser_revision":parsed.revision,
            "parser_diagnostics":[asdict(item) for item in parsed.diagnostics],
            "raw_output_sha256":hashlib.sha256(parsed.raw_output.encode()).hexdigest(),"raw_events":events}
        temporary=cache_path.with_suffix(".tmp");temporary.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
        os.replace(temporary,cache_path);return record
