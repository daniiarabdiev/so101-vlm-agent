"""Question-keyed calibration and readout for categorical grid policies.

Calibration artifacts are bound to the exact model/runtime/prompt/corpus and
acceptable-set assessment identities.  Option permutations are diagnostic
inputs only; deployment applies one selected correction to the frozen layout.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from so101_vlm.debiasing import (
    apply_contextual_correction,
    fit_diagonal_affine,
    score_sets,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def calibration_content_hash(calibration):
    return _digest({key:value for key,value in calibration.items() if key!="calibration_hash"})


JUDGMENT_KINDS=frozenset({"verified_holding","completion","anomaly"})


def question_stage(questions):
    """Return the policy stage while tolerating appended judgment questions."""
    kinds=[question["kind"] for question in questions]
    policy=[kind for kind in kinds if kind not in JUDGMENT_KINDS]
    if policy==["coarse_column","coarse_row"]:return "coarse"
    if policy==["fine_cell"]:return "fine"
    if policy==["macro"]:return "macro"
    if not policy and kinds and all(kind in JUDGMENT_KINDS for kind in kinds):return "judgment"
    raise ValueError(f"unsupported question layout {kinds}")


def question_stage_for(question,questions):
    return "judgment" if question["kind"] in JUDGMENT_KINDS else question_stage(questions)


def question_key(stage, question):
    options=question.get("options")
    if not isinstance(options,dict) or len(options)<2:
        raise ValueError("question requires at least two ordered options")
    payload={"stage":str(stage),"question_kind":str(question["kind"]),
             "ordered_options":[[str(label),str(semantic)] for label,semantic in options.items()]}
    return _canonical(payload)


@dataclass(frozen=True)
class CalibrationSignature:
    model_id: str
    model_revision: str
    quantization: str
    backend: str
    prompt_style_hash: str
    corpus_hash: str
    assessment_hash: str
    schema_version: int = 1

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key!="schema_version" and not str(value):raise ValueError(f"signature field {key} is required")

    def to_dict(self):return asdict(self)

    @property
    def hash(self):return _digest(self.to_dict())


def _row_values(row, key):
    labels=list(row["question"]["options"])
    scores=row.get("label_scores")
    if question_key(row["stage"],row["question"])!=key:
        raise ValueError("row question key mismatch")
    if not isinstance(scores,dict) or set(scores)!=set(labels):
        raise ValueError("row requires complete label scores")
    values=np.asarray([scores[label] for label in labels],dtype=float)
    if not np.isfinite(values).all():raise ValueError("row scores must be finite")
    accepted=set(row.get("acceptable_labels") or [])
    if not accepted:return None
    unknown=accepted-set(labels)
    if unknown:raise ValueError(f"unknown acceptable labels: {sorted(unknown)}")
    return values,{labels.index(label) for label in accepted}


def _metrics(rows,key,transform):
    parsed=[_row_values(row,key) for row in rows]
    parsed=[value for value in parsed if value is not None]
    if not parsed:return None
    logits=np.stack([transform(values) for values,_ in parsed])
    return score_sets(logits,[accepted for _,accepted in parsed])


def fit_grid_calibration(rows, signature: CalibrationSignature, *, min_affine_rows=20):
    """Select raw/contextual/diagonal-affine per exact semantic question key."""
    if min_affine_rows<2:raise ValueError("min_affine_rows must be at least two")
    rows=list(rows); groups={}; exclusions=[]
    for row in rows:
        if row.get("partition") not in ("fit","selection"):
            raise ValueError("calibration rows must be fit or selection only")
        key=question_key(row["stage"],row["question"])
        parsed=_row_values(row,key)
        if parsed is None:
            exclusions.append({"frame_id":row.get("frame_id"),"key":key,"reason":"empty_acceptable_set"})
            continue
        groups.setdefault(key,[]).append(row)
    artifacts={}
    for key,key_rows in sorted(groups.items()):
        fit=[row for row in key_rows if row["partition"]=="fit"]
        selection=[row for row in key_rows if row["partition"]=="selection"]
        if not fit:
            artifacts[key]={"supported":False,"selected_method":"raw","fit_n":0,"selection_n":len(selection),
                            "reason":"no_fit_rows"}
            continue
        candidates={"raw":{"transform":{"kind":"raw"}}}
        priors=[row.get("content_free_logits") for row in fit]
        if all(isinstance(prior,dict) for prior in priors):
            labels=list(fit[0]["question"]["options"])
            vectors=[np.asarray([prior[label] for label in labels],dtype=float) for prior in priors]
            if all(np.isfinite(vector).all() for vector in vectors) and all(np.allclose(vector,vectors[0]) for vector in vectors[1:]):
                candidates["contextual"]={"transform":{"kind":"contextual","content_free_logits":vectors[0].tolist()}}
        fit_parsed=[_row_values(row,key) for row in fit]
        if len(fit_parsed)>=min_affine_rows:
            calibration=fit_diagonal_affine(np.stack([value[0] for value in fit_parsed]),[value[1] for value in fit_parsed])
            candidates["affine"]={"transform":{"kind":"affine",**calibration.to_dict()}}

        def transform_for(candidate):
            transform=candidate["transform"]
            if transform["kind"]=="raw":return lambda values:values
            if transform["kind"]=="contextual":
                prior=np.asarray(transform["content_free_logits"],dtype=float)
                return lambda values:apply_contextual_correction(values,prior)
            scales=np.asarray(transform["scales"]);offsets=np.asarray(transform["offsets"])
            return lambda values:values*scales+offsets

        for candidate in candidates.values():
            fn=transform_for(candidate)
            candidate["fit_metrics"]=_metrics(fit,key,fn)
            candidate["selection_metrics"]=_metrics(selection,key,fn)
        eligible=[(name,value) for name,value in candidates.items() if value["selection_metrics"] is not None]
        priority={"raw":0,"contextual":1,"affine":2}
        if eligible:
            selected_name,_=max(eligible,key=lambda item:(item[1]["selection_metrics"]["agreement"],
                                                          -item[1]["selection_metrics"]["set_nll"],
                                                          -priority[item[0]]))
            reason="held_out_selection"
        else:selected_name,reason="raw","no_selection_rows"
        if len(fit)<min_affine_rows and selected_name=="raw":reason=f"raw_retained_fit_n_below_{min_affine_rows}"
        artifacts[key]={"supported":True,"selected_method":selected_name,"fit_n":len(fit),
                        "selection_n":len(selection),"reason":reason,"candidates":candidates,
                        "selected_transform":candidates[selected_name]["transform"],
                        "permutation_diagnostic_only":True}
    payload={"schema_version":1,"signature":signature.to_dict(),"signature_hash":signature.hash,
             "rows_sha256":_digest(rows),
             "row_counts":{"fit":sum(row.get("partition")=="fit" for row in rows),
                           "selection":sum(row.get("partition")=="selection" for row in rows)},
             "min_affine_rows":int(min_affine_rows),"questions":artifacts,"exclusions":exclusions}
    payload["calibration_hash"]=calibration_content_hash(payload)
    return payload


def apply_question_calibration(calibration,stage,question,label_scores):
    key=question_key(stage,question);entry=calibration.get("questions",{}).get(key)
    if not entry or not entry.get("supported"):
        raise ValueError(f"missing supported calibration for question key {key}")
    labels=list(question["options"])
    if set(label_scores)!=set(labels):raise ValueError("complete matching label scores required")
    values=np.asarray([label_scores[label] for label in labels],dtype=float)
    transform=entry["selected_transform"];kind=transform["kind"]
    if kind=="raw":corrected=values
    elif kind=="contextual":corrected=apply_contextual_correction(values,transform["content_free_logits"])
    elif kind=="affine":corrected=values*np.asarray(transform["scales"])+np.asarray(transform["offsets"])
    else:raise ValueError(f"unknown calibration transform {kind}")
    winner=int(np.argmax(corrected))
    return {label:float(corrected[index]) for index,label in enumerate(labels)},labels[winner],key


class CalibratedGridBackend:
    """Apply a frozen calibration to one exact backend/runtime identity."""
    def __init__(self,backend,calibration,runtime_signature: CalibrationSignature,*,artifact_bytes=None):
        if calibration.get("calibration_hash")!=calibration_content_hash(calibration):
            raise ValueError("calibration artifact content hash mismatch")
        if calibration.get("signature")!=runtime_signature.to_dict() or calibration.get("signature_hash")!=runtime_signature.hash:
            raise ValueError("calibration signature does not match runtime")
        actual={"model_id":getattr(backend,"model_id",None),"model_revision":getattr(backend,"model_revision",None),
                "quantization":getattr(backend,"quantization",None),
                "backend":getattr(backend,"backend",getattr(backend,"backend_id",None)),
                "prompt_style_hash":getattr(backend,"prompt_style_hash",None)}
        expected={key:getattr(runtime_signature,key) for key in actual}
        if {key:str(value) for key,value in actual.items()}!={key:str(value) for key,value in expected.items()}:
            raise ValueError("actual backend identity does not match calibration signature")
        if artifact_bytes is None:artifact_bytes=(json.dumps(calibration,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
        if json.loads(bytes(artifact_bytes))!=calibration:raise ValueError("calibration artifact bytes do not match parsed artifact")
        self.wrapped=backend;self.calibration=calibration;self.model_id=backend.model_id;self.model_revision=backend.model_revision
        self.backend=runtime_signature.backend;self.quantization=runtime_signature.quantization
        self.precision=getattr(backend,"precision",None);self.compute_dtype=getattr(backend,"compute_dtype",None)
        if hasattr(backend,"model_asset_identity"):
            self.model_asset_identity=copy.deepcopy(backend.model_asset_identity)
        if hasattr(backend,"model_asset_identity_sha256"):
            self.model_asset_identity_sha256=str(backend.model_asset_identity_sha256)
        self.prompt_style_hash=runtime_signature.prompt_style_hash;self.calibration_hash=calibration["calibration_hash"]
        self.calibration_artifact_bytes=bytes(artifact_bytes)
        self.calibration_artifact_file_sha256=hashlib.sha256(self.calibration_artifact_bytes).hexdigest()
        self._closed=False

    def close(self):
        if self._closed:return
        self._closed=True
        close=getattr(self.wrapped,"close",None)
        if close is not None:close()

    def decide_many(self,*,prompt,images,questions,mode):
        if mode!="readout":raise ValueError("calibrated grid backend requires readout mode")
        raw=self.wrapped.decide_many(prompt=prompt,images=images,questions=questions,mode=mode)
        response=copy.deepcopy(raw)
        for answer,question in zip(response.get("answers",[]),questions):
            stage=question_stage_for(question,questions)
            corrected,winner,key=apply_question_calibration(self.calibration,stage,question,answer.get("label_scores") or {})
            answer["raw_label"]=answer.get("label");answer["raw_label_scores"]=answer.get("label_scores")
            answer["label_scores"]=corrected;answer["label"]=winner;answer["calibration_question_key"]=key
        response["uncalibrated_response"]=raw;response["calibration_hash"]=self.calibration_hash
        return response
