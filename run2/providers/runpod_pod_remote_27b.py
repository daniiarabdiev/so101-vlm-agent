"""Canary-sealed remote-only RunPod Pod backend for the Run 2 grid.

This module creates only an HTTP client.  It never loads a local model or
creates, mutates, extends, or deletes a RunPod resource.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BACKEND_ID = "runpod-pod-worker-v1-vllm-0.29.0-qwen38-27b-bf16-exact-token-v1"
CANARY_IDENTITY_SHA256 = "271f6bec9d648dea2d095a10f6815a02ffee158da8911bff2296a7be10a72944"
DIRECT_SUFFIX = "Return exactly one allowed label."
THINKING_SUFFIX = (
    "Think briefly using visible evidence. End with exactly one separate line: "
    "FINAL: <label>, where <label> is one allowed label."
)
THINKING_MAX_TOKENS = 2048
LABEL_TOKEN_IDS = {
    **{str(index): 15 + index for index in range(1, 10)},
    **{chr(65 + index): 32 + index for index in range(5)},
}
_STYLE = {
    "schema": 1,
    "message": "one user message: formatted text then ordered PNG data URLs",
    "question": "prompt, blank line, suffix, ordered label=semantic lines, mode suffix",
    "direct_suffix": DIRECT_SUFFIX,
    "thinking_suffix": THINKING_SUFFIX,
    "readout": {
        "temperature": 0,
        "stream": False,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": "number of options",
        "allowed_token_ids": "ordered option ids",
        "logprob_token_ids": "ordered option ids",
        "return_tokens_as_token_ids": True,
        "enable_thinking": False,
    },
}
PROMPT_STYLE_HASH = hashlib.sha256(
    json.dumps(_STYLE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _png(image: Any) -> tuple[bytes, dict[str, Any]]:
    if isinstance(image, Image.Image):
        value = image.convert("RGB")
    elif isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            value = loaded.convert("RGB")
    else:
        value = Image.fromarray(np.asarray(image)).convert("RGB")
    stream = io.BytesIO()
    value.save(stream, format="PNG")
    data = stream.getvalue()
    return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "format": "PNG"}


def _formatted_prompt(prompt: str, question: dict[str, Any], mode: str) -> str:
    labels = list(question["options"])
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("question labels must be nonempty and distinct")
    unknown = [label for label in labels if label not in LABEL_TOKEN_IDS]
    if unknown:
        raise ValueError(f"question includes unverified labels: {unknown}")
    option_lines = "\n".join(f"{label} = {question['options'][label]}" for label in labels)
    suffix = DIRECT_SUFFIX if mode == "readout" else THINKING_SUFFIX
    return (
        f"{prompt}\n\n{question.get('prompt_suffix', '')}\nAllowed options in order:\n"
        f"{option_lines}\n{suffix}"
    )


def _scores(raw: dict[str, Any], labels: list[str]) -> dict[str, float]:
    try:
        rows = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("remote response lacks first-token logprobs") from exc
    wanted = {LABEL_TOKEN_IDS[label]: label for label in labels}
    found: dict[int, float] = {}
    for row in rows:
        token_id = row.get("token_id")
        match = re.fullmatch(r"token_id:(\d+)", str(row.get("token") or ""))
        if token_id is None and match:
            token_id = int(match.group(1))
        value = row.get("logprob")
        if not isinstance(token_id, int) or token_id not in wanted:
            raise ValueError("remote response contains an unrequested or ambiguous token")
        if token_id in found:
            raise ValueError("remote response repeats a label token")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("remote response contains a non-finite label score")
        found[token_id] = float(value)
    if set(found) != set(wanted):
        raise ValueError("remote response does not cover every requested label")
    return {label: found[LABEL_TOKEN_IDS[label]] for label in labels}


class RunPodPodRemoteBackend:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    backend = BACKEND_ID
    backend_id = BACKEND_ID
    quantization = "bf16"
    precision = "bf16"
    compute_dtype = "bfloat16"
    canary_identity_sha256 = CANARY_IDENTITY_SHA256
    prompt_style_hash = PROMPT_STYLE_HASH
    direct_suffix = DIRECT_SUFFIX
    thinking_suffix = THINKING_SUFFIX
    thinking_max_tokens = THINKING_MAX_TOKENS
    label_token_ids = dict(LABEL_TOKEN_IDS)
    thinking_supported = False
    endpoint_id = "ob7gz5empwhorr"
    remote_runtime_identity = {
        "endpoint_id": endpoint_id,
        "gpu": "NVIDIA A100 80GB PCIe",
        "cuda_catalog_version": "13.0",
        "image_index": "sha256:fd9e5c55c996361aad2543d96d9d85ca055625ee5d1fcefa2d213141deb45e17",
        "image_linux_amd64_manifest": "sha256:9547ad83fd9c6947d0b03749fe43839b360b1a7be188a4bd76249da63cddc5e6",
        "vllm_version": "0.29.0",
        "compatibility_env": "0",
        "hard_deadline_utc": "2026-09-21T10:54:41.650000+00:00",
        "component_timing": "unsupported by stock OpenAI API",
        "thinking": "operationally unsupported: unmeasured on this remote 27B backend",
    }
    model_asset_identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "bf16",
        "compute_dtype": "bfloat16",
        "canary_identity_sha256": CANARY_IDENTITY_SHA256,
    }

    def __init__(self) -> None:
        base_url = os.environ.get(
            "RUN2_REMOTE_VLLM_BASE_URL",
            "https://<pod-id>-8000.proxy.runpod.net",
        ).rstrip("/")
        key_path = Path(
            os.environ.get("RUN2_REMOTE_VLLM_API_KEY_FILE", "/tmp/so101-vllm-api-key-attempt4")
        ).expanduser()
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("remote vLLM API key file is empty")
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def decide_many(
        self,
        *,
        prompt: str,
        images: list[Any],
        questions: list[dict[str, Any]],
        mode: str = "readout",
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("remote backend is closed")
        if mode != "readout":
            raise RuntimeError(
                "thinking is operationally unsupported on this canary-sealed 27B backend"
            )
        if not questions:
            raise ValueError("questions must be nonempty")
        encoded = []
        audits = []
        for image in images:
            data, audit = _png(image)
            encoded.append("data:image/png;base64," + base64.b64encode(data).decode("ascii"))
            audits.append(audit)

        started = time.monotonic()
        answers = []
        raw_calls = []
        output_tokens = 0
        output_tokens_actual = True
        for question in questions:
            labels = list(question["options"])
            ids = [self.label_token_ids[label] for label in labels]
            text = _formatted_prompt(prompt, question, mode)
            audit_content = [
                {"type": "text", "text": text},
                *[{"type": "image_url", "image_url": audit} for audit in audits],
            ]
            request_audit = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": audit_content}],
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": len(ids),
                "logprob_token_ids": ids,
                "allowed_token_ids": ids,
                "return_tokens_as_token_ids": True,
            }
            request_body = json.loads(json.dumps(request_audit))
            for index, encoded_url in enumerate(encoded, start=1):
                request_body["messages"][0]["content"][index]["image_url"] = {"url": encoded_url}
            call_started = time.monotonic()
            response = self._client.post("/v1/chat/completions", json=request_body)
            call_s = time.monotonic() - call_started
            response.raise_for_status()
            raw = response.json()
            if raw.get("model") != self.model_id:
                raise ValueError("remote response model differs from pinned model")
            scores = _scores(raw, labels)
            winner = max(range(len(labels)), key=lambda index: (scores[labels[index]], -index))
            label = labels[winner]
            answers.append({"kind": question["kind"], "label": label, "label_scores": scores})
            raw_calls.append(
                {
                    "kind": question["kind"],
                    "label": label,
                    "label_scores": scores,
                    "revision": self.model_revision,
                    "request_audit": request_audit,
                    "raw_response": raw,
                    "request_id": raw.get("id") or response.headers.get("x-request-id"),
                    "timing_s": call_s,
                    "http_status": response.status_code,
                }
            )
            completion = (raw.get("usage") or {}).get("completion_tokens")
            if isinstance(completion, int) and completion >= 0:
                output_tokens += completion
            else:
                output_tokens_actual = False
        total_s = time.monotonic() - started
        return {
            "answers": answers,
            "raw_calls": raw_calls,
            "revision": self.model_revision,
            "passes": {"image_prefill": len(questions), "question_readouts": len(questions)},
            "shared_prefill": False,
            "timing": {
                "total_s": total_s,
                "prefill_s": None,
                "readout_s": None,
                "unsupported_reason": (
                    "stock vLLM OpenAI API does not expose disjoint image-prefill and "
                    "categorical-readout GPU intervals"
                ),
            },
            "tokens": {
                "status": "partial_actual",
                "input_total": None,
                "image": None,
                "text": None,
                "output": output_tokens if output_tokens_actual else None,
                "unavailable_reason": (
                    "provider usage exposes aggregate prompt tokens but not exact image/text split; "
                    "strict contract leaves all input components null"
                ),
            },
            "cost_usd": None,
            "cost_unavailable_reason": "dedicated Pod is billed by uptime, not per request",
        }


def backend_factory() -> RunPodPodRemoteBackend:
    return RunPodPodRemoteBackend()
