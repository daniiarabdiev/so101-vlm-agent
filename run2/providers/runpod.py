"""Pinned RunPod/vLLM backend for Run 2 grid decisions."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

import httpx
from PIL import Image

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BACKEND_ID = "runpod-worker-vllm-v2.27.0-vllm-0.29.0-bf16-exact-token-v1"
QUANTIZATION = "bf16"
DIRECT_SUFFIX = "Return exactly one label from the ordered allowed set."
THINKING_SUFFIX = "Reason carefully, then end the final answer with `FINAL: <label>` using one allowed label."
THINKING_MAX_TOKENS = 4096
PROMPT_STYLE_SPEC = {
    "schema": 3,
    "composition": ["base_prompt", "blank_line", "question.prompt_suffix",
                    "Allowed options in order:", "ordered <label> = <semantic> lines",
                    "mode_suffix"],
    "direct_suffix": DIRECT_SUFFIX,
    "thinking_suffix": THINKING_SUFFIX,
    "readout": {"allowed_token_ids": True, "logprob_token_ids": True, "max_tokens": 1},
    "thinking": {"enable_thinking": True, "max_tokens": THINKING_MAX_TOKENS,
                 "final_marker": "one separate line exactly FINAL: <label>",
                 "label_scores": None},
}
PROMPT_STYLE_HASH = hashlib.sha256(json.dumps(
    PROMPT_STYLE_SPEC,
    sort_keys=True, separators=(",", ":"),
).encode()).hexdigest()
LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36,
                   "1": 16, "2": 17, "3": 18, "4": 19, "5": 20,
                   "6": 21, "7": 22, "8": 23, "9": 24}


class RunPodCapabilityError(RuntimeError):
    pass


def _image_payload(image: object) -> tuple[str, dict]:
    if isinstance(image, Image.Image):
        value = image.convert("RGB")
    else:
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"image does not exist: {path}")
        with Image.open(path) as loaded:
            value = loaded.convert("RGB")
    buffer = io.BytesIO(); value.save(buffer, format="PNG"); data = buffer.getvalue()
    return ("data:image/png;base64," + base64.b64encode(data).decode(),
            {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "format": "PNG"})


def _labels(question: dict) -> list[str]:
    options = question.get("options")
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError("question requires at least two ordered options")
    labels = [str(label) for label in options]
    if len(labels) != len(set(labels)) or any(label not in LABEL_TOKEN_IDS for label in labels):
        raise ValueError("labels must be distinct pinned single-token grid labels")
    return labels


def _exact_scores(response: dict, labels: list[str]) -> dict[str, float]:
    try:
        rows = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RunPodCapabilityError("response lacks first-token logprobs") from exc
    by_id = {}
    for row in rows:
        match = re.fullmatch(r"token_id:(\d+)", row.get("token") or "")
        score = row.get("logprob")
        if match and isinstance(score, (int, float)) and math.isfinite(score):
            by_id[int(match.group(1))] = float(score)
    missing = [label for label in labels if LABEL_TOKEN_IDS[label] not in by_id]
    if missing:
        raise RunPodCapabilityError(f"incomplete exact label scores; missing {missing}")
    return {label: by_id[LABEL_TOKEN_IDS[label]] for label in labels}


def _thinking_label(response: dict, labels: list[str]) -> str:
    try:
        content = response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RunPodCapabilityError("thinking response lacks final content") from exc
    alternatives = "|".join(map(re.escape, labels))
    matches = re.findall(rf"(?mi)^\s*FINAL\s*:\s*({alternatives})\s*$", content)
    if len(matches) != 1 or matches[0].upper() not in labels:
        raise RunPodCapabilityError("thinking response lacks an unambiguous FINAL label")
    return matches[0].upper()


class RunPodGridBackend:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    backend = BACKEND_ID
    backend_id = BACKEND_ID
    quantization = QUANTIZATION
    prompt_style_hash = PROMPT_STYLE_HASH

    def __init__(self, endpoint_id: str, api_key: str, *, timeout_s: float = 600):
        if not endpoint_id or not api_key:
            raise ValueError("endpoint ID and API key are required")
        self.endpoint_id, self.api_key, self.timeout_s = endpoint_id, api_key, timeout_s
        self.url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1/chat/completions"

    def _call(self, *, prompt: str, images: list[object], question: dict, mode: str) -> dict:
        labels = _labels(question)
        if mode not in {"readout", "thinking"}:
            raise ValueError("mode must be readout or thinking")
        suffix = DIRECT_SUFFIX if mode == "readout" else THINKING_SUFFIX
        option_lines = "\n".join(f"{label} = {question['options'][label]}" for label in labels)
        text = (f"{prompt}\n\n{question.get('prompt_suffix', '')}\n"
                f"Allowed options in order:\n{option_lines}\n{suffix}")
        encoded = [_image_payload(image) for image in images]
        content = [{"type": "text", "text": text}]
        content.extend({"type": "image_url", "image_url": {"url": url}} for url, _ in encoded)
        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": mode == "thinking"}}
        if mode == "readout":
            ids = [LABEL_TOKEN_IDS[label] for label in labels]
            body.update(max_tokens=1, logprobs=True, top_logprobs=len(ids),
                        logprob_token_ids=ids, allowed_token_ids=ids,
                        return_tokens_as_token_ids=True)
        else:
            body.update(max_tokens=THINKING_MAX_TOKENS)
        start = time.monotonic()
        try:
            response = httpx.post(self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body, timeout=self.timeout_s)
        except Exception as exc:
            raise RuntimeError(f"RunPod request transport failure: {type(exc).__name__}") from exc
        elapsed = time.monotonic() - start
        if not response.is_success:
            raise RuntimeError(f"RunPod request failed with HTTP {response.status_code}")
        raw = response.json()
        if raw.get("model") != MODEL_ID:
            raise RunPodCapabilityError(f"served model mismatch: {raw.get('model')!r}")
        usage = raw.get("usage") or {}
        request_audit = {**body, "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            *[{"type": "image_url", "image_url": audit} for _, audit in encoded],
        ]}]}
        result = {"raw_response": raw, "request_audit": request_audit,
                  "request_id": raw.get("id"), "cost_usd": None,
                  "tokens": {"status": "partial", "image": None, "text": None,
                             "input_total": usage.get("prompt_tokens"),
                             "output": usage.get("completion_tokens"),
                             "unavailable_reason": "vLLM does not split image and text input tokens"},
                  "timing": {"total_s": elapsed, "prefill_s": None, "readout_s": None,
                             "unsupported_reason": "stock RunPod vLLM worker does not expose separated timings"},
                  "revision": MODEL_REVISION, "endpoint_id": self.endpoint_id}
        if mode == "readout":
            scores = _exact_scores(raw, labels)
            label = max(labels, key=lambda item: (scores[item], -labels.index(item)))
            result.update(label=label, label_scores=scores)
        else:
            finish_reason = raw.get("choices", [{}])[0].get("finish_reason")
            if finish_reason == "length":
                raise RunPodCapabilityError("thinking response truncated before a final label")
            result.update(label=_thinking_label(raw, labels), label_scores=None)
        return result

    def decide_many(self, *, prompt: str, images: list[object], questions: list[dict], mode: str) -> dict:
        calls = [self._call(prompt=prompt, images=images, question=q, mode=mode) for q in questions]
        outputs = [call["tokens"].get("output") for call in calls]
        return {"answers": [{"kind": q["kind"], "label": call["label"],
                             "label_scores": call["label_scores"]}
                            for q, call in zip(questions, calls)],
                "timing": {"total_s": sum(call["timing"]["total_s"] for call in calls),
                           "prefill_s": None, "readout_s": None,
                           "unsupported_reason": "stock RunPod vLLM worker does not expose separated timings"},
                "tokens": {"status": "partial", "image": None, "text": None,
                           "output": sum(int(v) for v in outputs) if all(v is not None for v in outputs) else None,
                           "unavailable_reason": "vLLM does not split image and text input tokens"},
                "passes": {"image_prefill": len(questions), "question_readouts": len(questions)},
                "shared_prefill": False, "raw_calls": calls, "cost_usd": None}


def backend_factory() -> RunPodGridBackend:
    project = Path(__file__).resolve().parents[2]
    state = json.loads((project / "run2/artifacts/runpod_endpoint/current.json").read_text())
    key = Path.home().joinpath(".config/runpod/so101-vlm-api-key").read_text().strip()
    return RunPodGridBackend(state["endpoint_id"], key)
