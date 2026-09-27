"""Categorical readout + pointing clients for a self-hosted vLLM OpenAI server (Run 3).

Request format is the Run 2 remote readout contract (run2/providers/runpod_pod_remote_27b.py):
one user message = formatted text then ordered PNG data URLs; text = prompt, blank line,
question suffix, "Allowed options in order:", "<label> = <semantic>" lines, direct suffix;
temperature 0, max_tokens 1, allowed_token_ids/logprob_token_ids restricted to the label ids.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from run2.providers.runpod_pod_remote_27b import DIRECT_SUFFIX

QWEN_LABEL_TOKEN_IDS = {**{str(i): 15 + i for i in range(1, 10)}, **{chr(65 + i): 32 + i for i in range(5)}}


def png_data_url(image: Any) -> tuple[str, str]:
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            value = im.convert("RGB")
    elif isinstance(image, Image.Image):
        value = image.convert("RGB")
    else:
        value = Image.fromarray(np.asarray(image)).convert("RGB")
    buf = io.BytesIO(); value.save(buf, format="PNG"); data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()


def formatted_prompt(prompt: str, question: dict) -> str:
    labels = list(question["options"])
    lines = "\n".join(f"{label} = {question['options'][label]}" for label in labels)
    return f"{prompt}\n\n{question.get('prompt_suffix', '')}\nAllowed options in order:\n{lines}\n{DIRECT_SUFFIX}"


class VllmClient:
    def __init__(self, base_url: str, api_key: str, model: str, label_token_ids: dict[str, int] | None = None,
                 extra_body: dict | None = None, timeout_s: float = 300):
        self.model = model
        self.label_token_ids = dict(label_token_ids or QWEN_LABEL_TOKEN_IDS)
        self.extra_body = dict(extra_body or {})
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"},
                                   timeout=httpx.Timeout(timeout_s, connect=30))

    def _post(self, body: dict) -> tuple[dict, float]:
        for attempt in range(4):
            t0 = time.monotonic()
            try:
                r = self.client.post("/v1/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1)); continue
            dt = time.monotonic() - t0
            if r.status_code >= 500 and attempt < 3:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json(), dt
        raise RuntimeError("unreachable")

    def readout(self, prompt: str, images: list, question: dict) -> dict:
        labels = list(question["options"])
        ids = [self.label_token_ids[label] for label in labels]
        text = formatted_prompt(prompt, question)
        urls = [png_data_url(im) for im in images]
        body = {"model": self.model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                              *[{"type": "image_url", "image_url": {"url": u}} for u, _ in urls]]}],
                "temperature": 0, "stream": False, "max_tokens": 1, "logprobs": True, "top_logprobs": len(ids),
                "logprob_token_ids": ids, "allowed_token_ids": ids, "return_tokens_as_token_ids": True,
                **self.extra_body}
        raw, dt = self._post(body)
        rows = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        wanted = {self.label_token_ids[l]: l for l in labels}
        found = {}
        for row in rows:
            tid = row.get("token_id")
            m = re.fullmatch(r"token_id:(\d+)", str(row.get("token") or ""))
            if tid is None and m:
                tid = int(m.group(1))
            if tid in wanted and isinstance(row.get("logprob"), (int, float)) and math.isfinite(row["logprob"]):
                found[wanted[tid]] = float(row["logprob"])
        if set(found) != set(labels):
            raise ValueError(f"incomplete label scores: got {sorted(found)} want {labels}")
        label = max(labels, key=lambda l: (found[l], -labels.index(l)))
        return {"label": label, "label_scores": found, "latency_s": dt, "image_sha256": [h for _, h in urls],
                "usage": raw.get("usage"), "request_id": raw.get("id"), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def generate(self, text: str, images: list, max_tokens: int = 64) -> dict:
        urls = [png_data_url(im) for im in images]
        body = {"model": self.model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                              *[{"type": "image_url", "image_url": {"url": u}} for u, _ in urls]]}],
                "temperature": 0, "stream": False, "max_tokens": max_tokens, **self.extra_body}
        raw, dt = self._post(body)
        return {"text": raw["choices"][0]["message"]["content"], "finish_reason": raw["choices"][0].get("finish_reason"),
                "latency_s": dt, "usage": raw.get("usage"), "request_id": raw.get("id"),
                "image_sha256": [h for _, h in urls]}

    def tokenize(self, text: str) -> list[int]:
        r = self.client.post("/tokenize", json={"model": self.model, "prompt": text, "add_special_tokens": False})
        r.raise_for_status()
        return r.json()["tokens"]
