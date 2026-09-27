"""Run 5 model clients.

Self-hosted (vLLM OpenAI server, on the Pod or through an SSH tunnel):
  readout(text, images, options, question, shots=None)
      one token restricted to the option labels; `shots` are fixed worked examples sent first as earlier chat
      turns (user: example images + prompt, assistant: its correct label). They are identical in every call,
      so vLLM's prefix cache serves them (few-shot readout).
  generate(text, images, max_tokens, reasoning, schema)
  point(noun, image, n=1 | 2)
`model` selects the base model or a LoRA adapter served by the same vLLM (e.g. "run5-lora").
Hosted (OpenRouter): generation and pointing only; spend goes to run5/budget/hosted_spend.jsonl (cap USD 30).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import httpx

from run3.readout import png_data_url
from run4.screen.clients import parse_answer  # noqa: F401  (re-exported for the agent)
from run5.agent.prompts import POINT_FOUR, POINT_ONE, POINT_TWO

KEY = Path(os.environ.get("RUN3_KEY_FILE", Path.home() / ".config/runpod/so101-run3-vllm-key"))
ON_POD = bool(os.environ.get("RUN3_ON_POD"))
DIRECT_SUFFIX = "Return exactly one allowed label."  # Run 2/3/4 readout contract
QWEN_REPO = "Qwen/Qwen3.8-27B"
BUDGET = Path(__file__).resolve().parents[1] / "budget"
HOSTED_LOG, HOSTED_CAP = BUDGET / "hosted_spend.jsonl", 30.0
HOSTED = {"gemini38flash": "google/gemini-3.8-flash"}


def formatted(text: str, options: dict, question: str) -> str:
    lines = "\n".join(f"{k} = {v}" for k, v in options.items())
    return f"{text}\n\n{question}\nAllowed options in order:\n{lines}\n{DIRECT_SUFFIX}"


def _content(text: str, images) -> list:
    if os.environ.get("VLM_IMAGE_FORMAT") == "jpeg" and len(images) > 1:   # readouts only: pointing (one image) stays PNG, it
        # moved 50 px with re-encoding alone and once flipped to a wrong spot with JPEG (ep_20260927-195800, 2026-09-27)
        # real-arm speed (2026-09-27): 3 PNGs per call were ~1 MB through the
        return [{"type": "text", "text": text}, *[{"type": "image_url", "image_url": {"url": _jpeg_data_url(im)}} for im in images]]  # proxy
    return [{"type": "text", "text": text}, *[{"type": "image_url", "image_url": {"url": png_data_url(im)[0]}} for im in images]]


def _jpeg_data_url(image, quality: int = 92) -> str:
    import base64, io
    import numpy as _np
    from PIL import Image as _Image
    im = image.convert("RGB") if isinstance(image, _Image.Image) else _Image.fromarray(_np.asarray(image)).convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_points(text: str, w: int, h: int, fmt: str = "qwen") -> list[tuple[float, float]]:
    """All points in pixel coordinates (Qwen/Gemini JSON formats, 0-1000 normalized)."""
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    pts = []
    key = r'"point(?:_2d)?"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    for a, b in re.findall(key, body):
        a, b = float(a), float(b)
        x, y = (b, a) if fmt == "gemini" else (a, b)
        x, y = x / 1000 * w, y / 1000 * h
        if 0 <= x <= w and 0 <= y <= h:
            pts.append((x, y))
    if not pts:  # bare [x, y] pairs
        for a, b in re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", body):
            x, y = float(a) / 1000 * w, float(b) / 1000 * h
            if 0 <= x <= w and 0 <= y <= h:
                pts.append((x, y))
    return pts


class Qwen:
    def __init__(self, model: str = QWEN_REPO, base: str | None = None):
        self.model = model
        base = base or ("http://localhost:8000" if ON_POD else "http://localhost:18000")
        self.client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {KEY.read_text().strip()}"},
                                   timeout=httpx.Timeout(600, connect=30))
        self.label_ids = {}
        for lab in [chr(65 + i) for i in range(9)]:
            toks = self._tokenize(lab)
            if len(toks) != 1:
                raise ValueError(f"label {lab!r} is not one token: {toks}")
            self.label_ids[lab] = toks[0]

    def _tokenize(self, text: str) -> list[int]:
        r = self.client.post("/tokenize", json={"model": QWEN_REPO, "prompt": text, "add_special_tokens": False})
        r.raise_for_status()
        return r.json()["tokens"]

    def _post(self, body: dict) -> tuple[dict, float]:
        for attempt in range(5):
            t0 = time.monotonic()
            try:
                r = self.client.post("/v1/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == 4:
                    raise
                time.sleep(4 * (attempt + 1)); continue
            if r.status_code >= 500 and attempt < 4:
                time.sleep(4 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json(), time.monotonic() - t0
        raise RuntimeError("unreachable")

    def readout(self, text: str, images, options: dict, question: str, shots: list[dict] | None = None) -> dict:
        labels = list(options)
        ids = [self.label_ids[l] for l in labels]
        messages = []
        for s in shots or []:
            messages.append({"role": "user", "content": _content(formatted(s["text"], s["options"], s["question"]), s["images"])})
            messages.append({"role": "assistant", "content": s["answer"]})
        messages.append({"role": "user", "content": _content(formatted(text, options, question), images)})
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 1, "logprobs": True,
                "top_logprobs": len(ids), "logprob_token_ids": ids, "allowed_token_ids": ids, "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        raw, dt = self._post(body)
        rows = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        want = {self.label_ids[l]: l for l in labels}
        found = {}
        for row in rows:
            tid = row.get("token_id")
            m = re.fullmatch(r"token_id:(\d+)", str(row.get("token") or ""))
            if tid is None and m:
                tid = int(m.group(1))
            if tid in want and isinstance(row.get("logprob"), (int, float)) and math.isfinite(row["logprob"]):
                found[want[tid]] = float(row["logprob"])
        if set(found) != set(labels):
            raise ValueError(f"incomplete label scores: {sorted(found)} vs {labels}")
        return {"label": max(labels, key=lambda l: (found[l], -labels.index(l))), "scores": found, "latency_s": dt,
                "prompt_tokens": (raw.get("usage") or {}).get("prompt_tokens"),
                "cached_tokens": ((raw.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}

    def generate(self, text: str, images, max_tokens: int = 256, reasoning: bool = False, schema: dict | None = None) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": _content(text, images)}], "temperature": 0,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": bool(reasoning)}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        raw, dt = self._post(body)
        return {"text": raw["choices"][0]["message"]["content"] or "", "latency_s": dt,
                "finish_reason": raw["choices"][0].get("finish_reason")}

    def point(self, noun: str, image, n: int = 1) -> dict:
        if noun.startswith("FOUR:"):
            prompt, n = POINT_FOUR.format(tray=noun[5:]), 4
        else:
            prompt = POINT_ONE.format(noun=noun) if n == 1 else POINT_TWO.format(what=noun)
        r = self.generate(prompt, [image], max_tokens=300)
        w, h = image.size
        pts = parse_points(r["text"], w, h)
        r["points"] = pts[:n] if len(pts) >= n else None
        return r


def hosted_spent() -> float:
    if not HOSTED_LOG.exists():
        return 0.0
    return sum(float(json.loads(l).get("usd") or 0) for l in HOSTED_LOG.read_text().splitlines())


class Hosted:
    """OpenRouter generation with reasoning (the general-model reference); every call's provider cost is logged."""
    fmt = "gemini"

    def __init__(self, name: str = "gemini38flash"):
        self.model = HOSTED[name]
        key = (Path.home() / ".config/openrouter/so101-vlm-api-key").read_text().strip()
        self.client = httpx.Client(base_url="https://openrouter.ai/api/v1", timeout=180, headers={"Authorization": f"Bearer {key}"})

    def generate(self, text: str, images, max_tokens: int = 2048, reasoning: bool = True, schema=None) -> dict:
        if hosted_spent() >= HOSTED_CAP:
            raise RuntimeError("Run 5 hosted API cap reached")
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens, "usage": {"include": True},
                "messages": [{"role": "user", "content": _content(text, images)}]}
        if not reasoning:
            body["reasoning"] = {"effort": "minimal", "exclude": True}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        r = None
        for attempt in range(5):
            t0 = time.monotonic()
            r = self.client.post("/chat/completions", json=body)
            dt = time.monotonic() - t0
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            raw = r.json()
            usd = float((raw.get("usage") or {}).get("cost") or 0)
            BUDGET.mkdir(parents=True, exist_ok=True)
            with HOSTED_LOG.open("a") as f:
                f.write(json.dumps({"time": time.time(), "model": self.model, "id": raw.get("id"), "usd": usd}) + "\n")
            return {"text": raw["choices"][0]["message"].get("content") or "", "latency_s": dt, "usd": usd,
                    "finish_reason": raw["choices"][0].get("finish_reason")}
        raise RuntimeError(f"hosted call failed: {r.status_code if r is not None else '?'}")

    def point(self, noun: str, image, n: int = 1) -> dict:
        if noun.startswith("FOUR:"):
            noun, n = f"the four inner corners of the {noun[5:]}", 4
        prompt = (f'Point to {noun}. Answer only with JSON [{{"point": [y, x], "label": "target"}}], coordinates normalized to 0-1000.'
                  if n == 1 else f'Point to {noun}. Answer only with JSON, one {{"point": [y, x], "label": "<n>"}} entry per point, '
                  f'coordinates normalized to 0-1000.' if n == 4 else
                  f'Point to {noun}. Answer only with JSON [{{"point": [y, x], "label": "1"}}, {{"point": [y, x], "label": "2"}}], '
                  f'coordinates normalized to 0-1000.')
        r = self.generate(prompt, [image], max_tokens=1024, reasoning=False)
        w, h = image.size
        pts = parse_points(r["text"], w, h, fmt="gemini")
        r["points"] = pts[:n] if len(pts) >= n else None
        return r
