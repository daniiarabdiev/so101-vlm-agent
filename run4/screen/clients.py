"""Model clients for the Run 4 screen.

Self-hosted (vLLM OpenAI server on the Pod): categorical readout (one token restricted to label ids),
free generation (optionally with reasoning), constrained JSON generation, and native pointing.
Hosted (OpenRouter): free generation only (no logprobs on these endpoints); spend is logged to
run4/budget/hosted_spend.jsonl and capped at USD 30.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import httpx

from run3.readout import VllmClient, png_data_url

KEY = Path(os.environ.get("RUN3_KEY_FILE", Path.home() / ".config/runpod/so101-run3-vllm-key"))
ON_POD = bool(os.environ.get("RUN3_ON_POD"))
LOCAL = {  # name: repo, port, readout extra body, reasoning extra body, pointing format
    "qwen27b": ("Qwen/Qwen3.8-27B", 8000, {"chat_template_kwargs": {"enable_thinking": False}},
                {"chat_template_kwargs": {"enable_thinking": True}}, "qwen"),
    "molmo2er": ("allenai/Molmo2-ER", 8001, {}, {}, "molmo"),
    "rynn9b": ("Alibaba-DAMO-Academy/RynnBrain1.1-9B", 8003, {"chat_template_kwargs": {"enable_thinking": False}},
               {"chat_template_kwargs": {"enable_thinking": True}}, "rynn"),
    "mimoemb7b": ("XiaomiMiMo/MiMo-Embodied-7B", 8004, {}, {}, "qwen25abs"),
}
HOSTED = {"gemini38flash": ("google/gemini-3.8-flash", "gemini"), "mimo26flash": ("xiaomi/mimo-v2.6-flash", "xy1000"),
          "mimo26pro": ("xiaomi/mimo-v2.6-pro", "xy1000")}
LABELS = [chr(65 + i) for i in range(9)] + [str(i) for i in range(1, 10)]
HOSTED_LOG = Path(__file__).resolve().parents[1] / "budget" / "hosted_spend.jsonl"
HOSTED_CAP = 30.0

POINT_PROMPT = {
    "molmo": "Point to {noun}.",
    "qwen": 'Locate {noun} in the image. Output its center point as JSON: [{{"point_2d": [x, y], "label": "target"}}], coordinates normalized to 0-1000.',
    "rynn": "Point to {noun}. Output the point as (x, y) with coordinates in [0, 1000].",
    "qwen25abs": 'Locate {noun} in the image and output its center point in JSON format: [{{"point_2d": [x, y], "label": "target"}}]. Answer only with the JSON.',
    "gemini": 'Point to {noun}. Answer only with JSON [{{"point": [y, x], "label": "target"}}], coordinates normalized to 0-1000.',
    "xy1000": 'Point to {noun}. Reply only with JSON {{"point": [x, y]}} where x (from the left) and y (from the top) are normalized to 0-1000.',
}


def _pairs(text: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in re.findall(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", text or "")]


def parse_point(text: str, fmt: str, w: int, h: int):
    """First point in pixel coordinates, or None."""
    if fmt == "molmo":
        from run3.pointing import parse_points
        pts = parse_points(text, w, h)
        return pts[0] if pts else None
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?))?\s*\]", body)
    nums = None
    if m:
        nums = [float(g) for g in m.groups() if g is not None]
    else:
        pr = _pairs(body)
        if pr:
            nums = [*pr[0], *pr[1]] if fmt == "rynn" and len(pr) >= 2 and "<object>" in body else [*pr[0]]
    if not nums:
        xy = re.search(r"x\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*,?\s*y\s*[:=]\s*(-?\d+(?:\.\d+)?)", body, flags=re.I)
        nums = [float(xy.group(1)), float(xy.group(2))] if xy else None
    if not nums:
        return None
    if len(nums) == 4:  # a box: use its centre
        nums = [(nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2]
    a, b = nums[:2]
    if fmt == "gemini":
        x, y = b / 1000 * w, a / 1000 * h
    elif fmt == "qwen25abs":
        x, y = a, b
    else:
        x, y = a / 1000 * w, b / 1000 * h
    return (x, y) if 0 <= x <= w and 0 <= y <= h else None


class Local:
    """Self-hosted model on the Pod."""

    def __init__(self, name: str):
        repo, port, self.extra, self.think_extra, self.fmt = LOCAL[name]
        self.name, self.repo = name, repo
        base = f"http://localhost:{port}" if ON_POD else f"http://localhost:{18000 + port % 1000}"
        self.c = VllmClient(base, KEY.read_text().strip(), repo, label_token_ids={}, extra_body=self.extra)
        self.label_ids = self._label_ids()
        self.c.label_token_ids = self.label_ids

    def _label_ids(self) -> dict:
        ids = {}
        for lab in LABELS:
            toks = self.c.tokenize(lab)
            if len(toks) != 1:
                raise ValueError(f"{self.name}: label {lab!r} is not one token: {toks}")
            ids[lab] = toks[0]
        if len(set(ids.values())) != len(ids):
            raise ValueError(f"{self.name}: label token ids collide")
        return ids

    def readout(self, prompt: str, images, question: dict) -> dict:
        return self.c.readout(prompt, images, question)

    def generate(self, text: str, images, max_tokens: int = 64, reasoning: bool = False, schema: dict | None = None) -> dict:
        extra = dict(self.think_extra if reasoning else self.extra)
        if schema is not None:
            extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        old = self.c.extra_body
        self.c.extra_body = extra
        try:
            return self.c.generate(text, images, max_tokens=max_tokens)
        finally:
            self.c.extra_body = old

    def point(self, noun: str, image) -> dict:
        r = self.generate(POINT_PROMPT[self.fmt].format(noun=noun), [image], max_tokens=512)
        w, h = image.size
        r["point"] = parse_point(r["text"], self.fmt, w, h)
        return r


def hosted_spent() -> float:
    if not HOSTED_LOG.exists():
        return 0.0
    return sum(float(json.loads(l).get("usd") or 0) for l in HOSTED_LOG.read_text().splitlines())


class Hosted:
    """OpenRouter free generation (images as PNG data URLs); provider-reported cost is logged per call."""

    def __init__(self, name: str):
        self.name = name
        self.model, self.fmt = HOSTED[name]
        self.key = (Path.home() / ".config/openrouter/so101-vlm-api-key").read_text().strip()
        self.client = httpx.Client(base_url="https://openrouter.ai/api/v1", timeout=180,
                                   headers={"Authorization": f"Bearer {self.key}"})

    def generate(self, text: str, images, max_tokens: int = 1024, reasoning: bool = False, schema=None) -> dict:
        if hosted_spent() >= HOSTED_CAP:
            raise RuntimeError("hosted API cap reached")
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens, "usage": {"include": True},
                "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                              *[{"type": "image_url", "image_url": {"url": png_data_url(im)[0]}} for im in images]]}]}
        if not reasoning:
            body["reasoning"] = {"effort": "minimal", "exclude": True}
        for attempt in range(5):
            t0 = time.monotonic()
            r = self.client.post("/chat/completions", json=body)
            dt = time.monotonic() - t0
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            raw = r.json()
            usd = float((raw.get("usage") or {}).get("cost") or 0)
            HOSTED_LOG.parent.mkdir(parents=True, exist_ok=True)
            with HOSTED_LOG.open("a") as f:
                f.write(json.dumps({"time": time.time(), "model": self.model, "id": raw.get("id"), "usd": usd}) + "\n")
            msg = raw["choices"][0]["message"]
            return {"text": msg.get("content") or "", "latency_s": dt, "usage": raw.get("usage"), "usd": usd,
                    "finish_reason": raw["choices"][0].get("finish_reason")}
        raise RuntimeError(f"hosted call failed: {r.status_code} {r.text[:300]}")

    def point(self, noun: str, image) -> dict:
        r = self.generate(POINT_PROMPT[self.fmt].format(noun=noun), [image], max_tokens=1024)
        w, h = image.size
        r["point"] = parse_point(r["text"], self.fmt, w, h)
        return r


def client(name: str):
    if name == "sam":
        from run4.sam_locator import SamLocator
        return SamLocator()
    return Local(name) if name in LOCAL else Hosted(name)


def parse_answer(text: str, allowed: list[str]) -> str | None:
    """Last 'ANSWER: X' line (X in allowed), else None."""
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    hits = re.findall(r"ANSWER\s*[:=]\s*\(?([A-Za-z0-9_]+)\)?", body)
    for h in reversed(hits):
        if h in allowed:
            return h
        if h.upper() in allowed:
            return h.upper()
    return None


def parse_plan(text: str, steps: list[str]) -> list[str] | None:
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for m in reversed(list(re.finditer(r"\{[^{}]*\"steps\"\s*:\s*\[[^\]]*\][^{}]*\}|\[[^\[\]]*\]", body))):
        try:
            v = json.loads(m.group(0))
        except (ValueError, TypeError):
            continue
        seq = v.get("steps") if isinstance(v, dict) else v
        if isinstance(seq, list) and seq and all(isinstance(s, str) and s in steps for s in seq):
            return seq
    return None


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
