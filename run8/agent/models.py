"""Run 8 model clients: the Run 5 clients plus
  QwenFast.point(..., json=True)   one-point pointing with a JSON-schema constrained answer (about 20 tokens, no markdown)
  hosted spend: the canonical ledger run5/budget/hosted_spend.jsonl, cumulative cap raised to USD 50 (user-approved
  2026-09-26; run8/BRIEF.md). The cap is enforced by run5.agent.models.Hosted.generate before every call.
  On a Pod the ledger is a copy of the Mac's canonical one; HOSTED_CAP (environment) then gives that Pod a lower cap so the
  Pods' combined spend stays under 50, and the Pod's new rows are merged back into the canonical ledger afterwards.
"""
from __future__ import annotations

import os

import run5.agent.models as M
from run5.agent.models import Hosted, Qwen, _content, parse_points  # noqa: F401
from run7.agent import prompts as P

M.HOSTED_CAP = min(50.0, float(os.environ.get("HOSTED_CAP") or 50.0))

POINT_SCHEMA = {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "object", "properties": {
    "point_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}, "required": ["point_2d"]}}


class QwenFast(Qwen):
    """Qwen with optional JSON-constrained single-point answers (run8 BRIEF A)."""

    def __init__(self, model: str, base: str | None = None, point_json: bool = False):
        super().__init__(model, base)
        self.point_json = point_json

    def point(self, noun: str, image, n: int = 1) -> dict:
        if not self.point_json or n != 1 or noun.startswith("FOUR:"):
            return super().point(noun, image, n)
        r = self.generate(P.POINT_ONE.format(noun=noun), [image], max_tokens=40, schema=POINT_SCHEMA)
        w, h = image.size
        pts = parse_points(r["text"], w, h)
        r["points"] = pts[:1] if pts else None
        return r
