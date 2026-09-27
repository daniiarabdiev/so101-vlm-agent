"""SAM 3.1 (official Meta API) as a locator for Run 4: text concept -> mask -> centroid pixel.

Reuses the reviewed Run 2 adapter and Run 3's ledger class with a Run 4 log (run4/budget/sam_spend.jsonl,
cap USD 25). The provider returns no mask scores, so the first mask is used; when several masks come back
the choice is logged. For a container spot the container itself is segmented and its centroid is used.
"""
from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from run2.completion_staging.sam3_adapters.meta_adapter import MetaSam31Adapter
from run3.sam_client import Run3SamLedger, decode_masks

RUN4 = Path(__file__).resolve().parent
_ADAPTER = None


def adapter() -> MetaSam31Adapter:
    global _ADAPTER
    if _ADAPTER is None:
        ledger = Run3SamLedger(path=RUN4 / "budget" / "sam_spend.jsonl", cap=Decimal("25"))
        _ADAPTER = MetaSam31Adapter(RUN4 / "sam_cache", remote_enabled=True, ledger=ledger, timeout_s=120)
    return _ADAPTER


def concept_for(noun: str) -> str:
    n = noun.removeprefix("the ").removeprefix("an empty spot inside the ")
    return n


class SamLocator:
    fmt = "sam"

    def point(self, noun: str, image) -> dict:
        t0 = time.monotonic()
        rec = adapter().segment_one(image, concept_for(noun), prompt_index=0)
        masks = decode_masks(rec)
        dt = time.monotonic() - t0
        if not masks:
            return {"point": None, "text": f"no mask ({rec.get('reason')})", "latency_s": dt, "usd": rec.get("cost_usd")}
        m = masks[0]
        vs, us = np.nonzero(m)
        return {"point": (float(us.mean()), float(vs.mean())), "text": f"{len(masks)} masks; first has {int(m.sum())} px",
                "latency_s": dt, "usd": rec.get("cost_usd"), "n_masks": len(masks)}

    def __call__(self, noun, image):  # agent interface: (px, info)
        r = self.point(noun, image)
        return r["point"], {"mode": "sam", "text": r["text"], "latency_s": r["latency_s"], "usd": r.get("usd")}
