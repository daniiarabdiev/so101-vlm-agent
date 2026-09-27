"""SAM 3.1 (official Meta API) labelling client for Run 3.

Reuses the reviewed Run 2 adapter (official meta_sam_parser, exact mask registration,
on-disk cache). Spend goes to a Run 3 log capped at the user's USD 10 SAM budget; the
Run 2 SAM ledger under run2/artifacts is not modified.
"""
from __future__ import annotations

import json
import time
import uuid
from decimal import Decimal
from pathlib import Path

import numpy as np

from run2.completion_staging.sam3_adapters.mask_normalization import decode_row_major_runs
from run2.completion_staging.sam3_adapters.meta_adapter import MetaSam31Adapter

RUN3 = Path(__file__).resolve().parent
SPEND_LOG = RUN3 / "budget" / "sam_spend.jsonl"
CAP_USD = Decimal("10")


class Run3SamLedger:
    """Minimal spend log: one line per request; refuses new requests at the cap."""

    def __init__(self, path: Path = SPEND_LOG, cap: Decimal = CAP_USD):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap

    def spent(self) -> Decimal:
        if not self.path.exists():
            return Decimal(0)
        total = Decimal(0)
        for line in self.path.read_text().splitlines():
            row = json.loads(line)
            if row["event"] in ("settle", "unknown_charge"):
                total += Decimal(str(row["usd"]))
        return total

    def _write(self, row: dict) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps({"time": time.time(), **row}, sort_keys=True) + "\n")

    def reserve(self, unit_cost, metadata):
        if self.spent() + Decimal(str(unit_cost)) > self.cap:
            raise RuntimeError("Run 3 SAM budget cap reached")
        return str(uuid.uuid4())

    def settle(self, key, usd, details):
        self._write({"event": "settle", "id": key, "usd": str(usd), "details": details})

    def record_pending_note(self, key, *, reason, details):
        # Unknown provider charge: count it as spent (conservative), keep the reason.
        self._write({"event": "unknown_charge", "id": key, "usd": "0.0025", "reason": reason, "details": details})


def make_adapter(cache_root: Path, remote: bool = True) -> MetaSam31Adapter:
    return MetaSam31Adapter(cache_root, remote_enabled=remote, ledger=Run3SamLedger(), timeout_s=120)


def decode_masks(record: dict) -> list[np.ndarray]:
    return [decode_row_major_runs(m["runs"], tuple(m["size"])) for m in record.get("masks", [])]
