"""Append-only accounting for Run 2 provider buckets.

The ledger intentionally keeps an unresolved reservation committed when a call
times out or when a successful response does not include provider-reported
cost. A human can reconcile that liability later from provider billing data.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path


class ProviderBudgetExceeded(RuntimeError):
    pass


def _money(value: object, *, positive: bool = False) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid USD amount") from exc
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        raise ValueError("invalid USD amount")
    return parsed


class ProviderBudgetLedger:
    """Process-safe ledger with a bucket cap and an optional sub-allocation."""

    def __init__(
        self,
        path: str | Path,
        *,
        cap_usd: float | str,
        allocation_usd: float | str | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_usd = _money(cap_usd, positive=True)
        self.allocation_usd = (
            _money(allocation_usd, positive=True)
            if allocation_usd is not None
            else self.cap_usd
        )
        if self.allocation_usd > self.cap_usd:
            raise ValueError("allocation cannot exceed bucket cap")

    @contextlib.contextmanager
    def _locked(self):
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                stream.seek(0)
                rows = [json.loads(line) for line in stream if line.strip()]
                yield stream, rows
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @staticmethod
    def _totals(rows: list[dict]) -> dict[str, Decimal | int]:
        reserved: dict[str, Decimal] = {}
        final: dict[str, Decimal] = {}
        for row in rows:
            if row.get("event") == "reserve":
                reserved[row["id"]] = _money(row["max_usd"])
            elif row.get("event") in {"settle", "release"}:
                final[row["id"]] = _money(row.get("actual_usd", "0"))
        spent = sum(final.values(), Decimal("0"))
        pending = sum(
            (amount for key, amount in reserved.items() if key not in final),
            Decimal("0"),
        )
        return {
            "actual_spend_usd": spent,
            "pending_reserved_usd": pending,
            "committed_usd": spent + pending,
            "requests": len(reserved),
        }

    @staticmethod
    def _append(stream, row: dict) -> None:
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps({"time": time.time(), **row}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def status(self) -> dict:
        with self._locked() as (_, rows):
            totals = self._totals(rows)
        return {
            **{key: str(value) if isinstance(value, Decimal) else value for key, value in totals.items()},
            "cap_usd": str(self.cap_usd),
            "allocation_usd": str(self.allocation_usd),
        }

    def reserve(self, max_usd: float | str, metadata: dict | None = None) -> str:
        amount = _money(max_usd, positive=True)
        with self._locked() as (stream, rows):
            committed = self._totals(rows)["committed_usd"]
            if committed + amount > self.allocation_usd:
                raise ProviderBudgetExceeded(
                    f"allocation exceeded: {committed} committed + {amount} requested "
                    f"> {self.allocation_usd} USD"
                )
            key = str(uuid.uuid4())
            self._append(
                stream,
                {
                    "event": "reserve",
                    "id": key,
                    "max_usd": str(amount),
                    "metadata": metadata or {},
                },
            )
            return key

    def settle(self, key: str, actual_usd: float | str, details: dict | None = None) -> Decimal:
        amount = _money(actual_usd)
        with self._locked() as (stream, rows):
            reservation = next(
                (row for row in rows if row.get("event") == "reserve" and row.get("id") == key),
                None,
            )
            if reservation is None:
                raise ValueError("unknown reservation")
            prior = next(
                (row for row in rows if row.get("event") in {"settle", "release"} and row.get("id") == key),
                None,
            )
            if prior is not None:
                return _money(prior.get("actual_usd", "0"))
            self._append(
                stream,
                {
                    "event": "settle",
                    "id": key,
                    "actual_usd": str(amount),
                    "details": details or {},
                    "exceeded_reservation": amount > _money(reservation["max_usd"]),
                },
            )
        return amount

    def record_pending_note(self, key: str, *, reason: str, details: dict | None = None) -> None:
        """Append evidence about an unresolved reservation without releasing it."""
        with self._locked() as (stream, rows):
            known = any(row.get("event") == "reserve" and row.get("id") == key for row in rows)
            done = any(
                row.get("event") in {"settle", "release"} and row.get("id") == key
                for row in rows
            )
            if known and not done:
                self._append(
                    stream,
                    {
                        "event": "pending_note",
                        "id": key,
                        "reason": reason,
                        "details": details or {},
                    },
                )

    def release_confirmed_no_charge(
        self,
        key: str,
        *,
        status_code: int,
        reason: str,
        details: dict | None = None,
    ) -> None:
        if status_code not in {400, 401, 403, 404, 422, 429}:
            return
        with self._locked() as (stream, rows):
            known = any(row.get("event") == "reserve" and row.get("id") == key for row in rows)
            done = any(
                row.get("event") in {"settle", "release"} and row.get("id") == key
                for row in rows
            )
            if known and not done:
                self._append(
                    stream,
                    {
                        "event": "release",
                        "id": key,
                        "actual_usd": "0",
                        "status_code": status_code,
                        "reason": reason,
                        "details": details or {},
                    },
                )
