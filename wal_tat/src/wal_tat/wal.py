"""Crash-detectable hash-chained write-ahead log for WAL-TAT."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


GENESIS_HASH = "0" * 64


class WALIntegrityError(RuntimeError):
    """Raised when a WAL record is malformed, reordered, or modified."""


@dataclass(frozen=True)
class WALRecord:
    sequence: int
    kind: str
    transaction_id: str
    timestamp_ns: int
    previous_hash: str
    payload: Dict[str, Any]
    digest: str


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _digest(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(fields)).hexdigest()


class HashChainWAL:
    """Append-only WAL v2 with sequence and SHA-256 chain verification.

    ``fsync=True`` makes every append a durability boundary. It is intentionally
    slower and should normally be used for begin/commit/rollback, not every
    optimizer step.
    """

    def __init__(self, path: Path | str, *, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = bool(fsync)
        records = self.verify()
        self._sequence = len(records)
        self._tail_hash = records[-1].digest if records else GENESIS_HASH

    def append(
        self,
        kind: str,
        transaction_id: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> WALRecord:
        if not kind or not transaction_id:
            raise ValueError("kind and transaction_id must be non-empty")
        fields: Dict[str, Any] = {
            "sequence": self._sequence,
            "kind": str(kind),
            "transaction_id": str(transaction_id),
            "timestamp_ns": time.time_ns(),
            "previous_hash": self._tail_hash,
            "payload": dict(payload or {}),
        }
        record = WALRecord(**fields, digest=_digest(fields))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if self.fsync:
                os.fsync(stream.fileno())
        self._sequence += 1
        self._tail_hash = record.digest
        return record

    def verify(self) -> List[WALRecord]:
        if not self.path.exists():
            return []
        records: List[WALRecord] = []
        previous = GENESIS_HASH
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                    record = WALRecord(**raw)
                except (json.JSONDecodeError, TypeError) as error:
                    raise WALIntegrityError(f"invalid record at line {line_number}") from error
                fields = asdict(record)
                digest = fields.pop("digest")
                if record.sequence != len(records):
                    raise WALIntegrityError(f"bad sequence at line {line_number}")
                if record.previous_hash != previous:
                    raise WALIntegrityError(f"broken chain at line {line_number}")
                if _digest(fields) != digest:
                    raise WALIntegrityError(f"digest mismatch at line {line_number}")
                records.append(record)
                previous = digest
        return records
