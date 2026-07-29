"""Checkpointing + resumable storage for the scraper (SPEC §2.1 rules 4-5).

Records are appended to a JSONL log as they are collected (crash-safe, resumable),
and a small marker file tracks which pages/ids are already done so a restart skips
them. A finalizer writes the deduped result to the configured parquet path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class CheckpointStore:
    """Append-only JSONL log + a done-marker set, with a parquet finalizer."""

    def __init__(self, records_jsonl: str | Path, done_marker: str | Path) -> None:
        self.records_jsonl = Path(records_jsonl)
        self.done_marker = Path(done_marker)
        self.records_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.done_marker.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.done_marker.exists():
            self._done = {
                line.strip()
                for line in self.done_marker.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

    # --- records ---------------------------------------------------------
    def append_records(self, records: list[dict[str, object]]) -> None:
        """Append records to the JSONL log (one JSON object per line)."""
        if not records:
            return
        with open(self.records_jsonl, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_records(self) -> list[dict[str, object]]:
        """Load all previously checkpointed records (empty if none)."""
        if not self.records_jsonl.exists():
            return []
        with open(self.records_jsonl, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # --- done markers (pages for Stage 1, ids for Stage 2) ---------------
    def _mark_done(self, key: str) -> None:
        if key in self._done:
            return
        self._done.add(key)
        with open(self.done_marker, "a", encoding="utf-8") as fh:
            fh.write(key + "\n")

    def is_page_done(self, url: str) -> bool:
        return url in self._done

    def mark_page_done(self, url: str) -> None:
        self._mark_done(url)

    def is_id_done(self, listing_id: str) -> bool:
        return listing_id in self._done

    def mark_id_done(self, listing_id: str) -> None:
        self._mark_done(listing_id)

    def record_failure(self, key: str, reason: str) -> None:
        """Log a permanently-failed url/id (post-retry) for transparency, and skip it."""
        failures = self.done_marker.with_suffix(".failures.txt")
        with open(failures, "a", encoding="utf-8") as fh:
            fh.write(f"{key}\t{reason}\n")
        self._mark_done(key)  # mark done so a resume does not re-attempt it forever

    # --- finalize --------------------------------------------------------
    def to_parquet(self, out_path: str | Path, dedup_key: str | None = None) -> Path:
        """Write the checkpointed records to parquet (optionally deduped on a key)."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.load_records())
        if dedup_key and not frame.empty:
            frame = frame.drop_duplicates(subset=[dedup_key], keep="first")
        frame.to_parquet(out, index=False)
        return out
