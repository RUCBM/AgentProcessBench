from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Progress:
    done: int
    skipped: int


class AnnotationStore:
    def __init__(self, db_path: Path, *, exports_dir: Path) -> None:
        self.db_path = db_path
        self.exports_dir = exports_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(annotations)").fetchall()
            if isinstance(row, sqlite3.Row)
        }
        desired: dict[str, str] = {
            "username": "TEXT",
            "data_source": "TEXT",
            "query_index": "INTEGER",
            "sample_index": "INTEGER",
            "final_label_touched": "INTEGER NOT NULL DEFAULT 1",
        }
        for col, decl in desired.items():
            if col in existing:
                continue
            conn.execute(f"ALTER TABLE annotations ADD COLUMN {col} {decl}")

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS annotations (
                  dataset TEXT NOT NULL,
                  record_id TEXT NOT NULL,
                  annotator TEXT NOT NULL,
                  index_in_dataset INTEGER NOT NULL,
                  username TEXT,
                  data_source TEXT,
                  query_index INTEGER,
                  sample_index INTEGER,
                  step_labels_json TEXT NOT NULL,
                  final_label INTEGER NOT NULL,
                  final_label_touched INTEGER NOT NULL DEFAULT 1,
                  status TEXT NOT NULL,
                  comment TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(dataset, record_id, annotator)
                );

                CREATE TABLE IF NOT EXISTS cursors (
                  dataset TEXT NOT NULL,
                  annotator TEXT NOT NULL,
                  cursor_index INTEGER NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(dataset, annotator)
                );

                CREATE INDEX IF NOT EXISTS idx_annotations_dataset_annotator
                  ON annotations(dataset, annotator);
                """
            )
            self._ensure_columns(conn)
            conn.commit()
        finally:
            conn.close()

    def get_annotation(self, *, dataset: str, record_id: str, annotator: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT dataset, record_id, annotator, index_in_dataset,
                       username, data_source, query_index, sample_index,
                       step_labels_json, final_label, final_label_touched, status, comment,
                       created_at, updated_at
                  FROM annotations
                 WHERE dataset = ? AND record_id = ? AND annotator = ?
                """,
                (dataset, record_id, annotator),
            ).fetchone()
            if row is None:
                return None
            return {
                "dataset": row["dataset"],
                "record_id": row["record_id"],
                "annotator": row["annotator"],
                "username": row["username"],
                "index_in_dataset": row["index_in_dataset"],
                "data_source": row["data_source"],
                "query_index": row["query_index"],
                "sample_index": row["sample_index"],
                "step_labels": json.loads(row["step_labels_json"]),
                "final_label": row["final_label"],
                "final_label_touched": bool(row["final_label_touched"]),
                "status": row["status"],
                "comment": row["comment"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def upsert_annotation(
        self,
        *,
        dataset: str,
        record_id: str,
        annotator: str,
        username: str | None,
        index_in_dataset: int,
        data_source: str | None,
        query_index: int | None,
        sample_index: int | None,
        step_labels: dict[str, Any],
        final_label: int,
        final_label_touched: bool,
        status: str,
        comment: str | None,
    ) -> None:
        now = _utc_now_iso()
        record = {
            "dataset": dataset,
            "record_id": record_id,
            "annotator": annotator,
            "username": username or annotator,
            "index_in_dataset": index_in_dataset,
            "data_source": data_source,
            "query_index": query_index,
            "sample_index": sample_index,
            "step_labels": step_labels,
            "final_label": final_label,
            "final_label_touched": bool(final_label_touched),
            "status": status,
            "comment": comment,
            "updated_at": now,
        }

        step_labels_json = json.dumps(step_labels, ensure_ascii=False, sort_keys=True)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO annotations (
                  dataset, record_id, annotator, index_in_dataset,
                  username, data_source, query_index, sample_index,
                  step_labels_json, final_label, final_label_touched, status, comment,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, record_id, annotator) DO UPDATE SET
                  index_in_dataset = excluded.index_in_dataset,
                  username = excluded.username,
                  data_source = excluded.data_source,
                  query_index = excluded.query_index,
                  sample_index = excluded.sample_index,
                  step_labels_json = excluded.step_labels_json,
                  final_label = excluded.final_label,
                  final_label_touched = excluded.final_label_touched,
                  status = excluded.status,
                  comment = excluded.comment,
                  updated_at = excluded.updated_at
                """,
                (
                    dataset,
                    record_id,
                    annotator,
                    index_in_dataset,
                    username or annotator,
                    data_source,
                    query_index,
                    sample_index,
                    step_labels_json,
                    final_label,
                    1 if final_label_touched else 0,
                    status,
                    comment,
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        self._append_export(record)

    def _append_export(self, record: dict[str, Any]) -> None:
        safe_dataset = "".join(ch for ch in record["dataset"] if ch.isalnum() or ch in ("_", "-", "."))
        username = str(record.get("username") or record.get("annotator") or "unknown")
        safe_username = "".join(ch for ch in username if ch.isalnum() or ch in ("_", "-", "."))
        path = self.exports_dir / f"{safe_dataset}__{safe_username}.jsonl"
        line = json.dumps(record, ensure_ascii=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    def get_progress(self, *, dataset: str, annotator: str) -> Progress:
        conn = self._connect()
        try:
            done = conn.execute(
                """
                SELECT COUNT(*) AS n
                  FROM annotations
                 WHERE dataset = ? AND annotator = ? AND status = 'done'
                """,
                (dataset, annotator),
            ).fetchone()["n"]
            skipped = conn.execute(
                """
                SELECT COUNT(*) AS n
                  FROM annotations
                 WHERE dataset = ? AND annotator = ? AND status = 'skipped'
                """,
                (dataset, annotator),
            ).fetchone()["n"]
            return Progress(done=int(done), skipped=int(skipped))
        finally:
            conn.close()

    def has_annotation(self, *, dataset: str, record_id: str, annotator: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT 1
                  FROM annotations
                 WHERE dataset = ? AND record_id = ? AND annotator = ?
                 LIMIT 1
                """,
                (dataset, record_id, annotator),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def has_done_annotation(self, *, dataset: str, record_id: str, annotator: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT 1
                  FROM annotations
                 WHERE dataset = ? AND record_id = ? AND annotator = ? AND status = 'done'
                 LIMIT 1
                """,
                (dataset, record_id, annotator),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def find_next_unannotated(
        self,
        *,
        dataset: str,
        annotator: str,
        record_ids: list[str],
        start_index: int,
    ) -> int | None:
        if start_index < 0:
            start_index = 0
        for idx in range(start_index, len(record_ids)):
            if not self.has_done_annotation(dataset=dataset, record_id=record_ids[idx], annotator=annotator):
                return idx
        for idx in range(0, min(start_index, len(record_ids))):
            if not self.has_done_annotation(dataset=dataset, record_id=record_ids[idx], annotator=annotator):
                return idx
        return None

    def get_cursor(self, *, dataset: str, annotator: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT cursor_index
                  FROM cursors
                 WHERE dataset = ? AND annotator = ?
                """,
                (dataset, annotator),
            ).fetchone()
            if row is None:
                return 0
            return int(row["cursor_index"])
        finally:
            conn.close()

    def set_cursor(self, *, dataset: str, annotator: str, cursor_index: int) -> None:
        now = _utc_now_iso()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO cursors (dataset, annotator, cursor_index, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dataset, annotator) DO UPDATE SET
                  cursor_index = excluded.cursor_index,
                  updated_at = excluded.updated_at
                """,
                (dataset, annotator, cursor_index, now),
            )
            conn.commit()
        finally:
            conn.close()
