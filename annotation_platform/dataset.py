from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _stable_record_id(dataset_name: str, obj: dict[str, Any]) -> str:
    data_source = str(obj.get("data_source") or dataset_name)
    query_index = obj.get("query_index")
    sample_index = obj.get("sample_index")
    if query_index is not None and sample_index is not None:
        return f"{data_source}:{query_index}:{sample_index}"
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{data_source}:{digest}"


def _content_preview(content: Any, limit: int = 140) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    else:
        try:
            text = json.dumps(content, ensure_ascii=False)
        except TypeError:
            text = str(content)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


@dataclass(frozen=True)
class DatasetIndex:
    name: str
    path: Path
    offsets: list[int]
    record_ids: list[str]
    stat_mtime_ns: int
    stat_size: int

    @property
    def size(self) -> int:
        return len(self.offsets)

    def is_stale(self) -> bool:
        stat = self.path.stat()
        return stat.st_mtime_ns != self.stat_mtime_ns or stat.st_size != self.stat_size

    def read_item(self, index: int) -> dict[str, Any]:
        if self.is_stale():
            raise RuntimeError(f"dataset file changed on disk: {self.path}")
        offset = self.offsets[index]
        with self.path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        if not line:
            raise RuntimeError(f"unexpected EOF while reading {self.path} at offset={offset}")
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            raise ValueError(f"empty/whitespace JSONL line in {self.path} at offset={offset}")
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object in {self.path} at offset={offset}, got {type(obj).__name__}")
        return obj

    def to_payload(
        self,
        *,
        item: dict[str, Any],
        dataset: str,
        index: int,
        record_id: str,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        messages = item.get("messages") or []
        if not isinstance(messages, list):
            messages = []

        assistant_indices: list[int] = []
        final_assistant_index: int | None = None
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_indices.append(i)
        if assistant_indices:
            final_assistant_index = assistant_indices[-1]

        assistant_previews: dict[str, str] = {}
        for i in assistant_indices:
            msg = messages[i] if isinstance(messages[i], dict) else {}
            assistant_previews[str(i)] = _content_preview(msg.get("content"))

        return {
            "dataset": dataset,
            "index_in_dataset": index,
            "record_id": record_id,
            "data_source": item.get("data_source"),
            "query_index": item.get("query_index"),
            "sample_index": item.get("sample_index"),
            "question": item.get("question"),
            "task_description": item.get("task_description"),
            "ground_truth": item.get("ground_truth"),
            "answer_text": item.get("answer_text"),
            "reward_info": item.get("reward_info"),
            "terminated": item.get("terminated"),
            "stop_reason": item.get("stop_reason"),
            "meta": item.get("meta"),
            "meta_info": item.get("meta_info"),
            "tool_metrics": item.get("tool_metrics"),
            "tools": item.get("tools"),
            "messages": messages,
            "assistant_message_indices": assistant_indices,
            "final_assistant_message_index": final_assistant_index,
            "assistant_previews": assistant_previews,
            "existing_annotation": existing,
        }


def _build_index(dataset_name: str, path: Path) -> DatasetIndex:
    offsets: list[int] = []
    record_ids: list[str] = []
    stat = path.stat()
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                text = line.decode("utf-8", errors="replace").strip()
            except UnicodeDecodeError:
                continue
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            offsets.append(offset)
            record_ids.append(_stable_record_id(dataset_name, obj))
    return DatasetIndex(
        name=dataset_name,
        path=path,
        offsets=offsets,
        record_ids=record_ids,
        stat_mtime_ns=stat.st_mtime_ns,
        stat_size=stat.st_size,
    )


def discover_datasets(dataset_dir: Path) -> dict[str, DatasetIndex]:
    datasets: dict[str, DatasetIndex] = {}
    for path in sorted(dataset_dir.glob("*.jsonl")):
        name = path.stem
        datasets[name] = _build_index(name, path)
    return datasets
