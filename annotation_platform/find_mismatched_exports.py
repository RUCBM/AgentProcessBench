from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
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


def _assistant_index_keys(item: dict[str, Any]) -> set[str]:
    messages = item.get("messages") or []
    if not isinstance(messages, list):
        return set()
    out: set[str] = set()
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            out.add(str(i))
    return out


def _build_expected_map(dataset_path: Path, dataset_name: str) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rid = _stable_record_id(dataset_name, item)
        expected[rid] = _assistant_index_keys(item)
    return expected


def _latest_done_by_key(export_path: Path) -> list[tuple[int, dict[str, Any]]]:
    """
    For append-only exports, keep only the latest (by updated_at) record for each key:
      (dataset, record_id)
    Returns a list of (line_no, obj) for the latest entries.
    """
    latest: dict[tuple[str, str], tuple[datetime, int, dict[str, Any]]] = {}
    for line_no, line in enumerate(export_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("status") != "done":
            continue
        dataset = str(obj["dataset"])
        record_id = str(obj["record_id"])
        dt = datetime.fromisoformat(str(obj["updated_at"]))
        key = (dataset, record_id)
        prev = latest.get(key)
        if prev is None or dt > prev[0]:
            latest[key] = (dt, line_no, obj)
    return [(line_no, obj) for (_, line_no, obj) in latest.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find export lines whose step_labels do not match current dataset messages.")
    parser.add_argument("--exports_dir", type=Path, required=True, help="e.g. annotation_platform/annotation_results/exports")
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        required=True,
        help="e.g. annotation_platform/raw_trajectories/annotation_file_diverse_queries (contains <dataset>.jsonl)",
    )
    args = parser.parse_args()

    exports_dir = args.exports_dir.expanduser()
    annotation_dir = args.annotation_dir.expanduser()

    expected_by_dataset: dict[str, dict[str, set[str]]] = {}
    for ds_path in sorted(annotation_dir.glob("*.jsonl")):
        dataset = ds_path.stem
        expected_by_dataset[dataset] = _build_expected_map(ds_path, dataset)

    total = 0
    mismatched = 0

    for export_path in sorted(exports_dir.glob("*.jsonl")):
        export_name = export_path.name
        for line_no, obj in _latest_done_by_key(export_path):
            dataset = str(obj["dataset"])
            record_id = str(obj["record_id"])
            step_labels_raw = obj.get("step_labels") or {}
            step_keys = set(map(str, step_labels_raw.keys())) if isinstance(step_labels_raw, dict) else set()

            total += 1
            expected_map = expected_by_dataset.get(dataset)
            if expected_map is None:
                mismatched += 1
                print(f"[NO_DATASET] {export_name}:{line_no} dataset={dataset} record_id={record_id}")
                continue

            expected_keys = expected_map.get(record_id)
            if expected_keys is None:
                mismatched += 1
                print(f"[NO_RECORD]  {export_name}:{line_no} dataset={dataset} record_id={record_id}")
                continue

            extra = sorted(step_keys - expected_keys, key=lambda s: int(s) if s.isdigit() else 10**9)
            missing = sorted(expected_keys - step_keys, key=lambda s: int(s) if s.isdigit() else 10**9)
            if extra or missing:
                mismatched += 1
                idx = obj.get("index_in_dataset")
                print(f"[MISMATCH] {export_name}:{line_no} dataset={dataset} index={idx} record_id={record_id}")
                if extra:
                    print(f"           extra_step_keys: {', '.join(extra)}")
                if missing:
                    print(f"           missing_step_keys: {', '.join(missing)}")

    print(f"\nDone. checked_done_records={total} mismatched={mismatched}")


if __name__ == "__main__":
    main()
