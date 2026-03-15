from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Default directories
MODELS_ROOT_DIR = Path("./eval/results")
REFERENCE_DIR = Path("./data/AgentProcessBench")
TARGET_DATASETS = ["bfcl", "hotpotqa", "tau2", "gaia_dev"]

TIMESTAMP_FIELDS = ("updated_at", "created_at", "timestamp")
DATASET_OFFSETS = {
    "hotpotqa": 0,
    "gaia_dev": 250,
    "bfcl": 500,
    "tau2": 750,
}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(obj)!r}")
            yield obj


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e15:
            v /= 1e9
        elif v > 1e12:
            v /= 1e3
        return datetime.fromtimestamp(v)
    if isinstance(value, str):
        ts = value.strip()
        if not ts:
            return None
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


def _to_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


def _normalize_step_labels(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        out: dict[str, int] = {}
        for k, v in value.items():
            iv = _to_int_or_none(v)
            if iv is None:
                continue
            out[str(k)] = iv
        return out
    if isinstance(value, list):
        out: dict[str, int] = {}
        for i, v in enumerate(value):
            iv = _to_int_or_none(v)
            if iv is None:
                continue
            out[str(i)] = iv
        return out
    raise TypeError(f"Unsupported step_labels type: {type(value)!r}")


def _first_neg1_index(step_labels: dict[str, int]) -> int:
    idxs: list[int] = []
    for k, v in step_labels.items():
        if v != -1:
            continue
        try:
            idxs.append(int(k))
        except ValueError:
            continue
    return min(idxs) if idxs else -1


def _record_key(record: dict[str, Any], dataset: str) -> str:
    ti = _to_int_or_none(record.get("total_index"))
    if ti is not None:
        return f"ti:{ti}"

    qi = _to_int_or_none(record.get("query_index"))
    si = _to_int_or_none(record.get("sample_index"))
    if qi is not None and si is not None:
        off = DATASET_OFFSETS.get(dataset)
        if off is not None:
            return f"ti:{off + qi * 5 + si}"
        return f"qs:{qi}:{si}"

    rid = record.get("record_id")
    if isinstance(rid, str) and rid.strip():
        return f"rid:{rid.strip()}"

    raise KeyError(
        "Cannot infer record key (need total_index or query_index+sample_index or record_id). "
        f"keys={sorted(record.keys())}"
    )


def _load_reference(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in _iter_jsonl(path):
        key = _record_key(rec, dataset)
        if key in out:
            raise ValueError(f"Duplicate reference key {key} in {path}")
        out[key] = rec
    return out


def _load_predictions_latest(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    # latest by timestamp; if no timestamp, keep last line occurrence
    latest: dict[str, tuple[datetime | None, int, dict[str, Any]]] = {}
    for line_no, rec in enumerate(_iter_jsonl(path), start=1):
        key = _record_key(rec, dataset)
        ts: datetime | None = None
        for field in TIMESTAMP_FIELDS:
            if field in rec:
                ts = _parse_timestamp(rec.get(field))
                if ts is not None:
                    break

        prev = latest.get(key)
        if prev is None:
            latest[key] = (ts, line_no, rec)
            continue

        prev_ts, prev_line_no, _prev_rec = prev

        # prefer records with valid timestamps; among same type, choose newer or later line
        if ts is not None and prev_ts is not None:
            if ts > prev_ts or (ts == prev_ts and line_no > prev_line_no):
                latest[key] = (ts, line_no, rec)
        elif ts is not None and prev_ts is None:
            latest[key] = (ts, line_no, rec)
        elif ts is None and prev_ts is None and line_no > prev_line_no:
            latest[key] = (ts, line_no, rec)

    return {k: v[2] for k, v in latest.items()}


@dataclass(frozen=True)
class ConsistencyMetrics:
    dataset: str
    run_name: str
    ref_records: int
    pred_records: int
    compared_records: int
    missing_or_failed: int
    step_matches: int
    step_total: int
    step_exact_matches: int
    first_neg1_index_matches: int

    @property
    def missing_or_failed_ratio(self) -> float:
        return self.missing_or_failed / self.compared_records if self.compared_records else 0.0

    @property
    def step_micro_accuracy(self) -> float:
        return self.step_matches / self.step_total if self.step_total else 0.0

    @property
    def step_exact_accuracy(self) -> float:
        return self.step_exact_matches / self.compared_records if self.compared_records else 0.0

    @property
    def first_neg1_index_accuracy(self) -> float:
        return self.first_neg1_index_matches / self.compared_records if self.compared_records else 0.0


def _compute_metrics(
    *,
    dataset: str,
    run_name: str,
    ref_by_key: dict[str, dict[str, Any]],
    pred_by_key: dict[str, dict[str, Any]],
) -> ConsistencyMetrics:
    keys = list(ref_by_key.keys())
    compared_records = len(keys)

    missing_or_failed = 0
    step_matches = 0
    step_total = 0
    step_exact_matches = 0
    first_neg1_index_matches = 0

    for key in keys:
        ref = ref_by_key[key]
        pred = pred_by_key.get(key)

        failed = False
        if pred is None:
            failed = True
        else:
            comment = pred.get("comment")
            if isinstance(comment, str) and comment.strip().startswith("llm_annotate_failed:"):
                failed = True
        if failed:
            missing_or_failed += 1

        ref_steps = _normalize_step_labels(ref.get("step_labels"))
        pred_steps = _normalize_step_labels(pred.get("step_labels")) if pred is not None else {}

        if _first_neg1_index(pred_steps) == _first_neg1_index(ref_steps):
            first_neg1_index_matches += 1

        step_total += len(ref_steps)
        for sk, sv in ref_steps.items():
            if pred_steps.get(sk) == sv:
                step_matches += 1

        if pred_steps == ref_steps:
            step_exact_matches += 1

    return ConsistencyMetrics(
        dataset=dataset,
        run_name=run_name,
        ref_records=len(ref_by_key),
        pred_records=len(pred_by_key),
        compared_records=compared_records,
        missing_or_failed=missing_or_failed,
        step_matches=step_matches,
        step_total=step_total,
        step_exact_matches=step_exact_matches,
        first_neg1_index_matches=first_neg1_index_matches,
    )


def _infer_run_name(path: Path) -> str:
    # keep legacy behavior, filename suffix after first "__"
    name = path.name
    if "__" in name:
        return name.split("__", 1)[1].removesuffix(".jsonl")
    return path.stem


def _run_name_group_key(run_name: str, mode: str) -> str:
    if mode == "raw":
        return run_name
    if mode == "casefold":
        return run_name.casefold()
    raise ValueError(f"Unknown run_name grouping mode: {mode!r}")


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _aggregate(group: list[ConsistencyMetrics]) -> ConsistencyMetrics:
    if not group:
        raise ValueError("empty metrics group")
    return ConsistencyMetrics(
        dataset="AVG",
        run_name=group[0].run_name,
        ref_records=sum(m.ref_records for m in group),
        pred_records=sum(m.pred_records for m in group),
        compared_records=sum(m.compared_records for m in group),
        missing_or_failed=sum(m.missing_or_failed for m in group),
        step_matches=sum(m.step_matches for m in group),
        step_total=sum(m.step_total for m in group),
        step_exact_matches=sum(m.step_exact_matches for m in group),
        first_neg1_index_matches=sum(m.first_neg1_index_matches for m in group),
    )


def _evaluate_one_dataset(
    *,
    dataset: str,
    reference_dir: Path,
    models_root_dir: Path,
    expected_reference_records: int,
) -> list[ConsistencyMetrics]:
    ref_path = reference_dir / f"{dataset}.jsonl"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference file: {ref_path}")

    ref_by_key = _load_reference(ref_path, dataset)
    if len(ref_by_key) != expected_reference_records:
        raise AssertionError(
            f"Reference {dataset} expected {expected_reference_records} records, got {len(ref_by_key)} (file={ref_path})"
        )

    model_paths = sorted(
        p
        for p in models_root_dir.rglob(f"{dataset}__*.jsonl")
        if p.is_file() and not any(part in {"raw", "_raw", "llm_annotations_raw"} for part in p.parts)
    )
    metrics: list[ConsistencyMetrics] = []
    for model_path in model_paths:
        pred_by_key = _load_predictions_latest(model_path, dataset)
        metrics.append(
            _compute_metrics(
                dataset=dataset,
                run_name=_infer_run_name(model_path),
                ref_by_key=ref_by_key,
                pred_by_key=pred_by_key,
            )
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare model annotations vs reference (step_labels only, ignore final_label).")
    parser.add_argument("--reference_dir", type=Path, default=REFERENCE_DIR, help="Reference folder with <dataset>.jsonl.")
    parser.add_argument("--models_root_dir", type=Path, default=MODELS_ROOT_DIR, help="Model predictions folder (recursive search).")
    parser.add_argument("--datasets", type=str, default=",".join(TARGET_DATASETS), help="Comma-separated dataset list.")
    parser.add_argument("--expected_reference_records", type=int, default=250, help="Expected records per dataset.")
    parser.add_argument(
        "--run_name_grouping",
        type=str,
        default="raw",
        choices=("raw", "casefold"),
        help="How to group run_name across datasets.",
    )
    parser.add_argument(
        "--score_metric",
        type=str,
        default="step_micro_acc",
        choices=("step_micro_acc", "step_exact_acc", "first_neg1_idx_acc"),
        help="Metric used as FINAL_SCORE.",
    )
    args = parser.parse_args(argv)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    if not datasets:
        print("No datasets provided.", file=sys.stderr)
        return 2

    all_metrics: list[ConsistencyMetrics] = []
    for ds in datasets:
        all_metrics.extend(
            _evaluate_one_dataset(
                dataset=ds,
                reference_dir=args.reference_dir,
                models_root_dir=args.models_root_dir,
                expected_reference_records=args.expected_reference_records,
            )
        )

    if not all_metrics:
        print("No model prediction files found.", file=sys.stderr)
        return 2

    by_run: dict[str, list[ConsistencyMetrics]] = {}
    by_run_display_names: dict[str, set[str]] = {}
    for m in all_metrics:
        k = _run_name_group_key(m.run_name, args.run_name_grouping)
        by_run.setdefault(k, []).append(m)
        by_run_display_names.setdefault(k, set()).add(m.run_name)

    def _print_table(headers: list[str], rows: list[list[str]]) -> None:
        if not rows:
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        numeric_cols = {
            "ref_n",
            "pred_n",
            "missing_or_failed_pct",
            "step_micro_acc",
            "step_exact_acc",
            "first_neg1_idx_acc",
        }
        print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
        for row in rows:
            cells: list[str] = []
            for i, cell in enumerate(row):
                if headers[i] in numeric_cols:
                    cells.append(cell.rjust(widths[i]))
                else:
                    cells.append(cell.ljust(widths[i]))
            print("  ".join(cells))

    for run_key in sorted(by_run.keys()):
        group = sorted(by_run[run_key], key=lambda x: x.dataset)
        avg = _aggregate(group)

        display_names = sorted(by_run_display_names.get(run_key) or {run_key})
        if len(display_names) == 1:
            run_name_display = display_names[0]
        else:
            run_name_display = display_names[0]
            print(f"[WARN] Collapsing run_name variants under {run_name_display!r}: {display_names}", file=sys.stderr)

        print(f"MODEL  {run_name_display}")

        headers = [
            "dataset",
            "ref_n",
            "pred_n",
            "missing_or_failed_pct",
            "step_micro_acc",
            "step_exact_acc",
            "first_neg1_idx_acc",
        ]
        rows: list[list[str]] = []
        for m in group:
            rows.append(
                [
                    m.dataset,
                    str(m.ref_records),
                    str(m.pred_records),
                    _format_pct(m.missing_or_failed_ratio),
                    _format_pct(m.step_micro_accuracy),
                    _format_pct(m.step_exact_accuracy),
                    _format_pct(m.first_neg1_index_accuracy),
                ]
            )
        rows.append(
            [
                avg.dataset,
                str(avg.ref_records),
                str(avg.pred_records),
                _format_pct(avg.missing_or_failed_ratio),
                _format_pct(avg.step_micro_accuracy),
                _format_pct(avg.step_exact_accuracy),
                _format_pct(avg.first_neg1_index_accuracy),
            ]
        )
        _print_table(headers, rows)

        if args.score_metric == "step_micro_acc":
            score = avg.step_micro_accuracy
        elif args.score_metric == "step_exact_acc":
            score = avg.step_exact_accuracy
        else:
            score = avg.first_neg1_index_accuracy

        print(f"FINAL_SCORE[{args.score_metric}]={score * 100:.2f}%")
        print(f"FINAL_SCORE_RAW={score:.6f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
