from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _coerce_label(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int) and v in (-1, 0, 1):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s in {"-1", "0", "1"}:
            return int(s)
        return s
    return v


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))

    def fmt_row(r: list[str]) -> str:
        parts = []
        for i, cell in enumerate(r):
            if i == 0:
                parts.append(cell.ljust(widths[i]))
            else:
                parts.append(cell.rjust(widths[i]))
        return "  ".join(parts)

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def _unique_trunc(names: list[str], *, width: int) -> dict[str, str]:
    base = {n: _truncate(n, width) for n in names}
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for n in names:
        b = base[n]
        k = seen.get(b, 0) + 1
        seen[b] = k
        if k == 1:
            out[n] = b
            continue
        suffix = f"#{k}"
        keep = max(1, width - len(suffix))
        out[n] = _truncate(n, keep) + suffix
    return out


@dataclass(frozen=True)
class Record:
    final_label: Any
    step_labels: dict[str, Any]


def _first_negative_index(step_labels: dict[str, Any]) -> int | None:
    indices: list[int] = []
    for k, v in step_labels.items():
        if v != -1:
            continue
        try:
            idx = int(str(k).strip())
        except Exception:
            continue
        indices.append(idx)
    return min(indices) if indices else None


def _load_dir(dir_path: Path) -> dict[str, dict[str, dict[str, Record]]]:
    """
    Returns: dataset -> model -> record_id -> Record
    Filename expected: <dataset>__<anything>.jsonl
    """
    out: dict[str, dict[str, dict[str, Record]]] = defaultdict(lambda: defaultdict(dict))
    for p in sorted(dir_path.glob("*.jsonl")):
        stem = p.stem
        if "__" not in stem:
            continue
        dataset, model = stem.split("__", 1)
        for obj in _iter_jsonl(p):
            rid = obj.get("record_id")
            if not isinstance(rid, str) or not rid:
                continue
            step_labels_raw = obj.get("step_labels") if isinstance(obj.get("step_labels"), dict) else {}
            step_labels = {str(k): _coerce_label(v) for k, v in step_labels_raw.items()}
            out[dataset][model][rid] = Record(
                final_label=_coerce_label(obj.get("final_label")),
                step_labels=step_labels,
            )
    return out


def _pairwise_agreement(
    a: dict[str, Record],
    b: dict[str, Record],
    common: list[str],
) -> tuple[float, float, float, int]:
    if not common:
        return 0.0, 0.0, 0.0, 0

    final_match = 0
    step_match = 0
    step_total = 0
    first_neg_match = 0

    for rid in common:
        ra = a[rid]
        rb = b[rid]
        if ra.final_label == rb.final_label:
            final_match += 1

        if _first_negative_index(ra.step_labels) == _first_negative_index(rb.step_labels):
            first_neg_match += 1

        if set(ra.step_labels.keys()) != set(rb.step_labels.keys()):
            raise Exception
        keys = set(ra.step_labels.keys()) | set(rb.step_labels.keys())

        for k in keys:
            va = ra.step_labels.get(k)
            vb = rb.step_labels.get(k)
            step_total += 1
            if va == vb:
                step_match += 1

    return (
        final_match / len(common),
        (step_match / step_total if step_total else 0.0),
        first_neg_match / len(common),
        step_total,
    )


def _all_same_final(models: list[dict[str, Record]], common: list[str]) -> float:
    if not common:
        return 0.0
    same = 0
    for rid in common:
        vals = [m[rid].final_label for m in models]
        if all(v == vals[0] for v in vals[1:]):
            same += 1
    return same / len(common)


def _all_same_first_negative(models: list[dict[str, Record]], common: list[str]) -> float:
    if not common:
        return 0.0
    same = 0
    for rid in common:
        vals = [_first_negative_index(m[rid].step_labels) for m in models]
        if all(v == vals[0] for v in vals[1:]):
            same += 1
    return same / len(common)


def _all_same_steps(models: list[dict[str, Record]], common: list[str]) -> tuple[float, int]:
    if not common:
        return 0.0, 0
    match = 0
    total = 0
    for rid in common:
        keys: set[str] = set()
        for m in models:
            keys |= set(m[rid].step_labels.keys())
        for k in keys:
            vals = [m[rid].step_labels.get(k) for m in models]
            total += 1
            if all(v == vals[0] for v in vals[1:]):
                match += 1
    return (match / total if total else 0.0), total


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM annotation consistency across datasets (intersection only).")
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing <dataset>__<model>.jsonl files (e.g. annotation_platform/annotation_results/llm_annotations).",
    )
    parser.add_argument("--name_width", type=int, default=28, help="Max display width for model names.")
    args = parser.parse_args()

    data = _load_dir(args.dir)
    if not data:
        raise SystemExit(f"No usable *.jsonl found in {args.dir}")

    all_datasets = sorted(data.keys())
    all_models = sorted({m for ds in data.values() for m in ds.keys()})
    model_disp = _unique_trunc(all_models, width=args.name_width)

    print("=== Records Count Summary ===")
    headers = ["dataset"] + [model_disp[m] for m in all_models] + ["n_models", "intersection"]
    summary_rows: list[list[str]] = []
    for dataset in all_datasets:
        models = sorted(data[dataset].keys())
        common_ids: set[str] | None = None
        for m in models:
            ids = set(data[dataset][m].keys())
            common_ids = ids if common_ids is None else (common_ids & ids)
        intersection = len(common_ids or set())

        row = [dataset]
        for m in all_models:
            row.append(str(len(data[dataset][m])) if m in data[dataset] else "")
        row.append(str(len(models)))
        row.append(str(intersection))
        summary_rows.append(row)
    _print_table(headers, summary_rows)

    any_printed = False
    for dataset in sorted(data.keys()):
        models = sorted(data[dataset].keys())
        if len(models) < 2:
            continue
        any_printed = True

        common_ids: set[str] | None = None
        sizes: dict[str, int] = {}
        for m in models:
            ids = set(data[dataset][m].keys())
            sizes[m] = len(ids)
            common_ids = ids if common_ids is None else (common_ids & ids)
        common = sorted(common_ids or set())

        print()
        print(f"=== {dataset} ===")
        print(f"models: {', '.join(model_disp[m] for m in models)}")
        print(
            "counts: "
            + ", ".join(f"{model_disp[m]}={sizes[m]}" for m in models)
            + f" | intersection={len(common)}"
        )
        if not common:
            print("No intersection records; skip.")
            continue

        all_final = _all_same_final([data[dataset][m] for m in models], common)
        all_first_neg = _all_same_first_negative([data[dataset][m] for m in models], common)
        all_steps, all_steps_total = _all_same_steps([data[dataset][m] for m in models], common)
        print(
            f"all-model agreement: final={_fmt_pct(all_final)} | first_-1={_fmt_pct(all_first_neg)} | "
            f"steps={_fmt_pct(all_steps)} (n_steps={all_steps_total})"
        )

        rows: list[list[str]] = []
        for a, b in itertools.combinations(models, 2):
            fa, sa, na, n_steps = _pairwise_agreement(data[dataset][a], data[dataset][b], common)
            rows.append(
                [
                    f"{_truncate(a, args.name_width)} vs {_truncate(b, args.name_width)}",
                    str(len(common)),
                    str(n_steps),
                    _fmt_pct(fa),
                    _fmt_pct(na),
                    _fmt_pct(sa),
                ]
            )

        _print_table(
            headers=["pair", "n_records", "n_steps", "final_agree", "first_-1_agree", "step_agree"],
            rows=rows,
        )

    if not any_printed:
        raise SystemExit("Found files, but no dataset has >=2 models to compare.")


if __name__ == "__main__":
    main()
