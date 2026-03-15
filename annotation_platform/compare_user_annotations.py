from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Ann:
    updated_at: datetime
    record_id: str
    index_in_dataset: int | None
    final_label: int | None
    step_labels: dict[str, int | None]


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _read_latest_by_record_id(path: Path) -> dict[str, Ann]:
    latest: dict[str, Ann] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("status") != "done":
            continue
        rid = str(obj["record_id"])
        updated_at = _parse_iso(str(obj["updated_at"]))
        step_labels_raw = obj.get("step_labels") or {}
        step_labels = {str(k): (v if v in (-1, 0, 1) else None) for k, v in step_labels_raw.items()}
        ann = Ann(
            updated_at=updated_at,
            record_id=rid,
            index_in_dataset=obj.get("index_in_dataset"),
            final_label=obj.get("final_label"),
            step_labels=step_labels,
        )
        prev = latest.get(rid)
        if prev is None or ann.updated_at > prev.updated_at:
            latest[rid] = ann
    return latest


def _find_user_files(dir_path: Path, username: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(dir_path.glob(f"*__{username}.jsonl")):
        dataset, u = p.stem.rsplit("__", 1)
        if u == username:
            out[dataset] = p
    return out


def _fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{(num / den) * 100:5.1f}%"


def _fmt_label(v: int | None) -> str:
    return "missing" if v is None else str(v)


def _int_key(s: str) -> tuple[int, str]:
    try:
        return int(s), s
    except Exception:
        return 10**9, s


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: list[str]) -> str:
        parts = []
        for i, cell in enumerate(r):
            parts.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return "  ".join(parts)

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def _cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """
    Cohen's kappa for two raters on nominal labels.
    pairs: list of (a, b) labels, both ints.
    Returns None if not enough data or degenerate denominator.
    """
    n = len(pairs)
    if n == 0:
        return None

    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n

    ca: dict[int, int] = {}
    cb: dict[int, int] = {}
    for a, b in pairs:
        ca[a] = ca.get(a, 0) + 1
        cb[b] = cb.get(b, 0) + 1

    p_e = 0.0
    for lab in set(ca) | set(cb):
        p_e += (ca.get(lab, 0) / n) * (cb.get(lab, 0) / n)

    denom = 1.0 - p_e
    if denom <= 0.0:
        return None
    return (p_o - p_e) / denom


def _fmt_float(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _percentile(sorted_x: list[float], q: float) -> float:
    """
    q in [0,1]. Linear interpolation between closest ranks.
    """
    if not sorted_x:
        raise ValueError("empty")
    if q <= 0:
        return sorted_x[0]
    if q >= 1:
        return sorted_x[-1]
    pos = q * (len(sorted_x) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_x) - 1)
    frac = pos - lo
    return sorted_x[lo] * (1 - frac) + sorted_x[hi] * frac


def _bootstrap_kappa_record_level(
    per_record_pairs: list[list[tuple[int, int]]],
    n_boot: int,
    ci: float,
    seed: int,
) -> tuple[float | None, float | None, float | None, int]:
    """
    Record-level bootstrap:
    - Resample records with replacement
    - Concatenate pairs within sampled records
    - Compute kappa
    Returns: (point_est, lo, hi, n_eff)
    """
    # point estimate (on full set)
    all_pairs: list[tuple[int, int]] = []
    for ps in per_record_pairs:
        all_pairs.extend(ps)
    point = _cohen_kappa(all_pairs)

    if n_boot <= 0 or not per_record_pairs:
        return point, None, None, 0

    alpha = (1.0 - ci) / 2.0
    rng = random.Random(seed)

    samples: list[float] = []
    m = len(per_record_pairs)

    for _ in range(n_boot):
        pairs: list[tuple[int, int]] = []
        for _j in range(m):
            ps = per_record_pairs[rng.randrange(m)]
            pairs.extend(ps)
        k = _cohen_kappa(pairs)
        if k is None:
            continue
        samples.append(k)

    if len(samples) < max(20, int(0.1 * n_boot)): 
        return point, None, None, len(samples)

    samples.sort()
    lo = _percentile(samples, alpha)
    hi = _percentile(samples, 1.0 - alpha)
    return point, lo, hi, len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two users' exported annotations (latest per record_id).")
    parser.add_argument("--dir", type=Path, default="./annotation_platform/annotation_results/exports", help="Exports directory (e.g. annotation_platform/annotation_results/exports).")
    parser.add_argument("--user_a", type=str, required=True)
    parser.add_argument("--user_b", type=str, required=True)
    parser.add_argument("--max_diffs", type=int, default=-1, help="Max differing records to print; -1 means no limit.")

    # NEW: bootstrap options (leave defaults; you can override)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap resamples for kappa CI (record-level). 0 disables.")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level for bootstrap CI (e.g., 0.95).")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for bootstrap.")

    args = parser.parse_args()

    dir_path = args.dir.expanduser()
    files_a = _find_user_files(dir_path, args.user_a)
    files_b = _find_user_files(dir_path, args.user_b)

    datasets = sorted(set(files_a) & set(files_b))
    print(f"Compare: {args.user_a} vs {args.user_b}")
    print(f"Dir: {dir_path}")
    print(f"Datasets (intersection): {', '.join(datasets) if datasets else '(none)'}")

    total_step_match = 0
    total_step = 0
    total_final_match = 0
    total_rec = 0

    # collect flattened pairs for your existing kappa line
    total_final_pairs: list[tuple[int, int]] = []
    total_step_pairs: list[tuple[int, int]] = []

    # NEW: per-record pairs (for record-level bootstrap)
    final_pairs_by_record: list[list[tuple[int, int]]] = []
    step_pairs_by_record: list[list[tuple[int, int]]] = []

    summary_rows: list[list[str]] = []
    all_diffs: list[tuple[str, str, str]] = []

    for dataset in datasets:
        a = _read_latest_by_record_id(files_a[dataset])
        b = _read_latest_by_record_id(files_b[dataset])
        common = sorted(set(a) & set(b))
        if not common:
            continue

        step_match = 0
        step_total = 0
        final_match = 0

        for rid in common:
            aa = a[rid]
            bb = b[rid]

            # for record-level bootstrap
            rec_final_pairs: list[tuple[int, int]] = []
            rec_step_pairs: list[tuple[int, int]] = []

            if aa.final_label == bb.final_label:
                final_match += 1

            if isinstance(aa.final_label, int) and aa.final_label in (-1, 0, 1) and isinstance(bb.final_label, int) and bb.final_label in (-1, 0, 1):
                total_final_pairs.append((aa.final_label, bb.final_label))
                rec_final_pairs.append((aa.final_label, bb.final_label))

            keys_a = set(aa.step_labels)
            keys_b = set(bb.step_labels)
            keys_common = sorted(keys_a & keys_b, key=_int_key)
            keys_missing = sorted((keys_a ^ keys_b), key=_int_key)
            per_rec_diffs: list[str] = []

            for k in keys_common:
                va = aa.step_labels.get(k)
                vb = bb.step_labels.get(k)
                step_total += 1
                if va == vb:
                    step_match += 1
                else:
                    per_rec_diffs.append(f"{k}: {_fmt_label(va)} vs {_fmt_label(vb)}")

                if isinstance(va, int) and va in (-1, 0, 1) and isinstance(vb, int) and vb in (-1, 0, 1):
                    total_step_pairs.append((va, vb))
                    rec_step_pairs.append((va, vb))

            for k in keys_missing:
                va = aa.step_labels.get(k)
                vb = bb.step_labels.get(k)
                per_rec_diffs.append(f"{k}: {_fmt_label(va)} vs {_fmt_label(vb)}")

            # store per-record units for bootstrap (even if empty, keep it: it reflects missingness)
            final_pairs_by_record.append(rec_final_pairs)
            step_pairs_by_record.append(rec_step_pairs)

            if aa.final_label != bb.final_label or per_rec_diffs:
                head = (
                    f"[{dataset}] {rid} "
                    f"(index_a={aa.index_in_dataset}, index_b={bb.index_in_dataset}) "
                    f"final: {aa.final_label} vs {bb.final_label}"
                )
                body = "  steps: " + (", ".join(per_rec_diffs) if per_rec_diffs else "(all same)")
                all_diffs.append((head, body, ""))

        summary_rows.append(
            [
                dataset,
                str(len(common)),
                _fmt_pct(step_match, step_total),
                str(step_total),
                _fmt_pct(final_match, len(common)),
            ]
        )

        total_step_match += step_match
        total_step += step_total
        total_final_match += final_match
        total_rec += len(common)

    print()
    _print_table(
        ["dataset", "n_records", "step_agree", "n_steps", "final_agree"],
        summary_rows or [["(none)", "0", "n/a", "0", "n/a"]],
    )
    print()
    print(f"Overall: n_records={total_rec}, step_agree={_fmt_pct(total_step_match, total_step)} (n_steps={total_step}), final_agree={_fmt_pct(total_final_match, total_rec)}")

    final_kappa = _cohen_kappa(total_final_pairs)
    step_kappa = _cohen_kappa(total_step_pairs)
    print(f"Overall (chance-corrected): final_kappa={_fmt_float(final_kappa)} (n={len(total_final_pairs)}), step_kappa={_fmt_float(step_kappa)} (n={len(total_step_pairs)})")

    # NEW: one extra line with bootstrap CI (record-level)
    n_boot = int(args.bootstrap)
    ci = float(args.ci)
    seed = int(args.seed)

    f_point, f_lo, f_hi, f_eff = _bootstrap_kappa_record_level(final_pairs_by_record, n_boot=n_boot, ci=ci, seed=seed)
    s_point, s_lo, s_hi, s_eff = _bootstrap_kappa_record_level(step_pairs_by_record, n_boot=n_boot, ci=ci, seed=seed)

    def _fmt_ci(lo: float | None, hi: float | None) -> str:
        if lo is None or hi is None:
            return "[n/a]"
        return f"[{lo:.4f}, {hi:.4f}]"

    if n_boot > 0:
        print(
            f"Overall (bootstrap {int(ci*100)}% CI; record-level): "
            f"final_kappa={_fmt_float(f_point)} {_fmt_ci(f_lo, f_hi)} (n_boot_eff={f_eff}/{n_boot}), "
            f"step_kappa={_fmt_float(s_point)} {_fmt_ci(s_lo, s_hi)} (n_boot_eff={s_eff}/{n_boot})"
        )

    if not all_diffs:
        print("\nNo differences found.")
        return

    max_diffs = int(args.max_diffs)
    if max_diffs == 0:
        return
    shown = all_diffs if max_diffs < 0 else all_diffs[:max_diffs]
    print(f"\nDifferences (showing {len(shown)}/{len(all_diffs)}):")
    for i, (h, b, _) in enumerate(shown, start=1):
        print(f"{i:>3}. {h}")
        print(f"     {b}")


if __name__ == "__main__":
    main()
