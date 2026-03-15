from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
                yield obj


def _extract_model_name(item: dict[str, Any]) -> str:
    meta = item.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    for key in ("model_name", "llm_model", "model", "model_id", "base_model"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for key in ("model_name", "llm_model", "model"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    tau2_info = meta.get("tau2_info")
    if isinstance(tau2_info, dict):
        agent_info = tau2_info.get("agent_info")
        if isinstance(agent_info, dict):
            llm = agent_info.get("llm")
            if isinstance(llm, str) and llm.strip():
                return llm.strip()

    source_path = meta.get("source_path")
    if isinstance(source_path, str) and source_path:
        # e.g. ".../agent_Qwen3-30B-A3B-Instruct-2507_user_Qwen3-30B-A3B-Instruct-2507_...json"
        marker_a = "agent_"
        marker_b = "_user_"
        a = source_path.find(marker_a)
        b = source_path.find(marker_b)
        if a != -1 and b != -1 and a + len(marker_a) < b:
            return source_path[a + len(marker_a) : b]

    return "unknown"


def _extract_score(item: dict[str, Any]) -> float | None:
    reward_info = item.get("reward_info")
    if isinstance(reward_info, dict):
        v = reward_info.get("reward")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-model scores on each benchmark from dataset jsonl files.")
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        default=Path("annotation_platform/raw_trajectories/annotation_file_diverse_queries"),
        help="Directory containing benchmark *.jsonl (bfcl.jsonl, gaia_dev.jsonl, hotpotqa.jsonl, tau2.jsonl).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["bfcl", "gaia_dev", "hotpotqa", "tau2"],
        help="Datasets to include (stems without .jsonl).",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    any_plotted = False
    for dataset in args.datasets:
        path = args.annotation_dir / f"{dataset}.jsonl"
        if not path.exists():
            continue

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for item in _iter_jsonl(path):
            score = _extract_score(item)
            if score is None:
                continue
            model = _extract_model_name(item)
            sums[model] = sums.get(model, 0.0) + score
            counts[model] = counts.get(model, 0) + 1

        models = sorted(counts.keys())
        if not models:
            continue
        means = [sums[m] / counts[m] for m in models]
        n = [counts[m] for m in models]

        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(models)), 4.5))
        ax.bar(models, means)
        ax.set_ylabel("Mean reward_info.reward")
        ax.set_title(f"{dataset}: score by model")
        ax.tick_params(axis="x", labelrotation=25)
        for i, (y, nn) in enumerate(zip(means, n, strict=False)):
            ax.text(i, y + (0.03 if y >= 0 else -0.05), f"n={nn}", ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
        fig.tight_layout()
        any_plotted = True

    if not any_plotted:
        raise SystemExit("No scores found (expected reward_info.reward in dataset jsonl).")

    plt.show()


if __name__ == "__main__":
    main()
