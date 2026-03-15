from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _json_dumps_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_args(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, (dict, list, int, float, bool)):
        return _json_dumps_canonical(arguments)
    text = str(arguments).strip()
    if text == "":
        return ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _json_dumps_canonical(obj)


def _normalize_tool_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None

    if call.get("type") == "function" and isinstance(call.get("function"), dict):
        fn = call["function"]
        name = str(fn.get("name") or "").strip()
        args = _normalize_args(fn.get("arguments"))
        if name == "":
            return None
        return {"type": "function", "name": name, "arguments": args}

    name = str(call.get("name") or "").strip()
    if name != "":
        return {
            "type": str(call.get("type") or "unknown"),
            "name": name,
            "arguments": _normalize_args(call.get("arguments") or call.get("input") or call.get("parameters")),
        }

    return None


def extract_tool_call_sequence(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages = item.get("messages") or []
    if not isinstance(messages, list):
        return []

    seq: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                norm = _normalize_tool_call(call)
                if norm is not None:
                    seq.append(norm)

        # legacy OpenAI-style single function call
        function_call = msg.get("function_call")
        if isinstance(function_call, dict):
            name = str(function_call.get("name") or "").strip()
            if name:
                seq.append(
                    {
                        "type": "function",
                        "name": name,
                        "arguments": _normalize_args(function_call.get("arguments")),
                    }
                )

    return seq


def _normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        m = _ANSWER_RE.search(text)
        if m:
            return m.group(1).strip()
        return text
    return _json_dumps_canonical(value)


def extract_final_answer(item: dict[str, Any]) -> str:
    if "answer_text" in item:
        return _normalize_answer_text(item.get("answer_text"))

    messages = item.get("messages") or []
    if not isinstance(messages, list):
        return ""

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        return _normalize_answer_text(msg.get("content"))

    return ""


def signature_key(item: dict[str, Any]) -> str:
    payload = {
        "tool_calls": extract_tool_call_sequence(item),
        "answer": extract_final_answer(item),
    }
    raw = _json_dumps_canonical(payload).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[str, dict[str, Any] | None]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if raw.strip() == "":
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield raw, None
                continue
            if not isinstance(obj, dict):
                yield raw, None
                continue
            yield raw, obj


def dedup_jsonl(input_path: Path, output_path: Path) -> None:
    seen: set[str] = set()
    total = 0
    kept = 0
    invalid = 0
    dropped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for raw, obj in iter_jsonl(input_path):
            total += 1
            if obj is None:
                invalid += 1
                out.write(raw + "\n")
                kept += 1
                continue

            key = signature_key(obj)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.write(raw + "\n")
            kept += 1

    print(
        _json_dumps_canonical(
            {
                "input": str(input_path),
                "output": str(output_path),
                "total_lines": total,
                "kept_lines": kept,
                "dropped_duplicates": dropped,
                "invalid_lines_kept": invalid,
                "unique_signatures": len(seen),
            }
        )
    )


def default_output_path(input_path: Path, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or input_path.parent
    suffix = input_path.suffix or ""
    stem = input_path.stem or input_path.name
    name = f"{stem}.dedup{suffix}" if suffix else f"{input_path.name}.dedup"
    return out_dir / name


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate trajectories by (tool-call sequence + final answer).")
    p.add_argument("--input", required=True, help="Input JSONL file path, or a directory containing *.jsonl.")
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL file path (if --input is a file), or output directory (if --input is a directory). "
            "If omitted, defaults to adding .dedup before the extension."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    if input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(input_path.glob("*.jsonl"))
        files = [p for p in files if not p.name.endswith(".dedup.jsonl")]
        if not files:
            raise SystemExit(f"no *.jsonl found under: {input_path}")
        for file_path in files:
            out_path = default_output_path(file_path, output_dir=output_dir)
            dedup_jsonl(input_path=file_path, output_path=out_path)
        return

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    dedup_jsonl(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
