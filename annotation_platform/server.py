from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from annotation_platform.dataset import DatasetIndex, discover_datasets
    from annotation_platform.storage import AnnotationStore
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from annotation_platform.dataset import DatasetIndex, discover_datasets
    from annotation_platform.storage import AnnotationStore


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


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


def _coerce_label(v: Any) -> int | None:
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
        return None
    return None


def _load_llm_annotations_dir(dir_path: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """
    Returns: dataset -> model_key -> record_id -> {"final_label", "step_labels", "explanations"}
    File naming convention: <dataset>__<model_key>.jsonl
    """
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for p in sorted(dir_path.glob("*.jsonl")):
        stem = p.stem
        if "__" not in stem:
            continue
        dataset, model_key = stem.split("__", 1)
        for obj in _iter_jsonl(p):
            rid = obj.get("record_id")
            if not isinstance(rid, str) or not rid:
                continue
            step_labels_raw = obj.get("step_labels") if isinstance(obj.get("step_labels"), dict) else {}
            step_labels = {str(k): _coerce_label(v) for k, v in step_labels_raw.items()}
            explanations_raw = obj.get("explanations") if isinstance(obj.get("explanations"), dict) else {}
            steps_expl_raw = explanations_raw.get("steps") if isinstance(explanations_raw.get("steps"), dict) else {}
            steps_expl = {str(k): (v if isinstance(v, str) or v is None else str(v)) for k, v in steps_expl_raw.items()}
            final_expl = explanations_raw.get("final")
            if not (isinstance(final_expl, str) or final_expl is None):
                final_expl = str(final_expl)

            out.setdefault(dataset, {}).setdefault(model_key, {})[rid] = {
                "final_label": _coerce_label(obj.get("final_label")),
                "step_labels": step_labels,
                "explanations": {"steps": steps_expl, "final": final_expl},
            }
    return out


class AppState:
    def __init__(self, dataset_dir: Path, data_dir: Path, llm_annotations_dir: Path | None) -> None:
        self.dataset_dir = dataset_dir
        self.data_dir = data_dir
        self.llm_annotations_dir = llm_annotations_dir
        self.static_dir = Path(__file__).resolve().parent / "static"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        exports_dir = self.data_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        self.store = AnnotationStore(self.data_dir / "annotations.sqlite3", exports_dir=exports_dir)
        self.datasets: dict[str, DatasetIndex] = discover_datasets(dataset_dir)
        self.llm_annotations: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        if self.llm_annotations_dir is not None and self.llm_annotations_dir.exists():
            self.llm_annotations = _load_llm_annotations_dir(self.llm_annotations_dir)
        self.lock = threading.Lock()

    def refresh_datasets(self) -> None:
        self.datasets = discover_datasets(self.dataset_dir)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "AgentProcessBenchAnnotation/0.1"

    @property
    def state(self) -> AppState:
        return self.server.state  # type: ignore[attr-defined]

    def _get_dataset_fresh(self, dataset: str) -> DatasetIndex | None:
        ds = self.state.datasets.get(dataset)
        if ds is None:
            return None
        try:
            stale = ds.is_stale()
        except OSError:
            stale = True
        if not stale:
            return ds
        with self.state.lock:
            self.state.refresh_datasets()
            return self.state.datasets.get(dataset)

    def _serve_static(self, rel_path: str) -> None:
        if rel_path == "" or rel_path == "/":
            rel_path = "index.html"
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]

        safe_path = (self.state.static_dir / rel_path).resolve()
        if not str(safe_path).startswith(str(self.state.static_dir.resolve())):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not safe_path.exists() or not safe_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        ctype, _ = mimetypes.guess_type(str(safe_path))
        ctype = ctype or "application/octet-stream"
        content = safe_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path.startswith("/static/"):
            rel = "index.html" if path == "/" else path[len("/static/") :]
            self._serve_static(rel)
            return

        if path == "/api/datasets":
            datasets = []
            for ds in sorted(self.state.datasets.values(), key=lambda d: d.name):
                datasets.append(
                    {
                        "name": ds.name,
                        "path": str(ds.path),
                        "size": ds.size,
                    }
                )
            _json_response(self, HTTPStatus.OK, {"datasets": datasets})
            return

        if path == "/api/progress":
            dataset = (qs.get("dataset", [""])[0] or "").strip()
            annotator = (qs.get("annotator", [""])[0] or "").strip()
            if dataset == "" or annotator == "":
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset and annotator required"})
                return
            ds = self._get_dataset_fresh(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return
            progress = self.state.store.get_progress(dataset=dataset, annotator=annotator)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "dataset": dataset,
                    "annotator": annotator,
                    "done": progress.done,
                    "skipped": progress.skipped,
                    "total": ds.size,
                },
            )
            return

        if path == "/api/item":
            dataset = (qs.get("dataset", [""])[0] or "").strip()
            annotator = (qs.get("annotator", [""])[0] or "").strip()
            index_str = (qs.get("index", [""])[0] or "").strip()
            if dataset == "" or index_str == "":
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset and index required"})
                return
            ds = self._get_dataset_fresh(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return
            try:
                index = int(index_str)
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index must be int"})
                return

            if index < 0 or index >= ds.size:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index out of range"})
                return
            try:
                item = ds.read_item(index)
            except Exception as e:
                _json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "failed to read dataset item", "dataset": dataset, "index": index, "detail": str(e)},
                )
                return
            record_id = ds.record_ids[index]

            existing = None
            if annotator != "":
                existing = self.state.store.get_annotation(dataset=dataset, record_id=record_id, annotator=annotator)
                if existing is not None:
                    messages = item.get("messages") or []
                    if not isinstance(messages, list):
                        messages = []
                    assistant_indices = [
                        i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"
                    ]
                    allowed = {str(i) for i in assistant_indices}
                    step_labels = existing.get("step_labels")
                    if isinstance(step_labels, dict) and allowed:
                        existing = dict(existing)
                        existing["step_labels"] = {k: v for k, v in step_labels.items() if str(k) in allowed}

            payload = ds.to_payload(item=item, dataset=dataset, index=index, record_id=record_id, existing=existing)
            if self.state.llm_annotations:
                refs_for_ds = self.state.llm_annotations.get(dataset) or {}
                models_out: dict[str, Any] = {}
                for model_key in sorted(refs_for_ds.keys()):
                    rec = refs_for_ds[model_key].get(record_id)
                    if rec is None:
                        continue
                    models_out[model_key] = rec
                if models_out:
                    payload["llm_references"] = {"models": models_out, "order": sorted(models_out.keys())}
            _json_response(self, HTTPStatus.OK, payload)
            return

        if path == "/api/next":
            dataset = (qs.get("dataset", [""])[0] or "").strip()
            annotator = (qs.get("annotator", [""])[0] or "").strip()
            if dataset == "" or annotator == "":
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset and annotator required"})
                return
            ds = self._get_dataset_fresh(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return

            cursor = self.state.store.get_cursor(dataset=dataset, annotator=annotator)
            next_index = self.state.store.find_next_unannotated(
                dataset=dataset, annotator=annotator, record_ids=ds.record_ids, start_index=cursor
            )
            if next_index is None:
                _json_response(self, HTTPStatus.OK, {"done": True, "dataset": dataset})
                return
            try:
                item = ds.read_item(next_index)
            except Exception as e:
                _json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "failed to read dataset item",
                        "dataset": dataset,
                        "index": next_index,
                        "detail": str(e),
                        "hint": "If you regenerated the JSONL while the server was running, refresh/restart the server.",
                    },
                )
                return
            record_id = ds.record_ids[next_index]
            existing = self.state.store.get_annotation(dataset=dataset, record_id=record_id, annotator=annotator)
            if existing is not None:
                messages = item.get("messages") or []
                if not isinstance(messages, list):
                    messages = []
                assistant_indices = [
                    i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"
                ]
                allowed = {str(i) for i in assistant_indices}
                step_labels = existing.get("step_labels")
                if isinstance(step_labels, dict) and allowed:
                    existing = dict(existing)
                    existing["step_labels"] = {k: v for k, v in step_labels.items() if str(k) in allowed}
            payload = ds.to_payload(
                item=item, dataset=dataset, index=next_index, record_id=record_id, existing=existing
            )
            if self.state.llm_annotations:
                refs_for_ds = self.state.llm_annotations.get(dataset) or {}
                models_out: dict[str, Any] = {}
                for model_key in sorted(refs_for_ds.keys()):
                    rec = refs_for_ds[model_key].get(record_id)
                    if rec is None:
                        continue
                    models_out[model_key] = rec
                if models_out:
                    payload["llm_references"] = {"models": models_out, "order": sorted(models_out.keys())}
            _json_response(self, HTTPStatus.OK, payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = _read_json_body(self)
        dataset = (body.get("dataset") or "").strip()
        record_id = (body.get("record_id") or "").strip()
        annotator = (body.get("annotator") or "").strip()
        index_in_dataset = body.get("index_in_dataset")
        step_labels = body.get("step_labels") or {}
        final_label = body.get("final_label")
        final_label_touched = body.get("final_label_touched")
        status = (body.get("status") or "in_progress").strip()
        comment = body.get("comment")
        data_source = body.get("data_source")
        query_index = body.get("query_index")
        sample_index = body.get("sample_index")
        username = body.get("username")

        if dataset == "" or record_id == "" or annotator == "":
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset, record_id, annotator required"})
            return
        if self._get_dataset_fresh(dataset) is None:
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
            return
        if not isinstance(step_labels, dict):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "step_labels must be dict"})
            return
        if final_label is None:
            final_label = 0
        if not isinstance(final_label, int) or final_label not in (-1, 0, 1):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "final_label must be -1/0/1"})
            return
        if isinstance(final_label_touched, bool):
            final_label_touched_bool = final_label_touched
        elif isinstance(final_label_touched, int) and final_label_touched in (0, 1):
            final_label_touched_bool = bool(final_label_touched)
        elif final_label_touched is None:
            final_label_touched_bool = False
        else:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "final_label_touched must be bool"})
            return
        if status not in ("in_progress", "done", "skipped"):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "status must be in_progress/done/skipped"})
            return
        if index_in_dataset is None:
            index_in_dataset = -1
        if not isinstance(index_in_dataset, int):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index_in_dataset must be int"})
            return
        if data_source is not None and not isinstance(data_source, str):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "data_source must be str"})
            return
        if query_index is not None and not isinstance(query_index, int):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "query_index must be int"})
            return
        if sample_index is not None and not isinstance(sample_index, int):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "sample_index must be int"})
            return
        if username is not None and not isinstance(username, str):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "username must be str"})
            return
        if username is None:
            username = annotator

        if status in ("done", "in_progress"):
            if not final_label_touched_bool:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"final_label_touched required for status={status}"},
                )
                return

            ds = self._get_dataset_fresh(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return
            if index_in_dataset < 0 or index_in_dataset >= ds.size:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index_in_dataset out of range"})
                return
            if ds.record_ids[index_in_dataset] != record_id:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "record_id does not match index_in_dataset"})
                return
            try:
                item = ds.read_item(index_in_dataset)
            except Exception as e:
                _json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "failed to read dataset item for validation",
                        "dataset": dataset,
                        "index": index_in_dataset,
                        "detail": str(e),
                    },
                )
                return
            messages = item.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            assistant_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
            missing_steps: list[int] = []
            invalid_steps: list[int] = []
            for idx in assistant_indices:
                v = step_labels.get(str(idx))
                if v not in (-1, 0, 1):
                    if v is None:
                        missing_steps.append(idx)
                    else:
                        invalid_steps.append(idx)
            if missing_steps:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "missing step_labels for assistant messages", "missing": missing_steps},
                )
                return
            if invalid_steps:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid step_labels for assistant messages", "invalid": invalid_steps},
                )
                return
            # Drop any stale extra keys that may exist from older dataset versions.
            step_labels = {str(idx): step_labels.get(str(idx)) for idx in assistant_indices}
        else:
            # For skipped items, do not persist step labels to avoid carrying stale keys.
            step_labels = {}

        with self.state.lock:
            self.state.store.upsert_annotation(
                dataset=dataset,
                record_id=record_id,
                annotator=annotator,
                username=username,
                index_in_dataset=index_in_dataset,
                data_source=data_source,
                query_index=query_index,
                sample_index=sample_index,
                step_labels=step_labels,
                final_label=final_label,
                final_label_touched=final_label_touched_bool,
                status=status,
                comment=comment,
            )
            if status in ("done", "skipped") and index_in_dataset >= 0:
                self.state.store.set_cursor(dataset=dataset, annotator=annotator, cursor_index=index_in_dataset + 1)

        _json_response(self, HTTPStatus.OK, {"ok": True})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentProcessBench annotation platform (stdlib server).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--annotation_dir",
        default=str(Path("annotation_platform/raw_trajectories/annotation_file_diverse_queries")),
        help="Directory containing *.jsonl to annotate.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(Path("annotation_platform/annotation_results")),
        help="Directory for sqlite + exports.",
    )
    parser.add_argument(
        "--llm_annotations_dir",
        default="./annotation_platform/annotation_results/llm_annotations",
        help="Optional directory containing LLM judge annotation JSONL files (named <dataset>__*.jsonl) to show as step-level references.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.annotation_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    llm_annotations_dir_arg = str(args.llm_annotations_dir or "").strip()
    llm_annotations_dir: Path | None
    if llm_annotations_dir_arg:
        llm_annotations_dir = Path(llm_annotations_dir_arg).expanduser().resolve()
    else:
        candidate = data_dir / "llm_annotations"
        llm_annotations_dir = candidate if candidate.exists() else None

    if not dataset_dir.exists():
        raise SystemExit(f"annotation_dir not found: {dataset_dir}")

    state = AppState(dataset_dir=dataset_dir, data_dir=data_dir, llm_annotations_dir=llm_annotations_dir)
    if len(state.datasets) == 0:
        raise SystemExit(f"no datasets found under: {dataset_dir} (expected *.jsonl)")

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.state = state  # type: ignore[attr-defined]

    print(f"[annotation] serving on http://{args.host}:{args.port}")
    print(f"[annotation] datasets: {', '.join(sorted(state.datasets))}")
    print(f"[annotation] sqlite: {state.store.db_path}")
    print(f"[annotation] exports: {state.store.exports_dir}")
    if state.llm_annotations_dir is not None:
        ds_cnt = len(state.llm_annotations)
        model_cnt = sum(len(v) for v in state.llm_annotations.values())
        print(f"[annotation] llm_annotations_dir: {state.llm_annotations_dir} (datasets={ds_cnt}, models={model_cnt})")
    server.serve_forever()


if __name__ == "__main__":
    main()
