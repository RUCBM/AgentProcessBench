# io_utils.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_parquet_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path}")
    return pd.read_parquet(path)


def safe_json_loads(s: Any, default: Any = None) -> Any:
    """Best-effort JSON parsing. Returns default on failure."""
    if s is None:
        return default
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8")
        except Exception:
            return default
    if not isinstance(s, str):
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _json_default(obj: Any) -> Any:
    """Best-effort fallback for json.dumps(default=...).

    Keeps rollouts/logs writable even if parquet rows or tool outputs contain numpy/pandas types.
    """
    try:
        import numpy as np  # local import to avoid hard dependency at import time

        if isinstance(obj, np.ndarray):
            return list(obj)
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, set):
        return list(obj)

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump(exclude_none=True)
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass

    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


def safe_json_dumps(obj: Any, *, ensure_ascii: bool = False, **kwargs: Any) -> str:
    """json.dumps with a default handler that supports numpy/pandas objects."""
    return json.dumps(obj, ensure_ascii=ensure_ascii, default=_json_default, **kwargs)


def to_native(obj: Any) -> Any:
    """Best-effort conversion of parquet/pyarrow-loaded nested values to native Python objects.

    Handles:
    - pyarrow scalars: .as_py()
    - pyarrow arrays: .to_pylist()
    - JSON-encoded str/bytes
    """
    if isinstance(obj, (list, dict)):
        return obj

    # pyarrow scalar-like
    if hasattr(obj, "as_py") and callable(getattr(obj, "as_py")):
        try:
            return obj.as_py()
        except Exception:
            pass

    # pyarrow list-like
    if hasattr(obj, "to_pylist") and callable(getattr(obj, "to_pylist")):
        try:
            return obj.to_pylist()
        except Exception:
            pass

    # JSON string/bytes
    if isinstance(obj, (str, bytes)):
        parsed = safe_json_loads(obj, default=None)
        return parsed if parsed is not None else obj

    return obj
