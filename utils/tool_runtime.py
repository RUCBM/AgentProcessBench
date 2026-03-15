# tool_runtime.py
from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from utils.io_utils import safe_json_dumps


logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: Optional[str]
    name: str
    arguments: Dict[str, Any]


def import_object(dotted: str) -> Any:
    """Import object by dotted path:
    - 'a.b.c:Obj' or 'a.b.c.Obj'
    """
    dotted = dotted.strip()
    if ":" in dotted:
        mod, attr = dotted.split(":", 1)
    else:
        mod, attr = dotted.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, attr)


def discover_tool_classes(
    module_name: str,
    *,
    base_class: Optional[Type[Any]] = None,
    recursive: bool = True,
) -> List[Type[Any]]:
    """Discover tool classes inside a module/package.

    - If module_name is a package and recursive=True, imports submodules via pkgutil.walk_packages.
    - Collects classes from loaded modules whose name starts with module_name.
    - If base_class is provided, keeps only subclasses of base_class (excluding base_class itself).
    """
    mod = importlib.import_module(module_name)

    # Best-effort import submodules for discovery
    if recursive and hasattr(mod, "__path__"):
        for _, subname, _ in pkgutil.walk_packages(mod.__path__, prefix=mod.__name__ + "."):
            try:
                importlib.import_module(subname)
            except Exception:
                continue

    classes: List[Type[Any]] = []
    prefix = module_name + "."
    for name, m in list(sys.modules.items()):
        if m is None:
            continue
        if name != module_name and not name.startswith(prefix):
            continue
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if base_class is None:
                classes.append(obj)
                continue
            try:
                if obj is base_class:
                    continue
                if issubclass(obj, base_class):
                    classes.append(obj)
            except Exception:
                continue

    # de-dup
    uniq: Dict[str, Type[Any]] = {}
    for c in classes:
        key = f"{c.__module__}.{c.__qualname__}"
        uniq[key] = c
    return list(uniq.values())


def _schema_to_openai_dict(schema: Any) -> Dict[str, Any]:
    """Normalize tool schema (dict or pydantic-like) into a plain dict for OpenAI tools=."""
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    # pydantic v2
    if hasattr(schema, "model_dump") and callable(getattr(schema, "model_dump")):
        try:
            return schema.model_dump(exclude_unset=True, exclude_none=True)
        except Exception:
            pass
    # pydantic v1
    if hasattr(schema, "dict") and callable(getattr(schema, "dict")):
        try:
            return schema.dict(exclude_unset=True, exclude_none=True)  # type: ignore[attr-defined]
        except Exception:
            pass
    return {}


def _convert_schema_via_model(schema_dict: Dict[str, Any], schema_model_path: str) -> Any:
    """Optionally convert schema dict to a schema model object (e.g., OpenAIFunctionToolSchema).

    schema_model_path: dotted path to a class in THIS repo folder (or installed deps),
                      e.g. 'tools.schemas:OpenAIFunctionToolSchema'
    """
    model_cls = import_object(schema_model_path)
    # pydantic v2
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(schema_dict)
    # pydantic v1
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(schema_dict)  # type: ignore[attr-defined]
    # fallback: constructor
    return model_cls(**schema_dict)


def _tool_response_to_text(resp: Any) -> str:
    """Convert tool response object into a string to be used as tool message content."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, (dict, list)):
        return safe_json_dumps(resp)

    # common pattern: ToolResponse(text=...)
    if hasattr(resp, "text"):
        try:
            t = getattr(resp, "text")
            if t is not None:
                return str(t)
        except Exception:
            pass

    # pydantic-like object
    if hasattr(resp, "model_dump") and callable(getattr(resp, "model_dump")):
        try:
            return safe_json_dumps(resp.model_dump(exclude_none=True))
        except Exception:
            pass

    return str(resp)


class ToolManager:
    """A small runtime that:
    - instantiates tool classes (possibly multiple instances per class)
    - aggregates OpenAI tool schemas
    - creates per-trajectory tool instances (instance_id)
    - routes tool_calls -> tool.execute
    - releases per-trajectory instances
    """

    def __init__(
        self,
        tool_classes: List[Type[Any]],
        tool_configs: Optional[Dict[str, Any]] = None,
        *,
        call_semaphore: Optional[threading.Semaphore] = None,
    ):
        self._tool_configs = tool_configs or {}
        self._tools_by_name: Dict[str, Any] = {}
        self._call_semaphore = call_semaphore

        for cls in tool_classes:
            cls_path = f"{cls.__module__}.{cls.__qualname__}"

            # Collect config entries for this class:
            # 1) old single entry: tool_configs[cls_path] = {...}
            # 2) list entry: tool_configs[cls_path] = [{...}, {...}]
            # 3) keyed entry: tool_configs[f"{cls_path}#<alias>"] = {...}
            cfg_entries = self._collect_cfg_entries_for_class(cls_path)

            # If no config entry exists, still instantiate one tool with empty config/schema
            if not cfg_entries:
                cfg_entries = [("default", {})]

            for alias, cfg_entry in cfg_entries:
                tool = self._build_tool_from_cfg_entry(cls, cfg_entry)

                # Determine tool name (priority: tool.name -> schema function.name -> alias fallback)
                tool_name = getattr(tool, "name", None)
                if not isinstance(tool_name, str) or not tool_name:
                    schema = getattr(tool, "tool_schema", None)
                    schema_d = _schema_to_openai_dict(schema)
                    tool_name = ((schema_d.get("function") or {}).get("name")) if isinstance(schema_d, dict) else None

                if not isinstance(tool_name, str) or not tool_name:
                    tool_name = f"{cls.__name__}_{alias}"

                # Safety: prevent silent overwrite
                if tool_name in self._tools_by_name:
                    raise ValueError(
                        f"Duplicate tool name detected: '{tool_name}'. "
                        f"Check your tool_config schema/function.name or cls_path#alias keys."
                    )

                self._tools_by_name[tool_name] = tool

    def _collect_cfg_entries_for_class(self, cls_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Return a list of (alias, cfg_entry_dict) for a tool class path.

        Supported:
        - tool_configs[cls_path] = {...}                       -> [("default", {...})]
        - tool_configs[cls_path] = [{...}, {...}]              -> [("0", {...}), ("1", {...})]
        - tool_configs[f"{cls_path}#<alias>"] = {...}          -> [("<alias>", {...}), ...]
        """
        entries: List[Tuple[str, Dict[str, Any]]] = []

        # 1) cls_path direct
        direct = self._tool_configs.get(cls_path)
        if isinstance(direct, list):
            for i, item in enumerate(direct):
                if isinstance(item, dict):
                    entries.append((str(i), item))
        elif isinstance(direct, dict):
            entries.append(("default", direct))

        # 2) cls_path#alias keys
        prefix = cls_path + "#"
        for k, v in self._tool_configs.items():
            if not isinstance(k, str) or not k.startswith(prefix):
                continue
            alias = k[len(prefix):].strip()
            if not alias:
                alias = "default"
            if isinstance(v, dict):
                entries.append((alias, v))

        # If both direct and #alias are provided, keep all; user likely intends multiple tools.
        # To make behavior deterministic, sort by alias.
        entries.sort(key=lambda x: x[0])
        return entries

    def _build_tool_from_cfg_entry(self, cls: Type[Any], cfg_entry: Dict[str, Any]) -> Any:
        """Build one tool instance from a single config entry."""
        # config: arbitrary dict passed to tool
        config = cfg_entry.get("config", cfg_entry)  # allow {config:{...}} or direct {...}
        if not isinstance(config, dict):
            config = {}

        # schema: optional dict in OpenAI format
        schema_dict = cfg_entry.get("schema")
        schema_obj: Any = None
        if isinstance(schema_dict, dict):
            schema_model_path = cfg_entry.get("schema_model")  # optional dotted path
            if isinstance(schema_model_path, str) and schema_model_path.strip():
                schema_obj = _convert_schema_via_model(schema_dict, schema_model_path.strip())
            else:
                schema_obj = schema_dict

        return self._instantiate_tool(cls, config=config, tool_schema=schema_obj)

    def _instantiate_tool(self, cls: Type[Any], *, config: Dict[str, Any], tool_schema: Any) -> Any:
        """Instantiate tool with flexible signatures:
        - (config, tool_schema)
        - (config)
        - ()
        """
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters.keys() if p != "self"]
        except Exception:
            params = []

        if len(params) >= 2:
            try:
                return cls(config, tool_schema)
            except Exception:
                pass
        if len(params) >= 1:
            try:
                return cls(config)
            except Exception:
                pass
        return cls()

    def get_openai_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool in self._tools_by_name.values():
            schema = None
            if hasattr(tool, "get_openai_tool_schema"):
                try:
                    schema = tool.get_openai_tool_schema()
                except Exception:
                    schema = None
            if schema is None:
                schema = getattr(tool, "tool_schema", None)

            schema_d = _schema_to_openai_dict(schema)
            if schema_d:
                schemas.append(schema_d)
        return schemas

    async def create_instances(self) -> Dict[str, str]:
        """Create per-trajectory tool instances. Supports sync or async tool.create()."""
        instances: Dict[str, str] = {}
        for name, tool in self._tools_by_name.items():
            if not hasattr(tool, "create"):
                continue
            try:
                out = tool.create()
                if inspect.isawaitable(out):
                    instance_id, _resp = await out
                else:
                    instance_id, _resp = out
                if isinstance(instance_id, str) and instance_id:
                    instances[name] = instance_id
            except Exception:
                continue
        return instances

    async def execute_call(self, instances: Dict[str, str], call: ToolCall) -> Tuple[str, Dict[str, Any]]:
        """Execute a tool call (sync or async tool.execute). Returns (tool_text, tool_metrics)."""
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            return safe_json_dumps({"error": f"Unknown tool: {call.name}"}), {"error": "unknown_tool"}

        instance_id = instances.get(call.name, "")
        acquired = False
        if self._call_semaphore is not None:
            self._call_semaphore.acquire()
            acquired = True
        try:
            out = tool.execute(instance_id, call.arguments)
            if inspect.isawaitable(out):
                tool_resp, tool_reward, tool_metrics = await out
            else:
                tool_resp, tool_reward, tool_metrics = out
        except Exception as e:
            logger.warning("Tool execute failed: %s: %s", call.name, e)
            return safe_json_dumps({"error": f"Tool execute failed: {e}"}), {"error": str(e)}
        finally:
            if acquired:
                self._call_semaphore.release()

        return _tool_response_to_text(tool_resp), (tool_metrics or {})

    async def release_instances(self, instances: Dict[str, str]) -> None:
        """Release per-trajectory tool instances. Supports sync or async tool.release()."""
        for name, tool in self._tools_by_name.items():
            instance_id = instances.get(name)
            if not instance_id:
                continue
            if not hasattr(tool, "release"):
                continue
            try:
                out = tool.release(instance_id)
                if inspect.isawaitable(out):
                    await out
            except Exception:
                continue
