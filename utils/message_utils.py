# message_utils.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from utils.io_utils import safe_json_loads
from utils.tool_runtime import ToolCall


ANSWER_SPAN_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)


def normalize_base_messages(prompt: Any) -> List[Dict[str, Any]]:
    """Normalize dataset prompt field into OpenAI-style messages."""
    if isinstance(prompt, str):
        prompt = safe_json_loads(prompt, default=None)

    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return []
    out: List[Dict[str, Any]] = []
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(role, str):
            continue
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def extract_question_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_contents:
        return None
    last = user_contents[-1]
    if not isinstance(last, str):
        last = str(last)
    if "Question:" in last:
        return last.split("Question:")[-1].strip()
    return last.strip() if last else None


def contains_answer(raw_text: str) -> bool:
    text = raw_text or ""
    return ANSWER_RE.search(text) is not None or extract_boxed_content(text) is not None


def extract_boxed_content(text: str):
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return None

def extract_answer_text(raw_text: str) -> Optional[str]:
    matches = ANSWER_SPAN_RE.findall(raw_text or "")
    if not matches:
        return extract_boxed_content(raw_text or "")
    return matches[-1].strip()


def _extract_code_block(text: str) -> Optional[str]:
    if not text:
        return None
    m = CODE_RE.search(text)
    if not m:
        return None
    code = (m.group(1) or "").strip()
    return code or None


def parse_native_tool_calls(assistant_msg: Dict[str, Any]) -> List[ToolCall]:
    """Parse OpenAI native tool_calls -> List[ToolCall]."""
    calls: List[ToolCall] = []
    tool_calls = assistant_msg.get("tool_calls") or []
    content = assistant_msg.get("content") if isinstance(assistant_msg.get("content"), str) else ""
    code = _extract_code_block(content or "")
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if not isinstance(name, str) or not name:
            continue

        if isinstance(args, str):
            args_obj = safe_json_loads(args, default={})
        elif isinstance(args, dict):
            args_obj = args
        else:
            args_obj = {}

        if code:
            if not isinstance(args_obj, dict):
                args_obj = {}
            args_obj = dict(args_obj)
            args_obj.setdefault("code", code)

        calls.append(ToolCall(
            id=tc.get("id") if isinstance(tc.get("id"), str) else None,
            name=name,
            arguments=args_obj if isinstance(args_obj, dict) else {},
        ))
    return calls


def parse_legacy_tool_call(raw_text: str) -> Optional[ToolCall]:
    """Parse legacy <tool_call>{...}</tool_call> -> ToolCall (single)."""
    m = TOOL_CALL_RE.search(raw_text or "")
    if not m:
        return None
    snippet = (m.group(1) or "").strip()
    code = _extract_code_block(snippet)
    if code:
        snippet = CODE_RE.sub("", snippet).strip()

    obj = safe_json_loads(snippet, default=None)
    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    args = obj.get("arguments", {}) or {}

    if not isinstance(name, str) or not name:
        return None
    if isinstance(args, str):
        args = safe_json_loads(args, default={})
    if not isinstance(args, dict):
        args = {}

    if code:
        args = dict(args)
        args.setdefault("code", code)

    return ToolCall(id=None, name=name, arguments=args)
