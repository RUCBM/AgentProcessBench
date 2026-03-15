# openai_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import os

import random
import threading
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


_client_local = threading.local()


def _is_retryable_openai_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    if any(s in name for s in ("RateLimit", "Timeout", "APIConnection", "ServiceUnavailable", "InternalServerError")):
        return True

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    return False


def _sleep_exponential_backoff(attempt: int, *, base_s: float, max_s: float) -> None:
    delay = min(max_s, base_s * (2 ** max(0, attempt - 1)))
    delay = delay * (0.75 + 0.5 * random.random())
    time.sleep(delay)


def get_openai_client(base_url: str, api_key: Optional[str]) -> OpenAI:
    """Thread-local OpenAI client to avoid creating too many HTTP sessions."""
    client: Optional[OpenAI] = getattr(_client_local, "client", None)
    cached_base = getattr(_client_local, "base_url", None)
    cached_key = getattr(_client_local, "api_key", None)

    if client is not None and cached_base == base_url and cached_key == api_key:
        return client

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    _client_local.client = client
    _client_local.base_url = base_url
    _client_local.api_key = api_key
    return client


# def openai_chat_completions(
#     *,
#     base_url: str,
#     model: str,
#     messages: List[Dict[str, Any]],
#     n: int = 1,
#     temperature: float = 0.8,
#     max_tokens: int = 1024,
#     timeout_s: int = 60,
#     api_key: Optional[str] = None,
#     tools: Optional[List[Dict[str, Any]]] = None,
#     max_retries: int = 5,
#     backoff_base_s: float = 0.5,
#     backoff_max_s: float = 8.0,
# ) -> List[Dict[str, Any]]:
#     """OpenAI-compatible ChatCompletions wrapper.
#
#     Returns a list of assistant messages as plain dicts:
#       {"role": "assistant", "content": "...", "tool_calls": [...]}
#
#     Notes on timeout:
#     - The OpenAI SDK timeout behavior depends on version and transport.
#     - We keep timeout_s in signature for consistency; if you need strict timeouts,
#       implement retry/timeout outside (e.g., via http client config).
#     """
#     client = get_openai_client(base_url, api_key)
#
#     outs: List[Dict[str, Any]] = []
#     for _ in range(max(1, n)):
#         last_exc: Optional[BaseException] = None
#         attempts = max(1, int(max_retries))
#
#         for attempt in range(1, attempts + 1):
#             try:
#                 create_kwargs: Dict[str, Any] = dict(
#                     model=model,
#                     messages=messages,
#                     n=1,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     tools=tools,
#                 )
#                 # Some OpenAI SDK versions accept per-request timeout, others do not.
#                 try:
#                     resp = client.chat.completions.create(**create_kwargs, timeout=timeout_s)
#                 except TypeError:
#                     resp = client.chat.completions.create(**create_kwargs)
#                 break
#             except Exception as exc:
#                 last_exc = exc
#                 if attempt >= attempts or not _is_retryable_openai_error(exc):
#                     raise
#                 print(f"LLM call failed (attempt {attempt}/{attempts}): {exc}", flush=True)
#                 _sleep_exponential_backoff(attempt, base_s=backoff_base_s, max_s=backoff_max_s)
#         else:
#             if last_exc is not None:
#                 raise last_exc
#
#         if not resp.choices:
#             outs.append({"role": "assistant", "content": "", "tool_calls": []})
#             continue
#
#         msg = resp.choices[0].message
#         out: Dict[str, Any] = {"role": msg.role or "assistant", "content": msg.content or ""}
#
#         tool_calls_out: List[Dict[str, Any]] = []
#         if getattr(msg, "tool_calls", None):
#             for tc in msg.tool_calls:
#                 tool_calls_out.append(
#                     {
#                         "id": tc.id,
#                         "type": tc.type,
#                         "function": {
#                             "name": tc.function.name,
#                             "arguments": tc.function.arguments,
#                         },
#                     }
#                 )
#         out["tool_calls"] = tool_calls_out
#         outs.append(out)
#
#     return outs



def openai_chat_completions(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    n: int = 1,
    temperature: float = 0.8,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    timeout_s: int = 60,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 5,
    backoff_base_s: float = 0.5,
    backoff_max_s: float = 8.0,
) -> List[Dict[str, Any]]:
    client = get_openai_client(base_url, api_key)

    def to_dict(x: Any) -> Dict[str, Any]:
        if x is None:
            return {}
        if hasattr(x, "model_dump"):  # pydantic v2
            return x.model_dump()
        if hasattr(x, "dict"):        # pydantic v1
            return x.dict()
        return dict(getattr(x, "__dict__", {}))

    outs: List[Dict[str, Any]] = []
    for _ in range(max(1, n)):
        attempts = max(1, int(max_retries))
        for attempt in range(1, attempts + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model,
                    messages=messages,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    tools=tools,
                    # extra_body={
                    #     "reasoning": {
                    #         "effort": "medium"
                    #         }
                    #     },
                    extra_body={
                        "reasoning_effort": "medium"
                    },
                )
                try:
                    resp = client.chat.completions.create(**kwargs, timeout=timeout_s)
                except TypeError:
                    resp = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_openai_error(exc):
                    raise
                _sleep_exponential_backoff(attempt, base_s=backoff_base_s, max_s=backoff_max_s)

        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        out = to_dict(msg)
        out.setdefault("role", "assistant")
        out.setdefault("content", "")
        outs.append(out)

    return outs


if __name__ == "__main__":
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compute 23 * 47 step by step and give the final answer."},
    ]

    outs = openai_chat_completions(
        base_url=base_url,
        api_key=api_key,
        model="deepseek-reasoner",
        messages=messages,
        temperature=0,
        max_tokens=32000,
    )

    print("\n===== RAW RESPONSE =====")
    for i, o in enumerate(outs):
        print(f"\n--- choice {i} ---")
        print(json.dumps(o, indent=2, ensure_ascii=False))