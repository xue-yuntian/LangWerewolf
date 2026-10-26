"""LLM 层：全项目唯一联网的地方。

对外只暴露 client 里的几个名字，调用方不要直接 import llm.client。
这样以后换实现（比如换成带流式的版本）只改这一个文件。
"""

from llm.client import (
    CallRecord,
    LLMError,
    TraceEntry,
    Usage,
    chat_json,
    records,
    reset_usage,
    set_backoff_base,
    set_client,
    set_trace,
    usage,
)

__all__ = [
    "CallRecord",
    "LLMError",
    "TraceEntry",
    "Usage",
    "chat_json",
    "records",
    "reset_usage",
    "set_backoff_base",
    "set_client",
    "set_trace",
    "usage",
]
