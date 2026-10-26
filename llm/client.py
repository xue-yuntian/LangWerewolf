"""LLM 客户端 —— 全项目唯一联网的地方。

────────────────────────────────────────────────────────────────
职责
────────────────────────────────────────────────────────────────
两段文本（system + user）→ 一次 HTTP 调用 → 一个 dict。

它不知道狼人杀是什么，也不知道人设、投票、验人。正因为它通用，
同一个函数会被两类东西共用：

    开局前的人设导演   "生成 N 个角色"      -> {"personas": [...]}
    开局后的 AI 玩家   "3 号的视角与要求"    -> {"speech": "...", "vote": 5}

────────────────────────────────────────────────────────────────
不负责
────────────────────────────────────────────────────────────────
  · 拼 prompt（agents/）
  · 判断答案对不对（agents/validator.py）——"投给已出局的人"不归它管
  · 并发调度（engine/batch.py）
  · 流式输出（D16 的 SSE 才需要，现在做没人消费）

────────────────────────────────────────────────────────────────
三个坑，写在最前面
────────────────────────────────────────────────────────────────
1. `response_format={"type": "json_object"}` 要求 prompt 里出现 "JSON" 字样，
   否则部分供应商直接返回 400。所以输出契约里那句"只输出一个 JSON 对象"
   不是文风问题，是功能必需。
2. 模型返回的 JSON 经常被 ```json 围栏或前后废话包住，必须先剥壳。
3. 4xx 绝不重试。参数错、key 错、模型名错——重试只会把同一个错误再犯一遍，
   还照样计费。
"""

import json
import os
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
CHAT_ENDPOINT = "/chat/completions"

# 价格（美元 / 百万 token）。改价只改这两行。
# 默认按 DeepSeek deepseek-flash 峰时价；缓存命中实际更便宜，这里按保守值算。
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 1.20

# 可重试的状态码：等一等可能就好。
# 400/401/403/404 不在里面——那是"你写错了"，不是"服务端忙"。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

# 退避基数（秒）。测试里会把它设成 0，避免真睡。
_BACKOFF_BASE = 0.5


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """调用失败：不可重试的错误，或重试次数耗尽。"""


class _Retryable(Exception):
    """内部用：这类错误值得再试一次。不对外暴露。"""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# 记账
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class CallRecord:
    """单次调用的轨迹。D28 接 Langfuse 时直接读这个列表。"""

    call_id: str
    model: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class TraceEntry:
    """一次调用的完整快照。只看不存：由 set_trace 注册的回调当场消费。

    content 是模型原始输出（剥壳之前），reasoning 是思维链（模型不吐就是空串）。
    把两者分开保留，是因为“答案对但思考错”和“思考对但格式写坏了”
    是两类完全不同的问题。
    """

    model: str
    system: str
    user: str
    content: str
    reasoning: str
    ok: bool
    error: str | None
    latency_ms: float
    tokens_in: int
    tokens_out: int


_lock = threading.Lock()
_usage = Usage()
_records: list[CallRecord] = []
_trace_hook: Callable[[TraceEntry], None] | None = None


def usage() -> Usage:
    """当前累计用量。返回副本，调用方改它不影响内部状态。"""
    with _lock:
        return Usage(
            calls=_usage.calls,
            tokens_in=_usage.tokens_in,
            tokens_out=_usage.tokens_out,
            cost_usd=_usage.cost_usd,
        )


def records() -> list[CallRecord]:
    with _lock:
        return list(_records)


def reset_usage() -> None:
    global _usage
    with _lock:
        _usage = Usage()
        _records.clear()


def set_trace(hook: Callable[[TraceEntry], None] | None) -> None:
    """注册/清除调用快照钩子。

    注册后每次 HTTP 尝试（含重试和失败）都会回调一次，CLI 的 --verbose
    靠它把提示词、思考过程、原始输出摊开给人看。默认不注册，零开销，
    也不会把一局的 prompt 全攒在内存里。
    """
    global _trace_hook
    _trace_hook = hook


def _trace(entry: TraceEntry) -> None:
    if _trace_hook is not None:
        _trace_hook(entry)


def set_backoff_base(seconds: float) -> None:
    """测试用：把退避基数调成 0，避免单元测试真的 sleep。"""
    global _BACKOFF_BASE
    _BACKOFF_BASE = seconds


def _record(
    *,
    model: str,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    ok: bool,
    error: str | None = None,
) -> None:
    cost = tokens_in / 1_000_000 * PRICE_IN_PER_M + tokens_out / 1_000_000 * PRICE_OUT_PER_M
    with _lock:
        _usage.calls += 1
        _usage.tokens_in += tokens_in
        _usage.tokens_out += tokens_out
        _usage.cost_usd += cost
        _records.append(
            CallRecord(
                call_id=uuid.uuid4().hex[:8],
                model=model,
                latency_ms=round(latency_ms, 1),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                ok=ok,
                error=error,
            )
        )


# ---------------------------------------------------------------------------
# 配置与 HTTP 客户端（可注入，便于离线测试）
# ---------------------------------------------------------------------------

_client: httpx.Client | None = None
_env_loaded = False


def _load_dotenv() -> None:
    """把项目根目录的 .env 读进 os.environ。

    自己写十行而不是引 python-dotenv：只支持 KEY=VALUE 和 # 注释，
    够用，且不给 CI 增加依赖。
    已存在的环境变量优先（setdefault），所以在 shell 里 export 的会赢。
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True

    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def set_client(client: httpx.Client | None) -> None:
    """替换底层 httpx 客户端。测试注入 MockTransport，或手工指向别的网关。"""
    global _client
    _client = client


def _get_client() -> httpx.Client:
    global _client
    if _client is not None:
        return _client

    _load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise LLMError("缺少 LLM_API_KEY：复制 .env.example 为 .env 并填入 key")

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    _client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(30.0, connect=5.0),
    )
    return _client


# ---------------------------------------------------------------------------
# 剥壳
# ---------------------------------------------------------------------------


def _short(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _extract_object(text: str) -> str:
    """从模型输出里抠出 JSON 对象文本。

    顺序：去围栏 -> 截取首个 { 到最后一个 }。
    不做正则修尾逗号——那类修补极易误伤字符串内容，宁可让它重试一次。
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise _Retryable(f"返回内容里找不到 JSON 对象：{_short(cleaned)}")
    return cleaned[start : end + 1]


def _parse_object(text: str) -> dict[str, object]:
    body = _extract_object(text)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise _Retryable(f"JSON 解析失败（{exc.msg}）：{_short(body)}") from exc

    if not isinstance(parsed, dict):
        raise _Retryable(f"顶层不是对象，收到 {type(parsed).__name__}")
    return {str(k): v for k, v in parsed.items()}


# ---------------------------------------------------------------------------
# 单次 HTTP
# ---------------------------------------------------------------------------


@dataclass
class _Raw:
    content: str
    reasoning: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


def _snapshot(
    model: str,
    system: str,
    user: str,
    raw: _Raw | None,
    error: str | None,
) -> TraceEntry:
    """拼一次调用的快照。raw=None 表示连 HTTP 都没成功。"""
    if raw is None:
        return TraceEntry(
            model=model,
            system=system,
            user=user,
            content="",
            reasoning="",
            ok=False,
            error=error,
            latency_ms=0.0,
            tokens_in=0,
            tokens_out=0,
        )
    return TraceEntry(
        model=model,
        system=system,
        user=user,
        content=raw.content,
        reasoning=raw.reasoning,
        ok=error is None,
        error=error,
        latency_ms=raw.latency_ms,
        tokens_in=raw.tokens_in,
        tokens_out=raw.tokens_out,
    )


def _parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _post(payload: dict[str, object], timeout: float) -> _Raw:
    client = _get_client()
    started = time.perf_counter()

    try:
        response = client.post(CHAT_ENDPOINT, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise _Retryable(f"请求超时：{exc}") from exc
    except httpx.HTTPError as exc:
        raise _Retryable(f"网络错误：{exc}") from exc

    latency_ms = (time.perf_counter() - started) * 1000

    if response.status_code in RETRYABLE_STATUS:
        raise _Retryable(
            f"HTTP {response.status_code}：{_short(response.text)}",
            retry_after=_parse_retry_after(response),
        )
    if response.status_code >= 400:
        # 4xx：参数/key/模型名错了，重试没意义
        raise LLMError(f"HTTP {response.status_code}：{_short(response.text)}")

    try:
        data = response.json()
    except ValueError as exc:
        raise _Retryable(f"响应不是 JSON：{_short(response.text)}") from exc

    content, reasoning, tokens_in, tokens_out = _read_completion(data)
    return _Raw(
        content=content,
        reasoning=reasoning,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )


def _read_completion(data: object) -> tuple[str, str, int, int]:
    """从 OpenAI 兼容响应里取 (content, reasoning, tokens_in, tokens_out)。"""
    if not isinstance(data, dict):
        raise _Retryable("响应顶层不是对象")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _Retryable(f"响应缺少 choices：{_short(str(data))}")

    first = choices[0]
    if not isinstance(first, dict):
        raise _Retryable("choices[0] 不是对象")

    message = first.get("message")
    if not isinstance(message, dict):
        raise _Retryable("choices[0].message 不是对象")

    content = message.get("content")
    if not isinstance(content, str):
        raise _Retryable("message.content 不是字符串")

    # 思维链。DeepSeek 放在 reasoning_content，一些网关用 reasoning；
    # 都没有就返回空串——不是所有模型都会吐思考过程
    reasoning = ""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            reasoning = value
            break

    tokens_in = 0
    tokens_out = 0
    usage_block = data.get("usage")
    if isinstance(usage_block, dict):
        raw_in = usage_block.get("prompt_tokens")
        raw_out = usage_block.get("completion_tokens")
        if isinstance(raw_in, int):
            tokens_in = raw_in
        if isinstance(raw_out, int):
            tokens_out = raw_out

    return content, reasoning, tokens_in, tokens_out


# ---------------------------------------------------------------------------
# 对外唯一入口
# ---------------------------------------------------------------------------


def _thinking_payload(thinking: bool) -> dict[str, object] | None:
    """把 thinking 开关翻译成本后端认的字段。认不出来就返回 None。

    只有 DeepSeek 官方 API 用 {"type": "enabled"|"disabled"}；换到 Qwen/GLM
    要叫 enable_thinking，或者要塞进 extra_body。判断错了不会静默降级——
    服务端回 400，调用方拿不到结果，一整轮投票全变成弃票。所以宁可不发：
    宁可慢一点，也不能因为多一个字段把票验废。
    """
    # 必须先读 .env：这个判断发生在构造 payload 阶段，比 _get_client 早，
    # 不主动加载的话进程内第一次调用读不到 LLM_BASE_URL，会静默跳过开关
    _load_dotenv()
    if "deepseek" not in os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).lower():
        return None
    return {"type": "enabled" if thinking else "disabled"}


def chat_json(
    system: str,
    user: str,
    *,
    model: str,
    temperature: float = 0.7,
    timeout: float = 30.0,
    retries: int = 2,
    thinking: bool | None = None,
) -> dict[str, object]:
    """调一次模型，返回 JSON 对象。

    参数:
        system/user: 两段提示词。user 段里必须出现 "JSON" 字样（见模块头坑 1）。
        model: 必须显式指定，不给默认值——这是 D25 模型路由能成立的前提。
        retries: 额外重试次数。总尝试次数 = retries + 1。
        thinking: 要不要先写思维链。None 表示不提这个字段，用服务端默认。
            DeepSeek 的 deepseek-flash 默认是开的，而投票、记笔记这类
            “从几个座位号里挑一个”的决策，思维链换不来更好的结果，
            只是把首字延迟拉长一倍、把输出 token 撑大几倍。

    抛出:
        LLMError: 4xx、或重试耗尽。
    """
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if thinking is not None:
        toggle = _thinking_payload(thinking)
        if toggle is not None:
            payload["thinking"] = toggle

    last_error = "未知错误"

    for attempt in range(retries + 1):
        try:
            raw = _post(payload, timeout)
        except _Retryable as exc:
            last_error = str(exc)
            _record(
                model=model,
                latency_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                ok=False,
                error=last_error,
            )
            _trace(_snapshot(model, system, user, None, last_error))
            if attempt >= retries:
                break
            _backoff(attempt, exc.retry_after)
            continue
        except LLMError as exc:
            # 4xx：参数/key/模型名错了，不重试，但也得让 trace 看到
            _trace(_snapshot(model, system, user, None, str(exc)))
            raise

        # 200 就已经产生费用，所以先记账，再判断内容质量
        _record(
            model=model,
            latency_ms=raw.latency_ms,
            tokens_in=raw.tokens_in,
            tokens_out=raw.tokens_out,
            ok=True,
        )

        try:
            parsed = _parse_object(raw.content)
        except _Retryable as exc:
            last_error = str(exc)
            _trace(_snapshot(model, system, user, raw, last_error))
            if attempt >= retries:
                break
            _backoff(attempt, None)
            continue

        _trace(_snapshot(model, system, user, raw, None))
        return parsed

    raise LLMError(f"重试 {retries} 次后仍失败：{last_error}")


def _backoff(attempt: int, retry_after: float | None) -> None:
    """指数退避。服务端给了 Retry-After 就听它的，别自己瞎猜。"""
    if retry_after is not None:
        delay = retry_after
    else:
        delay = _BACKOFF_BASE * (2**attempt) + random.uniform(0, 0.1)
    if delay > 0:
        time.sleep(delay)


# ---------------------------------------------------------------------------
# 手动冒烟：python -m llm.client
# ---------------------------------------------------------------------------


def _smoke() -> int:
    model = os.getenv("LLM_MODEL_FAST", DEFAULT_MODEL)
    try:
        result = chat_json(
            "你是狼人杀玩家。只输出 JSON，不要加代码块。",
            '请原样返回 {"ok": true, "msg": "连通"}，只输出 JSON。',
            model=model,
            temperature=0.0,
            retries=1,
        )
    except LLMError as exc:
        print(f"失败：{exc}")
        return 1

    print("成功：", result)
    totals = usage()
    print(
        f"调用 {totals.calls} 次 | token {totals.tokens_in} in / "
        f"{totals.tokens_out} out | 约 ${totals.cost_usd:.6f}"
    )
    for record in records():
        print(f"  {record.call_id} {record.model} {record.latency_ms}ms ok={record.ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
