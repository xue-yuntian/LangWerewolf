"""client 的离线测试：不联网、不需要 API key。

用 httpx.MockTransport 造假响应，所以在 CI 里也能验重试、剥壳、记账。
这也是为什么 client 的 httpx.Client 必须可注入——不可注入就只能靠真调用来测，
那既慢又花钱，还进不了 CI。
"""

import httpx
import pytest

from llm import client
from llm.client import LLMError, chat_json, records, reset_usage, usage

SYSTEM = "你是狼人杀玩家。只输出 JSON，不要加代码块。"
USER = "请原样返回一个 JSON 对象。"


@pytest.fixture(autouse=True)
def _clean_state():
    """每个测试前后都重置：客户端、退避、trace、记账。"""
    client.set_client(None)
    client.set_backoff_base(0.0)
    client.set_trace(None)
    reset_usage()
    yield
    client.set_client(None)
    client.set_backoff_base(0.5)
    client.set_trace(None)
    reset_usage()


def _install(handler) -> None:
    transport = httpx.MockTransport(handler)
    client.set_client(httpx.Client(base_url="https://fake.local", transport=transport))


def _ok(content: str, tokens_in: int = 100, tokens_out: int = 20) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        },
    )


def test_returns_dict() -> None:
    _install(lambda request: _ok('{"ok": true}'))

    assert chat_json(SYSTEM, USER, model="m") == {"ok": True}


def test_strips_markdown_fence() -> None:
    """模型经常会加 ```json 围栏。"""
    _install(lambda request: _ok('```json\n{"a": 1}\n```'))

    assert chat_json(SYSTEM, USER, model="m") == {"a": 1}


def test_strips_surrounding_chatter() -> None:
    """模型也会在前面加一句废话。"""
    _install(lambda request: _ok('好的，这是结果：{"a": 2} 希望有帮助'))

    assert chat_json(SYSTEM, USER, model="m") == {"a": 2}


def test_400_is_not_retried() -> None:
    """4xx 是"你写错了"，重试只会重复犯错还照样计费。"""
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(401, text="invalid api key")

    _install(handler)

    with pytest.raises(LLMError):
        chat_json(SYSTEM, USER, model="m", retries=3)

    assert counter["n"] == 1


def test_5xx_is_retried() -> None:
    codes = [500, 503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        code = codes.pop(0)
        if code == 200:
            return _ok('{"ok": 1}')
        return httpx.Response(code, text="upstream boom")

    _install(handler)

    assert chat_json(SYSTEM, USER, model="m", retries=3) == {"ok": 1}
    assert codes == []


def test_bad_json_is_retried() -> None:
    bodies = ["这不是 JSON", '{"fine": true}']
    _install(lambda request: _ok(bodies.pop(0)))

    assert chat_json(SYSTEM, USER, model="m", retries=1) == {"fine": True}


def test_gives_up_after_retries() -> None:
    _install(lambda request: httpx.Response(503, text="down"))

    with pytest.raises(LLMError, match="重试"):
        chat_json(SYSTEM, USER, model="m", retries=1)


def test_usage_is_accounted() -> None:
    _install(lambda request: _ok('{"a": 1}', tokens_in=1000, tokens_out=100))

    chat_json(SYSTEM, USER, model="m")
    chat_json(SYSTEM, USER, model="m")

    totals = usage()
    assert totals.calls == 2
    assert totals.tokens_in == 2000
    assert totals.tokens_out == 200
    assert totals.cost_usd > 0


def test_records_are_kept() -> None:
    _install(lambda request: _ok('{"a": 1}'))

    chat_json(SYSTEM, USER, model="deepseek-flash")

    logged = records()
    assert len(logged) == 1
    assert logged[0].model == "deepseek-flash"
    assert logged[0].ok is True


def test_trace_sees_prompt_and_raw_output() -> None:
    seen: list[client.TraceEntry] = []
    client.set_trace(seen.append)
    _install(lambda request: _ok('好的：{"a": 1}'))

    chat_json(SYSTEM, USER, model="m")

    assert len(seen) == 1
    assert seen[0].system == SYSTEM
    assert seen[0].user == USER
    # content 是剥壳之前的原文，不是解析后的 dict
    assert seen[0].content == '好的：{"a": 1}'
    assert seen[0].ok is True
    assert seen[0].reasoning == ""


def test_trace_reads_reasoning_content() -> None:
    """思维链是看“它怎么想的”的唯一入口，不能只留答案。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"a": 1}', "reasoning_content": "先看票型……"}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    seen: list[client.TraceEntry] = []
    client.set_trace(seen.append)
    _install(handler)

    chat_json(SYSTEM, USER, model="m")

    assert seen[0].reasoning == "先看票型……"
    assert seen[0].content == '{"a": 1}'
    assert seen[0].tokens_out == 5


def test_trace_marks_unparsable_output() -> None:
    """解析失败时也必须把模型原话留下来，否则无从排直。"""
    seen: list[client.TraceEntry] = []
    client.set_trace(seen.append)
    _install(lambda request: _ok("这不是 JSON"))

    with pytest.raises(LLMError):
        chat_json(SYSTEM, USER, model="m", retries=0)

    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].content == "这不是 JSON"
    assert seen[0].error


def test_trace_marks_http_error() -> None:
    seen: list[client.TraceEntry] = []
    client.set_trace(seen.append)
    _install(lambda request: httpx.Response(401, text="bad key"))

    with pytest.raises(LLMError):
        chat_json(SYSTEM, USER, model="m")

    assert seen[0].ok is False
    assert "401" in (seen[0].error or "")
    assert seen[0].content == ""


def test_trace_disabled_by_default() -> None:
    """没注册钩子时，重试多少次都不会因为 trace 多做事。"""
    bodies = ["坏", '{"fine": true}']
    _install(lambda request: _ok(bodies.pop(0)))

    assert chat_json(SYSTEM, USER, model="m", retries=1) == {"fine": True}
    assert len(records()) == 2
