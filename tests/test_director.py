"""人设导演的离线测试。

打桩方式和 test_agent 一致：httpx.MockTransport 造假响应，不联网。
重点验四件事：
  1. 场景进了 prompt，JSON 契约和用户点名的字段都在
  2. 正常返回时按顺序坐 1..N
  3. 数量不够用预制补，坏条目单条丢弃而不是整桌作废
  4. 调用失败返回空 dict，让调用方降级
"""

import json

import httpx
import pytest

from agents.director import generate_personas
from agents.personality import Persona, Personality
from llm import client


@pytest.fixture(autouse=True)
def _clean_state():
    """每个测试前后都重置：客户端、退避、记账。"""
    client.set_client(None)
    client.set_backoff_base(0.0)
    client.reset_usage()
    yield
    client.set_client(None)
    client.set_backoff_base(0.5)
    client.reset_usage()


def _install(handler) -> None:
    transport = httpx.MockTransport(handler)
    client.set_client(httpx.Client(base_url="https://fake.local", transport=transport))


def _ok(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _persona(name: str) -> dict[str, str]:
    return {
        "name": name,
        "background": f"{name}的处境",
        "traits": "急躁",
        "resilience": "被围攻时更凶",
        "style": "短句，爱反问",
        "logic": "中等",
        "quirk": "算了",
    }


def test_generates_one_persona_per_seat() -> None:
    payload = json.dumps({"personas": [_persona(f"P{i}") for i in range(1, 4)]})
    _install(lambda request: _ok(payload))

    seated = generate_personas("上海滩", 3, model="m")

    assert sorted(seated) == [1, 2, 3]
    assert all(isinstance(persona, Persona) for persona in seated.values())
    assert seated[1].name == "P1"


def test_prompt_carries_scene_and_json_contract() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"][1]["content"])
        return _ok(json.dumps({"personas": [_persona("张三")]}))

    _install(handler)
    generate_personas("1930 年代上海滩", 1, model="m")

    assert "1930 年代上海滩" in prompts[0]
    # 缺这两个字，部分供应商的 json_object 模式直接 400
    assert "JSON" in prompts[0]
    # 用户点名的字段必须出现在契约里，少一个模型就不写
    for field in ("抗压能力", "措辞习惯", "推理能力", "名字"):
        assert field in prompts[0]


def test_short_list_is_topped_up_with_presets() -> None:
    payload = json.dumps({"personas": [_persona("张三")]})
    _install(lambda request: _ok(payload))

    seated = generate_personas("上海滩", 3, model="m")

    assert isinstance(seated[1], Persona)
    assert isinstance(seated[2], Personality)
    assert isinstance(seated[3], Personality)


def test_broken_entry_is_skipped_not_fatal() -> None:
    """第一条缺 traits：丢它一条，李四照坐 1 号，2 号用预制补。"""
    payload = json.dumps({"personas": [{"name": "张三"}, _persona("李四")]})
    _install(lambda request: _ok(payload))

    seated = generate_personas("上海滩", 2, model="m")

    assert isinstance(seated[1], Persona)
    assert seated[1].name == "李四"
    assert isinstance(seated[2], Personality)


def test_failure_returns_empty_for_fallback() -> None:
    _install(lambda request: httpx.Response(401, text="bad key"))

    assert generate_personas("上海滩", 9, model="m") == {}


def test_empty_scene_makes_no_request() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"personas": []}')

    _install(handler)

    assert generate_personas("   ", 9, model="m") == {}
    assert calls["n"] == 0


def test_non_list_payload_returns_empty() -> None:
    _install(lambda request: _ok('{"personas": "九个"}'))

    assert generate_personas("上海滩", 9, model="m") == {}


def test_nobody_left_after_filtering_returns_empty() -> None:
    """全是不合法的条目：不是因为"补到 N 个"就算成功，而是真的没人。"""
    _install(lambda request: _ok('{"personas": [{}, "x", {"name": ""}]}'))

    assert generate_personas("上海滩", 3, model="m") == {}


def test_persona_text_lists_every_trait() -> None:
    persona = Persona.model_validate(_persona("沈知微"))
    text = persona.to_text()

    assert "沈知微" in text
    assert "性格：" in text
    assert "抗压能力：" in text
    assert "说话习惯：" in text
    assert "推理能力：" in text


def test_long_field_is_clipped() -> None:
    """模型爱写小作文。一个人设四百字，九个人就是三千六。"""
    persona = Persona.model_validate({**_persona("张三"), "traits": "很" * 500})

    assert len(persona.traits) <= 60
