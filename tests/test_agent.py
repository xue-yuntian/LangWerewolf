"""LLMAgentPolicy.choose_guard 的离线测试。

和 test_client 一样用 httpx.MockTransport 造假响应：不联网、不需要 key。
重点验三件事：
  1. 合法目标原样采纳；非法目标不会漏进引擎
  2. 模型给畸形答案（越界 / 连守 / 报错）时退化成空守，而不是抛异常
  3. 守卫已经死了就根本不发请求（省钱，也避免无意义的调用）
"""

import json

import httpx
import pytest

from agents.agent import LLMAgentPolicy
from engine.events import EventKind, EventLog
from engine.game import play_game
from engine.roles import Roles
from engine.state import DeathCause, GameState
from llm import client
from tests.support import FirstAlivePolicy, make_board, make_state, seed


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


def _guard_state() -> GameState:
    """3 号是守卫，第 1 夜。座位号故意不连续，防止代码把 pid 当列表下标。"""
    state = make_state(
        {
            1: Roles.WOLF,
            2: Roles.VILLAGER,
            3: Roles.GUARD,
            4: Roles.SEER,
            5: Roles.WITCH,
        }
    )
    state.night = 1
    return state


def _prompt_of(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return body["messages"][1]["content"]


def test_accepts_legal_target() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 5}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_guard(_guard_state()) == 5
    # 大提示词里必须有座位号，否则守卫守不了自己、投票也避不开自投
    assert "你是 3 号玩家" in prompts[0]
    # 任务段也要自带一遍：座位号 + 角色写在同一句里模型才不容易看漏
    assert "你是 3 号守卫" in prompts[0]
    # 任务段必须带上可选范围和 JSON 契约，否则模型无从下手
    assert "可选目标：1号、2号、3号、4号、5号" in prompts[0]
    assert '"target"' in prompts[0]
    # 视角段来自 view_for，身份是中文名 + 枚举值
    assert "你的身份是：守卫（guard）" in prompts[0]


def test_last_guarded_is_excluded_and_corrected() -> None:
    state = _guard_state()
    state.last_guarded = 5
    prompts: list[str] = []
    answers = ['{"target": 5}', '{"target": 2}']

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_guard(state) == 2
    assert answers == []  # 第一遍给了连守目标，第二遍才改对
    assert "可选目标：1号、2号、3号、4号" in prompts[0]
    assert "上一夜你守过 5 号" in prompts[0]
    assert "不合法" in prompts[1]


def test_null_means_empty_guard() -> None:
    _install(lambda request: _ok('{"target": null}'))

    assert LLMAgentPolicy(model="m").choose_guard(_guard_state()) is None


def test_numeric_string_is_accepted() -> None:
    """模型经常把整数写成字符串，不值得为这个浪费一次重试。"""
    _install(lambda request: _ok('{"target": "4"}'))

    assert LLMAgentPolicy(model="m").choose_guard(_guard_state()) == 4


def test_illegal_target_is_retried_once_then_gives_up() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 9}')  # 9 号根本不在场

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_guard(_guard_state()) is None
    assert calls["n"] == 2


def test_llm_error_falls_back_to_empty_guard() -> None:
    """401 不重试，也不应该冒泡出去把整局打断。"""
    _install(lambda request: httpx.Response(401, text="invalid api key"))

    assert LLMAgentPolicy(model="m").choose_guard(_guard_state()) is None


def test_no_request_when_guard_is_dead() -> None:
    state = _guard_state()
    state.kill(3, DeathCause.WOLF_KILL, "第1夜")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 1}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_guard(state) is None
    assert calls["n"] == 0


def test_nine_player_prompt_says_no_guard() -> None:
    """9p 没有守卫，必须在 prompt 里明说。

    不说的话模型会照着术语表里的“空守/连守/同守同救”悍跳一个
    根本不存在的守卫——它没有别的办法知道这局配了什么。
    """
    from engine.board import BOARDS_DIR, load_board
    from engine.state import create_state

    state = create_state(load_board(BOARDS_DIR / "9p.json"), seed())
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 1}')

    _install(handler)
    LLMAgentPolicy(model="m").vote(state, 2)

    assert "本局身份配置：" in prompts[0]
    assert "本局没有：守卫" in prompts[0]


def test_prompt_states_present_roles() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 5}')

    _install(handler)
    LLMAgentPolicy(model="m").choose_guard(_guard_state())

    # 板子上的身份要逐个列出来，带人数
    assert "- 守卫 1 人" in prompts[0]
    # _guard_state 里没有猎人
    assert "本局没有：猎人" in prompts[0]


def _wolf_state() -> GameState:
    """2/4/6 号是狼，7 人局。座位号与狼队不连续，防止把 pid 当列表下标。"""
    state = make_state(
        {
            1: Roles.VILLAGER,
            2: Roles.WOLF,
            3: Roles.SEER,
            4: Roles.WOLF,
            5: Roles.WITCH,
            6: Roles.WOLF,
            7: Roles.GUARD,
        }
    )
    state.night = 1
    return state


def _wolf_answers(*turns: tuple[int | None, str]) -> list[str]:
    """狼队会议的响应。发言顺序是 4 号、6 号、2 号（主狼最后）。"""
    return [json.dumps({"target": t, "plan": p}, ensure_ascii=False) for t, p in turns]


def test_wolf_kill_asks_every_alive_wolf() -> None:
    prompts: list[str] = []
    answers = _wolf_answers((5, "跟"), (3, "踩"), (5, "定 5"))

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state()) == 5
    assert answers == []
    # 三只狼三次调用：刀口和打法是同一个 JSON 里出来的
    assert len(prompts) == 3


def test_later_wolves_hear_earlier_intentions() -> None:
    prompts: list[str] = []
    answers = _wolf_answers((5, "跟"), (3, "踩"), (5, "定 5"))

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state())

    assert "在你之前表过态" not in prompts[0]
    assert "4号：刀 5 号。跟" in prompts[1]
    assert "6号：刀 3 号。踩" in prompts[2]


def test_tie_is_settled_by_lowest_seat_wolf() -> None:
    """三狼各说一个，平票时由主狼（2 号）拍板。"""
    answers = _wolf_answers((6, "a"), (7, "b"), (6, "c"))
    _install(lambda request: _ok(answers.pop(0)))

    assert LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state()) == 6


def test_failed_wolf_only_loses_its_vote() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, text="invalid api key")
        return _ok('{"target": 3, "plan": "踩 3"}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state()) == 3
    # 4 号失败只少它一票，剩下两只照样开完会
    assert calls["n"] == 3


def test_all_wolves_failing_means_empty_kill() -> None:
    _install(lambda request: httpx.Response(401, text="invalid api key"))

    assert LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state()) is None


def test_wolf_can_kill_a_teammate() -> None:
    """自刀是合法战术，不该被当成非法目标过滤掉。"""
    answers = _wolf_answers((4, "a"), (4, "b"), (4, "c"))
    _install(lambda request: _ok(answers.pop(0)))

    assert LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state()) == 4


def test_chief_speaks_last_and_sets_the_line() -> None:
    """主狼最后发言，一次调用同时出自己那一票和全队分工。"""
    prompts: list[str] = []
    answers = _wolf_answers(
        (5, "明天我跟 5 号的票"),
        (3, "我踩 3 号，逼他报验人"),
        (5, "4 号悍跳，6 号倒钩，2 号带节奏"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)
    policy = LLMAgentPolicy(model="m")

    assert policy.choose_wolf_kill(_wolf_state()) == 5

    # 4 号先说，6 号看到 4 号，主狼 2 号最后看完全队
    assert "你是主狼" not in prompts[0]
    assert "4号：刀 5 号。明天我跟 5 号的票" in prompts[1]
    assert "6号：刀 3 号。我踩 3 号，逼他报验人" in prompts[2]
    assert "你是主狼" in prompts[2]

    # 自己的打算写自己的笔记，主狼那份归所有人
    final = "第1夜 分工：刀5号：4 号悍跳，6 号倒钩，2 号带节奏"
    assert policy.notebook.get(4) == ["第1夜：明天我跟 5 号的票", final]
    assert policy.notebook.get(6) == ["第1夜：我踩 3 号，逼他报验人", final]
    assert policy.notebook.get(2) == [final]


def test_abstention_shows_up_in_the_transcript() -> None:
    """谁没投，后面的狼要看得见——不跟票本身就是分歧。"""
    prompts: list[str] = []
    answers = _wolf_answers((5, "跟"), (None, "我想空刀"), (5, "定 5"))

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_wolf_kill(_wolf_state()) == 5
    assert "6号：没定下刀口。我想空刀" in prompts[2]


def test_no_notes_when_every_wolf_passes() -> None:
    """全员空刀等于没定下刀口，也就没有可记的会议。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": null, "plan": "今晚空刀"}')

    _install(handler)
    policy = LLMAgentPolicy(model="m")

    assert policy.choose_wolf_kill(_wolf_state()) is None
    assert calls["n"] == 3
    assert policy.notebook.total() == 0


def test_god_hook_shows_the_whole_conference() -> None:
    """挂上 wolf_talk 后整场会议原样吐出来：谁投谁、谁没表态、主狼是谁。"""
    reported: list[str] = []
    answers = _wolf_answers(
        (5, "明天我跟 5 号的票"),
        (None, "我看不清，空刀"),
        (5, "4 号悍跳"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(answers.pop(0))

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    policy.wolf_talk = reported.append

    assert policy.choose_wolf_kill(_wolf_state()) == 5

    assert len(reported) == 1
    text = reported[0]
    assert text.startswith("[第1夜] 狼队会议")
    assert "4号 → 刀 5 号：明天我跟 5 号的票" in text
    assert "6号 → 空刀：我看不清，空刀" in text
    assert "2号（主狼，拍板） → 刀 5 号：4 号悍跳" in text
    assert text.endswith("定刀：5号")


def test_god_sees_a_conference_that_ends_in_no_kill() -> None:
    """全员空刀：笔记不写（没刀口可记），但上帝要看到他们确实商量过。"""
    reported: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok('{"target": null, "plan": "今晚空刀"}')

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    policy.wolf_talk = reported.append

    assert policy.choose_wolf_kill(_wolf_state()) is None
    assert len(reported) == 1
    assert reported[0].endswith("定刀：空刀")
    assert policy.notebook.total() == 0


def test_policy_is_silent_by_default() -> None:
    """没加 --god 就不该多出这些输出，默认对局和以前一模一样。"""
    assert LLMAgentPolicy(model="m").wolf_talk is None


def test_seed_gives_wolves_distinct_personalities() -> None:
    """狼队信息完全一样，人设是唯一的信息差。"""
    state = _wolf_state()
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 5, "plan": "跟"}')

    _install(handler)

    policy = LLMAgentPolicy(model="m", seed=89)
    policy.choose_wolf_kill(state)

    wolves = state.wolves_alive()
    assert len({policy.personalities[pid] for pid in wolves}) == len(wolves)
    assert all("你的身份特征如下" in prompt for prompt in prompts)


def _witch_state() -> GameState:
    """3 号女巫，第 2 夜（非首夜，不能自救）。"""
    state = make_state(
        {
            1: Roles.WOLF,
            2: Roles.VILLAGER,
            3: Roles.WITCH,
            4: Roles.SEER,
            5: Roles.GUARD,
        }
    )
    state.night = 2
    return state


def _seer_state() -> GameState:
    """3 号预言家，第 2 夜。"""
    state = make_state(
        {
            1: Roles.WOLF,
            2: Roles.VILLAGER,
            3: Roles.SEER,
            4: Roles.WITCH,
            5: Roles.GUARD,
        }
    )
    state.night = 2
    return state


def test_witch_save_sees_the_killed_seat() -> None:
    """killed 是女巫唯一的信息来源——当晚的 WOLF_KILL 还没 emit。"""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"save": true}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_witch_save(_witch_state(), 5) is True
    assert "今晚 5 号被狼刀" in prompts[0]
    assert "你的身份是：女巫（witch）" in prompts[0]


def test_witch_cannot_save_herself_after_first_night() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"save": true}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_witch_save(_witch_state(), 3) is False
    assert calls["n"] == 0


def test_witch_can_save_herself_on_first_night() -> None:
    state = _witch_state()
    state.night = 1
    _install(lambda request: _ok('{"save": true}'))

    assert LLMAgentPolicy(model="m").choose_witch_save(state, 3) is True


def test_witch_save_defaults_to_false_on_llm_error() -> None:
    _install(lambda request: httpx.Response(401, text="invalid api key"))

    assert LLMAgentPolicy(model="m").choose_witch_save(_witch_state(), 5) is False


def test_witch_poison_excludes_self() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 2}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_witch_poison(_witch_state()) == 2
    assert "可选目标：1号、2号、4号、5号" in prompts[0]


def test_witch_cannot_poison_herself() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 3}')  # 3 就是女巫自己

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_witch_poison(_witch_state()) is None
    assert calls["n"] == 2


def test_witch_skips_poison_after_saving() -> None:
    """一晚只能用一瓶药：救完就不该再问毒。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"save": true}')

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    state = _witch_state()

    assert policy.choose_witch_save(state, 5) is True
    assert policy.choose_witch_poison(state) is None
    assert calls["n"] == 1


def test_saved_slot_does_not_leak_across_games() -> None:
    """同一个 policy 实例跑第二局时，不能把上一局的解药算成今晚用的。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"save": true}' if calls["n"] == 1 else '{"target": 2}')

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    assert policy.choose_witch_save(_witch_state(), 5) is True

    # 新的一局，夜数同样是 2——旧实现会在这里误判成"今晚已经救过"
    assert policy.choose_witch_poison(_witch_state()) == 2


def test_seer_skips_checked_players() -> None:
    state = _seer_state()
    state.player(2).checked = True
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 4}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_seer_check(state) == 4
    assert "可选目标：1号、4号、5号" in prompts[0]


def test_seer_cannot_check_herself() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 3}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_seer_check(_seer_state()) is None
    assert calls["n"] == 2


def test_seer_returns_none_when_everyone_checked() -> None:
    state = _seer_state()
    for pid in (1, 2, 4, 5):
        state.player(pid).checked = True
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 1}')

    _install(handler)

    assert LLMAgentPolicy(model="m").choose_seer_check(state) is None
    assert calls["n"] == 0


def _vote_state() -> GameState:
    """5 人局：1 狼 / 2 民 / 3 民 / 4 预言家 / 5 守卫。第 1 天。"""
    state = make_state(
        {
            1: Roles.WOLF,
            2: Roles.VILLAGER,
            3: Roles.VILLAGER,
            4: Roles.SEER,
            5: Roles.GUARD,
        }
    )
    state.day = 1
    return state


def test_vote_returns_target() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 5}')

    _install(handler)

    assert LLMAgentPolicy(model="m").vote(_vote_state(), 3) == 5
    assert "可选目标：1号、2号、4号、5号" in prompts[0]
    assert "第1天" in prompts[0]


def test_vote_cannot_vote_self() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 3}')  # 3 号正在投票

    _install(handler)

    assert LLMAgentPolicy(model="m").vote(_vote_state(), 3) is None
    assert calls["n"] == 2


def test_vote_pk_round_limits_to_tied() -> None:
    """首轮 3/5 各 2 票，PK 轮只能投台上的人。"""
    state = _vote_state()
    prompts: list[str] = []
    answers = [
        '{"target": 3}',  # 1 号 -> 3
        '{"target": 3}',  # 2 号 -> 3
        '{"target": 5}',  # 3 号 -> 5
        '{"target": 5}',  # 4 号 -> 5
        '{"target": 1}',  # 5 号 -> 1
        '{"target": 5}',  # PK：1 号改投 5
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    for voter in (1, 2, 3, 4, 5):
        policy.vote(state, voter)

    assert policy.vote(state, 1, only=(3, 5)) == 5
    pk = prompts[5]
    assert "PK 重投" in pk
    assert "3号、5号 并列最高票" in pk
    assert "首轮票型：1号→3号、2号→3号、3号→5号、4号→5号、5号→1号" in pk
    assert "只能从 3号、5号 里选" in pk


def test_vote_pk_rejects_offstage_target() -> None:
    state = _vote_state()
    answers = [
        '{"target": 3}',
        '{"target": 3}',
        '{"target": 5}',
        '{"target": 5}',
        '{"target": 1}',
        '{"target": 2}',  # PK 第一遍投台下的人
        '{"target": 2}',  # PK 第二遍还是台下
    ]
    _install(lambda request: _ok(answers.pop(0)))

    policy = LLMAgentPolicy(model="m")
    for voter in (1, 2, 3, 4, 5):
        policy.vote(state, voter)

    # only 把候选限定在台上，模型报台下的 2 号问两次都拿不到合法答案
    assert policy.vote(state, 1, only=(3, 5)) is None


def test_vote_resets_when_day_changes() -> None:
    state = _vote_state()
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 5}')

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    policy.vote(state, 1)
    state.day = 2
    policy.vote(state, 1)

    assert "PK 重投" not in prompts[1]
    assert "第2天" in prompts[1]


def test_vote_does_not_leak_across_games() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 5}')

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    policy.vote(_vote_state(), 1)
    policy.vote(_vote_state(), 1)  # 新的一局，day 同样是 1

    assert "PK 重投" not in prompts[1]


def _speech_state() -> GameState:
    """5 人局，第 1 天投票前：state.day 还是 0，resolve_vote 里才会 +1。"""
    state = _vote_state()
    state.day = 0
    return state


def test_speak_returns_text() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"speech": "我怀疑 5 号，他昨天的票投得很奇怪。"}')

    _install(handler)

    text = LLMAgentPolicy(model="m").speak(_speech_state(), 3)
    assert text == "我怀疑 5 号，他昨天的票投得很奇怪。"
    assert "【发言】" in prompts[0]
    assert "第1天" in prompts[0]
    assert '"speech"' in prompts[0]


def test_speak_can_pass() -> None:
    _install(lambda request: _ok('{"speech": null}'))

    assert LLMAgentPolicy(model="m").speak(_speech_state(), 3) is None


def test_speech_prompt_warns_against_leaking_night_info() -> None:
    """不提醒的话，女巫会在发言里说出“几号被刀”，等于当众自曝。"""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"speech": "先听听别人的。"}')

    _install(handler)
    LLMAgentPolicy(model="m").speak(_speech_state(), 3)

    assert "信息纪律" in prompts[0]
    assert "说出刀口就是自曝" in prompts[0]


def test_wolf_speech_tells_it_to_disguise() -> None:
    """不提醒的话模型会老老实实认狼，发言环节反而让狼必输。"""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"speech": "我是好人。"}')

    _install(handler)
    LLMAgentPolicy(model="m").speak(_speech_state(), 1)  # 1 号是狼

    assert "必须把自己伪装成好人" in prompts[0]
    assert "不能供出队友" in prompts[0]


def test_witch_speech_offers_silver_water_without_forcing() -> None:
    """女巫要知道银水是什么、报不报的代价，但选哪条是她的自由。

    “银水可能是自刀骗药”和“报了自己会成刀口”都是真事实；
    一旦写成“不要投他”，就等于删掉了一条合法策略。
    """
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"speech": "2 号我保，投 1 号。"}')

    _install(handler)

    state = _guard_state()  # 5 号是女巫
    events = EventLog()
    # 真实的第 1 夜一定有 NIGHT_START：prompt 靠它才认定“这一轮从哪开始”
    events.emit(EventKind.NIGHT_START, "第1夜")
    events.emit(EventKind.WITCH_SAVE, "第1夜", target=2, visible_to="pid:5")
    policy = LLMAgentPolicy(model="m")
    policy.events = events

    policy.speak(state, 5)

    # 她得先看得见自己救过谁
    assert "女巫用解药救下 2 号" in prompts[0]
    # 再得知道这条信息是什么、用它的代价
    assert "银水" in prompts[0]
    assert "自刀骗药" in prompts[0]
    assert "你自己决定" in prompts[0]
    # 不能替她排除“怀疑银水”这条路
    assert "不要跟着投他" not in prompts[0]


def test_seer_speech_leaves_reporting_to_it() -> None:
    """预言家可以藏身份：报验人等于告诉狼该刀谁。"""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"speech": "4号预言家，首夜验3号是好人。"}')

    _install(handler)
    LLMAgentPolicy(model="m").speak(_guard_state(), 4)  # 4 号是预言家

    # 报与不报的代价都写出来
    assert "验人结果" in prompts[0]
    assert "你自己权衡" in prompts[0]
    # 但不规定必须报
    assert "等于白验" not in prompts[0]


def test_speeches_show_up_in_vote_prompt() -> None:
    """发言唯一的用途就是让后面的人看到。看不到等于没说。"""
    state = _speech_state()
    prompts: list[str] = []
    answers = [
        '{"speech": "1 号昨晚很可疑。"}',
        '{"speech": "我跟 1 号。"}',
        '{"target": 2}',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    assert policy.speak(state, 1) == "1 号昨晚很可疑。"
    assert policy.speak(state, 2) == "我跟 1 号。"

    # 后发言的人应该已经能看到前面的人说了什么
    assert "1 号：1 号昨晚很可疑。" in prompts[1]

    state.day = 1  # resolve_vote 里才会 +1，投票时已经是第 1 天
    policy.vote(state, 3)

    assert "以下是本轮已经发生过的发言" in prompts[2]
    assert "1 号：1 号昨晚很可疑。" in prompts[2]
    assert "2 号：我跟 1 号。" in prompts[2]


def test_speeches_reset_when_day_changes() -> None:
    state = _speech_state()
    prompts: list[str] = []
    answers = [
        '{"speech": "第一天的话"}',
        '{"target": 2}',
        '{"speech": "第二天的话"}',
        '{"target": 2}',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    policy.speak(state, 1)
    state.day = 1
    policy.vote(state, 3)
    assert "第一天的话" in prompts[1]

    policy.speak(state, 1)  # 新的一天：旧发言必须清掉
    state.day = 2
    policy.vote(state, 3)

    assert "第一天的话" not in prompts[3]
    assert "第二天的话" in prompts[3]


def test_speeches_do_not_leak_across_games() -> None:
    prompts: list[str] = []
    answers = ['{"speech": "上一局的话"}', '{"target": 2}']

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok(answers.pop(0))

    _install(handler)

    policy = LLMAgentPolicy(model="m")
    policy.speak(_speech_state(), 1)

    policy.vote(_vote_state(), 3)  # 新的一局，投票时还没人发言

    assert "上一局的话" not in prompts[1]


def _hunter_state() -> GameState:
    """3 号猎人，第 2 夜被刀死，场上还剩 1/2/4/5。"""
    state = make_state(
        {
            1: Roles.WOLF,
            2: Roles.VILLAGER,
            3: Roles.HUNTER,
            4: Roles.SEER,
            5: Roles.WITCH,
        }
    )
    state.night = 2
    state.kill(3, DeathCause.WOLF_KILL, "第2夜")
    return state


def test_hunter_shoots_a_candidate() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 1}')

    _install(handler)

    assert LLMAgentPolicy(model="m").hunter_shot(_hunter_state(), 3) == 1
    assert "【开枪】" in prompts[0]
    assert "可选目标：1号、2号、4号、5号" in prompts[0]


def test_hunter_can_decline() -> None:
    _install(lambda request: _ok('{"target": null}'))

    assert LLMAgentPolicy(model="m").hunter_shot(_hunter_state(), 3) is None


def test_hunter_cannot_shoot_himself() -> None:
    """猎人已经出局，存活名单里没有他，报自己的座位号只能被判非法。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok('{"target": 3}')

    _install(handler)

    assert LLMAgentPolicy(model="m").hunter_shot(_hunter_state(), 3) is None
    assert calls["n"] == 2


class _TapEvents(FirstAlivePolicy):
    """在无脑策略上加一个钩子，验证 play_game 确实把事件流接了过来。"""

    def __init__(self) -> None:
        self.seen: EventLog | None = None

    def attach_events(self, events: EventLog) -> None:
        self.seen = events


def test_play_game_injects_events() -> None:
    board = make_board({Roles.WOLF: 1, Roles.VILLAGER: 2, Roles.SEER: 1})
    policy = _TapEvents()

    result = play_game(board, policy, seed())

    assert policy.seen is result.events


def test_note_is_appended_not_overwritten() -> None:
    """笔记只能追加：覆盖一条就等于丢一天。

    历史被裁之后，笔记是这个玩家唯一的记忆载体。第二次写把第一次盖掉，
    第 3 天他就只记得“第 2 天发生了什么”。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok('{"note": "3 号在保 2 号，像是给 2 号拉票。"}')

    _install(handler)

    state = _guard_state()
    policy = LLMAgentPolicy(model="m")

    assert policy.note(state, 1, label="第1天") == "3 号在保 2 号，像是给 2 号拉票。"
    policy.note(state, 1, label="第2天")

    assert policy.notebook.get(1) == [
        "第1天：3 号在保 2 号，像是给 2 号拉票。",
        "第2天：3 号在保 2 号，像是给 2 号拉票。",
    ]


def test_note_says_the_details_will_be_dropped() -> None:
    """必须告诉模型“明天就看不到今天的详细记录了”。

    不说的话它会写一句“今天没什么特别的”——然后明天真的两眼一抹黑，
    而引擎这边以为它记住了。
    """
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"note": "记下 3 号。"}')

    _install(handler)

    LLMAgentPolicy(model="m").note(_guard_state(), 1, label="第1天")

    assert "今天的详细记录不会再出现在你的输入里" in prompts[0]


def test_note_is_skipped_for_the_dead() -> None:
    """出局的人不写总结：他明天没有任何调用，写了也没人读。"""
    state = _guard_state()
    state.kill(1, DeathCause.WOLF_KILL, "第1夜")
    policy = LLMAgentPolicy(model="m")

    assert policy.note(state, 1, label="第1天") is None
    assert policy.notebook.get(1) == []


class _NoteSpy(FirstAlivePolicy):
    """在无脑策略上挂 note，验证 play_game 真的每天都来收一次。"""

    def __init__(self) -> None:
        self.notes: list[tuple[str, int]] = []

    def note(self, state: GameState, pid: int, *, label: str) -> str:
        self.notes.append((label, pid))
        return f"{pid} 号记下了"


def test_play_game_collects_notes_every_day() -> None:
    """历史会被裁到只留当前轮，引擎不调 note 就等于没人记得昨天。

    第 1 天就分出胜负时不写笔记（明天没有任何调用），所以这里特意
    多放几个人，让对局至少能打满一天。
    """
    board = make_board(
        {Roles.WOLF: 1, Roles.VILLAGER: 4, Roles.SEER: 1, Roles.WITCH: 1, Roles.HUNTER: 1}
    )
    policy = _NoteSpy()

    result = play_game(board, policy, seed(0))

    assert result.days > 1, "板子构造问题：7 人局第一天就结束了"
    assert policy.notes, "play_game 没有调用 note()"

    labels = {label for label, _ in policy.notes}
    # 只写白天：夜间不总结。第一天的笔记一定有——那时还没分胜负
    assert "第1天" in labels
    assert all(label.endswith("天") for label in labels)
    # 同一天不会给同一个人收两次，否则就是同一个人的笔记被覆盖了
    assert len(policy.notes) == len(set(policy.notes))


def _bodies(handler_payloads: list[str]) -> tuple[list[dict[str, object]], object]:
    """按顺序回放预设 JSON，并把每个请求体存下来供断言。"""
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok(handler_payloads[len(bodies) - 1])

    return bodies, handler


def test_vote_and_note_disable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    """投票和记笔记关掉思维链，发言保留。

    这两步一天各跑 9 次，占全部调用一半以上；而“从几个座位号里挑一个”
    不需要推理过程——思维链只是把首字延迟和输出 token 一起拉长。
    发言不一样，它要组织逻辑、回应别人，留着思考。
    """
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    bodies, handler = _bodies(
        ['{"target": 2}', '{"note": "记下 2 号。"}', '{"speech": "2 号像好人。"}']
    )
    _install(handler)

    state = _guard_state()
    policy = LLMAgentPolicy(model="m")

    policy.vote(state, 1)
    policy.note(state, 1, label="第1天")
    policy.speak(state, 1)

    assert bodies[0]["thinking"] == {"type": "disabled"}  # 投票
    assert bodies[1]["thinking"] == {"type": "disabled"}  # 笔记
    # 发言不传这个字段——没传不等于开启，而是“听服务端默认”
    assert "thinking" not in bodies[2]


def test_thinking_is_omitted_on_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """认不出后端就不发 thinking：宁可慢，也不能让 400 把票全变成弃票。

    {"type": "disabled"} 是 DeepSeek 的写法；Qwen/GLM 叫 enable_thinking，
    或者要放进 extra_body。发错字段服务端会回 400，_ask 吃掉异常返回 None，
    结果是一整轮全员弃票——比慢严重得多。
    """
    monkeypatch.setenv("LLM_BASE_URL", "https://api.moonshot.cn")
    bodies, handler = _bodies(['{"target": 2}'])
    _install(handler)

    LLMAgentPolicy(model="m").vote(_guard_state(), 1)

    assert "thinking" not in bodies[0]


def test_persona_reaches_the_prompt() -> None:
    """按场景生成的人设要能进 prompt，而不只是预制风格。

    导演（agents/director.py）生成的是 Persona，预制的是 Personality，
    两种都走 personality_text。这里锁住“带名字的那一种真的会被渲染出来”，
    否则导演写了一堆人设，玩家端一行也看不到。
    """
    from agents.personality import Persona

    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(_prompt_of(request))
        return _ok('{"target": 2}')

    _install(handler)

    policy = LLMAgentPolicy(
        model="m",
        personalities={
            3: Persona(
                name="沈知微",
                background="退休刑警，被返聘来查这椿旧案",
                traits="嘴上不饶人，但护着自己人",
                resilience="被围攻时反而慢下来，一句一句反问",
                style="短句，爱用“你先把话说圆”开头",
                logic="强，习惯先从票型倒推动机",
            )
        },
    )
    policy.choose_guard(_guard_state())

    assert "沈知微" in prompts[0]
    assert "退休刑警" in prompts[0]
    assert "抗压能力：" in prompts[0]
