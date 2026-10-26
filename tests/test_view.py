"""玩家视角：术语按角色裁、历史只留当前这一轮。

这两个改动都是为了压 prompt。裁错了的后果比浪费更严重——裁掉模型
本该看到的私有信息，它会当场做出没法解释的决策，而且测试全绿
（因为断言的是"某段文字在不在"）。
"""

from engine.events import EventKind, EventLog
from engine.roles import Roles
from engine.state import DeathCause
from engine.view import PlayerView, render_event, round_start_seq, terms_text, view_for
from tests.support import make_state


def _plain_view(role: Roles) -> PlayerView:
    """除身份外什么都没有的视图，方便单看术语段。"""
    return PlayerView(
        pid=1,
        roles=role,
        day=1,
        alive=[1],
        dead=[],
        teammates=[],
        private_lines=[],
        public_lines=[],
        last_speeches=[],
        personality="",
        notes=[],
    )


def test_terms_are_scoped_to_the_role() -> None:
    """只发本角色用得上的术语。

    "金水""银水"必须留：那是别人发言里的词，听不懂就没法判断局势。
    "空守""解药"对平民没有任何用处——他既用不上，也不会从别人嘴里听到。

    断言要看条目头“- 解药：”而不是子串：自刀的描述里本来就有“骗女巫的解药”，
    那是解释自刀动机的必要文字，不能因此把整条自刀也删了。
    """
    villager = terms_text(Roles.VILLAGER)

    assert "- 金水：" in villager
    assert "- 银水：" in villager
    assert "- 空守：" not in villager
    assert "- 解药：" not in villager
    assert "- 开枪：" not in villager


def test_guard_terms_include_his_own_skills() -> None:
    """守卫要看到自己的三条，但别顺手把女巫的也发过去。"""
    guard = terms_text(Roles.GUARD)

    assert "- 空守：" in guard
    assert "- 连守：" in guard
    assert "- 同守同救：" in guard
    assert "- 解药：" not in guard


def test_terms_are_merged_into_the_identity_block() -> None:
    """术语不再单独占一段，跟着身份走。

    单独一段的代价是每次调用都发一份完整术语表；合并之后模型还是
    在同一处读到"我是谁、我能做什么、这些词什么意思"。
    """
    text = _plain_view(Roles.SEER).to_prompt_text()

    assert "以下是游戏术语的解释" not in text
    assert "你的能力是：" in text
    assert "相关术语：" in text
    assert "查验" in text
    # 身份段和术语段之间的空行没了，说明它们确实是一段
    assert "你的能力是：" in text.split("相关术语：")[0]
    assert text.index("你的能力是：") < text.index("相关术语：") < text.index("当前是第1天")


def test_history_only_covers_the_current_round() -> None:
    """第 2 天只带第 2 天：昨天的详细记录已经换成笔记了。

    不裁的话每一轮都把整局重发一遍——9p 到第 3 天，光历史就 3000 字，
    而其中大部分内容模型上一轮刚看过。
    """
    state = make_state({1: Roles.VILLAGER, 2: Roles.WOLF})
    state.night = 2

    events = EventLog()
    events.emit(EventKind.NIGHT_START, "第1夜")
    events.emit(EventKind.VOTE, "第1天", voter=1, target=2)
    events.emit(EventKind.NIGHT_START, "第2夜")
    events.emit(EventKind.VOTE, "第2天", voter=1, target=2)

    view = view_for(state, events, 1, since_seq=round_start_seq(events, 2))

    assert len(view.public_lines) == 1
    assert view.public_lines[0].startswith("第2天")


def test_speeches_do_not_enter_public_history() -> None:
    """发言只走 last_speeches，公开历史只留实际行为。

    两处都放的话，同一条发言在 prompt 里出现两次（而且公开历史那份
    因为早期 render_event 不读 pid，连说话人都没有）。
    """
    state = make_state({1: Roles.VILLAGER, 2: Roles.WOLF})
    events = EventLog()
    events.emit(EventKind.NIGHT_START, "第1夜")
    events.emit(EventKind.SPEECH, "第1天", pid=1, text="我怀疑 2 号")
    events.emit(EventKind.VOTE, "第1天", voter=1, target=2)

    view = view_for(state, events, 1, since_seq=round_start_seq(events, 1))

    assert view.public_lines == ["第1天：1 号投给 2 号"]
    assert not any("我怀疑 2 号" in line for line in view.public_lines)


def test_render_event_keeps_the_speaker() -> None:
    """事件渲染要带座位号，否则谁都能拿到一句无主的话。"""
    events = EventLog()
    event = events.emit(EventKind.SPEECH, "第1天", pid=3, text="我跳预言家")

    assert render_event(event) == "第1天：3 号：我跳预言家"


def test_view_day_is_the_round_being_played() -> None:
    """第 N 天发言时 state.day 还是 N-1，prompt 不能写成前一天。

    回归点：直接读 state.day 会让第一天的 prompt 写着“第0天”，
    而模型会照这个数去算“昨天发生了什么”。
    """
    state = make_state({1: Roles.VILLAGER, 2: Roles.WOLF})
    state.night = 1
    state.day = 0  # 第 1 天发言时就是这个状态：还没 resolve_vote

    view = view_for(state, EventLog(), 1)

    assert view.day == 1
    assert "当前是第1天" in view.to_prompt_text()


def test_dead_list_hides_the_night_cause() -> None:
    """夜里怎么死的不写进 prompt：那等于替女巫承认她出过手。

    狼刀还是毒杀，场上只有凶手和女巫知道。写出来之后，女巫“要不要
    认下这瓶毒”这个策略选择就被引擎提前替她做了。
    """
    state = make_state({1: Roles.WOLF, 2: Roles.VILLAGER, 3: Roles.WITCH})
    state.kill(2, DeathCause.WOLF_KILL, "第1夜")
    state.kill(3, DeathCause.POISON, "第2夜")

    view = view_for(state, EventLog(), 1)

    assert view.dead == [(2, "第1夜"), (3, "第2夜")]
    assert "毒杀" not in view.to_prompt_text()


def test_dead_list_keeps_the_public_causes() -> None:
    """投票出局和猎人开枪是公开事件（VOTE_RESULT / HUNTER_SHOT 全场可见）。

    这两条不该跟着藏起来：藏了模型会以为“突然少了一个人”，反而更难推理。
    """
    state = make_state({1: Roles.WOLF, 2: Roles.VILLAGER, 3: Roles.HUNTER})
    state.kill(2, DeathCause.VOTE, "第1天")
    state.kill(3, DeathCause.SHOT, "第1天")

    view = view_for(state, EventLog(), 1)

    assert view.dead == [(2, "第1天 投票"), (3, "第1天 枪杀")]


def test_round_start_is_empty_before_the_night_is_logged() -> None:
    """夜间结算前 NIGHT_START 还没发：什么都不留。

    这时玩家手里只有笔记。留着上一轮的详细记录并不会让他算得更准，
    只会把每一夜的调用成本翻一倍。
    """
    events = EventLog()
    events.emit(EventKind.SPEECH, "第1天", pid=1, text="旧发言")

    assert round_start_seq(events, 2) == len(events)


def test_first_night_does_not_trim() -> None:
    """第一夜没有"上一轮"可裁，不能把已有的私有事件也裁掉。

    回归点：早期实现找不到 NIGHT_START 就返回 len(events)，把第一夜
    女巫的 WITCH_SAVE 裁没了——她看不到自己救过谁，于是把银水当狼投。
    """
    events = EventLog()
    events.emit(EventKind.WOLF_KILL, "第1夜", target=2, visible_to="wolves")

    assert round_start_seq(events, 0) == -1
