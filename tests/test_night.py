"""D3 夜晚结算的单测。

重点覆盖那些"看起来能救回来其实救不回来"的组合规则：
同守同救、毒药无视守卫、猎人被毒不能开枪。
这些是狼人杀里最容易写错、也最容易在 D11 的胜率统计里暴露出来的地方。
"""

import pytest

from engine.night import resolve_night
from engine.roles import Roles
from engine.state import DeathCause, RuleViolation
from tests.support import ScriptNight, make_state, no_shot

# 常用阵容：1 狼、2 民、3 守卫、4 女巫、5 猎人、6 预言家
BASE = {
    1: Roles.WOLF,
    2: Roles.VILLAGER,
    3: Roles.GUARD,
    4: Roles.WITCH,
    5: Roles.HUNTER,
    6: Roles.SEER,
}


def _cause_of(state, pid: int) -> DeathCause | None:
    """某个玩家的死因；没死返回 None。"""
    for death in state.deaths:
        if death.pid == pid:
            return death.cause
    return None


def test_guard_blocks_wolf_kill() -> None:
    """守卫守中狼刀目标 -> 目标存活。"""
    state = make_state(BASE)
    resolve_night(state, ScriptNight(guard=2, kill=2), no_shot)

    assert state.is_alive(2)
    assert state.deaths == []


def test_witch_save_blocks_wolf_kill() -> None:
    """女巫救中狼刀目标 -> 目标存活，解药被消耗。"""
    state = make_state(BASE)
    resolve_night(state, ScriptNight(kill=2, save=True), no_shot)

    assert state.is_alive(2)
    assert state.witch_antidote_used is True


def test_guard_and_save_together_kills() -> None:
    """【同守同救】守卫和女巫同时保同一个人 -> 反而死。

    这是狼人杀最反直觉的规则：两个保护叠加成负向效果。
    """
    state = make_state(BASE)
    resolve_night(state, ScriptNight(guard=2, kill=2, save=True), no_shot)

    assert not state.is_alive(2)
    assert _cause_of(state, 2) is DeathCause.WOLF_KILL


def test_poison_ignores_guard() -> None:
    """毒药无视守卫。

    守卫只挡狼刀、不挡毒——所以守 2 号同时毒 2 号，2 号照死。
    """
    state = make_state(BASE)
    resolve_night(state, ScriptNight(guard=2, kill=1, poison=2), no_shot)

    assert not state.is_alive(2)
    assert _cause_of(state, 2) is DeathCause.POISON
    # 1 号是狼，被自己人刀了——"自刀"在规则上合法
    assert not state.is_alive(1)


def test_witch_cannot_use_both_potions() -> None:
    """一晚只能喝一瓶药。"""
    state = make_state(BASE)
    with pytest.raises(RuleViolation):
        resolve_night(state, ScriptNight(kill=2, save=True, poison=5), no_shot)


def test_guard_cannot_guard_same_target_twice() -> None:
    """守卫不能连续两晚守同一人。"""
    state = make_state(BASE)
    state.last_guarded = 2

    with pytest.raises(RuleViolation):
        resolve_night(state, ScriptNight(guard=2, kill=5), no_shot)


def test_witch_can_self_save_on_first_night() -> None:
    """首夜女巫可以自救。"""
    state = make_state(BASE)
    resolve_night(state, ScriptNight(kill=4, save=True), no_shot)

    assert state.is_alive(4)
    assert state.night == 1


def test_witch_cannot_self_save_later() -> None:
    """第二夜起女巫不能自救。"""
    state = make_state(BASE)
    state.night = 1  # 已经过完一夜，下一夜是第 2 夜

    with pytest.raises(RuleViolation):
        resolve_night(state, ScriptNight(kill=4, save=True), no_shot)


def test_hunter_killed_by_wolf_can_shoot() -> None:
    """猎人被刀可以开枪带走一人。"""
    state = make_state(BASE)
    resolve_night(state, ScriptNight(kill=5), lambda hunter: 1)

    assert not state.is_alive(5)
    assert not state.is_alive(1)
    assert _cause_of(state, 1) is DeathCause.SHOT


def test_hunter_poisoned_cannot_shoot() -> None:
    """猎人被毒不能开枪——毒药压制死亡技能。

    这条最容易漏：实现里如果只看 death_ability 不看死因，
    这里就会变成"1 号也被带走了"。
    """
    state = make_state(BASE)
    asked: list[int] = []

    def shot(hunter: int) -> int:
        asked.append(hunter)
        return 1

    resolve_night(state, ScriptNight(kill=None, poison=5), shot)

    assert not state.is_alive(5)
    assert state.is_alive(1)  # 没被带走
    assert asked == []  # 根本没问过猎人要打谁


def test_two_deaths_in_one_night() -> None:
    """一晚可以死两个人：被刀一个 + 被毒一个。"""
    state = make_state(BASE)
    resolve_night(state, ScriptNight(kill=2, poison=3), no_shot)

    assert not state.is_alive(2)
    assert not state.is_alive(3)
    assert len(state.deaths) == 2


def test_poison_wins_cause_when_same_target() -> None:
    """同一人既被刀又被毒：只死一次，死因判为毒。"""
    state = make_state(BASE)
    resolve_night(state, ScriptNight(kill=2, poison=2), no_shot)

    assert not state.is_alive(2)
    assert len([d for d in state.deaths if d.pid == 2]) == 1
    assert _cause_of(state, 2) is DeathCause.POISON


def test_seer_gets_faction_not_role() -> None:
    """预言家只拿到"是不是狼"，拿不到具体身份。"""
    wolf_state = make_state(BASE)
    wolf_result = resolve_night(wolf_state, ScriptNight(seer=1), no_shot)
    assert wolf_result.seer_saw_wolf is True

    good_state = make_state(BASE)
    good_result = resolve_night(good_state, ScriptNight(seer=2), no_shot)
    assert good_result.seer_saw_wolf is False
