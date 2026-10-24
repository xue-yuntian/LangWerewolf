"""D4 白天投票的单测。

四个分支必须都覆盖到：
  单一最高票 -> 出局
  平票 -> PK -> 出局
  平票 -> PK 再平票 -> 无人出局
  全员弃票 -> 无人出局（注意这和"平票"不是一回事）
"""

import pytest

from engine.roles import Roles
from engine.state import DeathCause, RuleViolation
from engine.vote import resolve_vote
from tests.support import ScriptVote, make_state

# 1 狼 + 4 民：结构简单，只关心票型，不关心阵营
LINEUP = {
    1: Roles.WOLF,
    2: Roles.VILLAGER,
    3: Roles.VILLAGER,
    4: Roles.VILLAGER,
    5: Roles.VILLAGER,
}


def test_single_highest_eliminated() -> None:
    state = make_state(LINEUP)
    result = resolve_vote(state, ScriptVote({1: 2, 2: 3, 3: 2, 4: 2, 5: 2}))

    assert result.tally[2] == 4
    assert result.tied == (2,)
    assert result.re_vote is False
    assert result.eliminated == 2


def test_tie_triggers_pk_then_eliminates() -> None:
    """2 号和 3 号都是 2 票 -> PK，台下三人重投 -> 2 号出局。"""
    state = make_state(LINEUP)
    # 首轮：2 得 2 票，3 得 2 票，4 得 1 票
    first = {1: 2, 2: 3, 3: 2, 4: 3, 5: 4}
    # PK 台下是 1、4、5 三个人
    pk = {1: 2, 4: 2, 5: 3}
    result = resolve_vote(state, ScriptVote({**first, **pk}))

    assert result.tied == (2, 3)
    assert result.re_vote is True
    # PK 台上的 2、3 没有投票权，所以 pk_ballots 里不该出现他们
    assert set(result.pk_ballots) == {1, 4, 5}
    assert result.eliminated == 2


def test_pk_tie_again_means_nobody_out() -> None:
    """PK 再平票 -> 当天无人出局。"""
    state = make_state(LINEUP)
    first = {1: 2, 2: 3, 3: 2, 4: 3, 5: 4}
    pk = {1: 2, 4: 3, 5: None}  # 2 和 3 各 1 票
    result = resolve_vote(state, ScriptVote({**first, **pk}))

    assert result.re_vote is True
    assert result.eliminated is None


def test_all_abstain_means_no_pk() -> None:
    """全员弃票不等于平票，不该进 PK。"""
    state = make_state(LINEUP)
    result = resolve_vote(state, ScriptVote({}))

    assert result.tally == {}
    assert result.tied == ()
    assert result.re_vote is False
    assert result.eliminated is None


def test_voting_self_rejected() -> None:
    state = make_state(LINEUP)
    with pytest.raises(RuleViolation):
        resolve_vote(state, ScriptVote({1: 1}))


def test_voting_dead_player_rejected() -> None:
    state = make_state(LINEUP)
    state.kill(3, DeathCause.WOLF_KILL, "第1夜")

    with pytest.raises(RuleViolation):
        resolve_vote(state, ScriptVote({1: 3}))


def test_day_counter_increments() -> None:
    state = make_state(LINEUP)
    assert state.day == 0
    resolve_vote(state, ScriptVote({}))
    assert state.day == 1
