"""D4 胜负判定的单测。

屠边必须能区分三种路径：
  神全灭 -> 狼胜
  民全灭 -> 狼胜
  狼全灭 -> 好人胜
搞错任意一条，D11 的胜率统计就全是错的。
"""

from engine.board import WinCondition
from engine.roles import Faction, Roles
from engine.state import DeathCause
from engine.win import Winner, check_winner
from tests.support import make_state

LINEUP = {
    1: Roles.WOLF,
    2: Roles.WOLF,
    3: Roles.SEER,
    4: Roles.WITCH,
    5: Roles.VILLAGER,
    6: Roles.VILLAGER,
}


def test_nobody_wins_at_start() -> None:
    assert check_winner(make_state(LINEUP)) is None


def test_all_wolves_dead_good_wins() -> None:
    state = make_state(LINEUP)
    state.kill(1, DeathCause.VOTE, "第1天")
    state.kill(2, DeathCause.VOTE, "第2天")

    assert check_winner(state) is Winner.GOOD


def test_all_gods_dead_wolf_wins() -> None:
    """屠边：神全灭即狼胜，平民还有人也救不回来。"""
    state = make_state(LINEUP)
    state.kill(3, DeathCause.WOLF_KILL, "第1夜")
    state.kill(4, DeathCause.POISON, "第1夜")

    # 平民还在场，但这不影响屠边判定
    assert state.alive_of(Faction.VILLAGER) == [5, 6]
    assert check_winner(state) is Winner.WOLF


def test_all_villagers_dead_wolf_wins() -> None:
    """屠边：民全灭同样狼胜。"""
    state = make_state(LINEUP)
    state.kill(5, DeathCause.WOLF_KILL, "第1夜")
    state.kill(6, DeathCause.WOLF_KILL, "第2夜")

    assert check_winner(state) is Winner.WOLF


def test_wolf_eliminated_has_priority_when_both_sides_wiped() -> None:
    """最后一狼与最后一神同夜死亡时，按好人胜处理（网杀惯例）。"""
    state = make_state(LINEUP)
    state.kill(1, DeathCause.WOLF_KILL, "第1夜")
    state.kill(2, DeathCause.WOLF_KILL, "第1夜")
    state.kill(3, DeathCause.WOLF_KILL, "第1夜")
    state.kill(4, DeathCause.WOLF_KILL, "第1夜")

    assert check_winner(state) is Winner.GOOD


def test_slaughter_all_needs_good_side_fully_wiped() -> None:
    """屠城板子：只死光神职不够，平民还在就没输。"""
    state = make_state(LINEUP, win_condition=WinCondition.S_ALL)
    state.kill(3, DeathCause.VOTE, "第1天")
    state.kill(4, DeathCause.VOTE, "第2天")
    assert check_winner(state) is None  # 平民还活着

    state.kill(5, DeathCause.VOTE, "第3天")
    state.kill(6, DeathCause.VOTE, "第4天")
    assert check_winner(state) is Winner.WOLF
