"""胜负判定。

按阵营聚合判，不按具体身份——所以加新身份、改板子都不影响这里。

屠边（slaughter_side）：神全灭 或 民全灭 -> 狼胜；狼全灭 -> 好人胜
屠城（slaughter_all）： 好人全灭 -> 狼胜；狼全灭 -> 好人胜
"""

from enum import StrEnum

from engine.board import WinCondition
from engine.roles import Faction
from engine.state import GameState


class Winner(StrEnum):
    """胜方。注意 "好人" 不是一个 Faction——好人是神 + 民两个阵营的合称。"""

    WOLF = "wolf"
    GOOD = "good"


def check_winner(state: GameState) -> Winner | None:
    """返回胜方；还没分出胜负返回 None。

    优先级：先判狼方是否全灭。如果某一刻双方同时归零（例如最后一狼和最后一神
    在同一晚死亡），按"狼全灭即好人胜"处理——网杀惯例。
    """
    if not state.wolves_alive():
        return Winner.GOOD

    gods = state.alive_of(Faction.GOD)
    villagers = state.alive_of(Faction.VILLAGER)

    if state.board.options.win_condition is WinCondition.S_ALL:
        # 屠城：好人（神 + 民）全灭才算狼胜
        if not gods and not villagers:
            return Winner.WOLF
        return None

    # 屠边：只要神全灭，或民全灭，狼就赢
    if not gods or not villagers:
        return Winner.WOLF
    return None
