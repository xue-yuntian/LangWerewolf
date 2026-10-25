"""游戏主循环：夜 -> 查胜负 -> 昼 -> 查胜负 -> 夜……直到分出胜负。

这是 D4 的终点：不做花活，只保证"一局能从头走到尾"。
D5 会把这里的策略换成随机机器人 + CLI 播放器，D6 再把过程包成事件流，
D20 会把 while 循环拆成可暂停的 step()。今天先让它跑通。
"""

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from engine.board import Board
from engine.night import NightChooser, NightResult, resolve_night
from engine.state import Death, DeathCause, GameState, apply_deaths, create_state
from engine.vote import VoteChooser, VoteResult, resolve_vote
from engine.win import Winner, check_winner

# 单局安全上限。正常 9 人局 5-8 天就结束；超过这个数说明策略陷入死循环
# （比如所有人每天互投弃票、夜里都空刀），早失败比卡死好。
DEFAULT_MAX_DAYS = 30


class GamePolicy(NightChooser, VoteChooser, Protocol):
    """驱动一局所需的全部策略。

    夜晚部分继承 NightChooser（D3），投票部分继承 VoteChooser（D4）。
    再加一个猎人开枪的选择——它既可能发生在夜里也可能发生在白天，
    所以不属于上面任何一个。
    """

    def hunter_shot(self, state: GameState, hunter: int) -> int | None:
        """猎人开枪打谁。None 表示不开枪。"""
        ...


@dataclass
class GameResult:
    """一局的结果。D5 的 CLI 播放器、D11 的批量仿真都消费它。"""

    winner: Winner
    state: GameState
    nights: list[NightResult] = field(default_factory=list)
    votes: list[VoteResult] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def days(self) -> int:
        return self.state.day


def play_game(
    board: Board,
    policy: GamePolicy,
    rng: Random,
    max_days: int = DEFAULT_MAX_DAYS,
) -> GameResult:
    """跑完一局，返回结果。

    参数:
        board: 已通过校验的板子
        policy: 所有决策的来源
        rng: 显式传入，便于用 Random(42) 复现同一局
        max_days: 安全上限
    """
    # 警长竞选是 D8 的内容。这里显式拒绝，而不是静默忽略，
    # 否则 12 人局会"看起来能跑但少了整个上警流程"，属于最难查的那种 bug。
    if board.options.sheriff:
        raise NotImplementedError("警长竞选在 D8 实现。现在请用 sheriff=false 的板子（如 9p）")

    state = create_state(board, rng)
    result = GameResult(winner=Winner.GOOD, state=state)  # winner 会被覆盖

    def shot_chooser(hunter: int) -> int | None:
        # 猎人的开枪对象由策略决定；apply_deaths 会用队列处理连环枪
        return policy.hunter_shot(state, hunter)

    def record_deaths(deaths: list[Death], label: str) -> None:
        for death in deaths:
            result.log.append(f"{label}：{death.pid}号 出局（{death.cause.value}）")

    winner: Winner | None = check_winner(state)

    while winner is None:
        if state.day >= max_days:
            raise RuntimeError(
                f"超过 {max_days} 天仍未分出胜负（存活 {state.alive_count()} 人），策略可能陷入循环"
            )

        # ---- 夜晚 ----
        night = resolve_night(state, policy, shot_chooser)
        result.nights.append(night)
        record_deaths(list(night.deaths), night.label)
        if not night.deaths:
            result.log.append(f"{night.label}：平安夜")

        winner = check_winner(state)
        if winner is not None:
            break

        # ---- 白天投票 ----
        vote = resolve_vote(state, policy)
        result.votes.append(vote)
        if vote.re_vote:
            result.log.append(f"{vote.label}：平票 PK，平票者 {vote.tied}")
        if vote.eliminated is None:
            result.log.append(f"{vote.label}：无人出局")
        else:
            deaths = apply_deaths(
                state,
                {vote.eliminated: DeathCause.VOTE},
                vote.label,
                shot_chooser,
            )
            record_deaths(deaths, vote.label)

        winner = check_winner(state)

    if winner is None:  # 仅为让类型收窄，逻辑上不可达
        raise RuntimeError("胜负判定异常：循环退出但没有胜方")

    result.winner = winner
    result.log.append(f"结束：{winner.value} 胜，共 {state.day} 天")
    return result
