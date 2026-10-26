"""白天投票结算。

流程：全体存活玩家投票 → 统计 → 单一最高票出局；
      多人并列最高 → PK 重投 → 再并列则当天无人出局。

PK 轮的三条约束：
  1. 台上的人没有投票权（他们正是被投的对象）
  2. 台下的人只能投台上的人，投台下直接抛 RuleViolation
  3. 出局者必须来自台上——即便校验被绕过，统计也只在场上的票里取最高

警长竞选（1.5 票）在 D8 实现。为了让 D8 只改一处，
票权已经走 _weight() 函数，而不是写死 1。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from engine.state import GameState, RuleViolation

# 每个投票者的票权。D8 有了警长之后，这个函数会读 state 里的警长是谁。
BASE_VOTE_WEIGHT = 1


class VoteChooser(Protocol):
    """每个投票者投给谁。None 表示弃票。

    only 非空表示 PK 重投：这一票只能投给台上的人。把候选集写进签名，
    而不是让策略自己看日历猜第几轮——猜错了就会投到台下，
    而引擎一旦不校验，没上 PK 台的人也会被推出去。
    """

    def vote(
        self, state: GameState, voter: int, *, only: Sequence[int] | None = None
    ) -> int | None: ...


@dataclass(frozen=True)
class VoteResult:
    """一天的投票结果。

    保留 ballots 和 tally 原文，因为复盘页要画"投票矩阵"和"谁抱团"，
    只留最终结果是不够的。
    """

    day: int
    ballots: dict[int, int | None]  # 投票者 -> 目标
    tally: dict[int, int]  # 目标 -> 票数
    tied: tuple[int, ...]  # 并列最高票的人（长度 > 1 才需要 PK）
    re_vote: bool  # 是否发生了 PK 重投
    pk_ballots: dict[int, int | None]
    pk_tally: dict[int, int]
    eliminated: int | None  # 最终出局者；None 表示无人出局

    @property
    def label(self) -> str:
        return f"第{self.day}天"


def _weight(state: GameState, voter: int) -> int:
    """投票者的票权。

    D8 之前恒为 1。用整数表示 1.5 票的做法是"内部 ×2"，
    但那是 D8 的事，现在不提前引入复杂度。
    """
    del state, voter  # 目前不使用，保留签名让 D8 只改函数体
    return BASE_VOTE_WEIGHT


def _collect(
    state: GameState,
    chooser: VoteChooser,
    voters: list[int],
    *,
    only: Sequence[int] | None = None,
) -> dict[int, int | None]:
    """收集一轮投票。当场校验目标合法性——非法票直接抛，不静默丢弃。

    only 非空表示 PK 重投：台上的人没有投票权（voters 已经排除了他们），
    台下的人只能投台上剩下的那几个。不拦的话，随机策略很容易把台下的人
    推成“最高票”，一轮下来会有人无缘无故出局。
    """
    ballots: dict[int, int | None] = {}
    for voter in voters:
        target = chooser.vote(state, voter, only=only)
        if target is None:
            ballots[voter] = None  # 弃票合法
            continue
        if target == voter:
            # 禁止自投。规则上可选，但禁掉能让机器人策略更简单、复盘更好读
            raise RuleViolation(f"{voter}号 不能投自己")
        if not state.is_alive(target):
            raise RuleViolation(f"{voter}号 投票给已出局的 {target}号")
        if only is not None and target not in only:
            raise RuleViolation(f"{voter}号 在 PK 轮投了台下的 {target}号")
        ballots[voter] = target
    return ballots


def _tally(state: GameState, ballots: dict[int, int | None]) -> dict[int, int]:
    """按票权累加。弃票不计入。"""
    tally: dict[int, int] = {}
    for voter, target in ballots.items():
        if target is None:
            continue
        tally[target] = tally.get(target, 0) + _weight(state, voter)
    return tally


def _top(tally: dict[int, int]) -> tuple[int, ...]:
    """得票最高的人。返回元组是为了让"并列"这件事在类型上就可见。"""
    if not tally:
        return ()
    best = max(tally.values())
    return tuple(sorted(pid for pid, n in tally.items() if n == best))


def resolve_vote(state: GameState, chooser: VoteChooser) -> VoteResult:
    """跑完一天的投票，返回结果。不负责把出局者标记死亡——
    那是调用方用 apply_deaths 做的，因为猎人被投出局时还要开枪。
    """
    state.day += 1
    alive = state.alive_ids()

    ballots = _collect(state, chooser, alive)
    tally = _tally(state, ballots)
    tied = _top(tally)

    # 只有一个最高票 -> 直接出局
    if len(tied) == 1:
        return VoteResult(
            day=state.day,
            ballots=ballots,
            tally=tally,
            tied=tied,
            re_vote=False,
            pk_ballots={},
            pk_tally={},
            eliminated=tied[0],
        )

    # 全员弃票 -> 无人出局（注意这和"平票"不同，没有 PK）
    if not tied:
        return VoteResult(
            day=state.day,
            ballots=ballots,
            tally=tally,
            tied=(),
            re_vote=False,
            pk_ballots={},
            pk_tally={},
            eliminated=None,
        )

    # 平票 -> PK 重投。
    # 规则：PK 台上的平票者**不能投票**，台下的人只能投他们。
    pk_voters = [pid for pid in alive if pid not in tied]
    pk_ballots = _collect(state, chooser, pk_voters, only=tied)
    pk_tally = _tally(state, pk_ballots)
    # 只在场上的票里取最高。_collect 已经拦了一道，这里再兜一道：
    # 即便以后有人绕开校验，也不可能把没上 PK 台的人判出局
    pk_top = _top({pid: n for pid, n in pk_tally.items() if pid in tied})

    # 再平票（或台下全员弃票）-> 当天无人出局
    eliminated = pk_top[0] if len(pk_top) == 1 else None

    return VoteResult(
        day=state.day,
        ballots=ballots,
        tally=tally,
        tied=tied,
        re_vote=True,
        pk_ballots=pk_ballots,
        pk_tally=pk_tally,
        eliminated=eliminated,
    )
