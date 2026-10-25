"""游戏主循环：夜 -> 查胜负 -> 昼 -> 查胜负 -> 夜……直到分出胜负。

D4 让它能跑通，D6 让它**每一件发生的事都留下一条带序号的事件**。
事件是 D12 的 prompt 输入、D16 的 SSE 推送内容、D23 的复盘时间轴。

本模块不做的事：
  · 不实现事件->状态的重放（见 engine/events.py 顶部的取舍说明）
  · 不实现警长竞选（D8）
  · 不做发言（D8 / D12 才有数据源）
"""

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from engine.board import Board
from engine.events import VIS_GOD, VIS_WOLVES, EventKind, EventLog, vis_pid
from engine.night import NightChooser, NightResult, resolve_night
from engine.roles import Roles
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
    """一局的结果。

    nights / votes 是"结构化摘要"（D11 统计、CLI 渲染用），
    events 是"逐条发生的事实"（D12 prompt、D16 推送、D23 复盘用）。
    两者不是重复：前者方便聚合，后者保证时序和可见性。
    """

    winner: Winner
    state: GameState
    nights: list[NightResult] = field(default_factory=list)
    votes: list[VoteResult] = field(default_factory=list)
    events: EventLog = field(default_factory=EventLog)

    @property
    def days(self) -> int:
        return self.state.day


# ---------------------------------------------------------------------------
# 事件发射：把已有的结构化结果"拆"成事件，不改动规则引擎
# ---------------------------------------------------------------------------


def _holder_pid(state: GameState, role: Roles) -> int | None:
    """某个唯一身份的持有者（不管死活）。

    用它在事件上标"只有这个人能看到"。为什么不传 pid：resolve_night 只回传
    "做了什么"，不回传"是谁做的"，从 state 反查比改 night.py 的返回值便宜得多。
    """
    holders = state.role_holders(role)
    return holders[0] if holders else None


def _emit_deaths(events: EventLog, deaths: list[Death], label: str) -> None:
    """死亡是全场可见事件——所有人都看到谁出局了。"""
    for death in deaths:
        events.emit(EventKind.DEATH, label, pid=death.pid, cause=death.cause.value)


def _emit_night(events: EventLog, night: NightResult, state: GameState) -> None:
    """把一夜拆成事件。顺序即日志顺序，复盘直接按它渲染时间轴。"""
    label = night.label
    events.emit(EventKind.NIGHT_START, label)

    if night.guard_target is not None:
        guard = _holder_pid(state, Roles.GUARD)
        events.emit(
            EventKind.GUARD,
            label,
            target=night.guard_target,
            visible_to=vis_pid(guard) if guard is not None else VIS_GOD,
        )

    if night.wolf_target is not None:
        # 狼刀只有狼队知道
        events.emit(EventKind.WOLF_KILL, label, target=night.wolf_target, visible_to=VIS_WOLVES)

    witch = _holder_pid(state, Roles.WITCH)
    witch_vis = vis_pid(witch) if witch is not None else VIS_GOD

    if night.witch_saved:
        events.emit(
            EventKind.WITCH_SAVE,
            label,
            target=night.wolf_target,
            visible_to=witch_vis,
        )
    if night.poison_target is not None:
        events.emit(
            EventKind.WITCH_POISON,
            label,
            target=night.poison_target,
            visible_to=witch_vis,
        )

    if night.seer_target is not None:
        seer = _holder_pid(state, Roles.SEER)
        events.emit(
            EventKind.SEER_CHECK,
            label,
            target=night.seer_target,
            saw_wolf=night.seer_saw_wolf,
            visible_to=vis_pid(seer) if seer is not None else VIS_GOD,
        )

    _emit_deaths(events, list(night.deaths), label)


def _emit_vote(events: EventLog, vote: VoteResult) -> None:
    """把一天的投票拆成事件。"""
    label = vote.label
    events.emit(EventKind.DAY_START, label)

    # 每人一票，前端画投票矩阵直接读这些
    for voter, target in sorted(vote.ballots.items()):
        events.emit(EventKind.VOTE, label, voter=voter, target=target)

    # PK 重投单独标 pk=True，否则和第一轮票混在一起看不出哪轮是哪轮
    for voter, target in sorted(vote.pk_ballots.items()):
        events.emit(EventKind.VOTE, label, voter=voter, target=target, pk=True)

    events.emit(
        EventKind.VOTE_RESULT,
        label,
        tally={str(pid): n for pid, n in vote.tally.items()},
        tied=list(vote.tied),
        re_vote=vote.re_vote,
        eliminated=vote.eliminated,
    )


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


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
    events = result.events

    # 身份表是上帝视角，绝不能进公共记录（D12 会按 visible_to 过滤）
    events.emit(
        EventKind.GAME_START,
        "",
        board=board.id,
        players={str(pid): p.role.value for pid, p in sorted(state.players.items())},
        visible_to=VIS_GOD,
    )

    # 记录猎人开枪的 (谁开, 打谁)。apply_deaths 内部调用 shot_chooser，
    # 在这里截一份，就能在不改 state.py 的前提下发出 HUNTER_SHOT 事件。
    shots: list[tuple[int, int | None]] = []

    def shot_chooser(hunter: int) -> int | None:
        target = policy.hunter_shot(state, hunter)
        shots.append((hunter, target))
        return target

    def drain_shots(label: str) -> None:
        while shots:
            hunter, target = shots.pop(0)
            events.emit(EventKind.HUNTER_SHOT, label, hunter=hunter, target=target)

    winner: Winner | None = check_winner(state)

    while winner is None:
        if state.day >= max_days:
            raise RuntimeError(
                f"超过 {max_days} 天仍未分出胜负（存活 {state.alive_count()} 人），策略可能陷入循环"
            )

        # ---- 夜晚 ----
        night = resolve_night(state, policy, shot_chooser)
        result.nights.append(night)
        _emit_night(events, night, state)
        drain_shots(night.label)

        winner = check_winner(state)
        if winner is not None:
            break

        # ---- 白天投票 ----
        vote = resolve_vote(state, policy)
        result.votes.append(vote)
        _emit_vote(events, vote)

        if vote.eliminated is not None:
            deaths = apply_deaths(
                state,
                {vote.eliminated: DeathCause.VOTE},
                vote.label,
                shot_chooser,
            )
            _emit_deaths(events, deaths, vote.label)
            drain_shots(vote.label)

        winner = check_winner(state)

    if winner is None:  # 仅为让类型收窄，逻辑上不可达
        raise RuntimeError("胜负判定异常：循环退出但没有胜方")

    result.winner = winner
    events.emit(EventKind.GAME_OVER, "", winner=winner.value, days=state.day)
    return result
