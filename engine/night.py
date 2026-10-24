"""夜晚结算。

夜晚顺序：守卫 → 狼人 → 女巫 → 预言家 → 结算。

为什么是这个顺序：女巫必须知道"今晚谁被刀了"才能决定救不救；
预言家最后，因为他只产出一条信息，不影响当晚死亡。

本模块只做**结算**。谁做什么决定，由传进来的 chooser 负责
（D5 是随机机器人，D12 换成 LLM Agent）。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from engine.roles import Faction, Roles
from engine.state import Death, DeathCause, GameState, RuleViolation, apply_deaths


class NightChooser(Protocol):
    """夜晚各阶段该做什么——由调用方实现。

    D4 只有 tests 里的脚本策略；D5 是随机机器人；
    D12 换成 Agent 时，每个方法内部会改成"按该玩家可见的信息调用 LLM"。
    """

    def choose_guard(self, state: GameState) -> int | None:
        """守卫守谁。None 表示空守。"""
        ...

    def choose_wolf_kill(self, state: GameState) -> int | None:
        """狼队刀谁。None 表示空刀。"""
        ...

    def choose_witch_save(self, state: GameState, killed: int | None) -> bool:
        """女巫是否用解药救今晚被刀的人。"""
        ...

    def choose_witch_poison(self, state: GameState) -> int | None:
        """女巫毒谁。None 表示不用毒。"""
        ...

    def choose_seer_check(self, state: GameState) -> int | None:
        """预言家验谁。"""
        ...


@dataclass(frozen=True)
class NightResult:
    """一晚的完整结果。既给上层驱动用，也是复盘时间轴的数据源。"""

    index: int  # 第几夜
    guard_target: int | None
    wolf_target: int | None
    witch_saved: bool
    poison_target: int | None
    seer_target: int | None
    seer_saw_wolf: bool | None  # 验出来了没，None 表示没验
    deaths: tuple[Death, ...]

    @property
    def label(self) -> str:
        return f"第{self.index}夜"


def _require_alive(state: GameState, pid: int, action: str) -> None:
    """动作目标必须在场。死者不能被刀、被守、被验、被毒。"""
    if not state.is_alive(pid):
        raise RuleViolation(f"{action} 的目标 {pid}号 已经出局")


def resolve_night(
    state: GameState,
    chooser: NightChooser,
    shot_chooser: Callable[[int], int | None],
) -> NightResult:
    """跑完一夜，返回结果并把死亡写入 state。

    参数:
        state: 会被就地修改（夜晚计数、死亡、药、守卫记录）
        chooser: 各阶段的决策来源
        shot_chooser: 猎人死亡时问它打谁（白天投票也要用，所以由外部传）
    """
    state.night += 1
    label = f"第{state.night}夜"

    # ---------------- 1. 守卫 ----------------
    guard_pid = state.alive_holder(Roles.GUARD)
    guard_target: int | None = None
    if guard_pid is not None:
        guard_target = chooser.choose_guard(state)
        if guard_target is not None:
            _require_alive(state, guard_target, "守卫守护")
            # 规则：不能连续两晚守同一个人。否则守卫能把自己或核心神职锁死。
            if guard_target == state.last_guarded:
                raise RuleViolation(
                    f"{guard_target}号 上一晚已被守护，守卫不能连续两晚守同一人"
                )
            # 允许守自己——这是标准规则，守卫首夜守自己是常见打法

    # ---------------- 2. 狼人 ----------------
    wolf_target: int | None = None
    if state.wolves_alive():
        wolf_target = chooser.choose_wolf_kill(state)
        if wolf_target is not None:
            _require_alive(state, wolf_target, "狼人刀人")
            # 允许狼刀狼（"自刀"），网杀里这是常见骗药战术

    # ---------------- 3. 女巫 ----------------
    witch_pid = state.alive_holder(Roles.WITCH)
    witch_saved = False
    poison_target: int | None = None

    if witch_pid is not None:
        # 解药：只有"有人被刀"且"解药还在"时才有意义
        if wolf_target is not None and not state.witch_antidote_used:
            witch_saved = chooser.choose_witch_save(state, wolf_target)
            if witch_saved and wolf_target == witch_pid and state.night > 1:
                # 首夜可自救，之后不可。这条是规则不变量，不是板子选项
                raise RuleViolation("女巫只有首夜能自救，之后不能救自己")

        # 毒药：每局一次
        if not state.witch_poison_used:
            poison_target = chooser.choose_witch_poison(state)

        # 同一晚只能用一瓶药。否则女巫等于救一个又毒一个，强度失衡
        if witch_saved and poison_target is not None:
            raise RuleViolation("女巫一晚只能用一瓶药：解药与毒药不能同时使用")

        if poison_target is not None:
            _require_alive(state, poison_target, "女巫下毒")
            if poison_target == witch_pid:
                raise RuleViolation("女巫不能毒自己")

    # ---------------- 4. 预言家 ----------------
    seer_pid = state.alive_holder(Roles.SEER)
    seer_target: int | None = None
    seer_saw_wolf: bool | None = None
    if seer_pid is not None:
        seer_target = chooser.choose_seer_check(state)
        if seer_target is not None:
            _require_alive(state, seer_target, "预言家验人")
            if seer_target == seer_pid:
                raise RuleViolation("预言家不能验自己")
            # 验的是阵营，不是具体身份——预言家只知道"是/不是狼"
            seer_saw_wolf = state.player(seer_target).faction is Faction.WOLF

    # ---------------- 5. 结算死亡 ----------------
    # 这里集中放所有"一晚死多人"的组合规则，是夜晚模块的核心
    causes: dict[int, DeathCause] = {}

    if wolf_target is not None:
        guarded = guard_target == wolf_target
        if guarded and witch_saved:
            # 【同守同救】守卫和女巫同时作用在同一个人身上 -> 救不回来，反而死
            # 这是狼人杀最经典的负向叠加规则，必须单独测
            causes[wolf_target] = DeathCause.WOLF_KILL
        elif guarded or witch_saved:
            # 只被守 或 只被救 -> 活
            pass
        else:
            causes[wolf_target] = DeathCause.WOLF_KILL

    if poison_target is not None:
        # 毒药无视守卫与解药，必然死亡。
        # 若同一人既被刀又被毒，判为"毒"——影响猎人能否开枪：
        # 中文规则里"被毒不能开枪"，所以死因判定成 poison 是保守且正确的选择。
        causes[poison_target] = DeathCause.POISON

    deaths = apply_deaths(state, causes, label, shot_chooser)

    # ---------------- 6. 收拾夜间副作用 ----------------
    if witch_saved:
        state.witch_antidote_used = True
    if poison_target is not None:
        state.witch_poison_used = True
    # 无论今晚有没有守人，都要记住；None 表示空守，就不会误判"连守"
    state.last_guarded = guard_target
    # 记下被验过的人，策略层据此避免重复验；写入只发生在引擎侧
    if seer_target is not None:
        state.player(seer_target).checked = True

    return NightResult(
        index=state.night,
        guard_target=guard_target,
        wolf_target=wolf_target,
        witch_saved=witch_saved,
        poison_target=poison_target,
        seer_target=seer_target,
        seer_saw_wolf=seer_saw_wolf,
        deaths=tuple(deaths),
    )
