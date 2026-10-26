"""测试共享工具（被各 test_*.py 导入，pytest 不会收集它）。

核心原则：**不依赖随机发牌**。
手工指定"1号是狼、2号是守卫"，测试才能精确断言谁死谁活。
随机发牌只用在一整局跑通那种测试里。

所有 *Chooser / *Policy 类都是 D5 / D12 的替身：它们是"按预设答案回答"的假策略，
把**规则结算**和**策略决策**解耦——这正是项目要讲的"规则引擎与智能体分离"。
"""

from collections import Counter
from collections.abc import Sequence
from random import Random

from engine.board import Board, BoardOptions, WinCondition
from engine.roles import Roles
from engine.state import GameState, Player


def make_board(
    counts: dict[Roles, int],
    *,
    sheriff: bool = False,
    win_condition: WinCondition = WinCondition.S_SIDE,
) -> Board:
    """直接构造 Board，跳过文件加载和语义校验（单测要能造出各种极端板子）。"""
    return Board(
        id="t",
        name="测试板",
        roles=counts,
        options=BoardOptions(sheriff=sheriff, win_condition=win_condition),
    )


def make_state(
    roles_by_pid: dict[int, Roles],
    *,
    win_condition: WinCondition = WinCondition.S_SIDE,
) -> GameState:
    """按"座位号 -> 身份"手工搭一个局。"""
    counts = Counter(roles_by_pid.values())
    board = make_board(dict(counts), win_condition=win_condition)
    players = {pid: Player(pid=pid, role=role) for pid, role in roles_by_pid.items()}
    return GameState(board=board, players=players)


class ScriptNight:
    """按预设答案回答夜晚的所有决策。"""

    def __init__(
        self,
        *,
        guard: int | None = None,
        kill: int | None = None,
        save: bool = False,
        poison: int | None = None,
        seer: int | None = None,
    ) -> None:
        self.guard = guard
        self.kill = kill
        self.save = save
        self.poison = poison
        self.seer = seer

    def choose_guard(self, state: GameState) -> int | None:
        return self.guard

    def choose_wolf_kill(self, state: GameState) -> int | None:
        return self.kill

    def choose_witch_save(self, state: GameState, killed: int | None) -> bool:
        return self.save

    def choose_witch_poison(self, state: GameState) -> int | None:
        return self.poison

    def choose_seer_check(self, state: GameState) -> int | None:
        return self.seer


class ScriptVote:
    """按预设映射投票；映射里没有的人弃票。

    引擎传了 only（PK 重投）就改用 pk_mapping，首轮票型和 PK 票型
    可以分开写，不会互相覆盖。
    """

    def __init__(
        self,
        mapping: dict[int, int | None],
        pk_mapping: dict[int, int | None] | None = None,
    ) -> None:
        self.mapping = mapping
        self.pk_mapping = pk_mapping

    def vote(
        self, state: GameState, voter: int, *, only: Sequence[int] | None = None
    ) -> int | None:
        if only is not None:
            return (self.pk_mapping or self.mapping).get(voter)
        return self.mapping.get(voter)


def no_shot(pid: int) -> int | None:
    """猎人不开枪。"""
    return None


def _first_alive_other(state: GameState, pid: int) -> int | None:
    for candidate in state.alive_ids():
        if candidate != pid:
            return candidate
    return None


class FirstAlivePolicy:
    """无脑策略：永远选第一个合法目标。

    它的作用不是"会玩"，而是让 play_game 能跑到底——
    D4 的验收就是"9 人局能分出胜负"。D5 会换成随机机器人。
    """

    def choose_guard(self, state: GameState) -> int | None:
        return None  # 空守，避免触发"连守"违规

    def choose_wolf_kill(self, state: GameState) -> int | None:
        wolves = set(state.wolves_alive())
        for pid in state.alive_ids():
            if pid not in wolves:
                return pid
        return None

    def choose_witch_save(self, state: GameState, killed: int | None) -> bool:
        return False

    def choose_witch_poison(self, state: GameState) -> int | None:
        return None

    def choose_seer_check(self, state: GameState) -> int | None:
        seer = state.alive_holder(Roles.SEER)
        if seer is None:
            return None
        return _first_alive_other(state, seer)

    def vote(
        self, state: GameState, voter: int, *, only: Sequence[int] | None = None
    ) -> int | None:
        # PK 重投时只能投台上的人：无脑策略也得照着这个约束选
        if only is not None:
            for pid in only:
                if pid != voter and state.is_alive(pid):
                    return pid
            return None
        return _first_alive_other(state, voter)

    def hunter_shot(self, state: GameState, hunter: int) -> int | None:
        return _first_alive_other(state, hunter)


def seed(value: int = 42) -> Random:
    """固定种子，保证同一局可复现。"""
    return Random(value)
