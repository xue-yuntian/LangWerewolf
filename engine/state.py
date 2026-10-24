"""游戏状态：谁在场、谁活着、谁怎么死的。

本模块只管**状态**，不管规则怎么算（night.py / vote.py）和谁赢（win.py）。

三条设计原则：
  1. 状态只由 kill() 改变，不允许外部直接改 player.alive
  2. 死因（DeathCause）是一等公民——猎人被毒能不能开枪全靠它
  3. 状态不引用 Board 之外的配置，规则常量集中在下面几个 frozenset
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from random import Random

from engine.board import Board
from engine.roles import ROLE_REGISTRY, DeathAbility, Faction, Roles


class RuleViolation(Exception):
    """玩家动作违反规则（守护连守、女巫毒解同晚、投票给死者……）。

    单独一个异常类型，是为了让测试能精确断言"被规则拦住了"，
    而不是笼统地捕获 ValueError。
    """


class DeathCause(StrEnum):
    """死因。决定死亡技能能否发动。"""

    WOLF_KILL = "wolf_kill"  # 被狼刀死
    POISON = "poison"  # 被女巫毒死
    VOTE = "vote"  # 白天被投出局
    SHOT = "shot"  # 被猎人开枪带走


# 能发动死亡技能的死因。
# 规则：被毒杀的猎人不能开枪（毒药压制一切被动技能），其余死因都可以。
# 这条是**不变量**，不是配置项——所以写成常量 + 单测，不给板子留开关。
SHOOTABLE_CAUSES: frozenset[DeathCause] = frozenset(
    {DeathCause.WOLF_KILL, DeathCause.VOTE, DeathCause.SHOT}
)


@dataclass(frozen=True)
class Death:
    """一次死亡记录。按发生顺序进 GameState.deaths，复盘时直接读。"""

    pid: int
    cause: DeathCause
    label: str  # "第2夜" / "第3天"

    def __str__(self) -> str:
        return f"{self.pid}号 死于 {self.cause.value}（{self.label}）"


@dataclass
class Player:
    """一名玩家。pid 为 1..N，就是局内座位号。"""

    pid: int
    role: Roles
    alive: bool = True

    @property
    def faction(self) -> Faction:
        return ROLE_REGISTRY[self.role].faction

    @property
    def death_ability(self) -> DeathAbility:
        return ROLE_REGISTRY[self.role].death_ability


@dataclass
class GameState:
    """一局游戏的全部可变状态。

    刻意不用 frozen：这是一局进行中的状态，本来就该变。
    但所有死亡必须走 kill()，这样 deaths 列表和 alive 标记不会不一致。
    """

    board: Board
    players: dict[int, Player]
    night: int = 0  # 已经过完的夜晚数
    day: int = 0  # 已经过完的白天数

    # 女巫的药，一辈子各一次
    witch_antidote_used: bool = False
    witch_poison_used: bool = False

    # 守卫上一晚守了谁。用于"不能连续两晚守同一人"
    last_guarded: int | None = None

    deaths: list[Death] = field(default_factory=list)

    # ---------------- 查询（都不改状态） ----------------

    def is_alive(self, pid: int) -> bool:
        player = self.players.get(pid)
        return player is not None and player.alive

    def player(self, pid: int) -> Player:
        return self.players[pid]

    def alive_ids(self) -> list[int]:
        """存活玩家，按座位号升序。发言顺序、投票顺序都用它。"""
        return [pid for pid, p in sorted(self.players.items()) if p.alive]

    def alive_count(self) -> int:
        return len(self.alive_ids())

    def alive_of(self, faction: Faction) -> list[int]:
        """某个阵营的存活玩家。屠边判定靠它。"""
        return [
            pid for pid, p in sorted(self.players.items()) if p.alive and p.faction is faction
        ]

    def wolves_alive(self) -> list[int]:
        return self.alive_of(Faction.WOLF)

    def role_holders(self, role: Roles) -> list[int]:
        """持有某身份的玩家（不管死活）。"""
        return [pid for pid, p in sorted(self.players.items()) if p.role is role]

    def sole_holder(self, role: Roles) -> int | None:
        """唯一身份的持有者；板子里没有这个身份则返回 None。

        只返回"还在场"的持有者。已死不返回，因为死了就不能行动了。
        """
        holders = self.role_holders(role)
        if not holders:
            return None
        return holders[0]

    def alive_holder(self, role: Roles) -> int | None:
        """唯一身份是否还活着，活着返回 pid。夜里行动前先问它。"""
        pid = self.sole_holder(role)
        return pid if pid is not None and self.is_alive(pid) else None

    # ---------------- 变更 ----------------

    def kill(self, pid: int, cause: DeathCause, label: str) -> Death:
        """标记一名玩家出局，并记录死因。

        调用方不要自己改 player.alive——统一走这里，
        保证 deaths 列表顺序就是真实发生顺序（复盘时间轴依赖它）。
        """
        player = self.players[pid]
        if not player.alive:
            raise RuleViolation(f"{pid}号 已经出局，不能再死一次")
        player.alive = False
        death = Death(pid=pid, cause=cause, label=label)
        self.deaths.append(death)
        return death


def create_state(board: Board, rng: Random) -> GameState:
    """按板子发牌开局。

    座位号 1..N 随机分配身份。用传入的 rng 而不是全局 random，
    这样测试传 Random(42) 就能复现同一局——D11 的批量仿真也靠这个。
    """
    deck: list[Roles] = []
    for role, count in board.roles.items():
        deck.extend([role] * count)
    rng.shuffle(deck)
    players = {pid: Player(pid=pid, role=role) for pid, role in enumerate(deck, start=1)}
    return GameState(board=board, players=players)


def apply_deaths(
    state: GameState,
    causes: dict[int, DeathCause],
    label: str,
    shot_chooser: Callable[[int], int | None],
) -> list[Death]:
    """结算一批死亡，并处理死亡技能（猎人开枪）。

    causes: {pid: 死因}。同一晚可能死多人，同一白天只死一人。
    shot_chooser: 给猎人 pid，返回他开枪打谁；None 表示不开枪。

    为什么要用队列而不是简单的 for 循环：
    猎人开枪可能打死另一个猎人，那个猎人也该能开枪——所以要能连环。
    用队列 + "每人只开一次枪" 保证不会无限循环。
    """
    new_deaths: list[Death] = []
    queue: list[tuple[int, DeathCause]] = list(causes.items())
    already_shot: set[int] = set()

    while queue:
        pid, cause = queue.pop(0)

        # 可能已经被前面的效果弄死了（比如被刀又被毒），跳过
        if not state.is_alive(pid):
            continue

        new_deaths.append(state.kill(pid, cause, label))

        # 死亡技能：只有死亡技能是 SHOOT、且这个死因允许发动时才开枪
        if state.player(pid).death_ability is not DeathAbility.SHOOT:
            continue
        if cause not in SHOOTABLE_CAUSES:
            continue
        if pid in already_shot:  # 防御性：每人只开一次
            continue
        already_shot.add(pid)

        target = shot_chooser(pid)
        if target is not None and state.is_alive(target):
            queue.append((target, DeathCause.SHOT))

    return new_deaths
