"""板子配置：模型定义与加载。

本模块只负责一件事——
把一份 board JSON 变成可信的 Board 对象，或者在不可信时一次性说清原因。

不负责：
  · 这一局谁死谁活（state.py）
  · 夜晚结算顺序（night.py）
  · 胜负判定（win.py）
"""

import json
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engine.roles import ROLE_REGISTRY, Faction, Roles

BOARDS_DIR = Path(__file__).resolve().parent.parent / "boards"

# 人数合法性区间。太小打不起来，太大单局 LLM 成本失控。
MIN_PLAYERS = 6
MAX_PLAYERS = 18

# 狼人占比的健康区间。超出只告警，不拦——平衡性属设计取舍。
WOLF_RATIO_MIN = 0.25
WOLF_RATIO_MAX = 0.45


class WinCondition(StrEnum):
    """胜负判定方式。"""

    S_SIDE = "slaughter_side"  # 屠边：神全灭或民全灭，狼人胜
    S_ALL = "slaughter_all"  # 屠城：好人全灭，狼人胜


class BoardOptions(BaseModel):
    """板子级规则开关。

    判定标准只有一句：这个开关在不同板子之间会取不同的值吗？
    会变的才放进来。

    全项目统一的不变量——女巫首夜可自救而之后不能、被毒不能发动死亡技能、
    守卫不能连续两晚守同一人、平票走 PK——属于 engine/constants.py 和
    各自的结算函数，不放这里。

    理由：不可配置的东西写成常量和单测，比写成配置项更安全。
    配置项意味着它可能被改错，而单测意味着改错会红。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 是否开启警长竞选。false 时第一天不走上警流程，直接入夜。
    sheriff: bool = False

    # 胜负判定方式。默认屠边，标准网杀板子都用屠边。
    win_condition: WinCondition = WinCondition.S_SIDE


class Board(BaseModel):
    """一份可信的板子。

    构造出来就代表已通过全部校验，所以下游（D3 结算、D4 仿真、D15 接口）
    可以直接信任它的字段，不需要再判一遍。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    # 键是 Roles 枚举成员：Pydantic 负责把 JSON 里的 "wolf" 转成 Roles.WOLF，
    # 同时自动挡掉"身份名拼错"和"数量小于 1"这两类错误。
    roles: dict[Roles, Annotated[int, Field(ge=1, le=20)]]
    options: BoardOptions = Field(default_factory=BoardOptions)

    @property
    def total_players(self) -> int:
        return sum(self.roles.values())

    def count(self, role: Roles) -> int:
        return self.roles.get(role, 0)

    def has(self, role: Roles) -> bool:
        return role in self.roles

    def faction_counts(self) -> dict[Faction, int]:
        """按阵营聚合人数。屠边判定和平衡性检查都靠它。"""
        counts: dict[Faction, int] = dict.fromkeys(Faction, 0)
        for role, n in self.roles.items():
            counts[ROLE_REGISTRY[role].faction] += n
        return counts


class BoardConfigError(Exception):
    """板子配置不可用。

    problems 里是**全部**问题，不是第一个。改配置的人不该试 8 次才发现 8 个错。
    """

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = list(problems)
        super().__init__(f"{source}: " + "; ".join(self.problems))


def _collect_problems(board: Board) -> list[str]:
    """收集所有"逻辑上打不起来"的问题。返回空列表表示通过。"""
    problems: list[str] = []
    present = set(board.roles)
    total = board.total_players

    # 1. 总人数区间
    if not MIN_PLAYERS <= total <= MAX_PLAYERS:
        problems.append(f"总人数 {total} 不在 {MIN_PLAYERS}-{MAX_PLAYERS} 之间")

    # 2. 必须有狼
    if not any(ROLE_REGISTRY[r].faction is Faction.WOLF for r in present):
        problems.append("板子里没有狼人")

    # 3. 唯一性 + 技能依赖（都来自注册表，加新身份不用改这里）
    for role, n in board.roles.items():
        spec = ROLE_REGISTRY[role]
        if spec.unique and n > 1:
            problems.append(f"{role.value} 是唯一身份，却配了 {n} 个")
        missing = spec.requires - present
        if missing:
            names = "、".join(sorted(r.value for r in missing))
            problems.append(f"{role.value} 依赖 {names}，但板子里没有（技能无目标）")

    # 4. 屠边的两个分支都要能成立
    if board.options.win_condition is WinCondition.S_SIDE:
        factions = board.faction_counts()
        if factions[Faction.GOD] == 0:
            problems.append("屠边需要至少 1 个神职，否则'神全灭'分支永远不成立")
        if factions[Faction.VILLAGER] == 0:
            problems.append("屠边需要至少 1 个平民，否则'民全灭'分支永远不成立")

    return problems


def _collect_warnings(board: Board) -> list[str]:
    """软告警：不拦，只提醒。平衡性是设计取舍，不是错误。"""
    messages: list[str] = []
    total = board.total_players
    if total <= 0:
        return messages

    ratio = board.faction_counts()[Faction.WOLF] / total
    if not WOLF_RATIO_MIN <= ratio <= WOLF_RATIO_MAX:
        messages.append(
            f"狼人占比 {ratio:.0%} 不在 {WOLF_RATIO_MIN:.0%}-{WOLF_RATIO_MAX:.0%}，"
            f"平衡性可能有问题"
        )
    return messages


def parse_board(raw: object, *, source: str = "<memory>") -> Board:
    """纯内存校验：任何 JSON 反序列化结果 -> Board。

    不碰文件，所以测试反例时不用创建临时文件。
    分两步：Pydantic 管"形状"，_collect_problems 管"语义"。
    """
    try:
        board = Board.model_validate(raw)
    except ValidationError as exc:
        shape_problems: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"]) or "<root>"
            shape_problems.append(f"{loc}: {err['msg']}")
        raise BoardConfigError(source, shape_problems) from exc

    problems = _collect_problems(board)
    if problems:
        raise BoardConfigError(source, problems)

    for message in _collect_warnings(board):
        warnings.warn(f"{source}: {message}", stacklevel=2)

    return board


def load_board(path: Path) -> Board:
    """从文件加载。

    比 parse_board 多管两件只有拿到路径才判得了的事：
    文件读不出来怎么报、id 与文件名是否一致。
    """
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BoardConfigError(str(path), [f"读取失败：{exc}"]) from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BoardConfigError(
            str(path),
            [f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行第 {exc.colno} 列）"],
        ) from exc

    board = parse_board(raw, source=str(path))

    if board.id != path.stem:
        raise BoardConfigError(str(path), [f"id={board.id!r} 与文件名 {path.stem!r} 不一致"])

    return board


def load_board_by_id(board_id: str, boards_dir: Path = BOARDS_DIR) -> Board:
    """按 id 加载板子。D4 的批量仿真和 D15 的列表接口会用到。"""
    boards_dir = Path(boards_dir)
    path = boards_dir / f"{board_id}.json"

    if not path.is_file():
        available = sorted(p.stem for p in boards_dir.glob("*.json"))
        raise BoardConfigError(
            str(path),
            [f"板子 {board_id!r} 不存在，可用：{'、'.join(available) or '（空）'}"],
        )

    return load_board(path)
