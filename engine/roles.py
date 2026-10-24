"""身份定义 —— 静态世界观的单一来源。

本模块只回答一个问题："某个身份是什么。"

不回答：
  · 某个板子有哪些身份、各几个（那是 boards/*.json）
  · 这一局谁死谁活（那是 state.py）
  · 夜晚按什么顺序结算（那是 night.py）

身份 id 一律小写。它会一路流到 API 返回、数据库、前端和 prompt，
统一小写能少一整类"大小写对不上"的 bug。
"""

from dataclasses import dataclass
from enum import Enum, StrEnum


class Faction(Enum):
    """阵营。胜负按阵营聚合判定，不按具体身份。

    刻意用 Enum 而不是 StrEnum：身份和阵营的取值有重叠（都是 "wolf" / "villager"），
    如果两者都是 str 子类，Faction.WOLF == Roles.WOLF 会成立，把它们当 dict 键时
    会互相覆盖。用普通 Enum 保证两者永不相等，需要字符串时取 .value。
    """

    WOLF = "wolf"  # 狼人阵营
    GOD = "god"  # 神职，属好人阵营
    VILLAGER = "villager"  # 平民，属好人阵营


class Roles(StrEnum):
    """全部身份。板子只能从这里挑，不能凭空造。"""

    WOLF = "wolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"


class DeathAbility(StrEnum):
    """死亡时能发动的技能。D3 的夜晚结算用它决定谁死了还能做事。"""

    NONE = "none"
    # 开枪带走一名存活玩家。注意：被毒杀时不能发动（规则见 state.SHOOTABLE_CAUSES）
    SHOOT = "shoot"


@dataclass(frozen=True)
class RoleSpec:
    """一个身份的静态定义。

    requires 做成数据，而不是写在校验器里的 if：
    以后加"骑士""白痴"这类新身份时，只改这张注册表，校验器一行不动。
    """

    name_zh: str
    faction: Faction
    # 全局最多 1 个。狼人、平民可多个。
    unique: bool = True
    # 该板子若包含本身份，必须同时包含这些身份。
    requires: frozenset[Roles] = frozenset()
    # 死亡技能。目前只有猎人有。
    death_ability: DeathAbility = DeathAbility.NONE


ROLE_REGISTRY: dict[Roles, RoleSpec] = {
    Roles.WOLF: RoleSpec("狼人", Faction.WOLF, unique=False),
    Roles.VILLAGER: RoleSpec("平民", Faction.VILLAGER, unique=False),
    # 没有狼，预言家验谁？所以依赖狼。
    Roles.SEER: RoleSpec("预言家", Faction.GOD, requires=frozenset({Roles.WOLF})),
    # 没有狼，女巫的毒药和解药都没有目标。
    Roles.WITCH: RoleSpec("女巫", Faction.GOD, requires=frozenset({Roles.WOLF})),
    Roles.HUNTER: RoleSpec("猎人", Faction.GOD, death_ability=DeathAbility.SHOOT),
    Roles.GUARD: RoleSpec("守卫", Faction.GOD),
}
