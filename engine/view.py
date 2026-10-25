"""玩家视角 —— 把引擎状态压缩成"这个玩家此刻知道什么"，供 D12 拼 prompt。

只读：这里拼出来的文本不会回写 GameState。
所有说明性文本（身份能力、术语表）都放在本模块，模型的行为约束只在这里维护一份。
"""

from collections.abc import Iterable
from dataclasses import dataclass

from engine.roles import ROLE_REGISTRY, Roles

# 身份说明直接喂给模型，写的是"能做什么 + 规则限制 + 白天怎么打"
roles_ability: dict[Roles, str] = {}

roles_ability[Roles.WOLF] = (
    "狼人属于坏人阵营，你的目的是杀光场上所有村民或所有神职来取得胜利。"
    "场上还有另外几个狼人作为你的队友，你们在游戏进行到夜晚的时候"
    "可以商讨杀死场上仍然活着的任何一名玩家。"
    "游戏进行到白天的时候，你需要干扰场上玩家的判断，以伪装成一名好人，"
    "尽量不要暴露你是一名狼人这件事。"
)
roles_ability[Roles.VILLAGER] = (
    "你属于好人阵营的平民，没有任何夜晚技能，白天只能靠听发言、看票型判断谁是狼人。"
    "你的票就是你的武器：每一轮都要投出你怀疑的狼人，被追问时把理由讲清楚。"
    "拿到平民牌时不要因为没信息就沉默或弃票——好人少一票，狼人就多一分胜率。"
)
roles_ability[Roles.SEER] = (
    "你属于好人阵营的预言家，每晚可以查验一名存活玩家的阵营，"
    "得到的结果只有「狼人」和「好人」，查不出具体身份。"
    "你是好人阵营最有力的信息来源，你需要评估你自己已知信息和自身安全两者之间的轻重关系。"
    "白天要挑合适时机报出验人结果和验人顺序，让好人能顺着验人链往下推。"
    "如果有人对跳预言家，要主动对线，把验人逻辑讲得比对方更可信。"
)
roles_ability[Roles.WITCH] = (
    "你属于好人阵营的女巫，有一瓶解药和一瓶毒药，各只能用一次，同一晚也只能用其中一瓶。"
    "首夜被刀可以自救，之后夜里被刀不能自救。"
    "解药优先留给预言家、猎人这类关键神职，毒药优先毒已经确认的狼人，"
    "毒错一瓶往往直接断送好人的节奏。"
    "两瓶药都用完之后你就等于平民，要提前把知道的用药信息交代清楚。"
)
roles_ability[Roles.HUNTER] = (
    "你属于好人阵营的猎人，活着的时候没有主动技能，"
    "但只要被狼人刀死或被投票出局，就能开枪带走一名存活玩家——被女巫毒死时不能开枪。"
    "正因为死后还能换走一个人，狼人通常不敢先刀你，"
    "你可以利用这一点在白天承担火力、逼狼人表态；"
    "同时要提前想好开枪目标，别把枪口对准已经确认的好人。"
)
roles_ability[Roles.GUARD] = (
    "你属于好人阵营的守卫，每晚可以守护一名存活玩家（包括你自己），"
    "被守的人当晚不会被狼刀死，但毒药无视你的守护。"
    "两条限制：不能连续两晚守同一个人；"
    "如果守卫和女巫的解药同时落在同一个人身上（同守同救），这个人依然会死。"
    "守人思路一般是保预言家、保猎人，或者在自己被怀疑时守自己；"
    "白天留意女巫透露的用药信息，尽量别和她的解药撞车。"
)


# 术语表：模型读事件流和别人的发言时会遇到这些词。
# 只描述本局真实存在的规则；没实现的机制要写明"没有"，否则模型会照网杀套路乱用。
glossary: dict[str, str] = {}

# ---------------- 夜晚 ----------------
glossary["夜晚顺序"] = (
    "每一夜的结算顺序固定为：守卫 → 狼人 → 女巫 → 预言家 → 结算死亡。"
    "女巫排在狼人之后，所以她知道今晚谁被刀。"
)
glossary["狼刀"] = "狼队每晚共同决定刀死一名存活玩家。可以刀自己的队友（自刀），也可以选择空刀。"
glossary["自刀"] = "狼人故意刀自己的队友，用来骗女巫的解药，或者制造好人内讧。"
glossary["空刀"] = "狼队当晚不刀任何人，通常是劣势局用来拖轮次。"
glossary["空守"] = "守卫当晚不守任何人。"
glossary["连守"] = "规则禁止守卫连续两晚守同一个人；空守一晚之后，下一晚又可以守他。"
glossary["同守同救"] = (
    "守卫和女巫的解药同时落在同一个人身上时救不回来，这个人反而会死。"
    "所以守卫要尽量避开女巫打算救的人。"
)

# ---------------- 女巫 ----------------
glossary["解药"] = "救回当晚被狼刀的人，一局一次；首夜可以自救，之后夜里被刀不能自救。"
glossary["毒药"] = "毒死任意一名存活玩家（不能毒自己），无视守卫和解药，一局一次。"
glossary["银水"] = "被女巫解药救下来的人。"

# ---------------- 预言家 ----------------
glossary["查验"] = (
    "预言家每晚验一名存活玩家，只能得到「狼人」或「好人」，验不出具体身份，也不能验自己。"
)
glossary["验人链"] = (
    "预言家报出的「第几夜验了谁、结果是什么」的完整顺序，好人靠它判断哪个预言家是真的。"
)
glossary["金水"] = "被预言家验出「好人」的玩家。"
glossary["查杀"] = "被预言家验出「狼人」的玩家。"

# ---------------- 猎人 ----------------
glossary["开枪"] = "猎人被狼刀或被投票出局后，可以带走一名存活玩家；被毒死时不能开枪。"

# ---------------- 白天投票 ----------------
glossary["弃票"] = "选择不投票，计 0 票。全体弃票时当天无人出局，也不会进入 PK。"
glossary["平票"] = "最高票并列，例如两个人同为 3 票。"
glossary["PK"] = "平票后进入的加时投票：台上平票的人不能投票，由台下存活玩家再投一轮。"
glossary["再平票"] = "PK 轮仍然并列，或者台下全员弃票，当天无人出局。"

# ---------------- 胜负 ----------------
glossary["屠边"] = "神职全灭、或平民全灭，狼人获胜；狼人全灭，好人获胜。"
glossary["屠城"] = "好人（神职加平民）全部阵亡，狼人才获胜。"

# ---------------- 发言用语：引擎不校验，但听发言判断局势要用 ----------------
glossary["起跳"] = "公开声称自己是某个身份，例如「我起跳预言家」。"
glossary["对跳"] = "两个人同时声称自己是同一个神职，最常见的是双预言家。"
glossary["悍跳"] = "狼人假冒预言家起跳，用编造的验人结果抢话语权。"
glossary["归票"] = "号召大家把票集中投到自己指定的人身上。"
glossary["踩"] = "明确表示某人是狼。"
glossary["保"] = "明确表示某人是好人。"
glossary["划水"] = "发言不表态、不给信息、不站边，等于没说话。"
glossary["抿身份"] = "通过别人的发言和语气推断他的真实身份。"
glossary["狼坑"] = "当前怀疑是狼的那几个人的集合。"
glossary["抗推"] = "被推上票台、快要被投出去的人。"
glossary["倒钩"] = "狼人故意站到真预言家或好人这边，骗取信任后再翻脸。"
glossary["深水狼"] = "藏得很深、发言跟好人几乎没区别的狼。"

# ---------------- 本版本没有的机制：写出来是为了让模型不要尝试 ----------------
glossary["警长"] = "D8 才实现的角色。当前版本没有警长竞选，也没有 1.5 票。"
glossary["警徽流"] = "警长临死前用警徽指示验人顺序的约定。没有警长，所以不存在。"
glossary["自爆"] = "狼人白天公开身份、直接结束当天的操作。当前版本未实现。"
glossary["发言记录"] = "当前版本还没有发言记录，只能根据夜间动作和投票判断局势。"


def glossary_text() -> str:
    """把术语表拼成 prompt 里的一段纯文本。"""
    return "\n".join(f"- {term}：{desc}" for term, desc in glossary.items())


def _join_pids(pids: Iterable[int]) -> str:
    """座位号拼成「1号、2号」；空集合返回「无」，避免 prompt 里出现 []。"""
    return "、".join(f"{pid}号" for pid in pids) or "无"


def _join_dead(dead: Iterable[tuple[int, str]]) -> str:
    """死者和死因拼成「1号（第2夜 被狼刀）」。"""
    return "、".join(f"{pid}号（{reason}）" for pid, reason in dead) or "无"


def _join_lines(lines: Iterable[str]) -> str:
    """每条信息占一行，避免模型把 list 的 repr 读成一整段。"""
    return "\n".join(f"- {line}" for line in lines)


@dataclass(frozen=True)
class PlayerView:
    pid: int
    roles: Roles
    day: int
    alive: list[int]
    dead: list[tuple[int, str]]
    teammates: list[int]  # 狼队友；非狼返回空
    private_lines: list[str]  # 只有他知道的（验人结果、用药、狼刀）
    public_lines: list[str]  # 全场可见的历史
    last_speeches: list[str]  # 上一轮发言（D12 之后才有）
    personality: str  # 角色设定，由开局之前的用户设定生成
    notes: list[str]  # 玩家笔记

    def to_prompt_text(self) -> str:
        # 用 blocks + join 拼，段落之间固定空一行；空字段直接不产生段落
        intro = "你正在进行一局狼人杀游戏。"
        if self.personality:
            intro += f"\n你的身份特征如下：\n{self.personality}"

        blocks = [
            intro,
            f"你的身份是：{ROLE_REGISTRY[self.roles].name_zh}（{self.roles}）\n"
            f"你的能力是：{roles_ability[self.roles]}",
            "以下是游戏术语的解释：\n" + glossary_text(),
            f"当前是第{self.day}天，以下是你知道的玩家信息：\n"
            f"存活玩家：{_join_pids(self.alive)}\n"
            f"死亡玩家：{_join_dead(self.dead)}",
        ]
        if self.teammates:
            blocks.append(f"你的队友是：{_join_pids(self.teammates)}")
        if self.private_lines:
            blocks.append("以下是你知道的私密信息：\n" + _join_lines(self.private_lines))
        if self.public_lines:
            blocks.append("以下是你知道的公开信息：\n" + _join_lines(self.public_lines))
        if self.notes:
            blocks.append("以下是你自己的笔记：\n" + _join_lines(self.notes))
        if self.last_speeches:
            blocks.append("以下是你上一轮的发言：\n" + _join_lines(self.last_speeches))
        blocks.append("注意，你的发言不能冗长，必须简明说明你想讲出的逻辑和结论。没有有效信息时可以选择过。")
        return "\n\n".join(blocks)
