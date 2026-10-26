"""人设 —— 让同一张牌在不同座位上打出不同的风格。

────────────────────────────────────────────────────────────────
为什么需要它
────────────────────────────────────────────────────────────────
只有身份信息时，模型会写出"标准答案式"的发言：所有人都一样谨慎、
一样理性、一样先听一轮再表态。九个人都在说"我倾向再观察"，
这场游戏既不好看，AI 之间也不会真的对抗。

人设的作用不是"加调味"，而是**制造策略多样性**：
激进的人先开枪、稳健的人后发力、多疑的人反复横跳——
这样悍跳、归票、对线这些行为才会自然发生。

────────────────────────────────────────────────────────────────
两种来源
────────────────────────────────────────────────────────────────
Personality  六种预制风格，靠 seed 分配。离线、可复现、零成本。
Persona      按用户给的场景现生成，带名字和背景。联网、每局不同。
             生成失败时回落到 Personality（见 agents/director.py）。

两者都只描述"这个人怎么说话"，不携带任何身份信息。导演不知道
牌怎么发，也不该知道——否则人设就成了身份泄漏的通道。

────────────────────────────────────────────────────────────────
一条边界：人设只管风格，不管策略
────────────────────────────────────────────────────────────────
反例（不要这样写）："你是预言家，所以你绝不公布验人结果。"
那会让某个身份直接退出游戏。

正例："你说话习惯先铺结论再给依据，不太在意铺垫是否礼貌。"
风格是**怎么说话**，不是**该不该做某事**。策略由身份和局势决定。

────────────────────────────────────────────────────────────────
一个关键约束：人设必须独立于发牌随机流
────────────────────────────────────────────────────────────────
如果 assign_personalities 和 create_state 共用同一个 rng，
那么以后你改一句人设文案、或者调整人设池的顺序，所有种子都会
发出不同的身份——D12 的 A/B 对照（同一个种子，换 prompt 看胜率变化）
直接失效。

所以这里用独立的种子偏移 PERSONALITY_SEED_OFFSET。
"""

from collections.abc import Sequence
from enum import StrEnum
from random import Random

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

# 人设随机流相对发牌随机流的种子偏移。
# 和 batch.POLICY_SEED_OFFSET 同理：两条流必须互不干涉。
PERSONALITY_SEED_OFFSET = 2_000_003


class Personality(StrEnum):
    """人设类型。id 用小写，文案单独放 _TEXT，两者分开便于以后做多语言。"""

    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    SUSPICIOUS = "suspicious"
    PERFORMER = "performer"
    ANALYST = "analyst"
    LOW_KEY = "low_key"


_TEXT: dict[Personality, str] = {
    Personality.AGGRESSIVE: (
        "你性子急、敢打敢冲，习惯第一个给场上定调。你倾向用最直接的措辞指认目标，"
        "被人反驳时不会退，反而会顺着对方的逻辑继续压。"
        "你相信压力能逼出破绽，但也容易被别人借力打成抗推位——你自己知道这个风险。"
    ),
    Personality.CAUTIOUS: (
        "你偏稳健，不喜欢在信息不足时表态。"
        "你习惯先听完一轮再决定站边，只在手上至少有两个独立依据时才明确指人。"
        "你的表达偏保守，常说“我再听听” ，代价是可能被归为划水。"
    ),
    Personality.SUSPICIOUS: (
        "你天生多疑，对每个人的发言都先找漏洞。"
        "你经常注意到别人忽略的细节——前后矛盾、改口、投票动机不一致。"
        "你的缺点是容易过度解读，把好人之间的正常分歧当成狼队配合。"
    ),
    Personality.PERFORMER: (
        "你享受场面，发言有起伏、会造梗、会反问。"
        "你擅长把复杂局面讲得生动，也擅长用情绪感染别人站你这边。"
        "你的风险是表演过头——明明没依据却说得像有铁证，被拆穿时很难看。"
    ),
    Personality.ANALYST: (
        "你习惯把局面拆成结构和票型来分析，喜欢排顺序、算轮次、列可能性。"
        "你的发言条理清楚但偏长，有时会被人嫌啰嗦。"
        "你相信逻辑能赢，但容易低估情绪和误导的作用。"
    ),
    Personality.LOW_KEY: (
        "你话不多，习惯点到为止。你不喜欢长篇大论，倾向用一两句话表明立场就收。"
        "你的信息量不一定少，但存在感偏低，容易被忽略、也容易被当成没贡献。"
        "你需要在关键时刻主动开口，否则关键的票会漏掉。"
    ),
}


# 每个字段的长度上限。模型很爱写小作文：一个人设四百字，九个人就是
# 三千六，而每个玩家每次调用都要带上自己那份。这个上限不是为了好看。
_FIELD_LIMITS: dict[str, int] = {
    "name": 8,
    "background": 40,
    "traits": 60,
    "resilience": 50,
    "style": 50,
    "logic": 50,
    "quirk": 20,
}


class Persona(BaseModel):
    """一个具体的人。按场景现生成，也可以手写进配置。

    和 Personality 的区别：Personality 是预制的六种风格，靠 seed 分配，
    同一种子永远是同一种；Persona 带名字和背景，每局可以完全不同。

    所有字段都会被压成单行并截断（_FIELD_LIMITS），所以这里不
    拒绝超长输入——模型多写两句是常态，拒掉整条才是浪费。
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    traits: str
    resilience: str
    style: str
    logic: str
    background: str = ""
    quirk: str = ""

    @field_validator("*")
    @classmethod
    def _clip(cls, value: str, info: ValidationInfo) -> str:
        """压掉换行和超长。字段名不在 _FIELD_LIMITS 里就用 60。"""
        limit = _FIELD_LIMITS.get(info.field_name or "", 60)
        return " ".join(value.split())[:limit]

    def to_text(self) -> str:
        """拼成 prompt 里那一段，字段顺序和生成时的提示词一致。"""
        head = f"你叫{self.name}。{self.background}".rstrip()
        lines = [
            head,
            f"性格：{self.traits}",
            f"抗压能力：{self.resilience}",
            f"说话习惯：{self.style}",
            f"推理能力：{self.logic}",
        ]
        if self.quirk:
            lines.append(f"口头禅：「{self.quirk}」")
        return "\n".join(lines)


def personality_text(kind: Persona | Personality) -> str:
    """把人设渲染成一段 prompt 文本。两种来源都走这里。"""
    if isinstance(kind, Persona):
        return kind.to_text()
    return _TEXT[kind]


def assign_personalities(
    pids: Sequence[int],
    rng: Random,
) -> dict[int, Personality]:
    """给每个座位分配一个人设。

    先洗牌再按位置取，所以前几个座位一定不重复——这避免了
    一整桌全是"多疑派"那种灾难。人数超过人设种类时循环复用。

    rng 必须是一个**独立于发牌**的随机源，通常这样构造：
        Random(seed + PERSONALITY_SEED_OFFSET)
    """
    pool = list(Personality)
    rng.shuffle(pool)
    return {pid: pool[index % len(pool)] for index, pid in enumerate(pids)}
