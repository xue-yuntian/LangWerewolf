"""人设导演 —— 开局前把一句场景变成一桌人。

────────────────────────────────────────────────────────────────
它做什么
────────────────────────────────────────────────────────────────
用户给一句场景（"1930 年代上海滩，一家快要倒闭的证券交易所"），
这里让模型生成 N 个具体的人：名字、处境、性格、抗压能力、
措辞习惯、推理能力。结果直接进 LLMAgentPolicy，每个玩家在
prompt 里只看到自己那份。

────────────────────────────────────────────────────────────────
它不做什么
────────────────────────────────────────────────────────────────
不碰身份。导演不知道牌怎么发，撞上也不该知道——人设和发牌是
两条独立的路，一旦混在一起，"人设"就会变成"身份泄漏"。
座位号按 1..N 顺序分配，和 create_state 的 pid 一一对应。

────────────────────────────────────────────────────────────────
失败怎么办
────────────────────────────────────────────────────────────────
返回空 dict，调用方回落到 Personality 的六种预制风格。一局
100+ 次调用的游戏，不该因为开局前这一次创作失败就起不来。
生成里单条人设坏了（缺字段、名字为空）也只丢那一条，剩下的
照用；不够的座位用预制风格补——桌面上不能有人没有性格。
"""

from random import Random

from pydantic import ValidationError

from agents.personality import Persona, Personality, assign_personalities
from llm import LLMError, chat_json

# 场景是给人看的一句话，不是文档。截断是为了让 prompt 长度可预期。
_MAX_SCENE = 200

SYSTEM = (
    "你是狼人杀对局的选角导演。你的任务不是写设定集，而是设计一桌"
    "吵得起来的人：彼此看不顺眼、说话方式对不上、遇到压力时反应不同。"
    "九个都冷静理性的人，这局游戏会很难看。"
)


def _build_prompt(scene: str, count: int) -> str:
    """拼生成提示词。

    几条要求不是文风偏好：
      · "不要提身份"——模型一旦写了"他是狼人"，人设就泄漏了发牌结果
      · "不要写策略"——和 personality._TEXT 顶部那条边界同源
      · "每个字段不超过 40 字"——九个人设每次调用都要重发一份，
        不设上限就等于把 token 预算交给模型的心情
    """
    return (
        f"【场景】\n{scene}\n\n"
        f"请为 {count} 位玩家各写一份人设。\n\n"
        "硬性要求：\n"
        f"1. 一共 {count} 个，一个不多一个不少；\n"
        "2. 名字用中文，两到三个字，互不重复；\n"
        "3. 性格必须能在发言里听出区别，不要出现两个都「沉稳理性」的人；\n"
        "4. 只写「这个人怎么说话、怎么反应」，不要写任何策略，"
        "也不要提狼人杀身份（谁是什么身份由发牌决定，不归你管）；\n"
        "5. 每个字段不超过 40 字，不要写小作文。\n\n"
        "字段说明：\n"
        "- name：名字\n"
        "- background：在这个场景里的身份或处境，一句话\n"
        "- traits：性格，一到两句\n"
        "- resilience：抗压能力——被怀疑、被围攻、被冤枉时会怎么表现\n"
        "- style：措辞习惯——句子长短、爱不爱反问、用不用书面语\n"
        "- logic：推理能力——强 / 中 / 弱，以及具体表现\n"
        "- quirk：一句口头禅，可以为空\n\n"
        '只输出 JSON，格式：{"personas": [{"name": "...", "background": "...", '
        '"traits": "...", "resilience": "...", "style": "...", "logic": "...", '
        '"quirk": "..."}]}'
    )


def generate_personas(
    scene: str,
    count: int,
    *,
    model: str,
    temperature: float = 1.0,
) -> dict[int, Persona | Personality]:
    """按场景生成 count 个人设，座位号从 1 开始连续。

    返回空 dict 表示彻底失败（场景为空、调用失败、结构不对），
    调用方改用 assign_personalities。

    temperature 给到 1.0：这是创作任务，要的是发散。对局里的
    vote / speak 用 0.7，那是判断任务，两者不该共用一个值。
    """
    scene = " ".join(scene.split())[:_MAX_SCENE]
    if not scene or count <= 0:
        return {}

    try:
        payload = chat_json(
            SYSTEM,
            _build_prompt(scene, count),
            model=model,
            temperature=temperature,
        )
    except LLMError:
        return {}

    raw = payload.get("personas")
    if not isinstance(raw, list):
        return {}

    personas: list[Persona] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            persona = Persona.model_validate(item)
        except ValidationError:
            # 缺字段的一条直接跳过，不能让一个人毁掉整桌
            continue
        if not persona.name:
            continue
        personas.append(persona)
        if len(personas) == count:
            break

    if not personas:
        return {}

    return _seat(personas, count)


def _seat(personas: list[Persona], count: int) -> dict[int, Persona | Personality]:
    """按顺序坐到 1..count 号，空位用预制人设补。

    补位用固定种子，所以同一次失败的结果可复现——调试
    "为什么 9 号总是同一个风格"时，随机补位会是个麻烦。
    """
    seated: dict[int, Persona | Personality] = {
        index + 1: persona for index, persona in enumerate(personas)
    }
    missing = list(range(len(seated) + 1, count + 1))
    if missing:
        seated.update(assign_personalities(missing, Random(0)))
    return seated
