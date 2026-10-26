"""玩家视角 —— 把引擎状态压缩成"这个玩家此刻知道什么"，供 D12 拼 prompt。

只读：这里拼出来的文本不会回写 GameState。
所有说明性文本（身份能力、术语表）都放在本模块，模型的行为约束只在这里维护一份。
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from engine.board import Board
from engine.events import VIS_ALL, VIS_WOLVES, Event, EventKind, EventLog, vis_pid
from engine.roles import ROLE_REGISTRY, Faction, Roles
from engine.state import Death, DeathCause, GameState

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

# ---------------- 本版本没有的机制：不发给模型 ----------------
# 警长竞选和自爆都没实现，写了只会诱导模型去用。本局配了哪些角色
# 由 setup_text 负责说明，术语表不需要兼这个职。

# 哪些术语发给谁：拆开之后每次调用只发本角色真正用得上的，
# 而不是把 36 条全塞进去。9p 没有守卫，空守/连守/同守同救三条
# 对任何人都没有意义。

# 所有人都要懂：一是自己会用（狼刀、弃票、PK），二是听别人说要用
# （金水、查杀、银水、起跳这些是别人发言里的词，听不懂就没法判断局势）
_COMMON_TERMS: tuple[str, ...] = (
    "夜晚顺序",
    "狼刀",
    "自刀",
    "空刀",
    "金水",
    "查杀",
    "验人链",
    "银水",
    "弃票",
    "平票",
    "PK",
    "再平票",
    "屠边",
    "屠城",
    "起跳",
    "对跳",
    "悍跳",
    "归票",
    "踩",
    "保",
    "划水",
    "抿身份",
    "狼坑",
    "抗推",
    "倒钩",
    "深水狼",
)

# 只有这个角色自己用得上的：别人说了也影响不到你的动作
_ROLE_TERMS: dict[Roles, tuple[str, ...]] = {
    Roles.GUARD: ("空守", "连守", "同守同救"),
    Roles.WITCH: ("解药", "毒药"),
    Roles.SEER: ("查验",),
    Roles.HUNTER: ("开枪",),
    Roles.WOLF: (),
    Roles.VILLAGER: (),
}


def terms_text(role: Roles) -> str:
    """这个角色用得上的术语，拼成一段纯文本。

    按角色裁而不是按板子裁：预言家的“查验”要发给预言家，
    但“金水”必须发给所有人——那是听别人说话时要用的词。
    """
    names = _COMMON_TERMS + _ROLE_TERMS.get(role, ())
    return "\n".join(f"- {name}：{glossary[name]}" for name in names)


def setup_text(board: Board) -> str:
    """本局有哪些身份、各几个，以及没有哪些。

    必须写进 prompt：术语表和能力说明是全局的，里面提到守卫、警长
    这些本局可能根本没上场的角色。不明确说“没有”，模型会照着术语表
    悍跳一个不存在的身份——它没有别的办法知道这局到底配了什么。
    """
    present = "\n".join(
        f"- {ROLE_REGISTRY[role].name_zh} {count} 人" for role, count in board.roles.items()
    )
    absent = [spec.name_zh for role, spec in ROLE_REGISTRY.items() if role not in board.roles]
    text = f"本局身份配置：\n{present}"
    if absent:
        text += f"\n本局没有：{'、'.join(absent)}。不要声称自己是其中任何一个角色。"
    return text


def _join_pids(pids: Iterable[int]) -> str:
    """座位号拼成「1号、2号」；空集合返回「无」，避免 prompt 里出现 []。"""
    return "、".join(f"{pid}号" for pid in pids) or "无"


def _join_dead(dead: Iterable[tuple[int, str]]) -> str:
    """死者和死法拼成「1号（第2夜）」「3号（第1天 投票）」。"""
    return "、".join(f"{pid}号（{reason}）" for pid, reason in dead) or "无"


def death_text(death: Death) -> str:
    """死者在公开信息里的写法。

    夜里怎么死的（被刀还是被毒）不能写。知道有人被毒杀，就等于知道
    女巫出过手——而她要不要认下这个动作本身就是策略。白天被投出去、
    被猎人带走是公开事件（VOTE_RESULT / HUNTER_SHOT 全场可见），照写。

    代价是猎人不知道自己死于毒。规则本来就如此：他仍然会尝试开枪，
    由 collect_hunter_shot 按死因拒掉。
    """
    if death.cause is DeathCause.WOLF_KILL or death.cause is DeathCause.POISON:
        return death.label
    return f"{death.label} {cause_text(death.cause.value)}"


def _join_lines(lines: Iterable[str]) -> str:
    """每条信息占一行，避免模型把 list 的 repr 读成一整段。"""
    return "\n".join(f"- {line}" for line in lines)


# ---------------------------------------------------------------------------
# 死因中文名
# ---------------------------------------------------------------------------

# sim.py 里原本有一份私有的 _CAUSE_ZH，这里是唯一应该存在的版本。
# sim.py 那份改成 from engine.view import cause_text 即可，别留两份。
_CAUSE_TEXT: dict[str, str] = {
    "wolf_kill": "狼刀",
    "poison": "毒杀",
    "vote": "投票",
    # 不带“被”字会被读成“这个人开的枪”，死因是“死于枪”不是“开了枪”
    "shot": "枪杀",
}


def cause_text(cause: str) -> str:
    """死因的中文名。未知死因原样返回，方便以后加新死因。"""
    return _CAUSE_TEXT.get(cause, cause)


# ---------------------------------------------------------------------------
# payload 取值：payload 是 Mapping[str, object]，直接索引会污染整条链的类型
# ---------------------------------------------------------------------------


def _get_int(payload: Mapping[str, object], key: str) -> int | None:
    """取整数字段。

    必须排掉 bool：Python 里 isinstance(True, int) 为 True，
    不排除的话 True 会被当成 1 号玩家。
    """
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _get_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _get_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# 事件 -> 一行中文
# ---------------------------------------------------------------------------


def render_event(event: Event) -> str | None:
    """把一条事件渲染成一行中文。

    返回 None 表示这条事件不产生玩家可见的文本
    （GAME_START / NIGHT_START / DAY_START / GAME_OVER 都是这类）。
    """
    payload = event.payload
    label = event.label

    if event.kind is EventKind.GUARD:
        target = _get_int(payload, "target")
        return f"{label}：守卫守护了 {target} 号" if target is not None else None

    if event.kind is EventKind.WOLF_KILL:
        target = _get_int(payload, "target")
        return f"{label}：狼队决定刀 {target} 号" if target is not None else None

    if event.kind is EventKind.WITCH_SAVE:
        target = _get_int(payload, "target")
        return f"{label}：女巫用解药救下 {target} 号" if target is not None else None

    if event.kind is EventKind.WITCH_POISON:
        target = _get_int(payload, "target")
        return f"{label}：女巫毒死 {target} 号" if target is not None else None

    if event.kind is EventKind.SEER_CHECK:
        target = _get_int(payload, "target")
        if target is None:
            return None
        verdict = "狼人" if _get_bool(payload, "saw_wolf") else "好人"
        return f"{label}：查验 {target} 号 → {verdict}"

    if event.kind is EventKind.DEATH:
        dead_pid = _get_int(payload, "pid")
        if dead_pid is None:
            return None
        # 不写死因：被刀还是被毒要玩家自己推，写出来等于替他们推理
        return f"{label}：{dead_pid} 号出局"

    if event.kind is EventKind.HUNTER_SHOT:
        hunter = _get_int(payload, "hunter")
        target = _get_int(payload, "target")
        if hunter is None or target is None:
            return None
        return f"{label}：{hunter} 号猎人开枪，带走 {target} 号"

    if event.kind is EventKind.VOTE:
        voter = _get_int(payload, "voter")
        target = _get_int(payload, "target")
        if voter is None:
            return None
        if target is None:
            return f"{label}：{voter} 号弃票"
        suffix = "（PK 轮）" if _get_bool(payload, "pk") else ""
        return f"{label}：{voter} 号投给 {target} 号{suffix}"

    if event.kind is EventKind.VOTE_RESULT:
        eliminated = _get_int(payload, "eliminated")
        if eliminated is None:
            return f"{label}：无人出局"
        return f"{label}：{eliminated} 号被投票出局"

    if event.kind is EventKind.SPEECH:
        # 发言不进公开历史（split_history 会跳过），但渲染本身要完整：
        # 不带座位号的话，任何调 render_event 的地方都只能拿到一句无主的话。
        speaker = _get_int(payload, "pid")
        text = _get_str(payload, "text")
        if speaker is None or text is None:
            return None
        return f"{label}：{speaker} 号：{text}"

    return None


# ---------------------------------------------------------------------------
# 事件流 -> 一个玩家的(私有历史, 公开历史)
# ---------------------------------------------------------------------------


def split_history(
    events: EventLog,
    pid: int,
    *,
    is_wolf: bool = False,
    since_seq: int = -1,
) -> tuple[list[str], list[str]]:
    """切出 (private_lines, public_lines)。

    做法是**差集**：
        visible = 他作为观众能看到的所有事件
        public  = 所有人都能看到的事件
        private = visible - public

    两个关键点：
      1. 差集必须按 seq 做。Event 的 payload 是 dict，不可哈希，
         set(events) 会抛 TypeError；而用 == 比较又是 O(n*m)。
      2. is_wolf=True 时要把 "wolves" 加进观众集合。WOLF_KILL 的
         visible_to 是 "wolves"，和 "pid:N" 匹配不上——只传 pid 的话，
         狼会看不到自己队伍昨晚刀了谁。

    since_seq 只保留这一轮的事件：早先的详细记录已经被模型自己的笔记
    取代（每次调用都把全程历史重发一遍，prompt 会随天数线性膨胀）。
    """
    audiences: set[str] = {VIS_ALL, vis_pid(pid)}
    if is_wolf:
        audiences.add(VIS_WOLVES)

    visible = [e for e in events.for_audiences(audiences) if e.seq >= since_seq]
    public = [e for e in events.for_audiences({VIS_ALL}) if e.seq >= since_seq]
    public_seqs = {event.seq for event in public}

    private_lines: list[str] = []
    for event in visible:
        if event.seq in public_seqs:
            continue
        text = render_event(event)
        if text is not None:
            private_lines.append(text)

    public_lines: list[str] = []
    for event in public:
        # 发言不进公开历史：同样的内容由 last_speeches 单独发（那里带座位号，
        # 而且就是一整天的）。两份并存的代价是每次调用多几百字，信息量却
        # 一模一样，而且公开历史那份还不带说话人。
        # 公开历史只留“实际发生了什么”：投票、出局、开枪、PK 结果。
        if event.kind is EventKind.SPEECH:
            continue
        text = render_event(event)
        if text is not None:
            public_lines.append(text)

    return private_lines, public_lines


@dataclass(frozen=True)
class PlayerView:
    pid: int
    roles: Roles
    day: int
    alive: list[int]
    dead: list[tuple[int, str]]
    teammates: list[int]  # 狼队友；非狼返回空
    private_lines: list[str]  # 只有他知道的（验人结果、用药、狼刀）
    public_lines: list[str]  # 全场可见的行为记录（投票、出局），不含发言
    last_speeches: list[str]  # 上一轮发言（D12 之后才有）
    personality: str  # 角色设定，由开局之前的用户设定生成
    notes: list[str]  # 玩家笔记
    setup: str = ""  # 本局身份配置，全局不变，所以带默认值方便测试构造

    def to_prompt_text(self) -> str:
        # 用 blocks + join 拼，段落之间固定空一行；空字段直接不产生段落
        intro = "你正在进行一局狼人杀游戏。"
        if self.personality:
            intro += f"\n你的身份特征如下：\n{self.personality}"

        blocks = [
            intro,
            # 座位号必须写在这里，不能只靠各个任务段自己补：
            # 模型不知道自己几号，就无法守自己、无法避开自投，
            # 也读不懂“谁在跟自己对话”。术语跟身份写在一起，
            # 从此不再单独占一段。
            f"你是 {self.pid} 号玩家。\n"
            f"你的身份是：{ROLE_REGISTRY[self.roles].name_zh}（{self.roles}）\n"
            f"你的能力是：{roles_ability[self.roles]}\n"
            f"相关术语：\n{terms_text(self.roles)}",
            self.setup,
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
            blocks.append("以下是本轮已经发生过的发言：\n" + _join_lines(self.last_speeches))
        blocks.append(
            "注意，你的发言不能冗长，必须简明说明你想讲出的逻辑和结论。没有有效信息时可以选择过。"
        )
        return "\n\n".join(block for block in blocks if block)


def round_start_seq(events: EventLog, night: int) -> int:
    """第 night 夜的起点 seq。用它把 prompt 裁到“只含当前这一轮”。

    夜间结算前 NIGHT_START 还没发，返回 len(events)——等于什么都不留：
    那时的玩家本来也只有自己的笔记可看。

    夜间决策时 state.night 还是“上一夜”（它记录的是已经过完的夜晚数），
    所以会多带一整轮。这是有意的取舍：判断“当前是第几夜”需要知道
    NIGHT_START 发没发，而夜间和白天共用同一个入口；多带一轮是常数开销，
    总比把整局重发一遍强。
    """
    for event in events:
        if event.kind is EventKind.NIGHT_START and event.label == f"第{night}夜":
            return event.seq
    # night == 0 表示第一夜还没结算，本来就没有“上一轮”可裁，
    # 返回 -1 才能保住已有的私有事件（测试里手搭的事件流没有 NIGHT_START）
    return -1 if night <= 0 else len(events)


def view_for(
    state: GameState,
    events: EventLog,
    pid: int,
    *,
    personality: str = "",
    notes: list[str] | None = None,
    last_speeches: list[str] | None = None,
    since_seq: int = -1,
) -> PlayerView:
    """把引擎状态压缩成"pid 这一刻知道什么"。

    只读，不回写 state。这是全项目唯一的信息出口——D12 拼 prompt
    只允许从这里取数据，不允许直接读 GameState，否则私有信息会漏。

    since_seq 由调用方用 round_start_seq 算：只带当前这一轮的历史，
    更早的靠笔记。不裁的话一次调用会把整局重发一遍。
    """
    role = state.player(pid).role
    is_wolf = ROLE_REGISTRY[role].faction is Faction.WOLF

    # 狼队友：包含已死的狼。狼从开局就知道彼此身份，队友死了也仍然知道。
    teammates = (
        [other for other in state.role_holders(Roles.WOLF) if other != pid] if is_wolf else []
    )

    # 死者只给“第几夜 + 公开死法”。夜里被刀还是被毒不能写：毒杀两个字
    # 等于替女巫承认她出过手，而她要不要说这件事是她的策略。
    dead = [(death.pid, death_text(death)) for death in state.deaths]

    private_lines, public_lines = split_history(events, pid, is_wolf=is_wolf, since_seq=since_seq)

    return PlayerView(
        pid=pid,
        roles=role,
        # 不能用 state.day：发言发生在 resolve_vote 之前，那时 day 还没 +1，
        # 第 N 天的发言会写成“第 N-1 天”（第一天就是“第0天”）。
        # night 在夜里和白天都等于当前轮号，开局前兜到 1。
        day=max(state.night, 1),
        alive=state.alive_ids(),
        dead=dead,
        teammates=teammates,
        private_lines=private_lines,
        public_lines=public_lines,
        last_speeches=list(last_speeches or []),
        personality=personality,
        notes=list(notes or []),
        setup=setup_text(state.board),
    )
