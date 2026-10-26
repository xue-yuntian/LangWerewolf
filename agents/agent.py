"""AI 玩家 —— 视角 → prompt → 一次模型调用 → 一个合法动作。

────────────────────────────────────────────────────────────────
职责边界
────────────────────────────────────────────────────────────────
    engine   决定"什么算合法"：越界、连守、药互斥都由 RuleViolation 兜底
    本模块   决定"想做什么"：拼 prompt、调模型、解析、把非法答案退化成保守动作

本模块只读引擎数据（state / events / view），不 import 任何结算函数，
所以它没有能力改变游戏状态——D12 的安全边界就画在这里。

────────────────────────────────────────────────────────────────
为什么非法答案要"退化"而不是抛异常
────────────────────────────────────────────────────────────────
引擎抛 RuleViolation 会中断整局，而模型给错座位号是常态：
写成 9 号、写成字符串 "5"、写一个已出局的人。一局 60+ 次调用，
哪怕 1% 出错率，不兜底就等于每局都会崩。所以这里的取舍是：

    合法        -> 采纳
    非法        -> 把可选范围再说一遍，最多纠正一次
    还不行/失败 -> 退化成"不行动"（空守），损失只有一个夜晚

────────────────────────────────────────────────────────────────
当前进度
────────────────────────────────────────────────────────────────
7 个策略方法全部实现：守卫、狼刀、女巫救/毒、预言家查验、白天投票、猎人开枪。
另有一个可选的 speak 钩子做白天发言——它不在 GamePolicy 协议里，
因为随机策略不会说话，而“会说话”是 LLMAgentPolicy 才需要的。
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from os import getenv
from random import Random

from agents.notebook import Notebook
from agents.personality import (
    PERSONALITY_SEED_OFFSET,
    Persona,
    Personality,
    assign_personalities,
    personality_text,
)
from engine.events import EventLog
from engine.roles import Roles
from engine.state import GameState
from engine.view import round_start_seq, view_for
from llm import LLMError, chat_json

# 夜间动作走 fast 模型：决策空间小、调用次数多，成本最敏感（见 .env.example）
DEFAULT_MODEL = "deepseek-flash"

# 一次决策最多问两遍：第二遍只用来纠正非法目标
MAX_ATTEMPTS = 2

# "不行动"的 JSON 写法。抽成常量是因为 f-string 里裸写花括号要转义，容易写错
NO_ACTION = '{"target": null}'

SYSTEM = (
    "你在参加一局狼人杀，是场上的一名玩家。"
    "你只输出一个 JSON 对象，不要解释、不要加 Markdown 代码块。"
)


class LLMAgentPolicy:
    """用模型决策的 GamePolicy。

    构造时只读环境变量、不建连接：第一次真正调用才需要 API key，
    所以 import 本模块和跑单测都不依赖网络或 .env。

    传 seed 时会按座位号自动分配人设。这不是装饰：三只狼拿到的信息
    完全相同，没有人设的话，问三遍等于问一遍。
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        personalities: Mapping[int, Persona | Personality] | None = None,
        seed: int | None = None,
    ) -> None:
        # 显式标注 str：getenv 的返回是 str | None，不兜底的话
        # 环境变量没设时会以 None 传进 chat_json
        self.model: str = model or getenv("LLM_MODEL_FAST") or DEFAULT_MODEL
        self.temperature = temperature
        self.personalities = dict(personalities or {})
        self.seed = seed
        self.notebook = Notebook()
        # play_game 开局后注入；没有它也能拼 prompt，只是看不到历史
        self.events: EventLog | None = None
        # 本夜是否已经用过解药。resolve_night 的局部变量，state 上查不到，
        # 只能自己记。带上 state 身份是为了防止同一个 policy 实例跨局复用时，
        # 上一局的夜数误命中新局
        self._saved_state: GameState | None = None
        self._saved_night = 0
        # 投票记账：(day, round) -> {voter: target}，round 0=首轮 1=PK
        # 签名里没有轮次，同一天第二次被问到就是 PK——只能自己记
        self._vote_state: GameState | None = None
        self._ballots: dict[tuple[int, int], dict[int, int | None]] = {}
        # 本轮已经发过的言。speak 往里写，vote 从里读。
        # 投票时 state.day 已经 +1，发言时还没，两边的天数算出来不一样，
        # 所以不按天数比对，而是“换了一天就把上一轮清掉”
        self._speeches_state: GameState | None = None
        self._speeches_day = -1
        self._speeches: list[str] = []
        # 上帝视角的观察口。sim.py 加 --god 时才挂，挂了就把每夜的狼队
        # 会议打到终端。这些内容不进事件流：事件流是给玩家的，而讨论
        # 里点名的是明天怎么骗人。
        self.wolf_talk: Callable[[str], None] | None = None

    # ------------------------------------------------------------------
    # 接线
    # ------------------------------------------------------------------

    def attach_events(self, events: EventLog) -> None:
        """接住 play_game 的事件流。

        这个方法不属于 GamePolicy 协议——它是可选钩子，只有需要读历史的
        策略才实现。RandomPolicy 不需要，所以它没有这个方法。
        """
        self.events = events

    # ------------------------------------------------------------------
    # 1. 守卫
    # ------------------------------------------------------------------

    def choose_guard(self, state: GameState) -> int | None:
        """守谁。None 表示空守。

        `state.last_guarded` 是守卫私有信息，view_for 里看不到，
        必须在这里写进任务描述，否则模型很可能连守同一人被引擎拦下。
        """
        self._ensure_personalities(state)
        guard = state.alive_holder(Roles.GUARD)
        if guard is None:
            return None

        # 和 night.py 的校验条件保持同一份口径：所有存活玩家，排除上一晚守过的
        candidates = [pid for pid in state.alive_ids() if pid != state.last_guarded]
        if not candidates:
            return None

        warning = (
            f"\n上一夜你守过 {state.last_guarded} 号，今晚不能再守他。"
            if state.last_guarded is not None
            else ""
        )
        question = (
            f"【今晚的任务】\n"
            f"现在是第{state.night}夜，你是 {guard} 号守卫，要决定守谁。{warning}\n"
            f"可选目标：{_pids(candidates)}（包括你自己）\n"
            f"也可以选择空守。\n"
            f'只输出 JSON：{{"target": 座位号}}；如果空守，输出 {NO_ACTION}。'
        )
        return self._ask_target(state, guard, question, candidates, thinking=False)

    # ------------------------------------------------------------------
    # 2. 狼人：每只活狼各投一票
    # ------------------------------------------------------------------

    def choose_wolf_kill(self, state: GameState) -> int | None:
        """今晚刀谁。None 表示空刀。

        引擎每夜只问一次（night.py 里不分是谁在问），整场狼队会议只能
        在这里自己开。四条约定：

          1. 每只狼一次调用，同时给出自己那一票和明天的打算。分两步问
             的话，第二遍要把几乎同样的上下文重发一次。
          2. 主狼（座位号最小的那只，也是平票时的拍板人）最后发言，
             听完全队的票和打算再定分工。
          3. 多数票获胜；平票由主狼拍板。
          4. 某只狼调用失败只是少一票少一句打算，不影响其他人。
        """
        self._ensure_personalities(state)
        wolves = state.wolves_alive()
        if not wolves:
            return None

        # 允许自刀，所以候选是全部存活玩家，不把狼排除在外
        candidates = state.alive_ids()
        if not candidates:
            return None

        chief = wolves[0]
        said: list[tuple[int, int | None, str | None]] = []
        for wolf in [*wolves[1:], chief]:
            target, plan = self._wolf_turn(state, wolf, said, candidates, chief=wolf == chief)
            # 失败的也记下来：后面的人看到“没定下刀口”才知道它没表态
            said.append((wolf, target, plan))

        votes = [(who, target) for who, target, _ in said if target is not None]
        decision = _tally(wolves, votes)
        if decision is not None:
            # 讨论不进事件流（WOLF_KILL 只记刀口），下一夜 round_start_seq
            # 又会把这一夜整体裁掉，所以商量完立刻落进笔记
            self._write_wolf_notes(state, wolves, said, decision)
        # 空刀也要报：上帝关心的是“到底商量过没有、谁想把刀口往哪引”，
        # 而笔记只在真定下刀口时才有意义
        self._report_wolf_talk(state, said, decision)
        return decision

    # ------------------------------------------------------------------
    # 3-4. 女巫：救 / 毒
    # ------------------------------------------------------------------

    def choose_witch_save(self, state: GameState, killed: int | None) -> bool:
        """救不救今晚被刀的人。

        `killed` 是唯一的信息来源：当晚的 WOLF_KILL 要等整夜结算完才 emit，
        事件流里翻不到"今晚刀了谁"，所以必须由参数喂进 prompt。
        """
        self._ensure_personalities(state)
        witch = state.alive_holder(Roles.WITCH)
        if witch is None or killed is None:
            return False

        # 非首夜不能自救。引擎那边会直接抛 RuleViolation，这里提前掐掉，
        # 既省一次调用，也不把违规答案交给引擎
        if killed == witch and state.night > 1:
            return False

        question = (
            f"【今晚的任务】\n"
            f"现在是第{state.night}夜，你是 {witch} 号女巫。\n"
            f"今晚 {killed} 号被狼刀。解药整局只有这一瓶。\n"
            f"救他则今晚无人出局；不救他就会死。狼也可能自刀骗药。\n"
            f'只输出 JSON：{{"save": true}} 表示救，{{"save": false}} 表示不救。'
        )

        prompt = question
        for attempt in range(MAX_ATTEMPTS):
            payload = self._ask(state, witch, prompt, thinking=False)
            if payload is None:
                return False  # 调不通就按"不救"，保守
            decision = _as_bool(payload.get("save"))
            if decision is not None:
                if decision:
                    # 记下"今晚用过解药"，choose_witch_poison 会据此投降
                    self._saved_state = state
                    self._saved_night = state.night
                return decision
            if attempt == MAX_ATTEMPTS - 1:
                break
            prompt = f"你上一轮没有给出合法的 save 字段。\n\n{question}"
        return False

    def choose_witch_poison(self, state: GameState) -> int | None:
        """毒谁。None 表示不用毒。

        引擎在同一夜里先问救、再问毒，但"今晚已用解药"只存在于
        resolve_night 的局部变量里，要等整夜结束才落库。所以只能靠
        _saved_night 自己记，否则会撞上"一晚只能用一瓶药"。
        """
        self._ensure_personalities(state)
        witch = state.alive_holder(Roles.WITCH)
        if witch is None:
            return None
        if self._saved_state is state and self._saved_night == state.night:
            return None

        # 不能毒自己；其余存活玩家都可以
        candidates = [pid for pid in state.alive_ids() if pid != witch]
        if not candidates:
            return None

        question = (
            f"【今晚的任务】\n"
            f"现在是第{state.night}夜，你是 {witch} 号女巫。\n"
            f"毒药整局只有这一瓶。\n"
            f"可选目标：{_pids(candidates)}\n"
            f'只输出 JSON：{{"target": 座位号}}；如果不用毒，输出 {NO_ACTION}。'
        )
        return self._ask_target(state, witch, question, candidates, thinking=False)

    # ------------------------------------------------------------------
    # 5. 预言家
    # ------------------------------------------------------------------

    def choose_seer_check(self, state: GameState) -> int | None:
        """验谁。

        引擎只拦"死者"和"验自己"，不拦重复验——所以"别验已经验过的人"
        得由这里保证，不然白丢一晚的信息。
        """
        self._ensure_personalities(state)
        seer = state.alive_holder(Roles.SEER)
        if seer is None:
            return None

        checked = set(state.checked_ids())
        candidates = [pid for pid in state.alive_ids() if pid != seer and pid not in checked]
        if not candidates:
            return None

        question = (
            f"【今晚的任务】\n"
            f"现在是第{state.night}夜，你是 {seer} 号预言家，要决定验谁。\n"
            f"可选目标：{_pids(candidates)}\n"
            f"你之前的查验结果在私密信息里。\n"
            f'只输出 JSON：{{"target": 座位号}}。'
        )
        return self._ask_target(state, seer, question, candidates, thinking=False)

    # ------------------------------------------------------------------
    # 6. 白天投票
    # ------------------------------------------------------------------

    def vote(
        self, state: GameState, voter: int, *, only: Sequence[int] | None = None
    ) -> int | None:
        """投谁。None 表示弃票。

        only 给出 PK 台上的座位号——引擎重投时会传，不用自己数第几轮。
        靠记账猜轮次太脆：只要有一轮没记上，就会把票投到台下。
        """
        self._ensure_personalities(state)
        if not state.is_alive(voter):
            return None

        # 新的一局：清空上一局的票，否则天数撞上会把第一轮当成 PK。
        # 发言列表只在“本局还没人发过言”时才清——本局刚说过的话
        # 就是这一票的依据，清了等于白说
        if state is not self._vote_state:
            self._vote_state = state
            self._ballots.clear()
            if state is not self._speeches_state:
                self._speeches_state = state
                self._speeches_day = -1
                self._speeches.clear()

        day = state.day
        if only is not None:
            # PK 重投：候选只剩台上还活着的人
            tied = list(only)
            candidates = [pid for pid in tied if pid != voter and state.is_alive(pid)]
            key = (day, 1)
            question = _pk_question(state, voter, tied, self._ballots.get((day, 0), {}), candidates)
        else:
            candidates = [pid for pid in state.alive_ids() if pid != voter]
            key = (day, 0)
            question = _vote_question(state, voter, candidates)

        if not candidates:
            return None

        target = self._ask_target(
            state, voter, question, candidates, with_speeches=True, thinking=False
        )
        self._ballots.setdefault(key, {})[voter] = target
        return target

    # ------------------------------------------------------------------
    # 7. 白天发言 + 每日总结（都是可选钩子，不在 GamePolicy 协议里）
    # ------------------------------------------------------------------

    def speak(self, state: GameState, pid: int) -> str | None:
        """发一句话。None 表示弃权不发言。

        引擎在投票前按座位号升序逐个调它（game._collect_speeches），
        说完立刻进事件流，所以后发言的人能看到前面的人说了什么。
        输出仍然走 JSON：重试、剥壳、trace 全是现成的，不用另开一个文本接口。
        """
        self._ensure_personalities(state)
        if not state.is_alive(pid):
            return None

        if state is not self._speeches_state:
            self._speeches_state = state
            self._speeches_day = -1
            self._speeches.clear()

        # 发言发生在 resolve_vote 之前，state.day 还没 +1
        day = state.day + 1
        if day != self._speeches_day:
            self._speeches_day = day
            self._speeches.clear()

        role = state.player(pid).role
        task = (
            f"【发言】\n"
            f"现在是第{day}天。\n"
            f"用不超过 100 字说明你的判断：怀疑谁、信谁、打算投谁，以及理由。\n"
            "信息纪律：夜间信息（谁被刀、救了谁、验了谁）只有你和你同阵营的人知道。"
            "把它原样说出来等于公开身份——狼说出刀口就是自曝，女巫说出救了谁就是自曝。"
            "想公开身份就直接起跳，想隐藏就只说结论、不报具体座位号。\n"
        )
        if role is Roles.WOLF:
            task += "你是狼人：必须把自己伪装成好人，不能承认是狼，不能供出队友。\n"
        elif role is Roles.WITCH:
            # 只给事实和代价，不替她做判断：银水可能是自刀骗药的狼，
            # 报银水能保人但自己会变成刀口——这些都是可选策略。
            # 和 personality.py 顶部那条边界一样：只管怎么说话，不管该不该做某事
            # prompt 里一旦写“不要投他”“不报就白用”，就把策略变成了禁令
            task += (
                "你是女巫。私密信息里的「女巫用解药救下 X 号」是你的银水："
                "他大概率是好人，但也可能是自刀骗药。\n"
                "报银水能保他，代价是你自己暴露成刀口；不报就是你一个人的判断，没人知道你手里有什么。\n"
                "怎么用这条信息，你自己决定。\n"
            )
        elif role is Roles.SEER:
            task += (
                "你是预言家。你的验人结果是好人手里最可靠的信息，"
                "但报出来等于告诉狼该刀谁，不报又只能靠发言推人。\n"
                "报不报、什么时候报，你自己权衡。但注意，选择不报的时候尽量不要提及你的预言家身份。\n"
            )
        task += '只输出 JSON：{"speech": "你的发言"}；不想发言就输出 {"speech": null}。'

        payload = self._ask(state, pid, task, with_speeches=True)
        if payload is None:
            return None

        text = _as_text(payload.get("speech"))
        if text is None:
            return None
        # 模型偶尔会整段倒豆子。截到 300 字，宁可截断也不能让一个人的
        # 发言把后面 8 个人的 prompt 撞爆
        if len(text) > 300:
            text = text[:300] + "…"

        self._speeches.append(f"{pid} 号：{text}")
        return text

    def note(self, state: GameState, pid: int, *, label: str) -> str | None:
        """写一条当天总结。只能追加，不能改前面的。

        引擎在一天结束、出局结算完之后调它（game._collect_notes）。
        明天起这一天的详细记录就从 prompt 里消失了，只靠这条笔记回忆——
        所以提示里必须把这件事说清楚，否则模型会写一句“今天没什么”。

        夜间不调它：狼队商量完打法后各自写的计划已经是笔记了，
        再总结一遍等于把同一件事花两份钱（见 _discuss_wolf_plan）。
        """
        self._ensure_personalities(state)
        if not state.is_alive(pid):
            return None

        task = (
            "【今日总结】\n"
            f"{label}结束了。用不超过 150 字写下以后还用得上的东西：\n"
            "谁说了什么、谁投了谁、谁出局、你现在怀疑谁、下一步打算怎么做。\n"
            "从明天起，今天的详细记录不会再出现在你的输入里，你只能靠这条笔记回忆——"
            "所以要写结论和依据，不要复述原话。\n"
            '只输出 JSON：{"note": "你的总结"}；确实没东西可写就输出 {"note": null}。'
        )
        payload = self._ask(state, pid, task, with_speeches=True, thinking=False)
        if payload is None:
            return None

        text = _as_text(payload.get("note"))
        if text is None:
            return None
        # 笔记会留到最后一轮，比发言更该压紧：150 字的需求给到200 字封顶
        if len(text) > 200:
            text = text[:200] + "…"

        self.notebook.add(pid, text, label=label)
        return text

    # ------------------------------------------------------------------
    # 8. 猎人开枪
    # ------------------------------------------------------------------

    def hunter_shot(self, state: GameState, hunter: int) -> int | None:
        """开枪带走谁。None 表示不开枪。

        只会在猎人被狼刀或被投出局时调到这里；被毒死不能开枪，引擎已经拦了。
        此刻猎人自己已经出局，所以存活名单天然不含他。
        """
        self._ensure_personalities(state)
        candidates = [pid for pid in state.alive_ids() if pid != hunter]
        if not candidates:
            return None

        question = (
            f"【开枪】\n"
            f"你是 {hunter} 号猎人，已经出局，现在可以带走一名存活玩家。\n"
            f"可选目标：{_pids(candidates)}\n"
            f"开枪机会只有一次，带错等于帮狼追一轮；也可以选择不开枪。\n"
            f'只输出 JSON：{{"target": 座位号}}；不开枪输出 {NO_ACTION}。'
        )
        return self._ask_target(state, hunter, question, candidates)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _wolf_turn(
        self,
        state: GameState,
        wolf: int,
        said: Sequence[tuple[int, int | None, str | None]],
        candidates: Sequence[int],
        *,
        chief: bool,
    ) -> tuple[int | None, str | None]:
        """问一只狼：你这一票投给谁，明天怎么打。

        一次调用出两样东西。拆成“先表态刀谁、再讨论打法”两步的话，
        第二遍要把整段上下文重发一次，而后半段的输入和前半段几乎一样。

        目标非法时重试一次（和 _ask_target 同一套规矩），plan 跟着重要
        一遍——它就在同一个 JSON 里。
        """
        question = _wolf_turn_question(state, wolf, said, candidates, chief=chief)
        prompt = question
        payload: dict[str, object] | None = None
        target: int | None = None

        for attempt in range(MAX_ATTEMPTS):
            payload = self._ask(state, wolf, prompt, thinking=False)
            if payload is None:
                return None, None

            raw = payload.get("target")
            if raw is None:
                break  # 明确选择空刀
            target = _as_pid(raw)
            if target is not None and target in candidates:
                break
            target = None
            if attempt == MAX_ATTEMPTS - 1:
                return None, None
            prompt = (
                f"你上一轮给出的目标「{_short(raw)}」不合法，它不在可选目标里。"
                f"只能从 {_pids(candidates)} 里选，或者输出空刀。\n\n{question}"
            )

        plan = _as_text(payload.get("plan")) if payload is not None else None
        if plan is not None and len(plan) > 200:
            plan = plan[:200] + "…"
        return target, plan

    def _write_wolf_notes(
        self,
        state: GameState,
        wolves: Sequence[int],
        said: Sequence[tuple[int, int | None, str | None]],
        decision: int,
    ) -> None:
        """把今晚商量出的东西写进笔记。

        自己的打算进自己的；主狼那句是全队口径，进所有人的——明天
        三只狼按同一套剧本演，散兵游勇才有一致性。主狼没说话就只剩
        各人自己那条，不至于一片空白。
        """
        label = f"第{state.night}夜"
        plans = {who: plan for who, _, plan in said if plan is not None}

        for wolf in wolves:
            if wolf != wolves[0] and wolf in plans:
                self.notebook.add(wolf, plans[wolf], label=label)

        final = plans.get(wolves[0])
        if final:
            for wolf in wolves:
                self.notebook.add(wolf, f"刀{decision}号：{final}", label=f"{label} 分工")

    def _report_wolf_talk(
        self,
        state: GameState,
        said: Sequence[tuple[int, int | None, str | None]],
        decision: int | None,
    ) -> None:
        """把整场会议原样吐给观察者。没挂回调就什么都不做。

        走回调而不是事件流：这些内容本来就不该让玩家看见（夜里刚商量
        完怎么骗人），而 EventLog 是面向玩家的。也不该塞进 NightResult：
        那是引擎的数据结构，没道理知道 policy 内部在想什么。
        """
        if self.wolf_talk is None:
            return

        lines = [f"[第{state.night}夜] 狼队会议"]
        for index, (wolf, target, plan) in enumerate(said):
            pick = f"刀 {target} 号" if target is not None else "空刀"
            # 主狼固定是最后一个发言的人（见 choose_wolf_kill 的约定）
            air = "（主狼，拍板）" if index == len(said) - 1 else ""
            lines.append(f"  {wolf}号{air} → {pick}：{plan or '（没表态）'}")
        lines.append(f"  定刀：{decision}号" if decision is not None else "  定刀：空刀")
        self.wolf_talk("\n".join(lines))

    def _ensure_personalities(self, state: GameState) -> None:
        """没显式指定人设、又给了 seed 时，按座位号分一轮。

        用独立的随机源（seed + PERSONALITY_SEED_OFFSET），所以分人设
        不会扰动发牌，也不会和策略里的其他随机性互相影响。
        """
        if self.personalities or self.seed is None:
            return
        rng = Random(self.seed + PERSONALITY_SEED_OFFSET)
        # 逐项写而不是整体赋值：assign_personalities 返回 dict[int, Personality]，
        # 而这里存的是 dict[int, Persona | Personality]，dict 在值类型上不变，
        # 直接赋值会被 mypy 拦下来。逐项写是子类型赋值，不需要 cast。
        for pid, kind in assign_personalities(sorted(state.players), rng).items():
            self.personalities[pid] = kind

    def _ask(
        self,
        state: GameState,
        pid: int,
        question: str,
        *,
        with_speeches: bool = False,
        thinking: bool | None = None,
    ) -> dict[str, object] | None:
        """问一次模型。None 表示这次调用失败（网络、额度、剥壳）。

        thinking=False 让后端跳过思维链。夜里全是“挑个人、说句话”，
        白天的投票和记笔记同理：这些步骤的正确答案不需要推理过程，
        思维链只把首字延迟拉长一倍、把输出 token 撑大几倍。真正值得
        开的是白天发言——那才需要组织语言、盘逻辑。
        """
        try:
            return chat_json(
                SYSTEM,
                self._full_prompt(state, pid, question, with_speeches=with_speeches),
                model=self.model,
                temperature=self.temperature,
                thinking=thinking,
            )
        except LLMError:
            return None

    def _ask_target(
        self,
        state: GameState,
        pid: int,
        question: str,
        candidates: Sequence[int],
        *,
        with_speeches: bool = False,
        thinking: bool | None = None,
    ) -> int | None:
        """问模型要一个座位号。None 表示"不行动"，或两次都没能拿到合法答案。"""
        prompt = question
        for attempt in range(MAX_ATTEMPTS):
            payload = self._ask(state, pid, prompt, with_speeches=with_speeches, thinking=thinking)
            if payload is None:
                # 网络/额度/剥壳失败：宁可这一晚不动作，也不能让整局崩掉
                return None

            raw = payload.get("target")
            if raw is None:
                return None  # 模型明确选择了不行动

            target = _as_pid(raw)
            if target is not None and target in candidates:
                return target

            if attempt == MAX_ATTEMPTS - 1:
                break
            prompt = (
                f"你上一轮给出的目标「{_short(raw)}」不合法，它不在可选目标里。"
                f"只能从 {_pids(candidates)} 里选，或者输出 {NO_ACTION}。\n\n{question}"
            )
        return None

    def _full_prompt(
        self,
        state: GameState,
        pid: int,
        question: str,
        *,
        with_speeches: bool = False,
    ) -> str:
        events = self.events if self.events is not None else EventLog()
        view = view_for(
            state,
            events,
            pid,
            personality=self._personality(pid),
            notes=self.notebook.get(pid),
            # 只有白天（发言/投票）才带本轮发言。夜间带上一轮发言会误导：
            # 模型会以为那是今晚刚发生的事
            last_speeches=self._speeches if with_speeches else None,
            # 只带当前这一轮：更早的详细记录已经被 note() 的总结取代。
            # 不裁的话每次调用都把整局重发一遍，prompt 随天数线性膨胀
            since_seq=round_start_seq(events, state.night),
        )
        return f"{view.to_prompt_text()}\n\n{question}"

    def _personality(self, pid: int) -> str:
        kind = self.personalities.get(pid)
        return personality_text(kind) if kind is not None else ""


def _pids(pids: Sequence[int]) -> str:
    return "、".join(f"{pid}号" for pid in pids) or "无"


def _wolf_turn_question(
    state: GameState,
    wolf: int,
    said: Sequence[tuple[int, int | None, str | None]],
    candidates: Sequence[int],
    *,
    chief: bool,
) -> str:
    """狼队会议的任务段。

    把前面队友的票和打算一起念出来：事件流里没有“当晚的狼刀讨论”
    （WOLF_KILL 要等整夜结算完才 emit），只能在这里口口相传。
    """
    lines = [
        "【今晚的任务】",
        f"现在是第{state.night}夜，你是 {wolf} 号。今晚两件事：刀谁，明天怎么打。",
    ]
    if said:
        lines.append("在你之前表过态的队友：")
        for who, target, plan in said:
            head = f"  {who}号："
            head += f"刀 {target} 号。" if target is not None else "没定下刀口。"
            if plan:
                head += plan
            lines.append(head)
    if chief:
        lines.append(
            "你是主狼，也是最后一个发言的人，由你定明天的分工：谁悍跳预言家、"
            "谁装好人倒钩、谁带节奏、谁在明面上护着谁。"
            "刀口按多数票定，你的 plan 里不要再复述刀谁。"
        )
    else:
        lines.append("说说你明天的打算，以及希望队友怎么配合你。")
    lines += [
        f"可选目标：{_pids(candidates)}",
        '只输出 JSON：{"target": 座位号, "plan": "明天怎么打，不超过 100 字"}；'
        "如果空刀，target 写 null。",
    ]
    return "\n".join(lines)


def _tally(wolves: Sequence[int], votes: Sequence[tuple[int, int]]) -> int | None:
    """多数票；平票时听主狼的，主狼没投就取座位号最小的那个目标。"""
    if not votes:
        return None

    counts = Counter(target for _, target in votes)
    best = max(counts.values())
    leaders = sorted(target for target, count in counts.items() if count == best)
    if len(leaders) == 1:
        return leaders[0]

    chief = wolves[0]
    for voter, target in votes:
        if voter == chief:
            return target
    return leaders[0]


def _vote_question(state: GameState, voter: int, candidates: Sequence[int]) -> str:
    return "\n".join(
        [
            "【投票】",
            f"现在是第{state.day}天，你是 {voter} 号，要投出今天出局的人。",
            f"可选目标：{_pids(candidates)}",
            "你可以投给任何存活玩家（不能投自己），也可以弃票。",
            f'只输出 JSON：{{"target": 座位号}}；弃票输出 {NO_ACTION}。',
        ]
    )


def _pk_question(
    state: GameState,
    voter: int,
    tied: Sequence[int],
    first: Mapping[int, int | None],
    candidates: Sequence[int],
) -> str:
    ballot = "、".join(
        f"{who}号→{target}号" for who, target in sorted(first.items()) if target is not None
    )
    lines = [
        "【PK 重投】",
        f"现在是第{state.day}天，首轮 {_pids(tied)} 并列最高票，进入 PK。",
    ]
    if ballot:
        lines.append(f"首轮票型：{ballot}")
    lines += [
        f"按规则台上的人不能投票，由台下的人决定。你是 {voter} 号。",
        f"只能从 {_pids(candidates)} 里选，也可以弃票。",
        f'只输出 JSON：{{"target": 座位号}}；弃票输出 {NO_ACTION}。',
    ]
    return "\n".join(lines)


def _short(value: object, limit: int = 20) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _as_bool(value: object) -> bool | None:
    """把模型给的 save 字段归一成 bool。

    None 表示无法判断，调用方按保守动作（不救/不毒）处理。
    字符串也接受：模型经常把 true 写成 "true" 或 "是"。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "是", "救"}:
            return True
        if text in {"false", "no", "否", "不救"}:
            return False
    return None


def _as_text(value: object) -> str | None:
    """把模型给的 speech 归一成一行文本。None 表示弃权或类型不对。"""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _as_pid(value: object) -> int | None:
    """把模型给的 target 归一成座位号。

    bool 必须先判：isinstance(True, int) 是 True，不排掉会把 True 当成 1 号。
    数字字符串也接受——模型经常把 5 写成 "5"。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
