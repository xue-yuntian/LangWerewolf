"""D5：生产用策略 —— 随机机器人。

────────────────────────────────────────────────────────────────
你要实现什么
────────────────────────────────────────────────────────────────
把这个文件里所有 `raise NotImplementedError` 换成真实逻辑，让它成为
`engine.game.GamePolicy` 的合法实现（协议定义在 engine/night.py、
engine/vote.py、engine/game.py 里，一共 7 个方法）：

    夜晚    choose_guard / choose_wolf_kill / choose_witch_save /
            choose_witch_poison / choose_seer_check
    白天    vote
    死亡    hunter_shot

────────────────────────────────────────────────────────────────
核心要求：随机，但**必须合法**
────────────────────────────────────────────────────────────────
"完全随机"会频繁抛 RuleViolation，一局都跑不完。正确做法是——
**先算出这个角色的合法候选集，再从候选集里随机**。
每个方法的注释里都写了它自己的合法条件，照着筛就行。

判断标准：`python -m engine.sim 9p --seed 0` 到 `--seed 99`
连续 100 个种子跑完，一条 RuleViolation 都不该出现。

────────────────────────────────────────────────────────────────
为什么需要这个文件
────────────────────────────────────────────────────────────────
D4 里驱动主循环的是 tests/support.py 的 FirstAlivePolicy——它只是为了
让 `play_game` 能跑到底，不会玩、也打不出好看的局。
RandomPolicy 是第一个"真正的玩家"，也是 D12 引入 LLM Agent 时替换掉的那一层。
这也是本项目"规则引擎与决策层分离"的实证：换策略不用改规则引擎一行代码。
"""

from random import Random

from engine.roles import Roles
from engine.state import GameState


class RandomPolicy:
    """在合法动作里随机挑一个。

    只依赖 GameState，不依赖任何引擎内部实现——D12 的 LLMAgentPolicy
    会保持同样的签名，只是"随机挑"换成"问模型"。
    """

    def __init__(self, rng: Random) -> None:
        # 显式注入 rng 而不是用全局 random：同一 seed 必须复现同一局。
        # D11 的批量仿真和 D27 的 CI 回归都靠这个。
        self.rng = rng
        # 记录"哪一局哪一夜已经用了救药"。用 (state, night) 而不是布尔值，
        # 避免跨夜或跨局时把上一夜的标记误当成本夜的。
        self._antidote_tonight: tuple[GameState, int] | None = None

    # ------------------------------------------------------------------
    # 夜间
    # ------------------------------------------------------------------

    def choose_guard(self, state: GameState) -> int | None:
        """守卫守谁。

        合法候选：所有存活玩家（**允许守自己**）
        额外约束：不能等于 `state.last_guarded`（不能连续两晚守同一人）
        返回 None 表示空守。
        """
        candidates = [pid for pid in state.alive_ids() if pid != state.last_guarded]
        return self._pick(candidates)

    def choose_wolf_kill(self, state: GameState) -> int | None:
        """狼队刀谁。

        合法候选：所有存活玩家（**允许自刀**，网杀里是骗药战术）
        返回 None 表示空刀。
        提示：真实狼队不会空刀，这里可以永远返回一个人。
        """
        candidates = list(state.alive_ids())
        return self._pick(candidates)

    def choose_witch_save(self, state: GameState, killed: int | None) -> bool:
        """女巫要不要用解药救今晚被刀的人。

        `killed` 已经由调用方保证"有人被刀"，你只需要决定救不救。
        可以简单用 self.rng.random() < 0.5。
        注意：引擎会拦住非法选择（解药毒药同晚、非首夜自救），
        但策略必须提前避开这些组合，否则一局都跑不完。
        """
        witch = state.alive_holder(Roles.WITCH)
        # 女巫只有首夜能自救；之后自己被刀时，策略不能选择救，否则引擎会抛错
        if witch is not None and killed == witch and state.night > 1:
            self._antidote_tonight = None
            return False

        saved = self.rng.random() >= 0.5
        self._antidote_tonight = (state, state.night) if saved else None
        return saved

    def choose_witch_poison(self, state: GameState) -> int | None:
        """女巫毒谁。

        合法候选：所有存活玩家，但**不能是女巫自己**
        返回 None 表示不用毒。
        提示：真随机毒人会大幅提高好人胜率，跑完 D11 的胜率统计后
        你会发现"随机女巫"很破坏平衡——这正是可以写进 README 的发现。
        """
        # 解药和毒药一晚只能二选一：今晚已经救了人，就不能再毒
        if (
            self._antidote_tonight is not None
            and self._antidote_tonight[0] is state
            and self._antidote_tonight[1] == state.night
        ):
            return None

        if self.rng.random() < 0.7:
            return None

        witch = state.alive_holder(Roles.WITCH)
        if witch is None:
            return None

        candidates = [pid for pid in state.alive_ids() if pid != witch]
        return self._pick(candidates)

    def choose_seer_check(self, state: GameState) -> int | None:
        """预言家验谁。

        合法候选：所有存活玩家，但**不能是预言家自己**
        已验过的人由引擎记在 Player.checked 上，这里用 state.checked_ids() 排除。
        """
        seer = state.alive_holder(Roles.SEER)
        if seer is None:
            return None

        checked = set(state.checked_ids())
        candidates = [pid for pid in state.alive_ids() if pid != seer and pid not in checked]
        return self._pick(candidates)

    # ------------------------------------------------------------------
    # 白天
    # ------------------------------------------------------------------

    def vote(self, state: GameState, voter: int) -> int | None:
        """第 voter 号投给谁。

        合法候选：所有存活玩家，但**不能是自己**（vote.py 会拦自投）
        返回 None 表示弃票。
        提示：一个能玩得起来的随机策略，应该尽量不弃票；
        全员弃票会让 day 计数虚增，D11 统计局数时会很难看。
        """
        candidates = [pid for pid in state.alive_ids() if pid != voter]
        return self._pick(candidates)

    # ------------------------------------------------------------------
    # 死亡技能
    # ------------------------------------------------------------------

    def hunter_shot(self, state: GameState, hunter: int) -> int | None:
        """猎人开枪打谁。

        合法候选：所有存活玩家，但**不能是猎人自己**
        返回 None 表示不开枪。
        注意：这个方法被 engine/state.py 的 apply_deaths 调用，
        只有当死因允许开枪时才会被问到（被毒杀不会问）。
        """
        candidates = [pid for pid in state.alive_ids() if pid != hunter]
        return self._pick(candidates)

    # ------------------------------------------------------------------
    # 工具函数（建议的私有方法，名字随你改）
    # ------------------------------------------------------------------

    def _pick(self, candidates: list[int]) -> int | None:
        """从候选里等概率随机取一个；候选为空返回 None。

        这就是全文件唯一的随机入口。把它抽出来的好处是：
        以后想换成"按权重挑"（D12 的 AI 就会按胜率挑），只改这一个函数。
        """
        if not candidates:
            return None
        return self.rng.choice(candidates)
