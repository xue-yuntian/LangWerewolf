"""批量仿真 —— 一局的"对照组"。

────────────────────────────────────────────────────────────────
它解决什么问题
────────────────────────────────────────────────────────────────
单局跑通只能证明"没崩"，不能证明"规则对、平衡合理"。
批量跑 N 局能给出三个数字：

    好人胜率 / 狼人胜率     —— 平衡性。一边倒说明规则或板子有问题
    平均天数               —— 节奏。太长说明推进慢或投票效率低
    撞上限的局数（aborted） —— 稳定性。必须为 0

这组数字是 **D12 的参照物**：AI 接入后你要能说"AI 把好人胜率从 51% 提到 58%"，
没有基线这句话就是空的。它同时是 **D27 的 CI 回归门禁的雏形**。

────────────────────────────────────────────────────────────────
两个关键设计
────────────────────────────────────────────────────────────────
① make_policy 是**工厂函数**，不是 policy 实例
   每局都要一个全新的 policy（RandomPolicy 内部有 _antidote_tonight 这类状态）。
   复用同一个实例会让上一局的状态泄漏到下一局。
   D12 换 LLM Agent 时只换这个工厂，run_batch 一行不动。

② 发牌和决策用**两个独立的随机流**
   如果两处都传 Random(seed)，策略的随机选择和发牌结果就同源了——
   同一个数字序列既决定"谁是什么身份"又决定"谁投谁"，
   胜率统计会带上相关性偏差。这里给策略种子加一个质数偏移。
"""

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random

from engine.board import Board, load_board_by_id
from engine.game import DEFAULT_MAX_DAYS, GamePolicy, play_game
from engine.policy import RandomPolicy
from engine.win import Winner

# 策略随机流相对发牌随机流的种子偏移。质数偏移让两条流尽量不相关。
POLICY_SEED_OFFSET = 1000003


@dataclass(frozen=True)
class BatchStats:
    """一批仿真的统计结果。

    只放"计数和总量"，比率全部做成 property——
    这样序列化时不会出现"比率和计数对不上"的情况。
    """

    board_id: str
    games: int  # 尝试的局数
    aborted: int  # 撞 max_days 而中止的局数，应当为 0
    wolf_wins: int
    good_wins: int
    days_total: int
    deaths_total: int
    events_total: int

    @property
    def completed(self) -> int:
        """正常分出胜负的局数。"""
        return self.games - self.aborted

    @property
    def good_win_rate(self) -> float:
        """好人胜率。分母用 completed，aborted 的局不算胜负。"""
        return self.good_wins / self.completed if self.completed else 0.0

    @property
    def wolf_win_rate(self) -> float:
        return self.wolf_wins / self.completed if self.completed else 0.0

    @property
    def avg_days(self) -> float:
        return self.days_total / self.completed if self.completed else 0.0

    @property
    def avg_deaths(self) -> float:
        return self.deaths_total / self.completed if self.completed else 0.0

    @property
    def avg_events(self) -> float:
        """平均事件数。D16 做 SSE 时用来估单局推送量。"""
        return self.events_total / self.completed if self.completed else 0.0

    def to_dict(self) -> dict[str, object]:
        """给 D27 的评测报告用。分子分母都留，方便复核。"""
        return {
            "board_id": self.board_id,
            "games": self.games,
            "completed": self.completed,
            "aborted": self.aborted,
            "good_wins": self.good_wins,
            "wolf_wins": self.wolf_wins,
            "good_win_rate": round(self.good_win_rate, 4),
            "wolf_win_rate": round(self.wolf_win_rate, 4),
            "avg_days": round(self.avg_days, 2),
            "avg_deaths": round(self.avg_deaths, 2),
            "avg_events": round(self.avg_events, 2),
        }

    def summary(self) -> str:
        """一行可读结论，直接贴进 README 或 CI 日志。"""
        return (
            f"{self.board_id}：{self.completed}/{self.games} 局完成 | "
            f"好人胜 {self.good_win_rate:.1%}  狼胜 {self.wolf_win_rate:.1%} | "
            f"平均 {self.avg_days:.1f} 天、{self.avg_deaths:.1f} 人出局 | "
            f"撞上限 {self.aborted} 局"
        )


def run_batch(
    board: Board,
    seeds: Sequence[int],
    make_policy: Callable[[Random], GamePolicy],
    *,
    max_days: int = DEFAULT_MAX_DAYS,
    progress: Callable[[int, BatchStats], None] | None = None,
) -> BatchStats:
    """跑一批对局并汇总。

    参数:
        board: 已校验的板子。带警长的板子会抛 NotImplementedError（D8 未实现），
               这里**故意不捕获**——静默跳过会让人误以为 12p 已经能跑。
        seeds: 种子序列。range(1000) 即可，固定种子保证可复现。
        make_policy: 策略工厂。每次调用返回一个全新的 policy 实例。
        max_days: 单局天数上限。撞上限的局计入 aborted，不参与胜率。
        progress: 可选回调，做进度显示用。

    注意: 只有 RuntimeError（撞天数上限）算 aborted。
          其它异常一律向上抛——"跑不完"和"跑错了"必须区分开，
          两种都吞掉的话，批量仿真会变成一个报喜不报忧的摆设。
    """
    wolf_wins = 0
    good_wins = 0
    aborted = 0
    days_total = 0
    deaths_total = 0
    events_total = 0

    for index, seed_value in enumerate(seeds):
        deal_rng = Random(seed_value)
        policy_rng = Random(seed_value + POLICY_SEED_OFFSET)
        policy = make_policy(policy_rng)

        try:
            result = play_game(board, policy, deal_rng, max_days=max_days)
        except RuntimeError:
            aborted += 1
            continue

        days_total += result.days
        deaths_total += len(result.state.deaths)
        events_total += len(result.events)

        if result.winner is Winner.WOLF:
            wolf_wins += 1
        else:
            good_wins += 1

        if progress is not None and (index + 1) % 100 == 0:
            progress(
                index + 1,
                BatchStats(
                    board_id=board.id,
                    games=index + 1,
                    aborted=aborted,
                    wolf_wins=wolf_wins,
                    good_wins=good_wins,
                    days_total=days_total,
                    deaths_total=deaths_total,
                    events_total=events_total,
                ),
            )

    return BatchStats(
        board_id=board.id,
        games=len(seeds),
        aborted=aborted,
        wolf_wins=wolf_wins,
        good_wins=good_wins,
        days_total=days_total,
        deaths_total=deaths_total,
        events_total=events_total,
    )


def _cli() -> int:
    """便捷入口：python -m engine.batch [局数]

    只是给你手动看基线数字用的，不参与生产逻辑。
    这里直接依赖 RandomPolicy 是刻意的——换成 Agent 时另有入口。
    """
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    board = load_board_by_id("9p")
    stats = run_batch(
        board,
        range(count),
        lambda rng: RandomPolicy(rng),
        progress=lambda done, s: print(f"... {done} 局  {s.summary()}"),
    )
    print()
    print(stats.summary())
    print(stats.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
