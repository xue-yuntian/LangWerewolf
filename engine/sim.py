"""D5：CLI 播放器 —— 把一局狼人杀打成人类能读的过程。

────────────────────────────────────────────────────────────────
你要实现什么
────────────────────────────────────────────────────────────────
让这条命令能跑通，并打印出完整的一局：

    python -m engine.sim 9p --seed 42

输出大概长这样（格式随你，但要能看懂"谁死了、怎么死的、谁被投出去"）：

    === 9人快速局（9p） seed=42 ===
    身份：1=狼 2=民 3=预言家 4=女巫 5=猎人 6=狼 7=民 8=狼 9=民

    [第1夜] 狼刀 4 号；女巫自救；预言家验 1 号 -> 狼
    [第1天] 投票 8 号：4 票 -> 8 号出局
    [第2夜] 狼刀 3 号；无药；预言家验 6 号 -> 狼
    ...
    === 好人胜，共 4 天 ===

────────────────────────────────────────────────────────────────
数据从哪来（不用你重新收集）
────────────────────────────────────────────────────────────────
play_game() 已经返回 GameResult，里面有：
    result.nights  list[NightResult]  每一夜干了什么
    result.votes   list[VoteResult]   每一天的票型和结果
    result.log     list[str]          引擎写的流水文本
    result.winner  Winner
    result.state   GameState          最终状态（能拿到每个玩家的真实身份）

NightResult 字段：index / guard_target / wolf_target / witch_saved /
                  poison_target / seer_target / seer_saw_wolf / deaths
VoteResult 字段：day / ballots / tally / tied / re_vote / pk_ballots /
                 pk_tally / eliminated

────────────────────────────────────────────────────────────────
一个设计提醒
────────────────────────────────────────────────────────────────
「渲染」和「运行」要分开：render_* 函数只吃数据、只返回字符串，不打印、不读状态。
这样以后 D15 要做 API、D23 要做复盘页时，同一份数据可以渲染成 HTML，
而不用把 CLI 的逻辑搬过去。
"""

import argparse
from collections.abc import Sequence

from engine.game import GameResult
from engine.night import NightResult
from engine.vote import VoteResult

# 实现时你还会需要这些（现在不用 import，否则 ruff 判 F401 未使用）：
#   from random import Random
#   from engine.board import load_board_by_id
#   from engine.game import DEFAULT_MAX_DAYS, play_game
#   from engine.policy import RandomPolicy

DEFAULT_SEED = 42


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    需要支持：
        位置参数  board_id    板子 id，如 9p / 12p（注意 12p 带警长，D8 之前会报错）
        --seed    int         随机种子，默认 DEFAULT_SEED，同种子必须复现同一局
        --max-days int        交给 play_game 的安全上限，默认用 game.DEFAULT_MAX_DAYS
        --players             可选：结尾额外打印每个玩家的真实身份
    """
    raise NotImplementedError("D5 待实现：build_parser")


def render_night(night: NightResult, state_players: object = None) -> list[str]:
    """把一夜渲染成若干行文本。

    要能看出：
        · 守卫守了谁（None 表示空守）
        · 狼刀了谁
        · 女巫救没救、毒了谁
        · 预言家验了谁、结果是"狼"还是"好人"
        · 这一夜死了谁、死因是什么（night.deaths 里每个 Death 有 pid / cause / label）

    第二参数先留着（打印玩家身份时可能要），用不上就删掉签名里那一项。
    """
    raise NotImplementedError("D5 待实现：render_night")


def render_vote(vote: VoteResult) -> list[str]:
    """把一天的投票渲染成若干行文本。

    要能看出：
        · 谁是最高票、几票
        · 发生平票时，哪些人进了 PK、PK 的票型如何
        · 最终谁出局；无人出局要明确写出来（而不是什么都不打）
    """
    raise NotImplementedError("D5 待实现：render_vote")


def render_summary(result: GameResult) -> list[str]:
    """开局/结尾的摘要。

    开头：板子名、种子、每个座位号的真实身份（调试用，D12 之后会藏起来）
    结尾：胜方、共几天、死亡顺序
    """
    raise NotImplementedError("D5 待实现：render_summary")


def main(argv: Sequence[str] | None = None) -> int:
    """入口。

    步骤建议：
        1. args = build_parser().parse_args(argv)
        2. board = load_board_by_id(args.board_id)
        3. policy = RandomPolicy(Random(args.seed))
        4. result = play_game(board, policy, Random(args.seed), max_days=...)
           注意：两处 Random(args.seed) 是两个独立实例，别复用同一个对象——
           复用会让"发牌"和"决策"共享随机流，复现难度上升。
        5. 依次 print 各段渲染结果
        6. return 0；出错时 print 到 stderr 并 return 1
    """
    raise NotImplementedError("D5 待实现：main")


if __name__ == "__main__":
    raise SystemExit(main())
