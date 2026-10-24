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
import sys
from collections.abc import Sequence
from random import Random

from engine.board import BoardConfigError, load_board_by_id
from engine.game import DEFAULT_MAX_DAYS, GameResult, play_game
from engine.night import NightResult
from engine.policy import RandomPolicy
from engine.roles import ROLE_REGISTRY, Roles
from engine.state import RuleViolation
from engine.vote import VoteResult
from engine.win import Winner

DEFAULT_SEED = 42

# 死因的展示名。
_CAUSE_ZH = {
    "wolf_kill": "狼刀",
    "poison": "毒杀",
    "vote": "投票",
    "shot": "开枪",
}


def _cause_text(cause: str) -> str:
    """死因的中文名。未知死因原样返回，方便以后加新死因。"""
    return _CAUSE_ZH.get(cause, cause)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    需要支持：
        位置参数  board_id    板子 id，如 9p / 12p（注意 12p 带警长，D8 之前会报错）
        --seed    int         随机种子，默认 DEFAULT_SEED，同种子必须复现同一局
        --max-days int        交给 play_game 的安全上限，默认用 game.DEFAULT_MAX_DAYS
        --players             可选：结尾额外打印每个玩家的真实身份
    """
    parser = argparse.ArgumentParser(
        prog="python -m engine.sim",
        description="跑一局狼人杀，把过程打印成人能读的文本",
    )
    parser.add_argument(
        "board_id", help="板子 id，如 9p、12p（12p 带警长，D8 前会报错）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子，同种子复现同一局（默认 {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=DEFAULT_MAX_DAYS,
        help=f"天数上限，防止策略死循环（默认 {DEFAULT_MAX_DAYS}）",
    )
    parser.add_argument("--players", action="store_true", help="结尾额外打印存活名单")
    return parser


def render_night(night: NightResult, *, has_guard: bool = True) -> list[str]:
    """把一夜渲染成若干行文本。

    要能看出：
        · 守卫守了谁（None 表示空守）
        · 狼刀了谁
        · 女巫救没救、毒了谁
        · 预言家验了谁、结果是"狼"还是"好人"
        · 这一夜死了谁、死因是什么（night.deaths 里每个 Death 有 pid / cause / label）

    has_guard=False 时（9p 这类没有守卫的板子）不打印守卫那一行，
    否则每夜都会出现"守卫没有发动技能"，像是守卫在挂机。
    """
    lines = [f"[{night.label}]"]
    if has_guard:
        if night.guard_target is None:
            lines.append("守卫没有发动技能。")
        else:
            lines.append(f"守卫守了 {night.guard_target} 号玩家。")
    if night.wolf_target is None:
        lines.append("狼人没有动刀。")
    else:
        lines.append(f"狼人对 {night.wolf_target} 号玩家动刀。")
    if night.witch_saved:
        lines.append(f"女巫用解药救下 {night.wolf_target} 号玩家。")
    elif night.poison_target is not None:
        lines.append(f"女巫毒死 {night.poison_target} 号玩家。")
    else:
        lines.append("女巫没有用药。")
    if night.seer_target is None:
        lines.append("预言家没有验人。")
    else:
        verdict = "狼人" if night.seer_saw_wolf else "好人"
        lines.append(f"预言家查验了 {night.seer_target} 号玩家：{verdict}")
    if not night.deaths:
        lines.append("平安夜。")
    else:
        deaths = "、".join(
            f"{death.pid}号（{_cause_text(death.cause.value)}）"
            for death in night.deaths
        )
        lines.append(f"今晚 {deaths} 出局。")
    return lines


def _format_ballots(ballots: dict[int, int | None]) -> str:
    """把票型渲染成 "1->4  2->弃" 这样的一行。"""
    return "  ".join(
        f"{voter}->{'弃' if target is None else target}"
        for voter, target in sorted(ballots.items())
    )


def _format_top(tally: dict[int, int], title: str) -> str:
    """渲染最高票。tally 为空表示全员弃票。"""
    if not tally:
        return f"{title}：无人得票"
    best = max(tally.values())
    top = sorted(pid for pid, n in tally.items() if n == best)
    names = "、".join(f"{pid} 号（{best} 票）" for pid in top)
    return f"{title}：{names}"


def render_vote(vote: VoteResult) -> list[str]:
    """把一天的投票渲染成若干行文本。

    要能看出：
        · 谁是最高票、几票
        · 发生平票时，哪些人进了 PK、PK 的票型如何
        · 最终谁出局；无人出局要明确写出来（而不是什么都不打）
    """
    lines = [f"[{vote.label}] 投票：{_format_ballots(vote.ballots)}"]
    lines.append(_format_top(vote.tally, "最高票"))

    if vote.re_vote:
        tied = "、".join(f"{pid} 号" for pid in vote.tied)
        lines.append(f"平票：{tied}，台下 PK 重投")
        lines.append(f"PK 投票：{_format_ballots(vote.pk_ballots)}")
        lines.append(_format_top(vote.pk_tally, "PK 最高票"))

    if vote.eliminated is None:
        lines.append("最终：无人出局")
    else:
        tally = vote.pk_tally if vote.re_vote else vote.tally
        votes = tally.get(vote.eliminated, 0)
        lines.append(f"最终：{vote.eliminated} 号出局（{votes} 票）")

    return lines


def render_header(result: GameResult, seed: int) -> list[str]:
    """开局：板子名、种子、每个座位号的真实身份（调试用，D12 之后会藏起来）。"""
    board = result.state.board
    roles = "  ".join(
        f"{pid}={ROLE_REGISTRY[player.role].name_zh}"
        for pid, player in sorted(result.state.players.items())
    )
    return [f"=== {board.name}（{board.id}） seed={seed} ===", f"身份：{roles}"]


def render_summary(result: GameResult) -> list[str]:
    """结尾：胜方、共几天、死亡顺序。"""
    winner = "狼人" if result.winner is Winner.WOLF else "好人"
    lines = [f"=== {winner}胜，共 {result.days} 天 ==="]

    if not result.state.deaths:
        lines.append("死亡顺序：无")
    else:
        lines.append("死亡顺序：")
        lines.extend(
            f"  {death.pid}号（{_cause_text(death.cause.value)}，{death.label}）"
            for death in result.state.deaths
        )

    return lines


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
    args = build_parser().parse_args(argv)

    try:
        board = load_board_by_id(args.board_id)
        policy = RandomPolicy(Random(args.seed))
        result = play_game(board, policy, Random(args.seed), max_days=args.max_days)
    except (BoardConfigError, NotImplementedError, RuleViolation, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    lines = render_header(result, args.seed)
    lines.append("")

    nights = {night.index: night for night in result.nights}
    votes = {vote.day: vote for vote in result.votes}
    total = max([*nights, *votes], default=0)

    for index in range(1, total + 1):
        if index in nights:
            lines.extend(render_night(nights[index], has_guard=board.has(Roles.GUARD)))
        if index in votes:
            lines.extend(render_vote(votes[index]))

    lines.append("")
    lines.extend(render_summary(result))
    if args.players:
        alive = "、".join(f"{pid}号" for pid in result.state.alive_ids()) or "无"
        lines.append(f"存活：{alive}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
