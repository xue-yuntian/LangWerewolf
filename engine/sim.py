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
    result.events  EventLog           逐条发生的事实（D6 起取代原来的 log 字段）
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
from os import getenv
from random import Random
from typing import TYPE_CHECKING

from engine.board import Board, BoardConfigError, load_board_by_id
from engine.game import DEFAULT_MAX_DAYS, GamePolicy, GameResult, play_game
from engine.night import NightResult
from engine.policy import RandomPolicy
from engine.roles import ROLE_REGISTRY, Roles
from engine.state import RuleViolation
from engine.view import cause_text
from engine.vote import VoteResult
from engine.win import Winner

if TYPE_CHECKING:
    from llm import TraceEntry

DEFAULT_SEED = 42


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    需要支持：
        位置参数  board_id    板子 id，如 9p / 12p（注意 12p 带警长，D8 之前会报错）
        --seed    int         随机种子，默认 DEFAULT_SEED，同种子必须复现同一局
        --max-days int        交给 play_game 的安全上限，默认用 game.DEFAULT_MAX_DAYS
        --players             可选：结尾额外打印每个玩家的真实身份
        --scene   str         人设场景（只对 --policy llm 有效）
    """
    parser = argparse.ArgumentParser(
        prog="python -m engine.sim",
        description="跑一局狼人杀，把过程打印成人能读的文本",
    )
    parser.add_argument("board_id", help="板子 id，如 9p、12p（12p 带警长，D8 前会报错）")
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
    parser.add_argument(
        "--god",
        action="store_true",
        help="上帝视角：打印身份表和夜晚每一步动作（默认只输出公开信息）",
    )
    parser.add_argument(
        "--policy",
        choices=["random", "llm"],
        default="random",
        help="决策来源：random=随机策略（默认，离线、秒出），llm=每个玩家各问一次模型",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每次调用的提示词、思维链和原始输出（只在 --policy llm 下有效）",
    )
    parser.add_argument(
        "--scene",
        default="",
        help=(
            "人设场景（只对 --policy llm 有效）：一句话设定，"
            "开局前让模型按它生成人设；不给就用预制风格"
        ),
    )
    return parser


def _print_trace(entry: "TraceEntry") -> None:
    """把一次调用摊开打：提示词 -> 思考 -> 原始输出。

    全部走 stderr，和 stdout 的最终报告分开。一局 100+ 次调用，
    user 段每次都是完整视角，输出会很长——想看的话建议
        --verbose 2> trace.txt
    把对局结果和调用日志分开存。
    """
    head = (
        f"\n{'=' * 72}\n"
        f"{entry.model}  {entry.latency_ms:.0f}ms  "
        f"{entry.tokens_in}in/{entry.tokens_out}out"
    )
    print(head, file=sys.stderr, flush=True)
    print(f"--- SYSTEM ---\n{entry.system}", file=sys.stderr, flush=True)
    print(f"--- USER ---\n{entry.user}", file=sys.stderr, flush=True)
    if entry.reasoning:
        print(f"--- THINK ---\n{entry.reasoning}", file=sys.stderr, flush=True)
    if entry.ok:
        print(f"--- OUTPUT ---\n{entry.content}", file=sys.stderr, flush=True)
    else:
        print(f"--- FAILED ---\n{entry.error}", file=sys.stderr, flush=True)


def _print_wolf_talk(text: str) -> None:
    """上帝视角专用：每夜狼队会议实时打到 stderr。

    和进度流走同一条路，但有 --god 才会挂上去。它不参与最后那份
    stdout 报告：会议内容进不了 render_night（那里只有 NightResult），
    而把讨论写进 NightResult 等于让引擎知道 policy 在想什么。
    """
    print(text, file=sys.stderr, flush=True)


def _build_policy(kind: str, seed: int, board: Board, scene: str) -> GamePolicy:
    """按 --policy 造策略。

    llm 分支的 import 放在函数内：engine 的规则模块不该依赖 agents，
    sim 作为最外层展示层可以，但关在函数里能让这条依赖在静态图上
    也看得见——只有用户显式要跑 LLM 局时才会碰到。

    seed 对 LLM 策略只用来分配人设，不参与发牌。给了 scene 就不看
    seed 了：人设由场景决定，同一个种子换个场景就是另一桌人。

    人设生成失败不报错，只打印一行提示后回落到预制风格。
    开局前的一次创作不该拦住整局。
    """
    if kind == "random":
        return RandomPolicy(Random(seed))

    from agents.agent import LLMAgentPolicy
    from llm.client import DEFAULT_MODEL, _load_dotenv

    _load_dotenv()
    if not getenv("LLM_API_KEY", "").strip():
        # 这里必须硬拦：策略层把调用失败退化成“不行动”，没 key 会跑出一局
        # 全员空刀、全员弃票的垃圾局，还不报错
        raise RuntimeError("缺少 LLM_API_KEY：复制 .env.example 为 .env 并填入 key")

    if not scene.strip():
        return LLMAgentPolicy(seed=seed)

    from agents.director import generate_personas

    # 人设是创作任务，一局只调一次，值得用强模型；对局里仍走 fast
    writer = getenv("LLM_MODEL_STRONG") or getenv("LLM_MODEL_FAST") or DEFAULT_MODEL
    personas = generate_personas(scene, board.total_players, model=writer)
    if not personas:
        print("人设导演：生成失败，回落到预制风格", file=sys.stderr, flush=True)
        return LLMAgentPolicy(seed=seed)

    print(
        f"人设导演：按「{scene}」生成了 {len(personas)} 个人设",
        file=sys.stderr,
        flush=True,
    )
    return LLMAgentPolicy(personalities=personas)


def render_night(night: NightResult, *, god: bool = False, has_guard: bool = True) -> list[str]:
    """把一夜渲染成若干行文本。

    默认是**公开视角**：只写谁出局、死因是什么。把刀口、用药、验人
    摆在观众面前，等于开局就摊底牌，后面的推理全部作废。
    god=True 才展开完整过程。

    has_guard=False 时（9p 这类没有守卫的板子）不打印守卫那一行，
    否则每夜都会出现“守卫没有发动技能”，像是守卫在挂机。
    """
    if not god:
        if not night.deaths:
            return [f"[{night.label}] 平安夜。"]
        # 死因也不写：被刀还是被毒是玩家要推理的事，写出来等于替他们推理
        deaths = "、".join(f"{death.pid}号" for death in night.deaths)
        return [f"[{night.label}] {deaths} 出局。"]

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
            f"{death.pid}号（{cause_text(death.cause.value)}）" for death in night.deaths
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


def render_header(result: GameResult, seed: int, *, god: bool = False) -> list[str]:
    """开局：板子名、种子。

    身份表是上帝视角，只有 god=True 才打印——默认输出要能直接给
    观众看，开局就把底牌翻出来，后面所有推理都没意义了。
    """
    board = result.state.board
    lines = [f"=== {board.name}（{board.id}） seed={seed} ==="]
    if god:
        roles = "  ".join(
            f"{pid}={ROLE_REGISTRY[player.role].name_zh}"
            for pid, player in sorted(result.state.players.items())
        )
        lines.append(f"身份：{roles}")
    return lines


def render_summary(result: GameResult, *, god: bool = False) -> list[str]:
    """结尾：胜方、共几天、死亡顺序。

    默认只写谁在哪一轮出局，不写死因——观众应该自己从发言和
    票型里推。god=True 才带死因。
    """
    winner = "狼人" if result.winner is Winner.WOLF else "好人"
    lines = [f"=== {winner}胜，共 {result.days} 天 ==="]

    if not result.state.deaths:
        lines.append("死亡顺序：无")
    else:
        lines.append("死亡顺序：")
        if god:
            lines.extend(
                f"  {death.pid}号（{cause_text(death.cause.value)}，{death.label}）"
                for death in result.state.deaths
            )
        else:
            lines.extend(f"  {death.pid}号（{death.label}）" for death in result.state.deaths)

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

    def progress(line: str) -> None:
        # flush 是关键：不刷的话管道里会攒到进程结束才吐出来，就失去“实时”的意义了
        print(line, file=sys.stderr, flush=True)

    try:
        board = load_board_by_id(args.board_id)
        if args.scene.strip() and args.policy != "llm":
            print("提示：--scene 只对 --policy llm 有效，本次忽略", file=sys.stderr)
        policy = _build_policy(args.policy, args.seed, board, args.scene)
        if args.god and hasattr(policy, "wolf_talk"):
            # 只有 LLM 策略会开狼队会议，所以用 hasattr 探一下而不是 isinstance：
            # 前者不必把 agents 拉进 main 的命名空间
            policy.wolf_talk = _print_wolf_talk
        if args.policy == "llm":
            model = getattr(policy, "model", "LLM")
            # 含白天发言：9 人局每天多 9 次，整局约 120 次
            print(
                f"正在用 {model} 跑一局，约 120 次调用（含白天发言），请稍等…",
                file=sys.stderr,
                flush=True,
            )
        if args.verbose:
            from llm import set_trace

            set_trace(_print_trace)
            print("trace 已开启：每次调用都会打印完整提示词", file=sys.stderr)
        result = play_game(
            board,
            policy,
            Random(args.seed),
            max_days=args.max_days,
            on_step=progress if args.policy == "llm" else None,
        )
    except (BoardConfigError, NotImplementedError, RuleViolation, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    lines = render_header(result, args.seed, god=args.god)
    lines.append("")

    nights = {night.index: night for night in result.nights}
    votes = {vote.day: vote for vote in result.votes}
    total = max([*nights, *votes], default=0)

    for index in range(1, total + 1):
        if index in nights:
            lines.extend(
                render_night(nights[index], god=args.god, has_guard=board.has(Roles.GUARD))
            )
        if index in votes:
            lines.extend(render_vote(votes[index]))

    lines.append("")
    lines.extend(render_summary(result, god=args.god))
    if args.players:
        alive = "、".join(f"{pid}号" for pid in result.state.alive_ids()) or "无"
        lines.append(f"存活：{alive}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
