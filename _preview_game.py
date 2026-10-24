"""临时脚本：渲染一局完整过程，看完即删。"""

import sys
from random import Random

_OUT = open("_preview.txt", "w", encoding="utf-8", newline="\n")
sys.stdout = _OUT

from engine.board import load_board_by_id
from engine.game import play_game
from engine.policy import RandomPolicy
from engine.roles import ROLE_REGISTRY

board = load_board_by_id("9p")

# 从 100 个种子里挑天数最多的一局，事件更完整，适合观察
best_seed = 0
best = None
for seed in range(100):
    candidate = play_game(board, RandomPolicy(Random(seed)), Random(seed))
    if best is None or candidate.days > best.days:
        best_seed, best = seed, candidate

SEED = best_seed
result = best
state = result.state

print(f"=== {board.name}（{board.id}） seed={SEED} ===")
print("（100 局中天数最多的一局）")
print(
    "身份："
    + "  ".join(
        f"{pid}={ROLE_REGISTRY[p.role].name_zh}"
        for pid, p in sorted(state.players.items())
    )
)
print()

for night in result.nights:
    parts = [f"[第{night.index}夜]"]
    if night.guard_target is not None:
        parts.append(f"守卫守 {night.guard_target} 号")
    if night.wolf_target is not None:
        parts.append(f"狼刀 {night.wolf_target} 号")
    else:
        parts.append("狼空刀")
    if night.witch_saved:
        parts.append("女巫用解药")
    if night.poison_target is not None:
        parts.append(f"女巫毒 {night.poison_target} 号")
    if night.seer_target is not None:
        verdict = "狼" if night.seer_saw_wolf else "好人"
        parts.append(f"预言家验 {night.seer_target} 号 -> {verdict}")
    if not night.deaths:
        parts.append("平安夜")
    print("；".join(parts))
    for death in night.deaths:
        print(f"        {death.pid} 号出局（{death.cause.value}）")

for vote in result.votes:
    ballots = "  ".join(
        f"{v}->{'弃' if t is None else t}" for v, t in sorted(vote.ballots.items())
    )
    print(f"[第{vote.day}天] 投票：{ballots}")
    if vote.re_vote:
        pk = "  ".join(
            f"{v}->{'弃' if t is None else t}" for v, t in sorted(vote.pk_ballots.items())
        )
        print(f"       平票 {vote.tied}，台下 PK 重投：{pk}")
    if vote.eliminated is None:
        print("       无人出局")
    else:
        tally = vote.pk_tally if vote.re_vote else vote.tally
        print(f"       {vote.eliminated} 号出局（{tally.get(vote.eliminated, 0)} 票）")
    others = [
        d for d in state.deaths if d.label == vote.label and d.pid != vote.eliminated
    ]
    for death in others:
        print(f"       {death.pid} 号被带走（{death.cause.value}）")

print()
print("死亡顺序：")
for death in state.deaths:
    print(f"  {death.pid} 号  {death.cause.value}  {death.label}")
print()
print(f"=== {result.winner.value} 胜，共 {result.days} 天 ===")
_OUT.close()
