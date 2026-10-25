"""D5：RandomPolicy 的验收测试。

两类测试：
    · 合法性测试（不能守同一人、不能毒自己……）——证明策略不会把引擎搞崩
    · 整体测试（100 个种子跑完都不抛异常）——证明是"合法随机"而不是"恰好没撞上"
"""

import pytest

from engine.board import BOARDS_DIR, load_board
from engine.game import play_game
from engine.policy import RandomPolicy
from engine.roles import Roles
from engine.sim import main
from engine.win import Winner
from tests.support import make_state, seed


def test_pick_returns_none_on_empty() -> None:
    """_pick([]) 必须返回 None，不能抛 IndexError。"""
    policy = RandomPolicy(seed(0))
    assert policy._pick([]) is None
    assert policy._pick([7]) == 7


def test_guard_never_repeats_last_guarded() -> None:
    """连续两夜：第二夜不能守第一夜守过的人。"""
    state = make_state({1: Roles.WOLF, 2: Roles.GUARD, 3: Roles.VILLAGER, 4: Roles.VILLAGER})
    policy = RandomPolicy(seed(0))

    first = policy.choose_guard(state)
    assert first is not None
    state.last_guarded = first

    for _ in range(100):
        target = policy.choose_guard(state)
        assert target is not None
        assert target != first


def test_witch_never_poisons_self() -> None:
    """女巫不能毒自己——跑 200 次都不该出现自己的座位号。"""
    state = make_state({1: Roles.WOLF, 2: Roles.WITCH, 3: Roles.VILLAGER, 4: Roles.VILLAGER})
    policy = RandomPolicy(seed(0))

    used = 0
    for _ in range(200):
        target = policy.choose_witch_poison(state)
        if target is not None:
            used += 1
            assert target != 2
    assert used > 0  # 固定种子下应当至少有一次用毒


def test_seer_never_checks_self() -> None:
    """预言家不能验自己，也不能重复验已经验过的人。"""
    state = make_state({1: Roles.WOLF, 2: Roles.SEER, 3: Roles.VILLAGER, 4: Roles.VILLAGER})
    state.player(1).checked = True  # 1 号已经被验过
    policy = RandomPolicy(seed(0))

    for _ in range(100):
        target = policy.choose_seer_check(state)
        assert target is not None
        assert target not in {1, 2}


def test_vote_never_self() -> None:
    """投票不能投自己，否则 vote.py 会抛 RuleViolation。"""
    state = make_state({1: Roles.WOLF, 2: Roles.VILLAGER, 3: Roles.VILLAGER})
    policy = RandomPolicy(seed(0))

    for voter in state.alive_ids():
        for _ in range(50):
            target = policy.vote(state, voter)
            assert target is not None
            assert target != voter


def test_hunter_never_shoots_self() -> None:
    """猎人开枪不能打自己。"""
    state = make_state({1: Roles.WOLF, 2: Roles.HUNTER, 3: Roles.VILLAGER})
    policy = RandomPolicy(seed(0))

    for _ in range(100):
        target = policy.hunter_shot(state, 2)
        assert target is not None
        assert target != 2


def test_witch_does_not_save_and_poison_same_night() -> None:
    """一晚只能用一瓶药：决定救人的同一夜不能再毒。"""
    state = make_state({1: Roles.WOLF, 2: Roles.WITCH, 3: Roles.VILLAGER, 4: Roles.VILLAGER})
    policy = RandomPolicy(seed(0))

    for _ in range(100):
        if policy.choose_witch_save(state, killed=3):
            assert policy.choose_witch_poison(state) is None
            return
    pytest.fail("100 次都没有选择救，随机逻辑可能有问题")


def test_witch_cannot_self_save_after_first_night() -> None:
    """女巫只有首夜能自救，之后自己被刀必须选择不救。"""
    state = make_state({1: Roles.WOLF, 2: Roles.WITCH, 3: Roles.VILLAGER})
    state.night = 2
    policy = RandomPolicy(seed(0))

    assert policy.choose_witch_save(state, killed=2) is False


def test_policy_finishes_100_games_without_violation() -> None:
    """核心验收：100 个种子连续跑完，一条 RuleViolation 都不许有。"""
    board = load_board(BOARDS_DIR / "9p.json")

    for value in range(100):
        result = play_game(board, RandomPolicy(seed(value)), seed(value))
        assert result.winner in (Winner.WOLF, Winner.GOOD)


def test_same_seed_reproduces_same_game() -> None:
    """同种子必须复现同一局：发牌、胜方和死亡顺序都要一致。"""
    board = load_board(BOARDS_DIR / "9p.json")

    first = play_game(board, RandomPolicy(seed(42)), seed(42))
    second = play_game(board, RandomPolicy(seed(42)), seed(42))

    assert first.winner == second.winner
    assert first.state.deaths == second.state.deaths
    assert {pid: p.role for pid, p in first.state.players.items()} == {
        pid: p.role for pid, p in second.state.players.items()
    }


def test_sim_cli_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI 冒烟测试：`python -m engine.sim 9p --seed 1` 返回 0 且有输出。"""
    assert main(["9p", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "第1夜" in out
    assert "胜" in out
