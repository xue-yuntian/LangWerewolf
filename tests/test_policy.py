"""D5：RandomPolicy 的测试 —— 现在全是红的，当作 TODO 清单用。

每实现一个方法，就删掉对应测试里的 NotImplementedError 并写真实断言。
全部变绿 = D5 完成。

注意区分两类测试：
    · 合法性测试（不能守同一人、不能毒自己……）——证明策略不会把引擎搞崩
    · 整体测试（100 个种子跑完都不抛异常）——证明是"合法随机"而不是"恰好没撞上"

写测试时你会用到下面这些 import（现在不 import，否则 ruff 判 F401 未使用）：

    from engine.board import BOARDS_DIR, load_board
    from engine.game import play_game
    from engine.policy import RandomPolicy
    from engine.roles import Roles
    from tests.support import make_state, seed
"""

import pytest


def test_pick_returns_none_on_empty() -> None:
    """_pick([]) 必须返回 None，不能抛 IndexError。"""
    raise NotImplementedError("D5 待实现：test_pick_returns_none_on_empty")


def test_guard_never_repeats_last_guarded() -> None:
    """连续两夜：第二夜不能守第一夜守过的人。

    写法建议：
        state = make_state({1: Roles.WOLF, 2: Roles.GUARD, ...})
        policy = RandomPolicy(seed(0))
        first = policy.choose_guard(state)
        state.last_guarded = first
        然后调用 100 次，断言每次都 != first
    """
    raise NotImplementedError("D5 待实现：test_guard_never_repeats_last_guarded")


def test_witch_never_poisons_self() -> None:
    """女巫不能毒自己——跑 200 次都不该出现自己的座位号。"""
    raise NotImplementedError("D5 待实现：test_witch_never_poisons_self")


def test_seer_never_checks_self() -> None:
    """预言家不能验自己。"""
    raise NotImplementedError("D5 待实现：test_seer_never_checks_self")


def test_vote_never_self() -> None:
    """投票不能投自己，否则 vote.py 会抛 RuleViolation。"""
    raise NotImplementedError("D5 待实现：test_vote_never_self")


def test_hunter_never_shoots_self() -> None:
    """猎人开枪不能打自己。"""
    raise NotImplementedError("D5 待实现：test_hunter_never_shoots_self")


def test_policy_finishes_100_games_without_violation() -> None:
    """核心验收：100 个种子连续跑完，一条 RuleViolation 都不许有。

    这是"合法随机"和"完全随机"的分界线。这里红了，
    说明某个方法的候选集筛得不对，去对应方法注释里核对条件。
    """
    raise NotImplementedError("D5 待实现：test_policy_finishes_100_games_without_violation")


def test_same_seed_reproduces_same_game() -> None:
    """同种子必须复现同一局：胜方和死亡顺序都要一致。"""
    raise NotImplementedError("D5 待实现：test_same_seed_reproduces_same_game")


@pytest.mark.skip(reason="等 sim.py 实现后打开")
def test_sim_cli_runs() -> None:
    """CLI 冒烟测试：`python -m engine.sim 9p --seed 1` 返回 0 且有输出。

    实现完 sim.py 后把 skip 去掉，改成：
        from engine.sim import main
        assert main(["9p", "--seed", "1"]) == 0
    """
