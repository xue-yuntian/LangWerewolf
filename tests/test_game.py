"""D4 验收：一局能不能从头发走到尾。

这是 W1 的门禁——"9 人局能分出胜负"。
不追求好看，只追求：有明确胜方、可复现、不死循环。
"""

import pytest

from engine.board import BOARDS_DIR, load_board
from engine.game import DEFAULT_MAX_DAYS, play_game
from engine.win import Winner
from tests.support import FirstAlivePolicy, seed

ALL_WINNERS = (Winner.WOLF, Winner.GOOD)


def test_9p_plays_to_the_end() -> None:
    board = load_board(BOARDS_DIR / "9p.json")
    result = play_game(board, FirstAlivePolicy(), seed(42))

    assert result.winner in ALL_WINNERS
    assert result.days >= 1
    assert result.days < DEFAULT_MAX_DAYS
    # 一局打完，必须有人出局，否则说明夜晚和白天都没生效
    assert any(not player.alive for player in result.state.players.values())
    # 夜数和天数要对得上：第一天之前有第一夜
    assert len(result.nights) == len(result.votes) or len(result.nights) == len(result.votes) + 1


def test_same_seed_is_reproducible() -> None:
    """固定种子必须复现同一局——D11 的批量仿真靠这个。"""
    board = load_board(BOARDS_DIR / "9p.json")
    first = play_game(board, FirstAlivePolicy(), seed(7))
    second = play_game(board, FirstAlivePolicy(), seed(7))

    assert first.winner is second.winner
    assert [d.pid for d in first.state.deaths] == [d.pid for d in second.state.deaths]


def test_different_seeds_give_different_roles() -> None:
    """不同种子应该分出不同身份，否则发牌逻辑有问题。"""
    board = load_board(BOARDS_DIR / "9p.json")
    a = play_game(board, FirstAlivePolicy(), seed(1))
    b = play_game(board, FirstAlivePolicy(), seed(2))

    roles_a = [p.role for p in a.state.players.values()]
    roles_b = [p.role for p in b.state.players.values()]
    assert roles_a != roles_b


def test_many_games_all_have_winner() -> None:
    """连跑 20 局，每局都必须有明确胜方，不能卡死。"""
    board = load_board(BOARDS_DIR / "9p.json")
    for i in range(20):
        result = play_game(board, FirstAlivePolicy(), seed(i))
        assert result.winner in ALL_WINNERS


def test_12p_rejected_until_sheriff_is_implemented() -> None:
    """12 人板带警长，D8 才实现。

    这里断言"显式报错"而不是"静默忽略"：
    静默忽略会让 12 人局看起来能跑，但整段上警流程消失了，是最难查的 bug。
    """
    board = load_board(BOARDS_DIR / "12p.json")
    assert board.options.sheriff is True

    with pytest.raises(NotImplementedError):
        play_game(board, FirstAlivePolicy(), seed(1))
