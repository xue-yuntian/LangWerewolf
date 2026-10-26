"""D4 验收：一局能不能从头发走到尾。

这是 W1 的门禁——"9 人局能分出胜负"。
不追求好看，只追求：有明确胜方、可复现、不死循环。
"""

import pytest

from engine.board import BOARDS_DIR, load_board
from engine.events import VIS_ALL, EventKind
from engine.game import DEFAULT_MAX_DAYS, play_game
from engine.state import GameState
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


def test_on_step_reports_every_phase() -> None:
    """CLI 靠 on_step 流式打印。漏一次就是“跑半天没动静”。"""
    board = load_board(BOARDS_DIR / "9p.json")
    lines: list[str] = []
    result = play_game(board, FirstAlivePolicy(), seed(3), on_step=lines.append)

    assert len(lines) == len(result.nights) + len(result.votes)
    assert lines[0].startswith("[第1夜]")
    assert lines[-1].startswith("[第")


class _Talker(FirstAlivePolicy):
    """会发言的策略。内容固定，只用来验证引擎把发言收进了事件流。"""

    def speak(self, state: GameState, pid: int) -> str | None:
        return f"{pid}号说了两句"


def test_speeches_happen_before_vote() -> None:
    """白天必须先发言再投票，而且发言是全场可见的。"""
    board = load_board(BOARDS_DIR / "9p.json")
    result = play_game(board, _Talker(), seed(3))

    speeches = [event for event in result.events if event.kind is EventKind.SPEECH]
    assert speeches
    # 发言必须全场可见，否则投票的人根本读不到
    assert all(event.visible_to == VIS_ALL for event in speeches)

    # DAY_START 后面紧跟着的就是发言，不能是投票
    kinds = [event.kind for event in result.events]
    first_day = kinds.index(EventKind.DAY_START)
    assert kinds[first_day + 1] is EventKind.SPEECH

    # 同一轮内按座位号升序——这是“依次发言”的前提
    label = speeches[0].label
    pids: list[int] = []
    for event in speeches:
        if event.label != label:
            continue
        pid = event.payload["pid"]
        assert isinstance(pid, int)
        pids.append(pid)
    assert pids == sorted(pids)


def test_strategy_without_speak_just_skips() -> None:
    """随机策略不会说话，引擎不能因此崩掉，也不该产生空的 SPEECH。"""
    board = load_board(BOARDS_DIR / "9p.json")
    result = play_game(board, FirstAlivePolicy(), seed(3))

    assert not [event for event in result.events if event.kind is EventKind.SPEECH]


def test_speeches_go_through_on_step() -> None:
    """发言必须能从进度回调里看到，否则一局跑下来全是结算，像没讨论过。"""
    board = load_board(BOARDS_DIR / "9p.json")
    lines: list[str] = []
    play_game(board, _Talker(), seed(3), on_step=lines.append)

    spoken = [line for line in lines if "号说了两句" in line]
    assert spoken
    assert spoken[0].startswith("[第1天]")


def test_on_step_does_not_leak_wolf_target() -> None:
    """进度流是公开的，报出刀口等于把夜间底牌给所有人看。"""
    board = load_board(BOARDS_DIR / "9p.json")
    lines: list[str] = []
    play_game(board, FirstAlivePolicy(), seed(3), on_step=lines.append)

    night_lines = [line for line in lines if line.startswith("[第") and "夜]" in line]
    assert night_lines
    assert all("狼刀" not in line for line in night_lines)


def test_12p_rejected_until_sheriff_is_implemented() -> None:
    """12 人板带警长，D8 才实现。

    这里断言"显式报错"而不是"静默忽略"：
    静默忽略会让 12 人局看起来能跑，但整段上警流程消失了，是最难查的 bug。
    """
    board = load_board(BOARDS_DIR / "12p.json")
    assert board.options.sheriff is True

    with pytest.raises(NotImplementedError):
        play_game(board, FirstAlivePolicy(), seed(1))
