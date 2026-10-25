"""D6 事件流的测试。

计划里的验收是"同一局重放两次结果一致"，下面拆成更可查的几条：
  · 序号连续（否则 SSE 的 event id 和断线重连会错位）
  · 同种子事件流完全一致（这就是"可重放"）
  · JSONL 往返不丢信息（D20 存档的地基）
  · 每次死亡都有对应事件（防漏发射）
  · 可见性过滤真的挡住私有事件（D12 的信息隔离靠它）
"""

from engine.board import BOARDS_DIR, load_board
from engine.events import VIS_ALL, VIS_GOD, VIS_WOLVES, EventKind, EventLog
from engine.game import GameResult, play_game
from engine.policy import RandomPolicy
from tests.support import seed


def _play(seed_value: int = 42) -> GameResult:
    board = load_board(BOARDS_DIR / "9p.json")
    return play_game(board, RandomPolicy(seed(seed_value)), seed(seed_value))


def test_event_seq_is_contiguous() -> None:
    """seq 必须从 0 开始连续，不能跳号也不能重复。"""
    game = _play()

    seqs = [event.seq for event in game.events]
    assert seqs == list(range(len(game.events)))


def test_game_has_basic_event_kinds() -> None:
    """一局里必须出现的骨架事件：开局、入夜、白天、结束。"""
    game = _play()
    kinds = {event.kind for event in game.events}

    assert EventKind.GAME_START in kinds
    assert EventKind.NIGHT_START in kinds
    assert EventKind.DAY_START in kinds
    assert EventKind.VOTE in kinds
    assert EventKind.VOTE_RESULT in kinds
    assert EventKind.GAME_OVER in kinds


def test_same_seed_produces_same_events() -> None:
    """同种子两次跑出的整条事件流必须逐条相同——这就是"可重放"。"""
    first = _play(7)
    second = _play(7)

    def snapshot(game: GameResult) -> list[tuple[str, str, str, dict[str, object]]]:
        return [
            (e.kind.value, e.label, e.visible_to, dict(e.payload)) for e in game.events
        ]

    assert snapshot(first) == snapshot(second)


def test_jsonl_roundtrip() -> None:
    """序列化再反序列化，一条都不能丢。"""
    game = _play()
    restored = EventLog.from_jsonl(game.events.to_jsonl())

    assert len(restored) == len(game.events)
    assert [e.to_dict() for e in restored] == [e.to_dict() for e in game.events]


def test_jsonl_detects_broken_sequence() -> None:
    """seq 断了必须报错，而不是静默接受——存档损坏要能被发现。"""
    text = "\n".join(
        [
            '{"seq": 0, "kind": "day_start", "label": "第1天"}',
            '{"seq": 5, "kind": "day_start", "label": "第1天"}',
        ]
    )

    try:
        EventLog.from_jsonl(text)
    except ValueError as exc:
        assert "seq" in str(exc)
    else:  # pragma: no cover - 只有实现错了才会走到这里
        raise AssertionError("seq 跳号时应当抛 ValueError")


def test_every_death_has_event() -> None:
    """state.deaths 里有几个人死了，就必须有几条 DEATH 事件。"""
    game = _play()

    death_events = [e for e in game.events if e.kind is EventKind.DEATH]
    assert len(death_events) == len(game.state.deaths)
    assert {e.payload["pid"] for e in death_events} == {
        death.pid for death in game.state.deaths
    }


def test_for_viewer_filters_private_events() -> None:
    """可见性过滤：公共记录里不能出现身份表、狼刀、验人结果。"""
    game = _play()

    public = game.events.for_viewer(VIS_ALL)
    assert public, "公共记录不能是空的"
    assert all(e.visible_to == VIS_ALL for e in public)

    # 身份表是上帝视角，绝不能出现在公共记录里
    assert all(e.kind is not EventKind.GAME_START for e in public)
    # 狼刀和验人结果同理
    assert all(e.kind is not EventKind.WOLF_KILL for e in public)
    assert all(e.kind is not EventKind.SEER_CHECK for e in public)

    # 上帝视角看得到全部
    assert len(game.events.for_viewer(VIS_GOD)) == len(game.events)

    # 狼队至少能看到自己的刀
    wolves = game.events.for_viewer(VIS_WOLVES)
    assert len(wolves) >= len(public)


def test_visibility_of_specific_kinds() -> None:
    """逐个核对容易记错的可见性归属。"""
    game = _play()

    for event in game.events:
        if event.kind is EventKind.WOLF_KILL:
            assert event.visible_to == VIS_WOLVES
        elif event.kind is EventKind.SEER_CHECK:
            assert event.visible_to.startswith("pid:")
        elif event.kind is EventKind.GAME_START:
            assert event.visible_to == VIS_GOD
        elif event.kind in (EventKind.DEATH, EventKind.VOTE, EventKind.GAME_OVER):
            assert event.visible_to == VIS_ALL
