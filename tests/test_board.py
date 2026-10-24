"""D2 板子配置的单测。

分两类：
  · 正向——合法的板子能加载，字段和派生值对得上
  · 反向——不合法的板子必须抛 BoardConfigError，且 errors 里说清原因

反向用例优先用 parse_board(dict)，不落磁盘：反例只关心"能不能拦住"，
不关心文件 IO。只有"id 与文件名一致"那条必须用真文件。
"""

import json
from pathlib import Path

import pytest

from engine.board import BOARDS_DIR, BoardConfigError, load_board, parse_board
from engine.roles import ROLE_REGISTRY, Faction, Roles

VALID_9P = {
    "id": "9p",
    "name": "9人快速局",
    "roles": {"wolf": 3, "villager": 3, "seer": 1, "witch": 1, "hunter": 1},
}


def _raw(**overrides: object) -> dict[str, object]:
    """在合法 9 人板基础上打补丁，构造反例。"""
    raw: dict[str, object] = json.loads(json.dumps(VALID_9P))
    raw.update(overrides)
    return raw


# --------------------------------------------------------------------------
# 正向
# --------------------------------------------------------------------------


def test_9p_loads() -> None:
    board = load_board(BOARDS_DIR / "9p.json")
    assert board.id == "9p"
    assert board.total_players == 9
    assert board.count(Roles.WOLF) == 3
    assert board.has(Roles.GUARD) is False
    assert board.options.sheriff is False


def test_12p_loads() -> None:
    board = load_board(BOARDS_DIR / "12p.json")
    assert board.id == "12p"
    assert board.total_players == 12
    assert board.has(Roles.GUARD) is True
    assert board.options.sheriff is True


def test_faction_counts() -> None:
    board = parse_board(VALID_9P)
    counts = board.faction_counts()
    assert counts[Faction.WOLF] == 3
    assert counts[Faction.GOD] == 3
    assert counts[Faction.VILLAGER] == 3
    assert sum(counts.values()) == board.total_players


def test_faction_and_role_never_equal() -> None:
    """阵营和身份的取值有重叠，必须保证两者不相等，否则当 dict 键会互相覆盖。"""
    assert Faction.WOLF != Roles.WOLF
    assert Faction.VILLAGER != Roles.VILLAGER


# --------------------------------------------------------------------------
# 反向：形状错误（由 Pydantic 拦）
# --------------------------------------------------------------------------


def test_unknown_role_rejected() -> None:
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles={"werewolf": 3, "villager": 3, "seer": 1}))
    assert any("werewolf" in p for p in exc.value.problems)


def test_zero_count_rejected() -> None:
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles={"wolf": 0, "villager": 3, "seer": 1}))
    assert any("roles.wolf" in p for p in exc.value.problems)


def test_extra_field_rejected() -> None:
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(option={"sheriff": True}))
    assert any("option" in p for p in exc.value.problems)


# --------------------------------------------------------------------------
# 反向：语义错误（由 _collect_problems 拦）
# --------------------------------------------------------------------------


def test_total_out_of_range() -> None:
    roles = {"wolf": 8, "villager": 15, "seer": 1}
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles=roles))
    assert any("总人数 24" in p for p in exc.value.problems)


def test_witch_without_wolf() -> None:
    roles = {"witch": 1, "villager": 4, "hunter": 1}
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles=roles))
    problems = exc.value.problems
    assert any("依赖 wolf" in p for p in problems)
    assert any("没有狼人" in p for p in problems)


def test_slaughter_side_needs_villager() -> None:
    roles = {"wolf": 2, "seer": 1, "witch": 1, "hunter": 1, "guard": 1}
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles=roles))
    assert any("平民" in p for p in exc.value.problems)


def test_unique_role_cannot_repeat() -> None:
    roles = {"wolf": 2, "villager": 3, "seer": 2}
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles=roles))
    assert any("唯一身份" in p for p in exc.value.problems)


# --------------------------------------------------------------------------
# 聚合：一次报全，不是遇到第一个就停
# --------------------------------------------------------------------------


def test_multiple_problems_aggregated() -> None:
    # 没有狼 + 女巫依赖狼 + 总人数 3 太少 —— 三个问题该一次报出来
    roles = {"witch": 1, "villager": 2}
    with pytest.raises(BoardConfigError) as exc:
        parse_board(_raw(roles=roles))
    assert len(exc.value.problems) >= 3
    text = str(exc.value)
    assert "总人数" in text
    assert "没有狼人" in text


# --------------------------------------------------------------------------
# 文件相关
# --------------------------------------------------------------------------


def test_id_must_match_filename(tmp_path: Path) -> None:
    path = tmp_path / "9p.json"
    path.write_text(json.dumps({**VALID_9P, "id": "wrong"}), encoding="utf-8")
    with pytest.raises(BoardConfigError) as exc:
        load_board(path)
    assert any("不一致" in p for p in exc.value.problems)


def test_missing_board_file() -> None:
    with pytest.raises(BoardConfigError) as exc:
        load_board(BOARDS_DIR / "99p.json")
    assert any("读取失败" in p for p in exc.value.problems)


# --------------------------------------------------------------------------
# 元测试：注册表必须覆盖所有身份
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role", list(Roles))
def test_registry_covers_all_roles(role: Roles) -> None:
    """以后加新身份忘了注册，这条会红。"""
    assert role in ROLE_REGISTRY
