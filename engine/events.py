"""事件流 —— 一局里"发生过什么"的唯一有序记录。

────────────────────────────────────────────────────────────────
设计取向：**只做 append-only 事件日志，不做完整 Event Sourcing**
────────────────────────────────────────────────────────────────
Event Sourcing 要求状态由事件重放算出来：`state = reduce(apply, events)`。
那意味着 night.py / vote.py 的规则会有第二份实现（reducer）。两份实现必然漂移，
漂移了就出 bug，而且是"两处都看起来对、合起来错"的 bug。

本项目单进程、状态小、复盘只需要**展示**事件，不需要回到任意时刻重建状态，
所以：**GameState 仍是权威状态，事件是它的追加记录。**
这也是面试可讲的取舍——"我评估过 Event Sourcing，但这个场景不需要时间旅行"。

────────────────────────────────────────────────────────────────
为什么现在必须做（三个已定的下游）
────────────────────────────────────────────────────────────────
· D12 的 prompt 要读"公共记录"——只能读事件流。读 GameState 会把
  每个人的真实身份、狼队成员、今晚刀谁全部暴露给模型。
· D16 的 SSE 逐条下发事件，seq 当 event id，断线重连靠它补齐。
· D23 的复盘时间轴 = 可拖动的事件流。

────────────────────────────────────────────────────────────────
visible_to：为什么现在就要有
────────────────────────────────────────────────────────────────
事件要当"公共记录"用，但夜间行动天然是私有的：
    狼刀      -> 只有狼队知道
    守卫守谁  -> 只有守卫知道
    女巫用药  -> 只有女巫知道
    验人结果  -> 只有预言家知道
    谁死了    -> 全场可见

D12 会按可见性给每个 AI 过滤信息。**这个信息必须在产生事件时就记下来**，
否则 D12 只能回头改事件定义——那等于 D6 重做。

viewer 的取值约定：
    "all"       -> 公共记录（所有人都能看到的部分）
    "god"       -> 上帝视角（复盘、调试用）
    "wolves"    -> 狼队
    f"pid:{n}"  -> 只有 n 号玩家
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# 可见性
# ---------------------------------------------------------------------------

VIS_ALL = "all"
VIS_GOD = "god"
VIS_WOLVES = "wolves"


def vis_pid(pid: int) -> str:
    """只有某个座位号能看到。"""
    return f"pid:{pid}"


# ---------------------------------------------------------------------------
# 事件类型
# ---------------------------------------------------------------------------


class EventKind(StrEnum):
    """事件种类。

    SPEECH 目前保留但**不发送**：随机策略不会发言，发言是 D8（警长竞选发言）
    和 D12（AI 发言）才有的数据源。枚举成员先占位，那时加发射方即可，不用改格式。
    """

    GAME_START = "game_start"
    NIGHT_START = "night_start"
    GUARD = "guard"
    WOLF_KILL = "wolf_kill"
    WITCH_SAVE = "witch_save"
    WITCH_POISON = "witch_poison"
    SEER_CHECK = "seer_check"
    DEATH = "death"
    HUNTER_SHOT = "hunter_shot"
    DAY_START = "day_start"
    SPEECH = "speech"  # 预留：D8 / D12
    VOTE = "vote"
    VOTE_RESULT = "vote_result"
    GAME_OVER = "game_over"


# 事件格式版本。以后改 payload 结构时递增，读侧按版本分支。
EVENT_VERSION = 1


# ---------------------------------------------------------------------------
# 单条事件
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """一条已发生的事实。不可变——事件只追加，永不修改。"""

    seq: int  # 从 0 开始、连续递增，前端当 SSE 的 event id 用
    kind: EventKind
    label: str  # "第1夜" / "第1天"，前端按它分组
    visible_to: str = VIS_ALL
    payload: Mapping[str, object] = field(default_factory=dict)
    v: int = EVENT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "v": self.v,
            "seq": self.seq,
            "kind": self.kind.value,
            "label": self.label,
            "visible_to": self.visible_to,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Event:
        """反序列化。字段不合法时抛 ValueError，不做静默兜底。"""
        seq = raw.get("seq")
        if not isinstance(seq, int):
            raise ValueError(f"seq 必须是整数，收到 {seq!r}")

        kind = raw.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"kind 必须是字符串，收到 {kind!r}")

        payload = raw.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError(f"payload 必须是对象，收到 {payload!r}")

        # 先收窄再赋值：mypy 不接受 int(raw["v"])，因为 object 不满足 SupportsInt
        raw_version = raw.get("v")
        version = raw_version if isinstance(raw_version, int) else EVENT_VERSION

        return cls(
            seq=seq,
            kind=EventKind(kind),
            label=str(raw.get("label", "")),
            visible_to=str(raw.get("visible_to", VIS_ALL)),
            payload={str(k): v for k, v in payload.items()},
            v=version,
        )

    def __str__(self) -> str:
        detail = "，".join(f"{k}={v}" for k, v in self.payload.items())
        head = f"[{self.seq:03d}] {self.label} {self.kind.value}"
        return f"{head}（{detail}）" if detail else head


# ---------------------------------------------------------------------------
# 事件日志
# ---------------------------------------------------------------------------


class EventLog:
    """追加式事件日志。seq 由 emit 自动分配，保证连续无缺口。"""

    def __init__(self) -> None:
        self._events: list[Event] = []

    # ---------------- 写入 ----------------

    def emit(
        self,
        kind: EventKind,
        label: str,
        *,
        visible_to: str = VIS_ALL,
        **payload: object,
    ) -> Event:
        """追加一条事件并返回它。

        为什么 seq 不让调用方传：序号是对外的游标（SSE event id），
        必须和列表长度严格对齐。让调用方传就有重复或跳号的可能。
        """
        event = Event(
            seq=len(self._events),
            kind=kind,
            label=label,
            visible_to=visible_to,
            payload=dict(payload),
        )
        self._events.append(event)
        return event

    # ---------------- 读取 ----------------

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, index: int) -> Event:
        return self._events[index]

    def all(self) -> list[Event]:
        """上帝视角：全部事件。"""
        return list(self._events)

    def for_viewer(self, viewer: str) -> list[Event]:
        """按可见性过滤。

        viewer="all"     -> 只有公共事件（D12 给所有 AI 的公共记录）
        viewer="god"     -> 全部
        viewer="wolves"  -> 公共 + 狼队私有
        viewer="pid:3"   -> 公共 + 3 号私有
        """
        if viewer == VIS_GOD:
            return list(self._events)
        return [e for e in self._events if e.visible_to in (VIS_ALL, viewer)]

    # ---------------- 序列化 ----------------

    def to_jsonl(self) -> str:
        """一行一条事件。选 JSONL 而不是数组，是因为它和 D16 的 SSE 一一对应。"""
        return "\n".join(
            json.dumps(e.to_dict(), ensure_ascii=False) for e in self._events
        )

    @classmethod
    def from_jsonl(cls, text: str) -> EventLog:
        """从 JSONL 还原。校验 seq 连续——序号断了说明文件损坏。"""
        log = cls()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, Mapping):
                raise ValueError(f"第 {lineno} 行不是 JSON 对象：{stripped[:60]}")
            event = Event.from_dict(raw)
            if event.seq != len(log._events):
                raise ValueError(
                    f"第 {lineno} 行的 seq={event.seq}，"
                    f"但按顺序应该是 {len(log._events)}，事件流已损坏"
                )
            log._events.append(event)
        return log
