"""笔记 —— AI 自己的判断累积。

────────────────────────────────────────────────────────────────
和 private_lines / public_lines 的区别
────────────────────────────────────────────────────────────────
那两段是**引擎知道的事实**：谁被刀、谁验了谁、谁投了谁。
笔记是**这个玩家自己的判断**："我认为 3 号是狼，因为他第 1 天站错队。"

它可能是错的——这正是它存在的意义。没有笔记，AI 每一轮都像刚开局，
发言之间没有连续性，也就谈不上"策略"，只剩下"每轮重新推理一遍"。

    引擎事实  →  private_lines / public_lines（来自事件流，可复现）
    个人判断  →  notes（来自模型自己的输出，可能错）

────────────────────────────────────────────────────────────────
谁写、谁读
────────────────────────────────────────────────────────────────
谁写：模型自己。D12 的输出契约里加一个可选字段 note，每轮追加一条。
谁读：下一次 view_for 的 notes 参数，拼进 prompt 的"你自己的笔记"一段。

现在（D12 之前）RandomPolicy 不会推理，所以这个容器是空的——
但接口先建好，D12 接线时不用改 view.py。

────────────────────────────────────────────────────────────────
为什么不进事件流
────────────────────────────────────────────────────────────────
事件流记录的是"比赛里发生过的事实"，公开、可复现、可回放。
笔记是私人心理活动，而且和具体 policy 实例绑定
（换一个 policy 就是另一个"人"，笔记不该跟着比赛走）。

如果将来 D23 的复盘页要展示 AI 的内心，那时再给事件流加
EventKind.NOTE（visible_to=pid:N）也不冲突，两条路可以并存。

────────────────────────────────────────────────────────────────
容量控制
────────────────────────────────────────────────────────────────
只在内存里保留最近 limit 条（默认 8）。9 人局大约 6 天，每人最多说 6 次，
8 条足够覆盖全局。无限累积会吃掉 prompt 预算，而更早的判断
基本已经被后来的结论覆盖了——留着只是给模型制造前后矛盾的素材。
"""

from collections.abc import Mapping, Sequence

DEFAULT_LIMIT = 8


class Notebook:
    """按座位号存私人笔记，每个座位一个队列，超出上限丢最早的。"""

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        if limit < 1:
            raise ValueError("limit 必须 ≥ 1")
        self._limit = limit
        self._notes: dict[int, list[str]] = {}

    def add(self, pid: int, note: str, *, label: str = "") -> None:
        """追加一条笔记。

        label 用来标注"第几天"，例如 add(3, "5号像狼", label="第2天")。
        没有 label 时原样存，便于测试和临时调试。
        空白笔记直接丢弃——模型经常返回 ""，没必要占一格容量。
        """
        text = note.strip()
        if not text:
            return
        line = f"{label}：{text}" if label else text
        bucket = self._notes.setdefault(pid, [])
        bucket.append(line)
        overflow = len(bucket) - self._limit
        if overflow > 0:
            del bucket[:overflow]

    def get(self, pid: int) -> list[str]:
        """取某个玩家的笔记。返回副本，调用方改它不会污染容器。"""
        return list(self._notes.get(pid, []))

    def clear(self) -> None:
        self._notes.clear()

    def total(self) -> int:
        """全部玩家的笔记总条数。日志里用它看"AI 到底有没有在思考"。"""
        return sum(len(bucket) for bucket in self._notes.values())

    def to_dict(self) -> dict[str, list[str]]:
        """导出。键转成 str，方便直接写进 JSON（D20 存档会用）。"""
        return {str(pid): list(bucket) for pid, bucket in sorted(self._notes.items())}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Sequence[str]],
        limit: int = DEFAULT_LIMIT,
    ) -> "Notebook":
        """还原。只保留最近 limit 条，和 add 的裁剪规则一致。"""
        notebook = cls(limit=limit)
        for key, bucket in raw.items():
            pid = int(key)
            notebook._notes[pid] = [str(item) for item in bucket][-limit:]
        return notebook
