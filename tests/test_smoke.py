"""D1 占位测试。

存在的唯一目的：让 CI 从第一天起就有测试可跑。

pytest 在"没有收集到任何测试"时退出码是 5，GitHub Actions 会判定为失败，
所以空仓库直接上 CI 会红。这层占位避免那个问题。

D2 开始写真实单测（板子配置加载、夜晚结算边界）后，可以删掉本文件。
"""


def test_engine_importable() -> None:
    import engine

    assert engine is not None


def test_api_importable() -> None:
    import api

    assert api is not None
