"""pytest 的 conftest（本目录为空插件）。

测试共享工具不放在这里，因为 tests/ 有 __init__.py（是一个包），
pytest 会把 conftest 当 tests.conftest 加载，测试文件里写
`from conftest import ...` 会 ImportError。

所以工具放在 tests/support.py，测试文件统一用：
    from tests.support import make_state, ScriptNight

（如果哪天把 tests/__init__.py 删掉，两个写法都能用。）
"""
