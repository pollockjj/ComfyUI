from __future__ import annotations

import asyncio
import inspect
import sys
import types


sys.modules.setdefault("comfy_aimdo.vram_buffer", types.ModuleType("vram_buffer"))


def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None
    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(test_func(**kwargs))
    return True
