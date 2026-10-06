# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for swapping edited function bodies into live modules."""

import importlib
import sys

import pytest

from vllm.v1.worker import hot_patch

STOCK = """
import functools


def helper(x):
    return x + 1


def wrap(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner


class Base:
    def forward(self, x):
        return x


class Layer(Base):
    def __init__(self):
        self.scale = 2

    def forward(self, x):
        return super().forward(x) * self.scale

    @wrap
    def decorated(self, x):
        return helper(x)
"""


@pytest.fixture
def module(tmp_path, monkeypatch):
    name = "rsii_hot_patch_case"
    path = tmp_path / f"{name}.py"
    path.write_text(STOCK)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    hot_patch._INSTALLED.pop(name, None)
    hot_patch._STOCK.pop(name, None)
    mod = importlib.import_module(name)
    yield mod
    sys.modules.pop(name, None)
    hot_patch._INSTALLED.pop(name, None)
    hot_patch._STOCK.pop(name, None)


def install(tmp_path, module, source, tag):
    path = tmp_path / f"revision-{tag}.py"
    path.write_text(source)
    return hot_patch.reload_sources({module.__name__: (source, str(path))})


def test_body_edit_keeps_identity(tmp_path, module):
    layer = module.Layer()
    cls, forward, bound = module.Layer, module.Layer.forward, layer.forward
    helper = module.helper
    assert layer.forward(3) == 6 and layer.decorated(3) == 4
    assert hot_patch.revision_factor() == ""
    source = STOCK.replace("* self.scale", "* self.scale + 100").replace(
        "return x + 1", "return x + 10"
    )
    patch = install(tmp_path, module, source, "a")
    assert sorted(c.qualname for c in patch.changed) == [
        "Layer.forward",
        "helper",
    ]
    assert patch.owners == {module.Layer}
    # Same objects, new bodies: aliases and bound methods see the edit.
    assert module.Layer is cls and module.Layer.forward is forward
    assert bound(3) == 106 and helper(3) == 13 and layer.decorated(3) == 13
    assert forward.__code__.co_filename.endswith("revision-a.py")
    assert hot_patch.revision_factor() != ""
    # Back to stock: diffed against the installed revision.
    patch = install(tmp_path, module, STOCK, "b")
    assert len(patch.changed) == 2 and layer.forward(3) == 6
    assert hot_patch.revision_factor() == ""


def test_decorated_method_body(tmp_path, module):
    layer = module.Layer()
    source = STOCK.replace("return helper(x)", "return helper(x) * 7")
    patch = install(tmp_path, module, source, "c")
    assert [c.qualname for c in patch.changed] == ["Layer.decorated"]
    assert layer.decorated(3) == 28


@pytest.mark.parametrize(
    "old,new",
    [
        ("self.scale = 2", "self.scale = 3"),
        (
            "def forward(self, x):\n        return super()",
            "def forward(self, y):\n        return super()",
        ),
        ("class Base:", "SCALE = 1\n\n\nclass Base:"),
        ("    @wrap\n", ""),
    ],
)
def test_refuses_restart_class_edits(tmp_path, module, old, new):
    assert STOCK.count(old) == 1
    layer = module.Layer()
    code = module.Layer.forward.__code__
    with pytest.raises(hot_patch.UnsupportedEdit):
        install(tmp_path, module, STOCK.replace(old, new), "d")
    assert module.Layer.forward.__code__ is code and layer.forward(3) == 6


def test_closure_change_refused(tmp_path, module):
    # Base.forward gains a super() call, so a __class__ cell it lacks.
    old = "class Base:\n    def forward(self, x):\n        return x"
    source = STOCK.replace(old, old.replace("return x", "return super() and x"))
    with pytest.raises(hot_patch.UnsupportedEdit, match="closure"):
        install(tmp_path, module, source, "e")
