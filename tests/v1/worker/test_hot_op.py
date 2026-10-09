# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for hot functions: install, swap, reload interop, routing."""

import sys
import types

import pytest
import torch
from torch import nn

from vllm.compilation import hot_op
from vllm.v1.worker import hot_patch

SOURCE = """
from torch import nn


class Toy(nn.Module):
    def forward(self, x):
        return x + 1


class Other(nn.Module):
    def forward(self, x):
        return x


class Attn(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states * 2
"""


@pytest.fixture
def toy(tmp_path, monkeypatch):
    path = tmp_path / "rsii_toy_model.py"
    path.write_text(SOURCE)
    module = types.ModuleType("rsii_toy_model")
    module.__file__ = str(path)
    exec(compile(SOURCE, str(path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, "rsii_toy_model", module)
    monkeypatch.setattr(hot_op, "_FUNCTIONS", {})
    monkeypatch.setattr(hot_op, "_SLOTS", [])
    monkeypatch.setattr(hot_op, "_GRADUATED", {})
    monkeypatch.setattr(hot_patch, "_INSTALLED", {})
    monkeypatch.setattr(hot_patch, "_STOCK", {})
    model = nn.Sequential(module.Toy(), module.Other(), module.Toy())
    return module, model, path


KEY = "rsii_toy_model:Toy.forward"


def test_install_replaces_method_with_stub(toy):
    module, model, _ = toy
    original = module.Toy.forward
    assert hot_op.install(model, [KEY]) == [KEY]
    assert hot_op.install(model, [KEY]) == []
    stub = module.Toy.forward
    assert stub is not original and stub.__wrapped__ is original
    assert [getattr(m, "_rsii_hot_id", None) for m in model] == [0, None, 1]
    seen = hot_op.witness()[KEY]
    assert seen["modules"] == 2 and seen["calls"] == 0
    assert seen["revision"] == hot_op.source_revision(original)
    assert hot_op.find_hot(original) == KEY
    assert hot_op.breaks_full_graphs()


def test_run_slot_calls_current_impl_and_counts(toy):
    module, model, _ = toy
    hot_op.install(model, [KEY])
    out = torch.empty(3)
    hot_op._run_slot(out, 0, [torch.zeros(3)])
    assert torch.equal(out, torch.ones(3))

    def edited(self, x):
        return x + 5

    hot_op.swap(KEY, edited)
    assert module.Toy.forward.__wrapped__ is edited
    hot_op._run_slot(out, 1, [torch.zeros(3)])
    assert torch.equal(out, torch.full((3,), 5.0))
    assert hot_op.witness()[KEY]["calls"] == 2
    assert hot_op.witness()[KEY]["swaps"] == 1


def test_wrong_output_shape_is_rejected(toy):
    _, model, _ = toy
    hot_op.install(model, [KEY])
    hot_op.swap(KEY, lambda self, x: x.sum())
    with pytest.raises(RuntimeError, match="shaped like its argument 0"):
        hot_op._run_slot(torch.empty(3), 0, [torch.zeros(3)])


def test_reload_patches_the_impl_not_the_stub(toy):
    module, model, path = toy
    hot_op.install(model, [KEY])
    stub = module.Toy.forward
    impl = stub.__wrapped__
    new = SOURCE.replace("return x + 1", "return x + 7")
    revision = path.with_name("rsii_toy_model_r1.py")
    revision.write_text(new)
    patch = hot_patch.reload_sources({"rsii_toy_model": (new, str(revision))})
    assert [c.function for c in patch.changed] == [impl]
    assert module.Toy.forward is stub
    out = torch.empty(2)
    hot_op._run_slot(out, 0, [torch.zeros(2)])
    assert torch.equal(out, torch.full((2,), 7.0))
    assert hot_op.witness()[KEY]["revision"] == hot_op.source_revision(impl)


def test_break_policy_rebuilds_nothing_for_hot_edits(toy, monkeypatch):
    module, model, path = toy
    hot_op.install(model, [KEY])
    new = SOURCE.replace("return x + 1", "return x + 2")
    patch = hot_patch.reload_sources({"rsii_toy_model": (new, str(path))})

    def fail(*args):
        raise AssertionError("recapture must not run")

    monkeypatch.setattr(hot_patch, "recapture", fail)
    report = hot_op.break_policy(object(), patch)
    assert report["rebuilt"] == "nothing"
    assert report["hot_unchanged_graphs"] == [KEY]


def test_break_policy_promotes_cold_methods_once(toy, monkeypatch):
    module, model, path = toy
    monkeypatch.setenv(hot_op.ENV_NAME, "")
    new = SOURCE.replace("return x\n", "return x * 3\n")
    patch = hot_patch.reload_sources({"rsii_toy_model": (new, str(path))})
    calls = []
    monkeypatch.setattr(
        hot_patch, "recapture", lambda w, p: calls.append(p) or {"x": 1}
    )
    config = types.SimpleNamespace(splitting_ops=["vllm::attn"])
    worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(get_model=lambda: model),
        vllm_config=types.SimpleNamespace(compilation_config=config),
    )
    report = hot_op.break_policy(worker, patch)
    other = "rsii_toy_model:Other.forward"
    assert report["promoted"] == [other] and len(calls) == 1
    assert hot_op.is_hot(other)
    assert config.splitting_ops == ["vllm::attn", hot_op.HOT_OP_NAME]
    assert other in hot_op.parse_keys(__import__("os").environ[hot_op.ENV_NAME])


def test_refused_install_changes_nothing(toy):
    module, model, _ = toy
    other = module.Other.forward
    keys = ["rsii_toy_model:Other.forward", "rsii_toy_model:Attn.forward"]
    with pytest.raises(RuntimeError, match="No module"):
        hot_op.install(model, keys)
    assert module.Other.forward is other
    assert not hot_op.hot_functions() and not hot_op._SLOTS
    assert all(getattr(m, hot_op._ID_ATTR, None) is None for m in model)


def test_break_policy_recaptures_when_promotion_is_refused(toy, monkeypatch):
    module, model, path = toy
    monkeypatch.setenv(hot_op.ENV_NAME, "")
    # The Other instance already carries a hot method: making Other.forward
    # hot is refused, and the edit must still be compiled in.
    setattr(model[1], hot_op._ID_ATTR, 99)
    new = SOURCE.replace("return x\n", "return x * 3\n")
    patch = hot_patch.reload_sources({"rsii_toy_model": (new, str(path))})
    calls = []
    monkeypatch.setattr(
        hot_patch, "recapture", lambda w, p: calls.append(p) or {"x": 1}
    )
    config = types.SimpleNamespace(splitting_ops=["vllm::attn"])
    worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(get_model=lambda: model),
        vllm_config=types.SimpleNamespace(compilation_config=config),
    )
    report = hot_op.break_policy(worker, patch)
    assert report["promoted"] == [] and "promotion_error" in report
    assert report["rebuilt"] == "recompile+recapture" and len(calls) == 1
    assert not hot_op._is_stub(module.Other.forward)
    assert module.Other.forward(model[1], torch.ones(2)).tolist() == [3.0, 3.0]
    assert config.splitting_ops == ["vllm::attn"]
    assert __import__("os").environ[hot_op.ENV_NAME] == ""


def test_resolve_rejects_bad_keys():
    with pytest.raises(ValueError):
        hot_op.resolve("no_colon_here")


def test_shorthand_key_and_output_argument(toy):
    module, model, _ = toy
    model.append(module.Attn())
    assert hot_op.canonical("Toy.forward", model) == KEY
    assert hot_op.canonical("Attn.forward@1", model) == (
        "rsii_toy_model:Attn.forward@1"
    )
    with pytest.raises(ValueError):
        hot_op.canonical("Missing.forward", model)
    hot_op.install(model, ["Toy.forward", "Attn.forward"])
    functions = hot_op.hot_functions()
    assert functions[KEY].out_index == 0
    # The argument named hidden_states is the output shape by default.
    assert functions["rsii_toy_model:Attn.forward"].out_index == 1


def test_graduated_revisions_change_the_cache_key(toy):
    module, model, _ = toy
    assert hot_patch.revision_factor() == ""
    hot_op._GRADUATED[KEY] = hot_op.source_revision(module.Toy.forward)
    first = hot_patch.revision_factor()
    hot_op._GRADUATED[KEY] = "other"
    assert first and hot_patch.revision_factor() not in ("", first)
