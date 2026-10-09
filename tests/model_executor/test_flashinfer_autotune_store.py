# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the per-entry FlashInfer autotune winner store."""

import importlib.util
import json
import sys

from vllm.model_executor.warmup.flashinfer_autotune_store import (
    WinnerStore,
    runner_digest,
)

ENV = {"flashinfer_version": "x", "gpu": "y", "schema": "rsii-v1"}
POLICY = ("cuda_graph_profile_replays", 1, "l2_cache_policy", "cold")
KEY = "('op', 'Runner', ((8, 16),), ())"


def _load_class(tmp_path, module, source):
    path = tmp_path / f"{module}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module] = mod
    spec.loader.exec_module(mod)
    return mod.Runner


def test_digest_changes_only_for_edited_runner(tmp_path):
    a = _load_class(tmp_path, "runner_a", "class Runner:\n    x = 1\n")
    b = _load_class(tmp_path, "runner_b", "class Runner:\n    x = 1\n")
    digest_a, digest_b = runner_digest(a), runner_digest(b)
    edited_a = _load_class(tmp_path, "runner_a", "class Runner:\n    x = 2\n")
    reloaded_b = _load_class(tmp_path, "runner_b", "class Runner:\n    x = 1\n")
    assert runner_digest(edited_a) != digest_a
    assert runner_digest(reloaded_b) == digest_b


def test_round_trip_and_mismatches(tmp_path):
    store = WinnerStore([tmp_path], ENV)
    store.publish(KEY, POLICY, "d1", "Runner", [8, 5])
    entry = store.lookup(KEY, POLICY, "d1")
    assert entry["tactic"] == [8, 5] and entry["runner"] == "Runner"
    hot = POLICY[:3] + ("hot",)
    assert store.lookup(KEY, hot, "d1") is None
    assert store.lookup(KEY, POLICY, "d2") is None
    assert store.lookup(KEY.replace("8", "9"), POLICY, "d1") is None
    other_env = WinnerStore([tmp_path], dict(ENV, gpu="z"))
    assert other_env.lookup(KEY, POLICY, "d1") is None


def test_entries_hold_no_paths_and_relocate(tmp_path):
    first = WinnerStore([tmp_path / "a"], ENV)
    first.publish(KEY, POLICY, "d1", "Runner", [8, 5])
    (tmp_path / "a").rename(tmp_path / "b")
    moved = WinnerStore([tmp_path / "b"], ENV)
    assert moved.lookup(KEY, POLICY, "d1")["tactic"] == [8, 5]
    for path in (tmp_path / "b").rglob("*.json"):
        assert str(tmp_path) not in path.read_text()


def test_read_only_layer_is_consulted_and_never_written(tmp_path):
    shared = WinnerStore([tmp_path / "shared"], ENV)
    shared.publish(KEY, POLICY, "d1", "Runner", [8, 5])
    layered = WinnerStore([tmp_path / "local", tmp_path / "shared"], ENV)
    assert layered.lookup(KEY, POLICY, "d1")["tactic"] == [8, 5]
    layered.publish(KEY, POLICY, "d2", "Runner", [16, 5])
    names = {p.name for p in (tmp_path / "shared").rglob("*.json")}
    assert len(names) == 2  # environment.json + the original entry
    local = list((tmp_path / "local").rglob("*.json"))
    assert any(json.loads(p.read_text()).get("tactic") == [16, 5] for p in local)


def test_malformed_entries_are_misses(tmp_path):
    store = WinnerStore([tmp_path], ENV)
    store.publish(KEY, POLICY, "d1", "Runner", [8, 5])
    entry = store.dirs[0] / (store.entry_key(KEY, POLICY, "d1") + ".json")
    no_tactic = {"file_key": KEY, "policy": list(POLICY), "runner_digest": "d1"}
    for text in ("{not json", "[1, 2]", '"x"', json.dumps(no_tactic)):
        entry.write_text(text)
        assert store.lookup(KEY, POLICY, "d1") is None
