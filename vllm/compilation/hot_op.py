# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot functions: model functions kept out of compiled code and CUDA graphs.

A developer iterating on a model usually edits one or two functions of the
forward pass. With torch.compile, those functions are inlined into the
Dynamo graph and fused into Inductor kernels, and the kernels are captured
into CUDA graphs. Rebinding the Python function afterwards has no effect: the
compiled code and the graphs still run the old version.

This module makes selected methods "hot". The method on the class is replaced
by a small stub that Dynamo traces as one opaque custom op,
``vllm::rsii_hot_call``. The op is a splitting op, so piecewise compilation
cuts the graph around it and runs it eagerly between piecewise CUDA graphs.
FULL CUDA graphs are captured with :class:`BreakableCUDAGraphCapture`, which
ends the graph segment at the op, records it as an eager step and resumes
capture after it. The op looks up the current implementation in a registry
on every call, so swapping the implementation needs no Dynamo trace, no
Inductor compile and no graph capture.

Hot functions are selected with ``VLLM_HOT_FUNCTIONS`` (comma-separated
``module:Class.method``) before the model is compiled, or later in a live
engine followed by a recompile. The set is part of the compile cache key
because the env var is a compile factor.

Constraints of a hot function (checked at call time):
    * Positional tensor arguments only (besides ``self``).
    * It returns one tensor with the shape, dtype and device of its first
      argument; the stub preallocates that output so its address is stable
      across graph replays.
"""

import dataclasses
import hashlib
import importlib
import inspect
import os
import textwrap
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op, weak_ref_tensor

logger = init_logger(__name__)

HOT_OP_NAME = "vllm::rsii_hot_call"
ENV_NAME = "VLLM_HOT_FUNCTIONS"


@dataclasses.dataclass
class HotFunction:
    """One hot method and its current implementation."""

    key: str
    cls: type
    name: str
    original: Callable[..., Any]
    impl: Callable[..., Any]
    stub: Callable[..., Any]
    num_modules: int = 0
    calls: int = 0
    swaps: int = 0


_FUNCTIONS: dict[str, HotFunction] = {}
# hot_id -> (function, module instance)
_SLOTS: list[tuple[HotFunction, nn.Module]] = []
_ID_ATTR = "_rsii_hot_id"


def source_revision(fn: Callable[..., Any]) -> str:
    """SHA-256 of a function's source, or of its code object if unavailable."""
    try:
        text = textwrap.dedent(inspect.getsource(fn))
        return hashlib.sha256(text.encode()).hexdigest()
    except (OSError, TypeError):
        code = fn.__code__
        return hashlib.sha256(code.co_code + repr(code.co_consts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# The custom op
# ---------------------------------------------------------------------------


def _run_slot(out: torch.Tensor, hot_id: int, args: list[torch.Tensor]) -> None:
    function, module = _SLOTS[hot_id]
    function.calls += 1
    result = function.impl(module, *args)
    if not isinstance(result, torch.Tensor) or result.shape != out.shape:
        raise RuntimeError(
            f"Hot function {function.key} must return a tensor shaped like its "
            f"first argument {tuple(out.shape)}, got "
            f"{getattr(result, 'shape', type(result))}"
        )
    out.copy_(result)


def rsii_hot_call(out: torch.Tensor, hot_id: int, args: list[torch.Tensor]) -> None:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    capture = BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        _run_slot(out, hot_id, args)
        return
    # Inside a breakable capture: end the graph segment, run eagerly, record
    # the eager step for replay and resume capture. Weak references keep the
    # replay closure from pinning graph-pool memory; the graph pool owns it.
    weak_out = weak_ref_tensor(out)
    weak_args = [weak_ref_tensor(a) for a in args]
    capture.add_eager(lambda: _run_slot(weak_out, hot_id, weak_args))


def rsii_hot_call_fake(
    out: torch.Tensor, hot_id: int, args: list[torch.Tensor]
) -> None:
    return None


direct_register_custom_op(
    op_name="rsii_hot_call",
    op_func=rsii_hot_call,
    mutates_args=["out"],
    fake_impl=rsii_hot_call_fake,
)


def _make_stub(original: Callable[..., Any]) -> Callable[..., Any]:
    """The method Dynamo traces in place of a hot function: one opaque op.

    ``__wrapped__`` points at the live implementation so source reloaders
    that unwrap decorators patch the implementation, not the stub.
    """

    def hot_stub(self: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(args[0])
        torch.ops.vllm.rsii_hot_call(out, getattr(self, _ID_ATTR), list(args))
        return out

    hot_stub.__wrapped__ = original  # type: ignore[attr-defined]
    hot_stub._rsii_hot_stub = True  # type: ignore[attr-defined]
    return hot_stub


def _is_stub(fn: Any) -> bool:
    return getattr(fn, "_rsii_hot_stub", False)


# ---------------------------------------------------------------------------
# Installing, swapping, inspecting
# ---------------------------------------------------------------------------


def parse_keys(value: str | None) -> list[str]:
    return [k.strip() for k in (value or "").split(",") if k.strip()]


def resolve(key: str) -> tuple[type, str]:
    """Resolve ``module:Class.method`` to the class and method name."""
    module_name, _, qualname = key.partition(":")
    cls_name, _, method = qualname.rpartition(".")
    if not module_name or not cls_name or not method:
        raise ValueError(f"Hot function key must be module:Class.method: {key}")
    obj: Any = importlib.import_module(module_name)
    for part in cls_name.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type) or not callable(getattr(obj, method, None)):
        raise ValueError(f"{key} does not name a class method")
    return obj, method


def is_hot(key: str) -> bool:
    return key in _FUNCTIONS


def hot_functions() -> dict[str, HotFunction]:
    return dict(_FUNCTIONS)


def install(model: nn.Module, keys: list[str]) -> list[str]:
    """Make ``keys`` hot on every module of ``model`` that uses them.

    Must run before the model is (re)traced by Dynamo; already compiled code
    keeps the inlined original until it is recompiled. Returns the new keys.
    """
    added = []
    for key in keys:
        if key in _FUNCTIONS:
            continue
        cls, name = resolve(key)
        original = getattr(cls, name)
        if _is_stub(original):
            raise RuntimeError(f"{key} already carries a hot stub")
        function = HotFunction(
            key=key,
            cls=cls,
            name=name,
            original=original,
            impl=original,
            stub=_make_stub(original),
        )
        for module in model.modules():
            if isinstance(module, cls) and getattr(type(module), name) is original:
                setattr(module, _ID_ATTR, len(_SLOTS))
                _SLOTS.append((function, module))
                function.num_modules += 1
        if function.num_modules == 0:
            raise RuntimeError(f"No module of {model.__class__.__name__} uses {key}")
        setattr(cls, name, function.stub)
        _FUNCTIONS[key] = function
        added.append(key)
        logger.info(
            "Hot function %s on %d modules (revision %s)",
            key,
            function.num_modules,
            source_revision(original)[:12],
        )
    return added


def ensure_splitting_op(compilation_config: Any) -> None:
    """Add the hot op to the piecewise splitting ops (idempotent)."""
    ops = compilation_config.splitting_ops
    if ops is not None and HOT_OP_NAME not in ops:
        ops.append(HOT_OP_NAME)


def install_from_env(model: nn.Module, vllm_config: Any) -> list[str]:
    keys = parse_keys(os.environ.get(ENV_NAME))
    if not keys:
        return []
    ensure_splitting_op(vllm_config.compilation_config)
    return install(model, keys)


def swap(key: str, impl: Callable[..., Any]) -> str:
    """Point a hot function at a new implementation; returns its revision.

    No compiled code or CUDA graph needs rebuilding: the op reads the
    registry on every call, including during graph replay. Reloaders that
    patch the implementation's ``__code__`` in place need no swap at all.
    """
    function = _FUNCTIONS[key]
    if _is_stub(impl):
        raise ValueError("Cannot swap in a hot stub")
    function.impl = impl
    function.stub.__wrapped__ = impl  # type: ignore[attr-defined]
    function.swaps += 1
    # A reload may have rebound the class attribute; the class must keep the
    # stub so a later re-trace still sees the opaque op.
    setattr(function.cls, function.name, function.stub)
    return source_revision(impl)


def find_hot(fn: Callable[..., Any]) -> str | None:
    """The key of the hot function whose implementation is ``fn``, if any."""
    for key, function in _FUNCTIONS.items():
        if function.impl is fn:
            return key
    return None


def witness() -> dict[str, dict[str, Any]]:
    """Per hot function: revision, eager call count, swaps, module count."""
    return {
        key: {
            "revision": source_revision(f.impl),
            "calls": f.calls,
            "swaps": f.swaps,
            "modules": f.num_modules,
        }
        for key, f in _FUNCTIONS.items()
    }


def reset_calls() -> None:
    for function in _FUNCTIONS.values():
        function.calls = 0


def breaks_full_graphs() -> bool:
    """Whether FULL graphs must be captured with eager breaks."""
    return bool(_FUNCTIONS)


# ---------------------------------------------------------------------------
# Graph policy for vllm.v1.worker.hot_patch
# ---------------------------------------------------------------------------


def break_policy(worker: Any, patch: Any) -> dict[str, Any]:
    """``graph_policy="break"``: rebuild nothing for hot functions.

    A changed function that is already the implementation of a hot function
    runs eagerly from the registry, so nothing compiled or captured is stale.
    A changed method that is not hot yet was inlined into compiled code: it
    is made hot, and the model is recompiled and recaptured once (the
    ``recapture`` policy) with the opaque op in its place. Later edits of it
    then rebuild nothing. Any other changed function falls back to a plain
    recompile and recapture.
    """
    import time

    from vllm.v1.worker.hot_patch import recapture

    started = time.perf_counter()
    hot, cold = [], []
    for change in patch.changed:
        key = find_hot(change.function)
        if key is not None:
            hot.append(key)
        else:
            cold.append(f"{change.module}:{change.qualname}")
    report: dict[str, Any] = {"hot_unchanged_graphs": hot, "promoted": []}
    if not cold:
        report["rebuilt"] = "nothing"
        report["break_seconds"] = time.perf_counter() - started
        return report
    model = worker.model_runner.get_model()
    promotable = []
    for key in cold:
        try:
            cls, _ = resolve(key)
        except (ValueError, AttributeError, ImportError):
            continue
        if any(isinstance(m, cls) for m in model.modules()):
            promotable.append(key)
    if promotable:
        ensure_splitting_op(worker.vllm_config.compilation_config)
        install(model, promotable)
        keys = parse_keys(os.environ.get(ENV_NAME)) + promotable
        # The hot set is a compile factor; keep the env in sync.
        os.environ[ENV_NAME] = ",".join(dict.fromkeys(keys))
    report["promoted"] = promotable
    if SPLICE and promotable and len(promotable) == len(cold):
        old_codes = {
            f"{c.module}:{c.qualname}": c.old_code for c in patch.changed
        }
        try:
            report.update(
                splice_promote(worker, {k: old_codes[k] for k in promotable})
            )
            report["rebuilt"] = "splice+recapture"
            report["break_seconds"] = time.perf_counter() - started
            return report
        except Exception as e:  # fall back to a full re-trace
            logger.warning("Splice failed, re-tracing the model: %s", e)
            report["splice_error"] = repr(e)
    report["rebuilt"] = "recompile+recapture"
    report.update(recapture(worker, patch))
    report["break_seconds"] = time.perf_counter() - started
    return report


# Promote by editing the saved Dynamo graph instead of re-tracing the model.
SPLICE = os.environ.get("VLLM_HOT_SPLICE", "1") == "1"


def _is_state(node: Any) -> bool:
    return node.op == "placeholder" and (
        "_parameters_" in node.name or "_buffers_" in node.name
    )


def splice_promote(worker: Any, traced_codes: dict[str, Any]) -> dict[str, Any]:
    """Make installed hot functions take effect without a Dynamo trace.

    ``traced_codes`` maps each newly hot key to the code object the model was
    traced with. Each traced call of it in the saved Dynamo graph is replaced
    by the hot op, vLLM's backend recompiles the edited graph (unchanged
    pieces hit the Inductor cache) and CUDA graphs are recaptured.
    """
    import copy
    import time

    from vllm.compilation import splice
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.eplb_utils import preserve_serving_state
    from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

    started = time.perf_counter()
    runner = worker.model_runner
    edited = []
    for fn in splice.serializable_functions():
        gm = fn.graph_module
        if any(n.op == "call_module" for n in gm.graph.nodes):
            gm = splice.flatten(gm)
        else:
            gm = copy.deepcopy(gm)
        found = 0
        for key, code in traced_codes.items():
            regions = splice.find_regions(gm, code, _is_state)
            if not regions:
                continue
            ids = [i for i, (f, _) in enumerate(_SLOTS) if f.key == key]
            if len(regions) != len(ids):
                raise ValueError(
                    f"{key}: {len(regions)} traced calls, {len(ids)} modules"
                )
            for hot_id, region in zip(ids, regions):
                splice.replace_with_hot_call(gm, region, hot_id)
            found += 1
        if found:
            edited.append((fn, gm))
    if not edited:
        raise ValueError("No traced call of the new hot functions found")
    spliced = time.perf_counter()
    tag = ",".join(sorted(_FUNCTIONS))
    manager = getattr(runner, "cudagraph_manager", None)
    if manager is not None:
        manager.release_graphs()
    CUDAGraphWrapper.clear_all_graphs()
    for fn, gm in edited:
        splice.recompile(fn, gm, worker.vllm_config, tag)
    compiled = time.perf_counter()
    torch.accelerator.synchronize()
    unlock_workspace()
    try:
        with preserve_serving_state(runner):
            worker.compile_or_warm_up_model()
    finally:
        lock_workspace()
    done = time.perf_counter()
    return {
        "splice_seconds": spliced - started,
        "inductor_seconds": compiled - spliced,
        "warmup_capture_seconds": done - compiled,
        "spliced_graphs": len(edited),
    }
