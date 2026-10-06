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
``module:Class.method`` or ``Class.method``, resolved against the model's
modules) before the model is compiled, or later in a live engine followed by
a recompile. The set is part of the compile cache key because the env var is
a compile factor. :func:`graduate` turns hot functions back into compiled
code once iteration is done.

Constraints of a hot function (checked at call time):
    * Positional tensor arguments only (besides ``self``).
    * It returns one tensor with the shape, dtype and device of one of its
      arguments: the one named ``hidden_states`` if there is one, else the
      first; a ``@N`` suffix on the key picks argument ``N`` (0-based). The
      stub preallocates that output so its address is stable across graph
      replays.
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
    out_index: int = 0
    num_modules: int = 0
    calls: int = 0
    swaps: int = 0


_FUNCTIONS: dict[str, HotFunction] = {}
# hot_id -> (function, module instance)
_SLOTS: list[tuple[HotFunction, nn.Module]] = []
_ID_ATTR = "_rsii_hot_id"
# key -> source revision of hot functions fused back by graduate(); a compile
# factor (hot_patch.revision_factor) so no stale artifact is loaded for them.
_GRADUATED: dict[str, str] = {}


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
            f"argument {function.out_index} {tuple(out.shape)}, got "
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


def _make_stub(original: Callable[..., Any], out_index: int) -> Callable[..., Any]:
    """The method Dynamo traces in place of a hot function: one opaque op.

    ``__wrapped__`` points at the live implementation so source reloaders
    that unwrap decorators patch the implementation, not the stub.
    """

    def hot_stub(self: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(args[out_index])
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


def split_out_index(key: str) -> tuple[str, int | None]:
    """``key@N`` -> (key, N); a key without suffix -> (key, None)."""
    base, sep, index = key.partition("@")
    return base, (int(index) if sep else None)


def canonical(key: str, model: nn.Module | None = None) -> str:
    """``Class.method`` -> ``module:Class.method`` via the model's modules.

    Keys that already name a module are returned unchanged. The class is
    looked up in the MRO of every module of ``model``, so a model that
    reuses another model's class (Qwen3's ``Qwen2MLP``) resolves to it.
    """
    base, index = split_out_index(key)
    if ":" in base or model is None:
        return key
    cls_name, _, method = base.rpartition(".")
    found = {
        klass
        for module in model.modules()
        for klass in type(module).__mro__
        if klass.__qualname__ == cls_name and method in vars(klass)
    }
    if len(found) != 1:
        raise ValueError(
            f"Hot function {key}: {len(found)} classes named {cls_name} with "
            f"{method} in {model.__class__.__name__}"
        )
    (klass,) = found
    suffix = "" if index is None else f"@{index}"
    return f"{klass.__module__}:{klass.__qualname__}.{method}{suffix}"


def _out_index(original: Callable[..., Any], index: int | None) -> int:
    if index is not None:
        return index
    try:
        params = list(inspect.signature(original).parameters)[1:]
    except (TypeError, ValueError):
        return 0
    return params.index("hidden_states") if "hidden_states" in params else 0


def resolve(key: str) -> tuple[type, str]:
    """Resolve ``module:Class.method`` to the class and method name."""
    key, _ = split_out_index(key)
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
    for spec in keys:
        spec = canonical(spec, model)
        key, index = split_out_index(spec)
        if key in _FUNCTIONS:
            continue
        cls, name = resolve(key)
        original = getattr(cls, name)
        if _is_stub(original):
            raise RuntimeError(f"{key} already carries a hot stub")
        out_index = _out_index(original, index)
        function = HotFunction(
            key=key,
            cls=cls,
            name=name,
            original=original,
            impl=original,
            stub=_make_stub(original, out_index),
            out_index=out_index,
        )
        for module in model.modules():
            if isinstance(module, cls) and getattr(type(module), name) is original:
                if getattr(module, _ID_ATTR, None) is not None:
                    raise RuntimeError(
                        f"{key}: {type(module).__name__} already has a hot "
                        "method (one hot method per module)"
                    )
                setattr(module, _ID_ATTR, len(_SLOTS))
                _SLOTS.append((function, module))
                function.num_modules += 1
        if function.num_modules == 0:
            raise RuntimeError(f"No module of {model.__class__.__name__} uses {key}")
        setattr(cls, name, function.stub)
        _FUNCTIONS[key] = function
        _GRADUATED.pop(key, None)
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
    keys = [canonical(k, model) for k in keys]
    # Canonical keys in the compile factor: shorthand and full spellings of
    # the same hot set share compiled artifacts.
    os.environ[ENV_NAME] = ",".join(keys)
    ensure_splitting_op(vllm_config.compilation_config)
    return install(model, keys)


def graduated_factor() -> str:
    """Hash of the graduated implementations (empty if none), for cache keys."""
    if not _GRADUATED:
        return ""
    return hashlib.sha256(repr(sorted(_GRADUATED.items())).encode()).hexdigest()


def graduate(worker: Any, keys: list[str] | None = None) -> dict[str, Any]:
    """Fuse hot functions back into compiled code, with their current edits.

    Each class gets its hot function's current implementation back as a
    plain method, the hot op leaves the splitting ops when no hot function
    is left, and the model is recompiled and recaptured in-process (the
    ``recapture`` policy of hot_patch). The result computes like a fresh,
    normally compiled engine started with the same source: Inductor fuses
    the edited body, and FULL graphs are one piece again.
    """
    import time

    from vllm.v1.worker.hot_patch import recapture

    runner = worker.model_runner
    if runner.req_states.req_id_to_index:
        raise RuntimeError("graduate needs an idle engine; drain requests first")
    started = time.perf_counter()
    keys = list(_FUNCTIONS) if keys is None else keys
    model = runner.get_model()
    revisions = {}
    for spec in keys:
        key, _ = split_out_index(canonical(spec, model))
        function = _FUNCTIONS.pop(key)
        setattr(function.cls, function.name, function.impl)
        revisions[key] = _GRADUATED[key] = source_revision(function.impl)
        for module in model.modules():
            hot_id = getattr(module, _ID_ATTR, None)
            if hot_id is not None and _SLOTS[hot_id][0] is function:
                # The slot stays (ids of other hot functions are positions in
                # _SLOTS); the module no longer refers to it.
                delattr(module, _ID_ATTR)
    os.environ[ENV_NAME] = ",".join(_FUNCTIONS)
    ops = worker.vllm_config.compilation_config.splitting_ops
    if not _FUNCTIONS and ops is not None and HOT_OP_NAME in ops:
        ops.remove(HOT_OP_NAME)
    report: dict[str, Any] = {"graduated": revisions, "still_hot": list(_FUNCTIONS)}
    report.update(recapture(worker, None))
    report["graduate_seconds"] = time.perf_counter() - started
    return report


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
    report["rebuilt"] = "recompile+recapture"
    report.update(recapture(worker, patch))
    report["break_seconds"] = time.perf_counter() - started
    return report
