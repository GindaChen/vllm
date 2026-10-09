# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swap edited model-forward code into a live engine.

A developer edits the body of one or two functions in a model file (for
example ``Qwen3Attention.forward``). ``hot_patch`` installs the new bodies
into the running process without reloading weights or rebuilding the model:

1. ``reload_sources`` compares each new source file with the source that is
   currently installed for that module, statement by statement. Only
   function bodies may differ. Each changed function keeps its identity:
   its ``__code__`` is replaced in place, so classes, registries, instances,
   bound methods and ``from x import f`` aliases all see the new body.
   Anything else (signatures, decorators, constructors, module or class
   statements) raises ``UnsupportedEdit``: such an edit needs a restart.
2. A graph policy then makes the compiled and captured state agree with the
   new code. ``"recapture"`` (the baseline) drops every CUDA graph and the
   compiled model, recompiles in-process and recaptures. Other policies can
   be registered in ``GRAPH_POLICIES``.

The engine must be idle: no running or waiting requests. Dummy runs during
recompilation write KV only to the null block, so KV memory is kept.

Compile caches: the new code objects carry the revision file's name, so
vLLM's traced-file hash changes. The AOT compile key does not hash source
files, so ``revision_factor()`` is added to it; without it a patched model
would silently reload the stock AOT artifact.
"""

import ast
import gc
import hashlib
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class UnsupportedEdit(RuntimeError):
    """The edit changes more than function bodies; restart the engine."""


@dataclass(frozen=True)
class ChangedFunction:
    """One function whose body was replaced in place."""

    module: str
    qualname: str
    function: FunctionType
    old_code: CodeType
    new_code: CodeType


@dataclass(frozen=True)
class KernelEdit:
    """A changed Triton kernel, left for a kernel handler to install."""

    module: str
    qualname: str
    # The outermost live object (JITFunction, Autotuner, Heuristics).
    live: Any
    # The new definition, from its first decorator line to its end.
    new_source: str
    module_source: str
    filename: str


@dataclass
class CodePatch:
    """What ``reload_sources`` changed."""

    # Module name -> SHA-256 of the installed source.
    sources: dict[str, str]
    changed: list[ChangedFunction]
    # Classes that own a changed method.
    owners: set[type] = field(default_factory=set)
    # Live module instances of those classes, by FQN.
    modules: list[tuple[str, nn.Module]] = field(default_factory=list)
    # Triton kernels, installed by KERNEL_HANDLERS, and their reports.
    kernels: list[KernelEdit] = field(default_factory=list)
    kernel_reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def needs_graph_policy(self) -> bool:
        """Python bodies changed, or a kernel handler asked for it."""
        return bool(self.changed) or any(
            report.get("needs_recapture") for report in self.kernel_reports
        )


# Module name -> (source text, filename) currently installed. Initialized
# from the module's file on first patch.
_INSTALLED: dict[str, tuple[str, str]] = {}
# Module name -> the source it was imported from.
_STOCK: dict[str, str] = {}
# Code objects of installed revisions that Dynamo inlined since the last
# patch. Fed by vllm.compilation.decorators while tracing.
_WATCHED: dict[CodeType, str] = {}
_TRACED: set[str] = set()

GRAPH_POLICIES: dict[str, Callable[[Any, CodePatch], dict[str, Any]]] = {}

# Handlers for edited Triton kernels: handler(worker, kernels) -> report.
# `worker` is None before the engine exists (a fresh engine with the edit).
# A handler installs the kernels in place (and may swap them into captured
# graphs) or raises UnsupportedEdit; nothing is installed then. A report with
# "needs_recapture": True makes hot_patch run the graph policy afterwards.
KERNEL_HANDLERS: list[Callable[[Any, list[KernelEdit]], dict[str, Any]]] = []

# Inductor's FX-graph cache and Triton's kernel cache are keyed by graph and
# kernel content, so a revision can safely reuse the stock compile's
# directories: unchanged subgraphs hit, changed ones miss. vLLM's own
# compiled-graph and AOT caches are not content-addressed and stay per
# revision (revision_factor).
SHARE_KERNEL_CACHE = True
# Serializing the AOT artifact after a hot-patch recompile only helps a later
# restart with exactly this revision; skip it by default.
SAVE_AOT_AFTER_PATCH = False
# vLLM dedupes piecewise graphs in memory by their AOTAutograd cache key,
# which covers the graph, its inputs and the compiler config. One pool for
# the process lets a recompile after a patch reuse every unchanged graph.
SHARE_COMPILED_ARTIFACTS = True
_ARTIFACTS: dict[str, Any] = {}
_KERNEL_CACHE_ENV = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
# The stock compile's kernel cache directories, captured at the first patch.
_KERNEL_CACHE: dict[str, str] = {}
_IN_RECAPTURE = False


def revision_factor() -> str:
    """Hash of every installed revision, for compile cache keys.

    Empty when nothing is patched, so stock cache keys are unchanged.
    """
    from vllm.compilation.hot_op import graduated_factor

    revised: list[tuple[str, str]] = [
        (name, hashlib.sha256(source.encode()).hexdigest())
        for name, (source, _) in sorted(_INSTALLED.items())
        if source != _STOCK[name]
    ]
    if graduated := graduated_factor():
        revised.append(("hot_op.graduated", graduated))
    if not revised:
        return ""
    return hashlib.sha256(repr(revised).encode()).hexdigest()


def shared_kernel_cache() -> dict[str, str]:
    """Kernel cache directories a patched model should compile into.

    Empty unless a revision is installed into an engine that had compiled
    the stock model (fresh engines keep vLLM's per-source directories).
    """
    if SHARE_KERNEL_CACHE and revision_factor():
        return dict(_KERNEL_CACHE)
    return {}


def compiled_artifact_pool() -> dict[str, Any]:
    """The compiled-graph dedupe table a new vLLM compiler should use."""
    return _ARTIFACTS if SHARE_COMPILED_ARTIFACTS else {}


def skip_aot_save() -> bool:
    """Whether to skip saving the AOT artifact of a hot-patch recompile."""
    return _IN_RECAPTURE and not SAVE_AOT_AFTER_PATCH


def note_traced(code: CodeType) -> None:
    """Record that Dynamo traced `code` (called while compiling)."""
    name = _WATCHED.get(code)
    if name is not None:
        _TRACED.add(name)


def traced_functions() -> set[str]:
    """Changed functions Dynamo inlined since the last patch."""
    return set(_TRACED)


def _statements(tree: ast.Module) -> Iterator[tuple[str, ast.stmt]]:
    """Yield (qualname, node) for module and class-body statements.

    Class bodies are flattened so a method is keyed "Class.method"; other
    statements are keyed by their position.
    """

    def walk(body: list[ast.stmt], prefix: str) -> Iterator[tuple[str, ast.stmt]]:
        for index, node in enumerate(body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + node.name, node
            elif isinstance(node, ast.ClassDef):
                header = ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=[],
                    decorator_list=node.decorator_list,
                    type_params=getattr(node, "type_params", []),
                )
                yield f"{prefix}{node.name}:<class>", header
                yield from walk(node.body, f"{prefix}{node.name}.")
            else:
                yield f"{prefix}<stmt {index}>", node

    yield from walk(tree.body, "")


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ast.dump(
        ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=[],
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_params=getattr(node, "type_params", []),
        )
    )


def _diff(module: str, old: str, new: str) -> list[str]:
    """Names of functions whose bodies differ; refuse any other change."""
    olds = dict(_statements(ast.parse(old)))
    news = dict(_statements(ast.parse(new)))
    if list(olds) != list(news):
        added = sorted(set(news) - set(olds))
        removed = sorted(set(olds) - set(news))
        raise UnsupportedEdit(
            f"{module}: definitions added {added} or removed {removed}, or reordered"
        )
    changed = []
    for name, node in news.items():
        before = olds[name]
        if ast.dump(before) == ast.dump(node):
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise UnsupportedEdit(f"{module}: {name} is not a function body edit")
        if _signature(before) != _signature(node):
            raise UnsupportedEdit(f"{module}: {name} signature or decorators")
        if node.name == "__init__":
            raise UnsupportedEdit(
                f"{module}: {name} changes construction; rebuild the model"
            )
        changed.append(name)
    return changed


def _codes(code: CodeType) -> Iterator[CodeType]:
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield const
            yield from _codes(const)


def _new_code(compiled: CodeType, qualname: str) -> CodeType:
    found = [c for c in _codes(compiled) if c.co_qualname == qualname]
    if len(found) != 1:
        raise UnsupportedEdit(f"{qualname}: expected one definition")
    return found[0]


def _unwrap(value: Any, qualname: str, filename: str) -> FunctionType:
    """Find the function defined as `qualname` behind decorators."""
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    seen: set[int] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, property):
            stack += [item.fget, item.fset, item.fdel]
            continue
        if isinstance(item, FunctionType):
            code = item.__code__
            if code.co_qualname == qualname and code.co_filename == filename:
                return item
            stack += [c.cell_contents for c in item.__closure__ or ()]
        wrapped = getattr(item, "__wrapped__", None)
        if wrapped is not None:
            stack.append(wrapped)
    raise UnsupportedEdit(f"{qualname}: no live function found")


def _live_value(module: ModuleType, qualname: str) -> tuple[Any, type | None]:
    owner: Any = module
    *path, name = qualname.split(".")
    for part in path:
        owner = owner.__dict__[part]
    return owner.__dict__[name], owner if path else None


def _is_kernel(value: Any) -> bool:
    try:
        from triton.runtime.jit import KernelInterface
    except ImportError:
        return False
    return isinstance(value, KernelInterface)


def _definition_source(source: str, qualname: str) -> str:
    """The text of `qualname`'s definition, decorators included."""
    node = dict(_statements(ast.parse(source)))[qualname]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    first = min([d.lineno for d in node.decorator_list] + [node.lineno])
    lines = source.splitlines(keepends=True)
    return "".join(lines[first - 1 : node.end_lineno])


def reload_sources(
    sources: dict[str, tuple[str, str]], worker: Any = None
) -> CodePatch:
    """Install new function bodies from `sources` into live modules.

    Args:
        sources: Module name -> (new source text, filename). The filename
            names the new code objects (tracebacks, compile cache hashes);
            pass a real file holding exactly that text.

        worker: The live worker, or None before the engine starts. Passed
            to kernel handlers.

    Returns:
        The changed functions. Unchanged functions keep their code. Edited
        Triton kernels are handed to KERNEL_HANDLERS before any function
        body is installed.

    Raises:
        UnsupportedEdit: The edit changes more than function bodies. Nothing
            is installed in that case.

    """
    if not _KERNEL_CACHE and all(k in os.environ for k in _KERNEL_CACHE_ENV):
        _KERNEL_CACHE.update({k: os.environ[k] for k in _KERNEL_CACHE_ENV})
    plans = []
    kernels: list[KernelEdit] = []
    for name, (source, filename) in sources.items():
        module = sys.modules[name]
        if name not in _INSTALLED:
            path = module.__file__
            assert path is not None
            _INSTALLED[name] = (Path(path).read_text(), path)
            _STOCK[name] = _INSTALLED[name][0]
        old_source, old_filename = _INSTALLED[name]
        compiled = compile(source, filename, "exec", dont_inherit=True)
        changes = []
        for qualname in _diff(name, old_source, source):
            value, owner = _live_value(module, qualname)
            if _is_kernel(value):
                kernels.append(
                    KernelEdit(
                        name,
                        qualname,
                        value,
                        _definition_source(source, qualname),
                        source,
                        filename,
                    )
                )
                continue
            function = _unwrap(value, qualname, old_filename)
            new = _new_code(compiled, qualname)
            if new.co_freevars != function.__code__.co_freevars:
                raise UnsupportedEdit(
                    f"{qualname}: closure variables changed "
                    f"{function.__code__.co_freevars} -> {new.co_freevars}"
                )
            changes.append((qualname, function, owner, new))
        plans.append((name, source, filename, changes))

    patch = CodePatch(sources={}, changed=[], kernels=kernels)
    if kernels:
        if not KERNEL_HANDLERS:
            names = [f"{k.module}:{k.qualname}" for k in kernels]
            raise UnsupportedEdit(f"No kernel handler for Triton edits {names}")
        # Kernels first: a refusal leaves everything uninstalled, and a
        # following recompile sees the new kernels.
        for handler in KERNEL_HANDLERS:
            patch.kernel_reports.append(handler(worker, kernels))
    _WATCHED.clear()
    _TRACED.clear()
    for name, source, filename, changes in plans:
        for qualname, function, owner, new in changes:
            patch.changed.append(
                ChangedFunction(name, qualname, function, function.__code__, new)
            )
            function.__code__ = new
            _WATCHED[new] = f"{name}:{qualname}"
            if owner is not None:
                patch.owners.add(owner)
        # Later patches diff against this source and its code objects.
        _INSTALLED[name] = (source, filename)
        patch.sources[name] = hashlib.sha256(source.encode()).hexdigest()
    return patch


def _find_instances(model: nn.Module, patch: CodePatch) -> None:
    owners = tuple(patch.owners)
    patch.modules = [
        (fqn, module)
        for fqn, module in model.named_modules()
        if owners and isinstance(module, owners)
    ]


def _release_compiled_state(worker: Any) -> int:
    """Drop CUDA graphs and every compiled submodule; return how many.

    The compiled module is not always the model or ``model.model``: a
    multimodal wrapper holds it deeper (``language_model.model``), so every
    ``support_torch_compile`` instance is reset.
    """
    from vllm.compilation.wrapper import (
        TorchCompileWithNoGuardsWrapper,
        reset_compile_wrapper,
    )
    from vllm.config import set_current_vllm_config

    runner = worker.model_runner
    manager = getattr(runner, "cudagraph_manager", None)
    if manager is not None:
        manager.release_graphs()
    torch.compiler.reset()
    compiled = [
        module
        for module in runner.get_model().modules()
        if isinstance(module, TorchCompileWithNoGuardsWrapper)
        and not getattr(module, "do_not_compile", True)
    ]
    with set_current_vllm_config(worker.vllm_config):
        for module in compiled:
            reset_compile_wrapper(module)
    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    return len(compiled)


class _CompileTimes(logging.Handler):
    """Collect vLLM's Dynamo and Inductor compile times from its log."""

    PATTERNS = {
        "dynamo_seconds": re.compile(r"Dynamo bytecode transform time: ([\d.]+) s"),
        "inductor_seconds": re.compile(r"Compiling a graph for .* takes ([\d.]+) s"),
    }

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.seconds = dict.fromkeys(self.PATTERNS, 0.0)

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for key, pattern in self.PATTERNS.items():
            match = pattern.search(message)
            if match:
                self.seconds[key] += float(match.group(1))


def recapture(worker: Any, patch: CodePatch | None) -> dict[str, Any]:
    """Baseline policy: recompile and recapture everything in-process."""
    global _IN_RECAPTURE
    from vllm.v1.worker.gpu.eplb_utils import preserve_serving_state
    from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

    runner = worker.model_runner
    started = time.perf_counter()
    num_compiled = _release_compiled_state(worker)
    from vllm.config import CompilationMode

    # Recapturing graphs around a stale compiled model would run old code.
    mode = worker.vllm_config.compilation_config.mode
    if mode == CompilationMode.VLLM_COMPILE and num_compiled == 0:
        raise RuntimeError("hot_patch found no compiled submodule to reset")
    released = time.perf_counter()
    from torch._dynamo.utils import counters

    cache_keys = ("fxgraph_cache_hit", "fxgraph_cache_miss", "fxgraph_cache_bypass")
    before = {k: counters["inductor"][k] for k in cache_keys}
    artifacts_before = len(_ARTIFACTS)
    times = _CompileTimes()
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(times)
    unlock_workspace()
    _IN_RECAPTURE = True
    try:
        with preserve_serving_state(runner):
            # The first call compiles. Not a profile run: KV memory exists
            # now, and hybrid models must address real state slots.
            runner._dummy_run(runner.max_num_tokens, skip_eplb=True)
            compiled = time.perf_counter()
            worker.compile_or_warm_up_model()
    finally:
        _IN_RECAPTURE = False
        vllm_logger.removeHandler(times)
        lock_workspace()
    done = time.perf_counter()
    compile_seconds = compiled - released
    return {
        "compiled_modules_reset": num_compiled,
        "release_seconds": released - started,
        "compile_seconds": compile_seconds,
        **times.seconds,
        "compile_other_seconds": compile_seconds - sum(times.seconds.values()),
        "warmup_capture_seconds": done - compiled,
        "kernel_cache": "shared" if shared_kernel_cache() else "per revision",
        **{k: counters["inductor"][k] - before[k] for k in cache_keys},
        "compiled_artifacts_new": len(_ARTIFACTS) - artifacts_before,
        "compiled_artifacts_shared": SHARE_COMPILED_ARTIFACTS,
        "aot_saved": SAVE_AOT_AFTER_PATCH,
    }


GRAPH_POLICIES["recapture"] = recapture


def _break(worker: Any, patch: CodePatch) -> dict[str, Any]:
    from vllm.compilation.hot_op import break_policy

    return break_policy(worker, patch)


# Hot functions (vllm/compilation/hot_op.py): edits rebuild nothing.
GRAPH_POLICIES["break"] = _break


def hot_patch(
    worker: Any,
    sources: dict[str, tuple[str, str]],
    graph_policy: str = "recapture",
) -> dict[str, Any]:
    """Install edited function bodies into an idle engine's model.

    Weights, KV memory, the CUDA context and the process are kept.

    Returns:
        A report: changed functions, source hashes, live instances whose
        class changed, Dynamo-traced functions and timings.

    """
    if not getattr(worker, "use_v2_model_runner", False):
        raise UnsupportedEdit("hot_patch supports the V2 model runner only")
    runner = worker.model_runner
    if runner.req_states.req_id_to_index:
        raise RuntimeError("hot_patch needs an idle engine; drain requests first")
    if graph_policy not in GRAPH_POLICIES:
        raise ValueError(f"Unknown graph policy {graph_policy!r}")
    started = time.perf_counter()
    patch = reload_sources(sources, worker)
    _find_instances(runner.get_model(), patch)
    patched = time.perf_counter()
    timings: dict[str, Any] = {}
    if patch.needs_graph_policy:
        timings = GRAPH_POLICIES[graph_policy](worker, patch)
    torch.accelerator.synchronize()
    done = time.perf_counter()
    return {
        "graph_policy": graph_policy,
        "sources": patch.sources,
        "changed": [f"{c.module}:{c.qualname}" for c in patch.changed],
        "kernels": [f"{k.module}:{k.qualname}" for k in patch.kernels],
        "kernel_reports": patch.kernel_reports,
        "graph_policy_ran": patch.needs_graph_policy,
        "modules": len(patch.modules),
        "traced": sorted(traced_functions()),
        "revision_factor": revision_factor(),
        "patch_seconds": patched - started,
        "policy_seconds": done - patched,
        **timings,
        "total_seconds": done - started,
    }
