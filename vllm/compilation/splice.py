# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edit the Dynamo graph of a live compiled model without re-tracing.

vLLM keeps the graph Dynamo traced next to the compiled callable
(``VllmSerializableFunction.graph_module``). After a model method changes,
re-tracing the whole model with Dynamo takes seconds. When the change only
needs a region of that graph replaced, this module edits the saved graph
instead and runs vLLM's backend on it again: the graph is split at the
splitting ops, unchanged pieces hit the Inductor cache, and only pieces whose
content changed are compiled.

Regions are found from node metadata: the source lines each node was traced
from (``stack_trace``), matched against the code object of the method as it
was when the graph was traced.
"""

import contextlib
import hashlib
import operator
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import CodeType
from typing import Any

import torch
import torch.fx as fx

from vllm.logger import init_logger

logger = init_logger(__name__)

_FRAME = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<name>\S+)')


def serializable_functions(model: torch.nn.Module) -> list[Any]:
    """The compiled callables of every compiled module inside ``model``.

    Covers each ``@support_torch_compile`` instance, nested ones included,
    whose compiled function is held by its AOT-compiled wrapper (the default
    with torch >= 2.10).
    """
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    found = []
    for module in model.modules():
        if not isinstance(module, TorchCompileWithNoGuardsWrapper):
            continue
        aot = getattr(module, "aot_compiled_fn", None)
        artifacts = getattr(aot, "_artifacts", None)
        fn = getattr(artifacts, "compiled_fn", None)
        if fn is None:
            raise ValueError(
                f"{type(module).__name__} has no AOT-compiled function to edit"
            )
        found.append(fn)
    return found


# ---------------------------------------------------------------------------
# Flattening a split graph
# ---------------------------------------------------------------------------


def flatten(split_gm: fx.GraphModule) -> fx.GraphModule:
    """Inline the submodules of a ``split_module`` graph into one graph.

    Inverse of vLLM's ``split_graph`` (with ``keep_original_order``): node
    order and metadata are kept, submodule outputs are forwarded to their
    ``getitem`` users.
    """
    graph = fx.Graph()
    env: dict[fx.Node, Any] = {}
    root = torch.nn.Module()

    def lookup(n: fx.Node) -> Any:
        return env[n]

    for node in split_gm.graph.nodes:
        if node.op == "call_module":
            sub = split_gm._modules[node.target]
            sub_env: dict[fx.Node, Any] = {}
            placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
            if len(placeholders) != len(node.args) or node.kwargs:
                raise ValueError(f"Unexpected call of {node.target}")
            for ph, arg in zip(placeholders, node.args):
                sub_env[ph] = fx.map_arg(arg, lookup)
            for sn in sub.graph.nodes:
                if sn.op == "placeholder":
                    continue
                if sn.op == "output":
                    env[node] = fx.map_arg(sn.args[0], lambda n: sub_env[n])
                    continue
                if sn.op == "get_attr":
                    name = f"{node.target}_{sn.target}".replace(".", "_")
                    setattr(root, name, getattr(sub, sn.target))
                    copy = graph.get_attr(name)
                    copy.meta = dict(sn.meta)
                    sub_env[sn] = copy
                    continue
                sub_env[sn] = graph.node_copy(sn, lambda n: sub_env[n])
        elif (
            node.op == "call_function"
            and node.target is operator.getitem
            and isinstance(env.get(node.args[0]), (tuple, list))
        ):
            env[node] = env[node.args[0]][node.args[1]]
        elif node.op == "get_attr":
            setattr(root, node.target, getattr(split_gm, node.target))
            env[node] = graph.node_copy(node, lookup)
        else:
            env[node] = graph.node_copy(node, lookup)
    graph.lint()
    return fx.GraphModule(root, graph)


# ---------------------------------------------------------------------------
# Finding a traced method's regions
# ---------------------------------------------------------------------------


@dataclass
class Region:
    """One inlined call of a method: its nodes, input and output."""

    nodes: list[fx.Node]
    inputs: list[fx.Node]
    output: fx.Node


def _in_code(node: fx.Node, code: CodeType, lines: range) -> bool:
    trace = node.meta.get("stack_trace") or ""
    for m in _FRAME.finditer(trace):
        if (
            m["name"] == code.co_name
            and int(m["line"]) in lines
            and os.path.basename(m["file"]) == os.path.basename(code.co_filename)
        ):
            return True
    return False


def code_lines(code: CodeType) -> range:
    lines = [line for _, _, line in code.co_lines() if line is not None]
    return range(min(lines), max(lines) + 1)


def find_regions(
    gm: fx.GraphModule,
    code: CodeType,
    is_state: Callable[[fx.Node], bool],
) -> list[Region]:
    """Contiguous runs of nodes traced from ``code``, one per call.

    ``is_state`` tells placeholders that are module state (parameters,
    buffers), which the method reads through ``self`` rather than as
    arguments.
    """
    lines = code_lines(code)
    runs: list[list[fx.Node]] = []
    current: list[fx.Node] = []
    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method") and _in_code(node, code, lines):
            current.append(node)
        elif current and node.op not in ("placeholder", "get_attr"):
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    regions = []
    for nodes in runs:
        members = set(nodes)
        inputs: list[fx.Node] = []
        for n in nodes:
            for a in n.all_input_nodes:
                if a not in members and a not in inputs and not is_state(a):
                    inputs.append(a)
        outputs = [n for n in nodes if any(u not in members for u in n.users)]
        if len(outputs) != 1:
            raise ValueError(
                f"{code.co_qualname}: a call has {len(outputs)} outputs "
                f"({[n.name for n in outputs]}); expected one"
            )
        regions.append(Region(nodes, inputs, outputs[0]))
    return regions


def replace_with_hot_call(gm: fx.GraphModule, region: Region, hot_id: int) -> None:
    """Replace a region by ``out = empty_like(x); rsii_hot_call(out, id, [x])``."""
    if len(region.inputs) != 1:
        raise ValueError(f"Expected one tensor input, got {region.inputs}")
    (x,) = region.inputs
    graph = gm.graph
    y = region.output
    x_val, y_val = x.meta.get("example_value"), y.meta.get("example_value")
    if x_val is not None and y_val is not None and x_val.shape != y_val.shape:
        raise ValueError("Hot functions must return their input's shape")
    with graph.inserting_before(y):
        out = graph.call_function(torch.empty_like, (x,))
        out.meta = {k: v for k, v in y.meta.items() if k != "stack_trace"}
        call = graph.call_function(
            torch.ops.vllm.rsii_hot_call.default, (out, hot_id, [x])
        )
        call.meta = {"example_value": None}
    y.replace_all_uses_with(out, delete_user_cb=lambda u: u not in (out, call))
    for n in reversed(region.nodes):
        if not n.users:
            graph.erase_node(n)
    leftover = [n.name for n in region.nodes if n.graph is graph and n.users]
    if leftover:
        raise ValueError(f"Region nodes still used outside: {leftover}")
    graph.lint()
    gm.recompile()


# ---------------------------------------------------------------------------
# Recompiling the edited graph
# ---------------------------------------------------------------------------


def _fake_mode(gm: fx.GraphModule) -> Any:
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("example_value")
            mode = getattr(val, "fake_mode", None)
            if mode is not None:
                return mode
    raise ValueError("Graph placeholders carry no fake tensors")


def recompile(fn: Any, gm: fx.GraphModule, vllm_config: Any, tag: str) -> None:
    """Run vLLM's backend on ``gm`` and make ``fn`` call the result.

    The vLLM-level compiled-graph cache is indexed by piece number, so the
    edited graph gets its own cache directory, derived from ``tag``.
    Inductor's content-addressed caches are shared.
    """
    from torch._guards import TracingContext, tracing

    from vllm.compilation.backends import VllmBackend
    from vllm.config import set_current_vllm_config

    config = vllm_config.compilation_config
    base = config.cache_dir
    digest = hashlib.sha256(tag.encode()).hexdigest()[:10]
    config.cache_dir = os.path.join(base.rstrip("/"), f"splice-{digest}")
    functorch = (
        torch._functorch.config.patch(fn.aot_autograd_config)
        if getattr(fn, "aot_autograd_config", None) is not None
        else contextlib.nullcontext()
    )
    backend = VllmBackend(vllm_config, fn.prefix, fn.is_encoder)
    try:
        with (
            set_current_vllm_config(vllm_config),
            tracing(TracingContext(_fake_mode(gm))),
            functorch,
        ):
            result = backend(gm, list(fn.example_inputs))
    finally:
        config.cache_dir = base
    fn.optimized_call = result.optimized_call
    fn.vllm_backend = backend
    fn.graph_module = gm
