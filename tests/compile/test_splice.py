# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for editing a Dynamo graph without re-tracing."""

import operator

import pytest
import torch
import torch.fx as fx
from torch import nn

from vllm.compilation import hot_op  # noqa: F401  (registers the op)
from vllm.compilation import splice


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(4, 8, bias=False)
        self.down = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        h = self.up(x)
        h = torch.relu(h)
        return self.down(h)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x):
        y = x * 2
        return y + self.mlp(y)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def dynamo_graph(model, x):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    torch.compile(model, backend=backend, fullgraph=True)(x)
    return graphs[0]


def is_state(node):
    return node.op == "placeholder" and "parameters" in node.name


@pytest.fixture
def traced():
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(3, 4)
    with torch.no_grad():
        gm = dynamo_graph(model, x)
    return model, x, gm


def test_find_regions_one_per_call(traced):
    _, _, gm = traced
    regions = splice.find_regions(gm, MLP.forward.__code__, is_state)
    assert len(regions) == 2
    for region in regions:
        assert len(region.inputs) == 1
        assert region.output in region.nodes


def test_flatten_inverts_split(traced):
    model, x, gm = traced
    ids = {}
    part = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.relu:
            part += 1
        ids[node] = part
    split = fx.passes.split_module.split_module(
        gm, None, lambda n: ids[n], keep_original_order=True
    )
    flat = splice.flatten(split)
    assert not any(n.op == "call_module" for n in flat.graph.nodes)
    args = [p for p in gm.graph.nodes if p.op == "placeholder"]
    inputs = [
        x if "x" in a.name else dict(model.named_parameters())[
            a.name.replace("l_self_modules_", "")
            .replace("_modules_", ".")
            .replace("_parameters_", ".")
            .rstrip("_")
        ]
        for a in args
    ]
    with torch.no_grad():
        assert torch.equal(flat(*inputs)[0], gm(*inputs)[0])


def test_replace_with_hot_call_rewires_users(traced):
    _, _, gm = traced
    regions = splice.find_regions(gm, MLP.forward.__code__, is_state)
    for hot_id, region in enumerate(regions):
        splice.replace_with_hot_call(gm, region, hot_id)
    calls = [
        n
        for n in gm.graph.nodes
        if n.target is torch.ops.vllm.rsii_hot_call.default
    ]
    assert [c.args[1] for c in calls] == [0, 1]
    assert not splice.find_regions(gm, MLP.forward.__code__, is_state)
    # Each hot call writes the buffer the layer's residual add reads.
    for call in calls:
        out = call.args[0]
        assert out.target is torch.empty_like
        assert any(u.target is operator.add for u in out.users)
