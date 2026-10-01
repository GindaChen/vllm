# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-parameter weight digests and reset, for verifying weight updates."""

import hashlib
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from vllm.model_executor.parameter import SharedWeightParameter


def _tensors(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    # Both keep their bytes in inner tensors (e.g. TorchAO's qdata and scale).
    if isinstance(tensor, SharedWeightParameter):
        for index, partition in tensor.partitions.items():
            yield from _tensors(f"{name}.{index}", partition)
    elif is_traceable_wrapper_subclass(tensor):
        for attr in tensor.__tensor_flatten__()[0]:
            yield from _tensors(f"{name}.{attr}", getattr(tensor, attr))
    else:
        yield name, tensor


def _weights(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        yield from _tensors(name, param)


def compute_tensor_digests(model: nn.Module, hash_workers: int = 0) -> dict[str, str]:
    """Hash all weight bytes while weights are quiescent.

    Zero workers retains serial behavior. One to four CPU workers overlap hashing
    with caller-thread tensor copies, retaining at most that many queued payloads.
    This bounds payload count, not total host memory or producer temporaries.
    """
    if type(hash_workers) is not int or not 0 <= hash_workers <= 4:
        raise ValueError("hash_workers must be an integer between zero and four")
    if hash_workers:
        digests = {}
        pending = deque()
        with ThreadPoolExecutor(max_workers=hash_workers) as pool:
            for name, weight in _weights(model):
                if len(pending) == hash_workers:
                    first_name, future = pending.popleft()
                    digests[first_name] = future.result().hexdigest()
                payload = (
                    weight.detach()
                    .cpu()
                    .contiguous()
                    .view(-1)
                    .view(torch.uint8)
                    .numpy()
                )
                pending.append((name, pool.submit(hashlib.sha256, payload)))
                del payload
            for name, future in pending:
                digests[name] = future.result().hexdigest()
        return digests
    return {
        name: hashlib.sha256(
            weight.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, weight in _weights(model)
    }


def zero_weights(model: nn.Module) -> None:
    for _, weight in _weights(model):
        weight.data.zero_()
