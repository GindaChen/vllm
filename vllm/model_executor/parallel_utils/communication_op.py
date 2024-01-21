from collections import namedtuple
from typing import Any, Dict, List, Optional, Union

from torch.distributed import ProcessGroup

import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_global_rank,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    torch.distributed.all_reduce(input_,
                                 group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size,) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim],) +
                                          input_size[dim + 1:])
    return output_tensor


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    Args:
        input_: Input tensor to gather.
        dst: Destination rank in the tensor parallel group.
            Defaults to 0 (the group-rank of the leader proc).
        dim: Dimension to gather. Defaults to -1.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    group = get_tensor_model_parallel_group()
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    rank = get_tensor_model_parallel_rank()
    if rank == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    dst_global_rank = get_tensor_model_parallel_global_rank(dst)
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst_global_rank,
                             group=group)
    output_tensor = None
    if rank == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    return output_tensor


def broadcast(input_: torch.Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None):
    """Broadcast the input tensor.

    Args:
        input_: Input tensor to broadcast.
        src: Source rank with respect to `group`. Defaults to 0.
        group: Process group. Defaults to the world group.

    """
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_


def broadcast_object_list(obj_list: List[Any],
                          src: int = 0,
                          group: Optional[ProcessGroup] = None):
    """Broadcast the input object list.

    Args:
        obj_list: Input object list to broadcast.
        src: Source rank with respect to `group`. Defaults to 0.
        group: Process group. Defaults to the world group.
    """
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list


TensorMetadata = namedtuple("TensorMetadata", ["dtype", "size"])


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
) -> Dict[Any, Union[torch.Tensor, Any]]:
    """Broadcast the input tensor dictionary.

    Args:
        tensor_dict: Input tensor dictionary to broadcast.
        src: Source rank with respect to `group`. Defaults to 0.
        group: Process group. Defaults to the world group.

    """
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict

    rank = torch.distributed.get_rank(group=group)
    src_global_rank = ranks[src]
    if rank == src:
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list = []
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.is_cuda, (
                    f"Tensor {key}: {value} is not on cuda. Currently we only "
                    f"support broadcasting tensors on cuda.")
                metadata_list.append(
                    (key, TensorMetadata(value.dtype, value.size())))
            else:
                metadata_list.append((key, value))
        torch.distributed.broadcast_object_list([metadata_list],
                                                src=src_global_rank,
                                                group=group)
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = tensor_dict[key]
                torch.distributed.broadcast(tensor, src=src, group=group)
    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list,
                                                src=src_global_rank,
                                                group=group)
        metadata_list = recv_metadata_list[0]
        tensor_dict = {}
        async_handles = []
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device="cuda")
                async_handle = torch.distributed.broadcast(tensor,
                                                           src=src_global_rank,
                                                           async_op=True,
                                                           group=group)
                async_handles.append(async_handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
