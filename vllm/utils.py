import enum
import os
import socket
import time
import uuid
from platform import uname
from typing import List

import psutil
import torch


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def is_hip() -> bool:
    return torch.version.hip is not None


def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # NOTE: This import statement should be executed lazily since
    # the Neuron-X backend does not have the `cuda_utils` module.
    from vllm._C import cuda_utils

    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74
    max_shared_mem = cuda_utils.get_device_attribute(
        cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu)
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
    return s.getsockname()[0]


def get_distributed_init_method(ip: str, port: int) -> str:
    return f"tcp://{ip}:{port}"


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


# debug_pront = print
debug_pront = lambda *_, **__: None
debug_pront_2 = lambda *_, **__: None

logger_g = None


# debug_pront_3 = lambda *_, **__: None
def debug_pront_3(*args, **kwargs):
    global logger_g
    if not logger_g:
        from vllm.logger import init_logger
        logger_g = init_logger(__name__)
    logger_g.info(*args, **kwargs)
    return


# debug_slept = time.sleep
debug_slept = lambda *_, **__: None


def tensor_size_in_bytes(tensor):
    """
    Calculate the size of a tensor in bytes.

    Args:
    tensor (torch.Tensor): The tensor for which to calculate the size.

    Returns:
    int: The size of the tensor in bytes.
    """
    return tensor.nelement() * tensor.element_size()


def human_readable_size(size_in_bytes):
    """
    Convert a size in bytes to a human readable string.

    Args:
    size_in_bytes (int): The size in bytes.

    Returns:
    str: The human readable size.
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 ** 2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 ** 3:
        return f"{size_in_bytes / 1024 ** 2:.2f} MB"
    elif size_in_bytes < 1024 ** 4:
        return f"{size_in_bytes / 1024 ** 3:.2f} GB"
    elif size_in_bytes < 1024 ** 5:
        return f"{size_in_bytes / 1024 ** 4:.2f} TB"
    else:
        return f"{size_in_bytes / 1024 ** 5:.2f} PB"


def tensor_size_in_bytes_human_readable(tensor):
    return human_readable_size(tensor_size_in_bytes(tensor))
