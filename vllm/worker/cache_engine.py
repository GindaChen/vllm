"""CacheEngine class for managing the KV cache."""
import random
import time
from typing import Dict, List, Tuple

import torch

from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import get_pipeline_model_parallel_next_rank, \
    get_pipeline_model_parallel_prev_rank, get_tensor_model_parallel_rank
from vllm.utils import in_wsl, debug_pront, debug_pront_3, tensor_size_in_bytes, human_readable_size

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        self.gpu_cache: List[KVCache] = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        debug_pront_3(f"At allocate_gpu_cache(), got {self.num_layers = }")
        debug_pront_3(f"At allocate_gpu_cache(), got {key_block_shape = }")
        value_block_shape = self.get_value_block_shape()
        debug_pront_3(f"At allocate_gpu_cache(), got {value_block_shape = }")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def send_blocks(self, block_ids: List[int]) -> None:
        tasks = []
        rank = get_pipeline_model_parallel_next_rank()
        start_time = time.perf_counter()
        debug_pront_3(f"Sending blocks {len(block_ids) = } to {rank = }")
        total_size = 0
        for block_id in block_ids:
            for i in range(self.num_layers):
                # debug_pront_3(f"Sending block: {block_id} from layer {i}")
                for j in [0, 1]:
                    a = self.gpu_cache[i][j][block_id]
                    x = torch.distributed.isend(a, dst=rank)
                    total_size += tensor_size_in_bytes(a)
        #             tasks.append(x)
        #         pass
        # for task in tasks:
        #     debug_pront(f"Waiting for task: {task}")
        #     task.wait()
        end_time = time.perf_counter()
        duration = end_time - start_time
        duration *= 1000
        total_size = human_readable_size(total_size)
        debug_pront_3(
            f"Done sending blocks {len(block_ids) = } ({total_size = }) to {rank = } in {duration} ms"
        )
        return

    def recv_blocks(self, block_ids: List[int]) -> None:
        tasks = []
        rank = get_pipeline_model_parallel_prev_rank()
        start_time = time.perf_counter()
        # debug_pront_3(f"Receiving blocks {block_ids = } from {rank = }")
        debug_pront_3(f"Receiving blocks {len(block_ids) = } from {rank = }")
        total_size = 0
        for block_id in block_ids:
            for i in range(self.num_layers):
                # debug_pront_3(f"Receiving block: {block_id} from layer {i}")
                for j in [0, 1]:
                    a = self.gpu_cache[i][j][block_id]
                    x = torch.distributed.irecv(a, src=rank)
                    total_size += tensor_size_in_bytes(a)
                    tasks.append(x)
                pass
        end_time = time.perf_counter()
        duration = end_time - start_time
        duration *= 1000
        total_size = human_readable_size(total_size)
        debug_pront_3(
            f"Done receiving (irecv) blocks {len(block_ids) = } ({total_size = }) from {rank = } in {duration} ms"
        )
        for i, task in enumerate(tasks):
            task.wait()
        end_time = time.time()
        duration = end_time - start_time
        duration *= 1000
        debug_pront_3(
            f"Done waiting (irecv) blocks {len(block_ids) = } ({total_size = }) from {rank = } in {duration} ms"
        )
        return

    def send_blocks_batch_layer(self, block_ids):
        # TODO: Batch and chunk the blocks such that it reuses big-pages to do the block transfer.
        tasks = []
        rank = get_pipeline_model_parallel_next_rank()
        start_time = time.perf_counter()
        total_size = 0

        N = len(block_ids)
        k_tensor = torch.empty(N, *self.get_key_block_shape(), device="cuda")
        v_tensor = torch.empty(N, *self.get_value_block_shape(), device="cuda")

        for i in range(self.num_layers):
            for j, block_id in enumerate(block_ids):
                blocks = self.gpu_cache[i]
                block = blocks[block_id]
                k_block, v_block = block
                k_tensor[j, :] = k_block
                v_tensor[j, :] = v_block
            torch.distributed.isend(k_tensor, dst=rank)
            torch.distributed.isend(v_tensor, dst=rank)

        end_time = time.perf_counter()
        duration = end_time - start_time
        duration *= 1000
        total_size = human_readable_size(total_size)
        debug_pront_3(
            f"Done sending blocks {len(block_ids) = } ({total_size = }) to {rank = } in {duration} ms"
        )
        return

    def recv_blocks_batch_layer(self, block_ids):
        tasks = []
        rank = get_pipeline_model_parallel_next_rank()
        start_time = time.perf_counter()
        total_size = 0

        N = len(block_ids)
        k_tensor = torch.empty(N, *self.get_key_block_shape(), device="cuda")
        v_tensor = torch.empty(N, *self.get_value_block_shape(), device="cuda")

        for i in range(self.num_layers):
            e1 = torch.distributed.irecv(k_tensor, src=rank)
            e1.wait()
            e2 = torch.distributed.irecv(v_tensor, src=rank)
            e2.wait()
            for j, block_id in enumerate(block_ids):
                self.gpu_cache[i][block_id][0][:] = k_tensor[j, :]
                self.gpu_cache[i][block_id][1][:] = v_tensor[j, :]

        end_time = time.perf_counter()
        duration = end_time - start_time
        duration *= 1000
        total_size = human_readable_size(total_size)
        debug_pront_3(
            f"Done sending blocks {len(block_ids) = } ({total_size = }) to {rank = } in {duration} ms"
        )
        return

    def retrieve_blocks(self, src_block_ids: List[int],
                        dst_block_ids: List[int]):
        """Retrieve the blocks from the another GPU (that has exposed memory handler for me)."""
        # context_tp_size = decoding_tp_size = self.parallel_config.tensor_parallel_size
        decoding_tp_rank = get_tensor_model_parallel_rank()
        decoding_worker_k_caches = [k for k, v in self.gpu_cache]
        decoding_worker_v_caches = [v for k, v in self.gpu_cache]

        # Call the kernel
        x = 16 // torch.tensor([], dtype=self.dtype).element_size()

        cache_ops.migrate_blocks(
            int(self.num_layers),
            int(self.num_gpu_blocks),
            int(self.num_heads),
            int(self.head_size),
            int(self.block_size),
            int(x),
            decoding_tp_rank,
            src_block_ids,
            dst_block_ids,
            decoding_worker_k_caches,
            decoding_worker_v_caches,
        )
        return

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
