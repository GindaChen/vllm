"""A GPU worker class."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, get_tensor_model_parallel_src_rank,
    get_tensor_model_parallel_group, get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_first_rank, ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        # Uninitialized model runner. Will be initialized by self.init_model().
        self.is_lead_worker: bool = None  # lead worker of the TP-group
        self.model_runner: ModelRunner = None  # ModelRunner

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine: 'CacheEngine' = None
        self.cache_events = None
        self.gpu_cache = None

    def init_model(self) -> None:
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        _check_if_gpu_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        is_lead_worker = (self.rank == get_tensor_model_parallel_src_rank())
        self.is_lead_worker = is_lead_worker
        self.model_runner = ModelRunner(self.model_config,
                                        self.parallel_config,
                                        self.scheduler_config, is_lead_worker)
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()
        # FIXME: Hack - if I'm the leader of the TP-group, set the model_runner's driver flag to true.

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def execute_lambda(self, lambda_fn, *args, **kwargs):
        return lambda_fn(self, *args, **kwargs)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    # FIXME: (hack) Hack the way out to identify prefill/decode role of the worker.
    def transfer_kv_cache(self,
                          send_blocks: List[int] = None,
                          recv_blocks: List[int] = None):
        """Migrate KV cache from prefill to decode. If the worker is
        responsible for prefill, then send; otherwise, receive.
        """

        if not send_blocks or not recv_blocks:
            return

        print(
            f"Worker {self.rank} executes transfer_kv_cache with {send_blocks = } and {recv_blocks = }"
        )

        def is_prefill_worker():
            leader_rank = get_pipeline_model_parallel_first_rank()
            rank = torch.distributed.get_rank()
            return leader_rank == rank

        if is_prefill_worker():
            self.cache_engine.send_blocks(send_blocks)
        else:
            self.cache_engine.recv_blocks(recv_blocks)

        torch.cuda.synchronize()
        return

    # FIXME: (hack) SHOULD NOT BE IN MASTER!

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        # TODO: Decouple data transfer from the send/recv logic.
        send_blocks: List[int] = None,
        recv_blocks: List[int] = None,
    ) -> Optional[SamplerOutput]:
        # FIXME: (hack) pre-execution metadata from driver node to this worker.

        # Transfer blocks from prefill worker to this worker (if any)
        self.transfer_kv_cache(send_blocks=send_blocks,
                               recv_blocks=recv_blocks)

        # Perform cache swapping
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        num_seq_groups = len(seq_group_metadata_list)
        print(
            f"Worker {self.rank} executes model with {seq_group_metadata_list = }"
        )
        print(f"Worker {self.rank} executes model with {num_seq_groups = }")
        if num_seq_groups == 0:
            return {}

        # Execute the model in the same tensor-parallel group.
        lead_worker_rank = get_tensor_model_parallel_src_rank()
        group = get_tensor_model_parallel_group()
        print(f"Worker {self.rank} executes model with {lead_worker_rank = } "
              f"and {torch.distributed.get_process_group_ranks(group) = }")
        print(f"Worker {self.rank} property: \n"
              f"{torch.distributed.get_rank() = }\n"
              f"{self.rank = }\n"
              f"{self.is_driver_worker = }\n"
              f"{self.model_runner.is_driver_worker = }\n")

        output = self.model_runner.execute_model(
            seq_group_metadata_list,
            self.gpu_cache,
            lead_worker_rank=lead_worker_rank,
            group=group,
        )
        return output


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
