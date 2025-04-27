# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.worker.worker_base import WorkerBase as WorkerBaseV0

import vllm.v1.kv.kv_transfer
from vllm.v1.kv.kv_transfer import KVTransferAgent

logger = init_logger(__name__)
print("\033[92mInside vllm.v1.worker.worker_base.py\033[0m")
logger.debug_learning("Inside vllm.v1.worker.worker_base.py")

class WorkerBase(WorkerBaseV0):
    """
    Abstract class for v1 worker, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in v0 WorkerBase
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """
        Initialize common worker components.
        
        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver 
            responsibilities
        """
        # Configuration storage
        super().__init__(vllm_config=vllm_config)

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: Optional[torch.device] = None
        self.model_runner: Optional[nn.Module] = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        super().init_worker(all_kwargs)
        pass

    def init_kv_transfer(self) -> None:
        # Setup the KV trasnfer configuration
        logger.debug_learning(f"Initializing KV transfer configuration...?")
        scheduler_config = self.vllm_config.scheduler_config
        logger.debug_learning(f"Scheduler config: {scheduler_config.kv_transfer_role = }")
        if scheduler_config.kv_transfer_role is None:
            return
        
        self.kv_transfer_agent = KVTransferAgent(
            self.rpc_rank,
            self.kv_transfer_role,
            self.kv_transfer_init_port_base,
        )

        logger.debug_learning(
            f"Worker {self.rpc_rank} (role = {self.kv_transfer_agent.kv_transfer_role}) with KV transfer init port {self.kv_transfer_agent.kv_transfer_init_port}"
        )
        
        # Start the server in a separate thread
        self.kv_transfer_agent.start_kv_transfer_server()
        return