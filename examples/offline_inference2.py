import asyncio
from collections import deque
from typing import List, Optional, Any, Dict

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
from vllm.config import ParallelConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, Sequence
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import get_open_port, get_ip

logger = init_logger(__name__)

# Create a dummy LLM to capture the configurations.
llm = LLM(model="facebook/opt-125m", tensor_parallel_size=2, dummy_llm=True)
engine_args = llm.engine_args
engine_configs = engine_args.create_engine_configs()
model_config, cache_config, parallel_config, scheduler_config = engine_configs

# Create the workers (from ground up)
num_gpus = 1
placement_group = initialize_cluster(
    # `initialize_cluster` function only uses
    # parallel_config's `world_size` and `worker_use_ray`.
    # It does not think about the TP ranking and placement of engines.
    # We use it to construct the ray workers first.
    parallel_config,
    engine_use_ray=True,
)
ray_remote_kwargs = {}
# for bundle_id, bundle in enumerate(placement_group.bundle_specs):
prefill_scheduling_strategy = PlacementGroupSchedulingStrategy(
    placement_group=placement_group,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)
decode_scheduling_strategy = PlacementGroupSchedulingStrategy(
    placement_group=placement_group,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=1,
)
prefill_ray_worker = ray.remote(
    num_cpus=0,
    num_gpus=num_gpus,
    scheduling_strategy=prefill_scheduling_strategy,
    **ray_remote_kwargs,
)(RayWorkerVllm).remote(model_config.trust_remote_code)
decode_ray_worker = ray.remote(
    num_cpus=0,
    num_gpus=num_gpus,
    scheduling_strategy=decode_scheduling_strategy,
    **ray_remote_kwargs,
)(RayWorkerVllm).remote(model_config.trust_remote_code)

all_workers: List[RayWorkerVllm] = [prefill_ray_worker, decode_ray_worker]


def run_workers(
    workers,
    method: str,
    *args,
    **kwargs,
) -> Any:
    result = [
        worker.execute_method.remote(method, *args, **kwargs)
        for worker in workers
    ]
    if result:
        result = ray.get(result)
    return result


async def run_workers_async(
    workers,
    method: str,
    *args,
    **kwargs,
) -> Any:
    raise NotImplementedError
    result = [
        worker.execute_method.remote(method, *args, **kwargs)
        for worker in workers
    ]
    return result


run_workers(all_workers, "set_cuda_visible_devices", [0, 1])

# Initialize the workers.
# TODO: Need to modify the Worker code with the distributed environment init code
# - Worker.rank = global_gpu_rank
# - Worker.local_rank = local_gpu_rank
# - Worker.role_id = 0 / 1 (prefill / decode)
# - Worker.role_pp_rank = ...
# - Worker.role_tp_rank = ...
# - Worker.role_pp_size = parallel_config.pipeline_parallel_size
# - Worker.role_tp_size = parallel_config.tensor_parallel_size

from vllm.worker.worker import Worker, WorkerRankConfig

# The parallel config is now wrt prefill / decode
prefill_parallel_config = ParallelConfig(
    pipeline_parallel_size=1,
    tensor_parallel_size=1,
    worker_use_ray=True,
)
decode_parallel_config = ParallelConfig(
    pipeline_parallel_size=1,
    tensor_parallel_size=1,
    worker_use_ray=True,
)

# Initialize the communication between GPUs.
driver_ip = get_ip()
distributed_init_method = f"tcp://{driver_ip}:{get_open_port()}"

# Worker.local_rank:
#   Use to setup the GPU device.
#   In fact, it uses only once: `torch.device(self.local_rank)`
# TODO: Probably we should get a
#   global_parallel_config that specifies the world_size
#

prefill_worker_rank_config = WorkerRankConfig(
    global_gpu_rank=0, global_world_size=2,
    local_gpu_rank=0, local_world_size=2,
    role_id=0,
    role_pp_rank=0, role_pp_size=1, role_tp_rank=0, role_tp_size=1,
)
decode_worker_rank_config = WorkerRankConfig(
    global_gpu_rank=0, global_world_size=2,
    local_gpu_rank=1, local_world_size=2,
    role_id=1,
    role_pp_rank=0, role_pp_size=1, role_tp_rank=0, role_tp_size=1,
)
prefill_ray_worker.init_worker.remote(
    lambda: Worker(
        model_config,
        prefill_parallel_config,
        scheduler_config,
        0,
        0,
        distributed_init_method,
        True,  # is_driver_worker
        prefill_worker_rank_config,
    )
)
decode_ray_worker.init_worker.remote(
    lambda: Worker(
        model_config,
        decode_parallel_config,
        scheduler_config,
        1,
        1,
        distributed_init_method,
        # True,
        False,
        decode_worker_rank_config,
    )
)


def init_cache():
    """Initialize the cache engine for a worker. Adapted from LLMEngine."""
    num_blocks = run_workers(
        all_workers,
        "profile_num_available_blocks",
        block_size=cache_config.block_size,
        gpu_memory_utilization=cache_config.gpu_memory_utilization,
        cpu_swap_space=cache_config.swap_space_bytes,
    )
    num_gpu_blocks = min(b[0] for b in num_blocks)
    num_cpu_blocks = min(b[1] for b in num_blocks)

    # ... ignored some assertion checking as they are too noisy ...
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks

    run_workers(all_workers, "init_cache_engine", cache_config=cache_config)
    run_workers(all_workers, "warm_up_model")
    return


r1 = prefill_ray_worker.init_worker.remote()
ray.get(r1)
r2 = decode_ray_worker.init_worker.remote()
ray.get(r2)
# run_workers(all_workers, "init_model")
run_workers(all_workers, "load_model")
init_cache()

# ... and here we are done with the Worker initialization.

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Just want to check the remote environment is initialized.

# Now initialize the scheduler for prefill and decode worker.
tokenizer = get_tokenizer(
    model_config.tokenizer,
    tokenizer_mode=model_config.tokenizer_mode,
    trust_remote_code=model_config.trust_remote_code,
    tokenizer_revision=model_config.tokenizer_revision,
    revision=model_config.revision
)
prefill_scheduler = Scheduler(scheduler_config, cache_config)
decode_scheduler = Scheduler(scheduler_config, cache_config)

prefill_ray_worker_group = [prefill_ray_worker]
decode_ray_worker_group = [decode_ray_worker]

migration_queue = deque()


# Do one prefill
def prefill_step():
    seq_group_metadata_list, scheduler_outputs = prefill_scheduler.schedule()
    # Run worker with lead worker...
    all_outputs = run_workers(
        prefill_ray_worker_group,
        "execute_model",
        seq_group_metadata_list,
        scheduler_outputs,
        tokenizer,
    )
    output = all_outputs[0]
    # TODO: This output need to get processed by complicated functions
    #   LLMEngine._process_sequence_group_outputs()
    #   LLMEngine._process_model_outputs()
    return output


def decode_step():
    seq_group_metadata_list, scheduler_outputs = decode_scheduler.schedule()
    # Run worker with lead worker...
    all_outputs = run_workers(
        decode_ray_worker_group,
        "execute_model",
        seq_group_metadata_list,
        scheduler_outputs,
        tokenizer,
    )
    output = all_outputs[0]
    # TODO: This output need to get processed by complicated functions
    #   LLMEngine._process_sequence_group_outputs()
    #   LLMEngine._process_model_outputs()
    return output


for request_id, prompt in enumerate(prompts):
    # Add request to prefill scheduler.
    prompt_token_ids = tokenizer.encode(prompt)
    block_size = cache_config.block_size
    seq = Sequence(request_id, prompt, prompt_token_ids, block_size)
    seqs = []
    seq_group = SequenceGroup(request_id=str(request_id), seqs=seqs, sampling_params=sampling_params, arrival_time=0)
    prefill_scheduler.add_seq_group(seq_group)

    # Run prefill step.
    output = prefill_step()
    # migration output fro prefill to decode
    decode_step()

    pass

# # Create an LLM.
# llm = LLM(model="facebook/opt-125m")
# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
