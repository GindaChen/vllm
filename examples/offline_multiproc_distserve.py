import asyncio
import dataclasses
import multiprocessing
import socket
from multiprocessing import Process
from typing import Union, Callable, List

from vllm import LLM, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=1,  #
    # tensor_parallel_size=2,
    is_dummy=True,
)
engine_args = llm.engine_args
model_config, cache_config, parallel_config, scheduler_config, lora_config = engine_args.create_engine_configs()


# FIXME: Hack - avoid using ray naturally.


def get_distributed_method() -> str:
    def get_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]

    def get_open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def get_distributed_init_method(ip: str, port: int) -> str:
        return f"tcp://{ip}:{port}"

    return get_distributed_init_method(get_ip(), get_open_port())


# Then use the configs to construct workers in a different process.
# The worker should also have a mechanism to communicate with the main process.

def worker_process(task_queue, result_queue, local_rank, rank, distributed_init_method):
    from vllm.worker.worker import Worker
    # Create a worker.
    logger.info(f"Creating a worker with local_rank={local_rank}, rank={rank}")
    worker = Worker(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        lora_config=lora_config,
    )

    # Then for each task from the task queue,
    # call the methods of the worker to process the task.
    logger.info(f"Worker({worker.rank}) Waiting for tasks...")
    while True:
        task = task_queue.get()
        func_name, args, kwargs = task
        if task is None or func_name is None:
            return
        logger.info(f"Worker({worker.rank}) Received task: {task}")

        func = getattr(worker, func_name)
        result = func(*args, **kwargs)
        logger.info(f"Worker({worker.rank}) Finished task: {task}.")
        result_queue.put(result)

    return


distributed_init_method = get_distributed_method()


class WorkerProcess:
    def __init__(self, local_rank, rank, distributed_init_method):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_idle = False

        self.proc = Process(
            target=worker_process,
            args=(
                self.task_queue, self.result_queue,
                self.local_rank, self.rank,
                self.distributed_init_method
            )
        )

    def start_worker_loop(self):
        self.proc.start()
        return self

    def invoke(self, func_name: Union[str, Callable], *args, **kwargs):
        if callable(func_name):
            func_name = func_name.__name__
        self.is_idle = False
        self.task_queue.put((func_name, args, kwargs))
        result = self.result_queue.get()
        self.is_idle = True
        return result

    async def invoke_async(self, func_name: Union[str, Callable], *args, **kwargs):
        logger.info(f"[Worker({self.rank})] Invoking {func_name} with args={args}, kwargs={kwargs}")
        if callable(func_name):
            func_name = func_name.__name__
        self.is_idle = False
        self.task_queue.put((func_name, args, kwargs))
        while True:
            try:
                result = self.result_queue.get(timeout=0)
                break
            except multiprocessing.queues.Empty:
                await asyncio.sleep(0)
        self.is_idle = True
        return result

    def kill(self):
        _none = (None, None, None)
        self.task_queue.put(_none)
        self.proc.join()
        return


def setup_workers(workers: List[WorkerProcess]):
    async def _setup_comm(worker):
        await worker.invoke_async("init_model")

    async def _setup_worker(worker):
        await worker.invoke_async("load_model")
        cache_config.num_gpu_blocks = 10000
        cache_config.num_cpu_blocks = 10000
        await worker.invoke_async("init_cache_engine", cache_config)
        await worker.invoke_async("warm_up_model")
        return

    main_loop = asyncio.get_event_loop()
    main_loop.run_until_complete(asyncio.gather(*[_setup_comm(worker) for worker in workers]))
    main_loop.run_until_complete(asyncio.gather(*[_setup_worker(worker) for worker in workers]))
    return


if __name__ == '__main__':
    logger.info("Creating a worker process...")
    parallel_config.pipeline_parallel_size = 2
    parallel_config.world_size = 2
    parallel_config.is_dist_serve = True

    prefill_worker = WorkerProcess(
        local_rank=0, rank=0,
        distributed_init_method=distributed_init_method,
    ).start_worker_loop()
    decode_worker = WorkerProcess(
        local_rank=1, rank=1,
        distributed_init_method=distributed_init_method,
    ).start_worker_loop()

    logger.info("Worker process created. Start to setup the worker.")
    setup_workers([prefill_worker, decode_worker])
    prefill_worker.kill()
