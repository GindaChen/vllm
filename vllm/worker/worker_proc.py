import asyncio
import multiprocessing
from multiprocessing import Process
from typing import Union, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)


def worker_process(
    task_queue,
    result_queue,
    local_rank,
    rank,
    distributed_init_method,
    model_config,
    parallel_config,
    scheduler_config,
    lora_config,
):
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


class WorkerProcess:

    def __init__(
        self,
        local_rank,
        rank,
        distributed_init_method,
        model_config,
        parallel_config,
        scheduler_config,
        lora_config,
    ):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_idle = False

        self.proc = Process(
            target=worker_process,
            args=(
                self.task_queue,
                self.result_queue,
                self.local_rank,
                self.rank,
                self.distributed_init_method,
                # Config
                model_config,
                parallel_config,
                scheduler_config,
                lora_config,
            ))

    def start_worker_loop(self):
        self.proc.start()
        return self

    async def invoke_async(self, func_name: Union[str, Callable], *args,
                           **kwargs):
        logger.info(
            f"[Worker({self.rank})] Invoking {func_name} with args={args}, kwargs={kwargs}"
        )
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
        logger.info(f"[Worker({self.rank})] Killing the worker...")
        _none = (None, None, None)
        self.task_queue.put(_none)
        self.proc.join()
        logger.info(f"[Worker({self.rank})] Worker killed.")
        return
