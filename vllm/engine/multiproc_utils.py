import asyncio
import multiprocessing
from multiprocessing import Process

from vllm.config import ParallelConfig, SchedulerConfig, ModelConfig


def init_worker_loop(
    inbound_queue: multiprocessing.Queue,  # input
    outbound_queue: multiprocessing.Queue,  # output
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    is_driver_worker: bool = False,
):
    from vllm.worker.worker import Worker
    worker = Worker(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        is_driver_worker=is_driver_worker,
    )

    while True:
        task = inbound_queue.get()
        if task is None:
            print(f"Worker {rank} received None, exiting.")
            break
        method, args, kwargs = task
        result = getattr(worker, method)(*args, **kwargs)
        outbound_queue.put(result)
    return


class WorkerFuture:

    def __init__(self, queue):
        self.queue = queue

    def done(self):
        return not self.queue.empty()

    def result(self):
        return self.queue.get()


class WorkerProcess:

    def __init__(self, model_config, parallel_config, scheduler_config,
                 local_rank, rank, distributed_init_method, is_driver_worker):
        self.inbound_queue = multiprocessing.Queue()
        self.outbound_queue = multiprocessing.Queue()
        self.worker = Process(
            target=init_worker_loop,
            args=(
                self.inbound_queue,
                self.outbound_queue,
                model_config,
                parallel_config,
                scheduler_config,
                local_rank,
                rank,
                distributed_init_method,
                is_driver_worker,
            ),
        )
        pass

    def start_worker(self):
        self.worker.start()
        return self

    def kill_worker(self):
        self.inbound_queue.put(None)
        self.worker.join()
        pass

    def execute_method_future(self, method, *args, **kwargs):
        self.inbound_queue.put((method, args, kwargs))
        # FIXME: This is a hack!
        return WorkerFuture(self.outbound_queue)

    def execute_method(self, method, *args, **kwargs):
        self.inbound_queue.put((method, args, kwargs))
        result = self.outbound_queue.get()
        return result

    async def execute_method_async(self, method, *args, **kwargs):
        # FIXME: There is a better way to write this.
        self.inbound_queue.put((method, args, kwargs))
        while self.outbound_queue.empty():
            await asyncio.sleep(0.001)
        result = self.outbound_queue.get()
        return result
