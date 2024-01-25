import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
                    Union, AsyncIterator)

from vllm.config import ModelConfig
from vllm.core.dist_scheduler import DistScheduler, DistScheduleOutput
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import debug_pront, debug_pront_2, debug_pront_3

logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(task: asyncio.Task,
                               request_tracker: "RequestTracker") -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        stream = self._request_streams[request_id]
        stream.put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            all_outputs = await self._run_workers_async(
                "execute_model",
                driver_kwargs={
                    "seq_group_metadata_list": seq_group_metadata_list,
                    "blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
                    "blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
                    "blocks_to_copy": scheduler_outputs.blocks_to_copy,
                })

            # Only the driver worker returns the sampling results.
            output = all_outputs[0]
        else:
            output = []

        return self._process_model_outputs(output, scheduler_outputs)

    async def _invoke_dist_workers(
            self, dist_output: DistScheduleOutput, is_prefill: bool,
            step) -> Tuple[List[RequestOutput], bool, bool, Tuple['float', 'float']]:

        # See the DistScheduler.schedule() as of how the scheduling actually happened
        # to avoid complex prefill / decode communication logic.
        start_time = time.time()
        is_transfer = dist_output.is_transfer_schedule
        if is_prefill:
            seq_group_metadata_list = dist_output.prefill_metadata
            scheduler_outputs = dist_output.prefill_output
            worker_group = self.prefill_workers
        else:
            seq_group_metadata_list = dist_output.decode_metadata
            scheduler_outputs = dist_output.decode_output
            worker_group = self.decode_workers

        # Invoke execute model
        output = []
        if not scheduler_outputs.is_empty(
        ) or dist_output.is_transfer_schedule:
            data = {
                "seq_group_metadata_list": seq_group_metadata_list,
                "blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
                "blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
                "blocks_to_copy": scheduler_outputs.blocks_to_copy,
                "send_blocks": dist_output.send_blocks,
                "recv_blocks": dist_output.recv_blocks,
            }
            debug_pront(
                f"{'Prefill pool' if is_prefill else 'Decode pool'} invoking execute_model with data: {data}"
            )

            all_outputs = await self._run_dist_worker_group_async(
                worker_group,
                "execute_model",
                **data,
            )
            output = all_outputs[0]

        step_output = self._process_model_outputs(output, scheduler_outputs)
        return step_output, is_prefill, is_transfer, (start_time, step)

    async def step_dist_async(self) -> Tuple[List[RequestOutput], bool]:
        """
        Step once in distserve mode.

        Returns: (step_output: List[RequestOutput], is_running: bool)

        """
        self.iteration_counter += 1
        debug_pront_2("\n-------------------\n")
        debug_pront_3(
            f"Starting step_dist_async() step {self.iteration_counter}.")
        assert self.parallel_config.is_disaggregate

        scheduler: DistScheduler = self.scheduler
        assert isinstance(scheduler, DistScheduler)

        scheduler_outputs: DistScheduleOutput = scheduler.schedule()
        debug_pront_2(f"Scheduler outputs properties: \n"
                      f"{scheduler_outputs.is_transfer_schedule = },\n"
                      f"{scheduler_outputs.has_prefill_schedule = },\n"
                      f"{scheduler_outputs.has_decode_schedule = },\n")
        debug_pront_2(f"Prefill scheduler: \n"
                      f"{len(scheduler.prefill_scheduler.waiting) = } \n"
                      f"{len(scheduler.prefill_scheduler.running) = } \n"
                      f"{len(scheduler.prefill_scheduler.swapped) = } \n")
        debug_pront_2(f"Decode scheduler: \n"
                      f"{len(scheduler.decode_scheduler.waiting) = } \n"
                      f"{len(scheduler.decode_scheduler.running) = } \n"
                      f"{len(scheduler.decode_scheduler.swapped) = } \n")

        prefill_future = None
        decode_future = None

        # Case 1: Block migration - must schedule both prefill and decode.
        if scheduler_outputs.is_transfer_schedule:
            assert scheduler.is_prefill_in_progress and scheduler.is_decode_in_progress, \
                "Block migration must schedule both prefill and decode."
            debug_pront_3(f"Block migration is invoked.")
            prefill_future = self._invoke_dist_workers(
                scheduler_outputs,
                is_prefill=True,
                step=self.iteration_counter)
            decode_future = self._invoke_dist_workers(
                scheduler_outputs,
                is_prefill=False,
                step=self.iteration_counter)
            pass
        # Case 2: Normal - can schedule either prefill or decode (or both)
        elif scheduler_outputs.has_prefill_schedule or scheduler_outputs.has_decode_schedule:
            if scheduler_outputs.has_prefill_schedule:
                assert scheduler.is_prefill_in_progress, \
                    "Prefill schedule must be invoked when prefill is in progress."
                debug_pront_3(f"Prefill schedule is invoked.")
                prefill_future = self._invoke_dist_workers(
                    scheduler_outputs,
                    is_prefill=True,
                    step=self.iteration_counter)
            if scheduler_outputs.has_decode_schedule:
                assert scheduler.is_decode_in_progress, \
                    "Decode schedule must be invoked when decode is in progress."
                debug_pront_3(f"Decode schedule is invoked.")
                decode_future = self._invoke_dist_workers(
                    scheduler_outputs,
                    is_prefill=False,
                    step=self.iteration_counter)

        # Add the futures to the pending futures set.
        if prefill_future:
            self.pending_futures.add(prefill_future)
        if decode_future:
            self.pending_futures.add(decode_future)

        if not self.pending_futures:
            return [], False

        finished, pending = await asyncio.wait(
            self.pending_futures, return_when=asyncio.FIRST_COMPLETED)
        self.pending_futures = pending

        # Prepare return result.
        result = []
        for future in finished:
            output, is_prefill, is_transfer, (start_time, step) = await future
            duration = time.time() - start_time
            task_name = 'prefill' if is_prefill else 'decode'
            if is_transfer:
                task_name += '_transfer'
            debug_pront_3(f"Accepted a finished task {step = } {task_name = } (step finished in {duration = })")
            if is_prefill:
                scheduler.on_prefill_finish(is_transfer=is_transfer)
            else:
                scheduler.on_decode_finish(is_transfer=is_transfer)
            result += output
        debug_pront_2(
            f"Finished step_dist_async() step {self.iteration_counter}.")
        debug_pront_2(f"Obtain result: {result = }.")

        debug_pront_2(
            f"Scheduler properties: \n"
            f"{scheduler.is_prefill_in_progress = }, \n"
            f"{scheduler.is_decode_in_progress = }, \n"
            f"{scheduler._in_progress_prefill_requests = }, \n"
            f"{scheduler._in_progress_prefill_requests_metadatas = }, \n"
            f"{scheduler.prefill_memblocks = }, \n"
            f"{scheduler.pending_migration_requests = }, \n")

        return result, True

    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Run the driver worker asynchronously.
        driver_executor = getattr(self.driver_worker, method)
        coros.append(asyncio.get_event_loop().run_in_executor(
            None, partial(driver_executor, *driver_args, **driver_kwargs)))

        # Run the ray workers asynchronously.
        for worker in self.workers:
            coros.append(worker.execute_method.remote(method, *args, **kwargs))

        all_outputs = await asyncio.gather(*coros)
        return all_outputs

    async def _run_dist_worker_group_async(
        self,
        worker_group: 'List[Union[RayWorkerVllm, Worker]]',
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ):
        """Runs the given method on all workers in the given worker group."""
        assert self.parallel_config.is_disaggregate

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        def _is_local(worker) -> bool:
            """Returns true if the worker is a local worker (wrt the driver)."""
            return getattr(worker, 'is_driver_worker', False)

        def _execute(worker, method, *args, **kwargs):
            """Executes the given method on the worker. Returns a handler if
            the worker is remote, otherwise returns the output directly.
            """
            if _is_local(worker):
                method = getattr(worker, method)
                func = partial(method, *args, **kwargs)
                coro = asyncio.get_event_loop().run_in_executor(None, func)
                return coro
            return worker.execute_method.remote(method, *args, **kwargs)

        def _get_return_value(worker, output):
            """Auxiliary function to get the return value of the worker."""
            if _is_local(worker):
                return output
            return ray.get(output)

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        lead_worker, rest_workers = worker_group[0], worker_group[1:]

        # Start the lead worker
        lead_worker_output = _execute(lead_worker, method, *driver_args,
                                      **driver_kwargs)

        # Start the rest of the workers
        # TODO: Is the type actually right for ray outputs?
        #  Don't we need to use ray.get()?
        rest_worker_outputs = [
            _execute(worker, method, *args, **kwargs)
            for worker in rest_workers
        ]

        coros = [lead_worker_output] + rest_worker_outputs

        all_outputs = await asyncio.gather(*coros)
        return all_outputs


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for LLMEngine.
        *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop: bool = start_engine_loop
        self._request_tracker = RequestTracker()

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = args[1]
            parallel_config = args[2]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            if self.engine_use_ray:
                await self.engine.add_request.remote(**new_request)
            else:
                self.engine.add_request(**new_request)

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
            is_running = len(request_outputs) > 0
        else:
            if self.engine.parallel_config.is_disaggregate:
                request_outputs, is_running = await self.engine.step_dist_async(
                )
            else:
                request_outputs = await self.engine.step_async()
                is_running = len(request_outputs) > 0

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)

        return is_running

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self, return_at_finish=False):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                if return_at_finish:
                    return
                # Wait for new requests if there are no requests in progress.
                debug_pront_2("Waiting for new requests...")
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        prefix_pos: Optional[int] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"prefix_pos: {prefix_pos},"
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            prefix_pos=prefix_pos)

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        prefix_pos: Optional[int] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        try:
            stream = await self.add_request(request_id,
                                            prompt,
                                            sampling_params,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time,
                                            prefix_pos=prefix_pos)

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs,
                         start_engine_loop: bool = True) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        placement_group = initialize_cluster(parallel_config,
                                             engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(parallel_config.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop)
        return engine

    async def do_log_stats(self) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote()
        else:
            self.engine.do_log_stats()
