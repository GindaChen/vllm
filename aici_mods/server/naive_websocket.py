# TODO: Refactor this file into AICI
import asyncio
from dataclasses import asdict

import fastapi

from vllm import AsyncLLMEngine
from vllm.engine.async_llm_engine import AsyncStream
from vllm.engine.async_llm_engine import AsyncStream
from vllm.entrypoints.openai import api_server as openai_api_server_module
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.sampling_params import SamplingParams
from vllm.utils import FlexibleArgumentParser


def get_engine() -> AsyncLLMEngine:
    return openai_api_server_module.engine


# Patch AsyncStream with a few properties
def AsyncStream__is_ready(self: AsyncStream):
    return not self._queue.empty()


def AsyncStream__get(self: AsyncStream):
    assert self.is_ready()
    return self._queue.get_nowait()


AsyncStream.is_ready = AsyncStream__is_ready
AsyncStream.get = AsyncStream__get


# TODO: Refactor the data structure used for requests.
@openai_api_server_module.router.websocket("/v1/session")
async def websocket_session(websocket: fastapi.WebSocket):
    await websocket.accept()
    active_sequences = {}
    active_sequences_lock = asyncio.Lock()

    # TODO: Refactor this logic into a class?
    async def handle_data():

        engine: AsyncLLMEngine = get_engine()
        try:
            while True:
                # await asyncio.sleep(0.01)
                data = await websocket.receive_json()
                print(f"Received data: {data}")
                action = data['action']
                kwargs = data['kwargs']
                print(f"Action: {action}, kwargs: {kwargs}")

                print(f"{active_sequences = }")

                if action == 'create':
                    request_id = kwargs['request_id']
                    assert request_id not in active_sequences

                    prompt_text = kwargs.get('prompt_text', None)
                    prompt_token_ids = kwargs.get('prompt_token_ids', None)

                    sampling_params = SamplingParams(
                        **kwargs.get('sampling_params', {})
                    )

                    generator: 'AsyncStream' = await engine.add_request(
                        request_id, {
                            'prompt': prompt_text,
                            'prompt_token_ids': prompt_token_ids,
                        },
                        sampling_params,
                    )

                    # async with active_sequences_lock:
                    active_sequences[request_id] = generator

                elif action == 'abort':
                    request_id = kwargs['request_id']
                    await engine.abort(request_id)
                    active_sequences.pop(request_id, None)
                    print(f"Aborted request_id: {request_id}")

        except fastapi.WebSocketDisconnect:
            return

    async def send_responses():
        try:
            while True:
                to_delete = []
                keys = active_sequences.keys()

                for request_id in keys:
                    generator: AsyncStream = active_sequences.get(request_id, None)

                    assert isinstance(generator, AsyncStream) or generator is None
                    if generator is None:
                        print(f"Generator is None for request_id: {request_id}")
                        continue

                    if not generator.is_ready():
                        continue

                    print(f"Handle send_responses for request_id: {request_id}")

                    if generator.is_ready():
                        print(f"Generator is ready for request_id: {request_id}")
                        try:
                            response = generator.get()
                        except Exception as e:
                            print(f"Request finished generation: {request_id =}, or with exception: {e}")
                            # Delete the handler
                            to_delete.append(request_id)
                            # Send the abort message
                            await websocket.send_json(dict(
                                request_id=request_id,
                                outputs=None,
                                finished=True,
                                metrics={},
                            ))
                            continue

                        print(f"Sending response for request_id: {request_id} -> {response}")
                        # TODO: Only send the prefix, not the whole stuff.
                        outputs = response.outputs
                        outputs_ = [asdict(i) for i in outputs]
                        await websocket.send_json(dict(
                            request_id=request_id,
                            outputs=outputs_,
                            finished=response.finished,
                            metrics=asdict(response.metrics),
                        ))
                        print(f"Sent response for request_id: {request_id}")

                for request_id in to_delete:
                    active_sequences.pop(request_id, None)

                await asyncio.sleep(0.01)

        except fastapi.WebSocketDisconnect:
            return

    await asyncio.gather(handle_data(), send_responses())

    return


if __name__ == "__main__":
    # NOTE(simon)(GindaChen):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    openai_api_server_module.run_server(args)
