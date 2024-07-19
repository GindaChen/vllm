import asyncio
import fastapi

from vllm.sampling_params import SamplingParams
from dataclasses import asdict


def register_websocket_session(router, get_engine):
    @router.websocket("/v1/session")
    async def websocket_session(websocket: fastapi.WebSocket):

        await websocket.accept()
        active_sequences = {}
        active_sequences_lock = asyncio.Lock()

        async def handle_data():
            engine = get_engine()
            try:
                while True:
                    data = await websocket.receive_json()
                    print(f"Received data: {data}")
                    action = data['action']
                    kwargs = data['kwargs']
                    print(f"Action: {action}, kwargs: {kwargs}")

                    if action == 'create':
                        request_id = kwargs['request_id']
                        assert request_id not in active_sequences

                        prompt_text = kwargs.get('prompt_text', "")
                        prompt_ids = kwargs.get('prompt_ids', [])
                        sampling_params = SamplingParams(
                            **kwargs.get('sampling_params', {})
                        )

                        generator: 'AsyncStream' = await engine.add_request(
                            request_id,
                            prompt_text,
                            sampling_params,
                        )

                        async with active_sequences_lock:
                            active_sequences[request_id] = generator

                    elif action == 'abort':
                        request_id = kwargs['request_id']
                        engine.abort(request_id)
                        if request_id in active_sequences:
                            async with active_sequences_lock:
                                if request_id in active_sequences:
                                    del active_sequences[request_id]
                        print(f"Aborted request_id: {request_id}")

            except fastapi.WebSocketDisconnect:
                return

        async def send_responses():
            try:
                while True:
                    to_delete = []
                    async with active_sequences_lock:
                        keys = active_sequences.keys()

                    for request_id in keys:
                        print(f"Handle send_responses for request_id: {request_id}")
                        generator = active_sequences.get(request_id, None)
                        from vllm.engine.async_llm_engine import AsyncStream
                        assert isinstance(generator, AsyncStream) or generator is None
                        if generator is None:
                            continue

                        if generator.is_ready():
                            print(f"Generator is ready for request_id: {request_id}")
                            try:
                                response = await generator.get()
                            except Exception as e:
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

                            print(response)
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
                        async with active_sequences_lock:
                            if request_id in active_sequences:
                                del active_sequences[request_id]

                    await asyncio.sleep(0.05)

            except fastapi.WebSocketDisconnect:
                return

        await asyncio.gather(handle_data(), send_responses())

