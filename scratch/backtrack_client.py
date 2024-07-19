import asyncio
import json
import random
import uuid

import websockets


async def connect_to_server():
    uri = "ws://localhost:8000/v1/session"  # Replace with the actual URL of your WebSocket server

    async with websockets.connect(uri) as websocket:
        async def send_create_request(request_id, prompt_text, prompt_ids, sampling_params):
            request = {
                "action": "create",
                "kwargs": {
                    "request_id": request_id,
                    "prompt_text": prompt_text,
                    "prompt_ids": prompt_ids,
                    "sampling_params": sampling_params
                }
            }
            await websocket.send(json.dumps(request))
            print(f"Sent create request: {request}")

        async def send_abort_request(request_id):
            request = {
                "action": "abort",
                "kwargs": {
                    "request_id": request_id
                }
            }
            await websocket.send(json.dumps(request))
            print(f"Sent abort request: {request}")

        async def control_loop(base_prompt_len,
                               backtrack_per_token,
                               backtrack_len,
                               splice_len,
                               max_tokens):

            assert backtrack_len < splice_len, f"Otherwise nothing is going to go on..."
            sampling_params = dict(
                n=1, temperature=0.0, max_tokens=max_tokens,
            )
            prompt_ids = [10] * base_prompt_len
            prompt_text = ' a' * base_prompt_len

            buffer_token_ids = [i for i in prompt_ids]

            cur_request_id = None
            while len(buffer_token_ids) <= max_tokens:

                request_id = f"example_request_1-{uuid.uuid4()}"
                cur_request_id = request_id


                # Create request
                print(f"Creating request with id: {request_id}. Length = {len(buffer_token_ids)}")
                await send_create_request(
                    request_id, prompt_text,
                    prompt_ids, sampling_params,
                )

                # Receive response for n tokens
                print(f"Backtracking {backtrack_per_token} tokens.")
                for _ in range(backtrack_per_token):
                    # Receive response
                    response = await websocket.recv()
                    data = json.loads(response)
                    # Get 1 token from the data
                    # TODO: Fix this when we have multiple outputs...
                    token = data['outputs'][0]['token_ids'][-1]
                    buffer_token_ids.append(token)
                    if data['finished']:
                        return

                # Abort request
                print(f"Aborting request with id: {request_id}")
                await send_abort_request(request_id)

                # Then backtrack a few tokens
                buffer_token_ids = buffer_token_ids[:-backtrack_len]

                # Insert a few more tokens
                buffer_token_ids += [
                    random.randint(20, 2048) for _ in range(splice_len)
                ]

                print(f"Adjust buffer length and tokens: {len(buffer_token_ids)}")

            if cur_request_id is not None:
                await send_abort_request(cur_request_id)
            pass

        await control_loop(
            base_prompt_len=128,
            backtrack_per_token=5,
            backtrack_len=2,
            splice_len=5,
            max_tokens=192,
        )

        return


asyncio.run(connect_to_server())
