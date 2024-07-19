import asyncio
import uuid
import websockets
import json

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

        async def receive_responses():
            try:
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    print(f"Received response: {data}")
                    if data['finished']:
                        return
            except websockets.ConnectionClosed:
                print("Connection closed")

        # Example usage
        request_id = f"example_request_1-{uuid.uuid4()}"
        prompt_text = "This is a test prompt."
        prompt_ids = []
        sampling_params = dict(
            n=1, temperature=0.0, max_tokens=128,
        )

        # Send a create request
        await send_create_request(request_id, prompt_text, prompt_ids, sampling_params)

        # Start receiving responses
        await receive_responses()

        # Send an abort request (if needed)
        await send_abort_request(request_id)

# Run the client
asyncio.run(connect_to_server())