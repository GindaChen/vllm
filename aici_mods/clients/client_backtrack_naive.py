import argparse
import asyncio
import json
import random
import threading
import time
import uuid
from typing import Dict, List

import websockets


class MetricStore:
    """
    >>> metric_store = MetricStore()
    >>> metric_store.log_request_sent("req-1", {"prompt": "This is a test prompt."})
    >>> metric_store.log_response_received("req-1")
    >>> metrics = metric_store.get_metrics()
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricStore, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.metrics = {}

    def log_request_sent(self, request_id: str, metadata: Dict):
        if request_id not in self.metrics:
            self.metrics[request_id] = {}
        self.metrics[request_id]['sent_time'] = time.time()
        self.metrics[request_id]['metadata'] = metadata

    def log_response_received(self, request_id: str):
        if request_id in self.metrics:
            self.metrics[request_id]['received_time'] = time.time()

    def get_metrics(self) -> Dict[str, Dict]:
        return self.metrics


async def send_create_request(websocket, request_id, prompt_text, prompt_token_ids, sampling_params):
    request = {
        "action": "create",
        "kwargs": {
            "request_id": request_id,
            "prompt_text": prompt_text,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params
        }
    }
    await websocket.send(json.dumps(request))
    print(f"Sent create request: {request}")


async def send_abort_request(websocket, request_id):
    request = {
        "action": "abort",
        "kwargs": {
            "request_id": request_id
        }
    }
    await websocket.send(json.dumps(request))
    print(f"Sent abort request: {request}")


metric_store = MetricStore()


async def control_loop(uri, base_prompt_len, backtrack_per_token, backtrack_len, splice_len, max_tokens):
    assert backtrack_len < splice_len, "Backtrack length should be less than splice length."

    async with websockets.connect(uri) as websocket:
        sampling_params = {
            "n": 1,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        get_rand_token = lambda: random.randint(20, 2048)
        prompt_token_ids = [get_rand_token() for _ in range(base_prompt_len)]
        cur_request_id = None

        while len(prompt_token_ids) <= max_tokens:
            request_id = f"bt-req-{uuid.uuid4()}"
            cur_request_id = request_id

            # Create request
            print(f"Creating request with id: {request_id}. Length = {len(prompt_token_ids)}")
            await send_create_request(websocket, request_id, None, prompt_token_ids, sampling_params)
            metric_store.log_request_sent(request_id, {
                "prompt_text": None,
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": sampling_params,
            })

            # Receive response for n tokens
            print(f"Backtracking after {backtrack_per_token} tokens.")
            for _ in range(backtrack_per_token):
                response = await websocket.recv()
                data = json.loads(response)
                print(data)
                token = data['outputs'][0]['token_ids'][-1]
                print(f"Received token: {token}")
                prompt_token_ids.append(token)
                if data['finished']:
                    return

            # Abort request
            print(f"Aborting request with id: {request_id}")
            await send_abort_request(websocket, request_id)

            # Backtrack and splice tokens
            buffer_token_ids = prompt_token_ids[:-backtrack_len]
            print(f"Backtracked {backtrack_len} tokens. Length = {len(buffer_token_ids)}")
            buffer_token_ids += [get_rand_token() for _ in range(splice_len)]
            print(f"Spliced {splice_len} tokens. Length = {len(buffer_token_ids)}")

        if cur_request_id is not None:
            await send_abort_request(websocket, cur_request_id)


async def send_requests_in_batch(uri, requests: List[Dict]):
    tasks = [
        control_loop(
            uri,
            base_prompt_len=req["base_prompt_len"],
            backtrack_per_token=req["backtrack_per_token"],
            backtrack_len=req["backtrack_len"],
            splice_len=req["splice_len"],
            max_tokens=req["max_tokens"]
        ) for req in requests
    ]
    await asyncio.gather(*tasks)


async def connect_to_server(args):
    uri = args.uri
    if args.file:
        with open(args.file, "r") as file:
            requests = json.load(file)
    else:
        request = dict(
            base_prompt_len=args.base_prompt_len,
            backtrack_per_token=args.backtrack_per_token,
            backtrack_len=args.backtrack_len,
            splice_len=args.splice_len,
            max_tokens=args.max_tokens,
        )
        requests = [request]

    await send_requests_in_batch(uri, requests)
    return


def parse_args():
    parser = argparse.ArgumentParser(description="WebSocket Client for Sending Requests")
    parser.add_argument("--uri", type=str, default="ws://localhost:8000/v1/session", help="WebSocket server URI")

    # Group 1: One-off request parameters
    parser.add_argument("--base_prompt_len", type=int, help="Length of the base prompt")
    parser.add_argument("--backtrack_per_token", type=int, help="Number of tokens to backtrack per token")
    parser.add_argument("--backtrack_len", type=int, help="Number of tokens to backtrack in total")
    parser.add_argument("--splice_len", type=int, help="Number of tokens to splice")
    parser.add_argument("--max_tokens", type=int, help="Maximum number of tokens")

    # Group 2: Specify the path to the JSON file containing list of request parameters
    parser.add_argument("--file", type=str,
                        help="Path to JSON file containing list of request parameters")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(connect_to_server(args))
