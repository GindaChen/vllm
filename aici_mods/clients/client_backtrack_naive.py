import argparse
import asyncio
import json
import random
import uuid
from typing import Dict, List

import websockets

from .utils.metric import MetricStore

metric_store = MetricStore()


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

        metric_id = f"{uuid.uuid4()}"
        metric_store.register_request(metric_id, dict(
            base_prompt_len=base_prompt_len,
            backtrack_per_token=backtrack_per_token,
            backtrack_len=backtrack_len,
            splice_len=splice_len,
            max_tokens=max_tokens,
        ))

        while len(prompt_token_ids) <= max_tokens:
            request_id = f"bt-req-{uuid.uuid4()}"
            cur_request_id = request_id

            # Create request
            print(f"Creating request with id: {request_id}. Length = {len(prompt_token_ids)}")
            await send_create_request(websocket, request_id, None, prompt_token_ids, sampling_params)
            metric_store.log_request_sent(metric_id, {
                "type": "create",
                "length": len(prompt_token_ids),
            })

            # Receive response for n tokens
            print(f"Backtracking after {backtrack_per_token} tokens.")
            for _ in range(backtrack_per_token):
                response = await websocket.recv()
                metric_store.log_response_received(metric_id)
                data = json.loads(response)
                token = data['outputs'][0]['token_ids'][-1]
                print(f"[{metric_id}] Received token: {token}")
                prompt_token_ids.append(token)
                if data['finished']:
                    return

            # Abort request
            print(f"[{metric_id}] Aborting request with id: {request_id}")
            await send_abort_request(websocket, request_id)
            metric_store.log_request_sent(metric_id, {
                "type": "abort",
            })

            # Backtrack and splice tokens
            buffer_token_ids = prompt_token_ids[:-backtrack_len]
            buffer_token_ids += [get_rand_token() for _ in range(splice_len)]
            print(f"[{metric_id}] Spliced {splice_len} tokens. Length = {len(buffer_token_ids)}")

        if cur_request_id is not None:
            await send_abort_request(websocket, cur_request_id)
            metric_store.log_request_sent(metric_id, {
                "type": "abort",
            })


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


async def main(args):
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
    metrics = metric_store.get_metrics()
    stats = metric_store.get_stats()

    if output_metric_path := args.output_metric:
        with open(output_metric_path, "w+") as f:
            json.dump({
                "metric_metadata": metric_store.metric_metadata,
                "metrics": metrics,
            }, f)
            print(f"Saved metric to file {output_metric_path}")
        pass

    print(stats)
    if output_metric_stat_path := args.output_metric_stat:
        with open(output_metric_stat_path, "w+") as f:
            json.dump(stats, f)
            print(f"Saved metric stat to file {output_metric_stat_path}")
        pass

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

    # Group 3: Output metric
    parser.add_argument('--output_metric', type=str, help='Path to the metric output.')
    parser.add_argument('--output_metric_stat', type=str, help='Path to the metric stat output.')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
