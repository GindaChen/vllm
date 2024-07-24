"""
python client_backtrack_rest_asyncio.py --stream --max_tokens 200 --base_prompt_len 128 --backtrack_per_token 5 --backtrack_len 5 --splice_len 10
"""
import argparse
import asyncio
import json
import random
import time
import uuid
from typing import Dict, List, Union, AsyncGenerator

import aiohttp


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


async def post_http_request(
    session: aiohttp.ClientSession, prompt: Union[str, List[int]],
    api_url: str,
    stream: bool = False,
    max_tokens: int = 16,
    model="facebook/opt-125m",
) -> aiohttp.ClientResponse:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model": model,
        "prompt": prompt,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = await session.post(api_url, headers=headers, json=pload)
    return response


async def get_streaming_response(response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
    async for chunk in response.content.iter_any():
        if not chunk:
            continue
        if b"[DONE]" in chunk:
            return
        chunk = chunk[6:]  # strip the `data: ` prefix
        chunk = json.loads(chunk.decode("utf-8"))
        yield chunk


def get_token():
    return random.randint(20, 4096)


async def generate_tokens(
    api_url: str, stream: bool,
    base_prompt_len: int = 0,
    max_tokens: int = 0,
    backtrack_per_token: int = 0,
    backtrack_len: int = 0,
    splice_len: int = 0,
    model='facebook/opt-125m',
) -> None:
    start = time.time()
    assert splice_len > backtrack_len

    metric_id = f"{uuid.uuid4()}"

    prompt_ids = [get_token() for _ in range(base_prompt_len)]

    async with aiohttp.ClientSession() as session:
        while len(prompt_ids) < max_tokens:
            print(f"{len(prompt_ids)!r}\n", flush=True)
            response = await post_http_request(session, prompt_ids, api_url, stream, max_tokens, model=model)
            num_printed_lines = 0

            # Forward `backtrack_per_token` tokens
            i = 0
            async for h in get_streaming_response(response):
                clear_line(num_printed_lines)
                num_printed_lines = 0
                token = h['choices'][0]['token']
                prompt_ids.append(token)
                print(f"[{metric_id}] {token = }")
                i += 1
                if i >= backtrack_per_token:
                    break

            # Abort the request so vLLM knows to stop
            await response.release()

            # Backtrack a few tokens, and add a few splice tokens
            prompt_ids = prompt_ids[:-backtrack_len]
            prompt_ids += [get_token() for _ in range(splice_len)]

    end = time.time()
    print(f"Time taken: {end - start:.2f}s")


async def main(args):
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n
    file = args.file
    stream = args.stream
    model = args.model
    assert stream

    if file:
        with open(file, "r") as file:
            rs = json.load(file)
    else:
        rs = [
            dict(
                base_prompt_len=args.base_prompt_len,
                backtrack_per_token=args.backtrack_per_token,
                backtrack_len=args.backtrack_len,
                splice_len=args.splice_len,
                max_tokens=args.max_tokens,
            )
        ]

    tasks = [generate_tokens(api_url, stream, model=model, **r) for r in rs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")

    # Group 1: Single request
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--base_prompt_len", type=int, default=128)
    parser.add_argument("--backtrack_per_token", type=int, default=5)
    parser.add_argument("--backtrack_len", type=int, default=5)
    parser.add_argument("--splice_len", type=int, default=10)
    # Group 2: File-based request
    parser.add_argument("--file", type=str,
                        help="Path to JSON file containing list of request parameters")

    # Group 3: Output metric
    # parser.add_argument("--output_metric", type=str,
    #                     help="Path to store the metrics")

    args = parser.parse_args()

    asyncio.run(main(args))
