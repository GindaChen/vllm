"""Example Python client for `vllm.entrypoints.api_server`
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""

import argparse
import json
import random
import time
from typing import Dict, Iterable, List, Union

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: Union[str, List[int]],
                      api_url: str,
                      n: int = 1,
                      stream: bool = False,
                      max_tokens: int = 16) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model": "facebook/opt-125m",
        "prompt": prompt,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[Dict]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\n"):
        if not chunk:
            continue
        # print(f"{chunk = }")
        # h = b'data: [DONE]'
        if b"[DONE]" in chunk:
            return
        # h = b'data: {"id":"cmpl-a52c27c28d9f4a4dac20e164e7863089",
        # "object":"text_completion","created":1721795130,"model":"facebook/opt-125m",
        # "choices":[{"index":0,"text":" a","token":10,"logprobs":null,"finish_reason":null,
        # "stop_reason":null}],"usage":null}'

        # strip the `data: ` prefix
        chunk = chunk[6:]
        chunk = json.loads(chunk.decode("utf-8"))
        yield chunk


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt_ids", type=str, default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=16)
    args = parser.parse_args()

    get_token = lambda: random.randint(20, 4096)
    prompt_ids = args.prompt_ids
    prompt_ids = eval(prompt_ids)

    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n
    stream = args.stream
    max_tokens = args.max_tokens
    assert stream

    print(f"prompt_ids: {prompt_ids!r}\n", flush=True)

    start = time.time()

    base_prompt_len = len(prompt_ids)
    backtrack_per_token = 5
    backtrack_len = 5
    splice_len = 10


    while len(prompt_ids) < max_tokens:
        print(f"{len(prompt_ids)!r}\n", flush=True)
        response = post_http_request(prompt_ids, api_url, n, stream, max_tokens)
        num_printed_lines = 0

        # Forward `backtrack_per_token` tokens
        i = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            token = h['choices'][0]['token']
            prompt_ids.append(token)
            print(f"{token = }")
            i += 1
            if i >= backtrack_per_token:
                break

        # Abort the request so vLLM knows to stop
        response.close()

        # Backtrack a few tokens, and add a few splice tokens
        prompt_ids = prompt_ids[:-backtrack_len]
        prompt_ids += [get_token() for _ in range(splice_len)]

    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
