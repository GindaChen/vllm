import json
from typing import List, Dict, Iterable

import requests

from .utils.metric import MetricStore

# localhost:8000/v1/completions

metric_store = MetricStore()


def send_create_request(url, request_id, prompt_text, prompt_token_ids, sampling_params):
    headers = {"User-Agent": "Test Client"}
    pload = {
        'prompt': prompt_token_ids,
        "max_tokens": sampling_params['max_tokens'],
        "stream": True,
        'stop_token_ids': [],
        'ignore_eos': True,
        'skip_special_tokens': False,
        # ---
        "n": 1,
        "use_beam_search": True,
        "temperature": 0.0,

    }
    response = requests.post(url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def send_abort_request(url, request_id):
    pass


def control_loop(url, base_prompt_len, backtrack_per_token, backtrack_len, splice_len, max_tokens):
    pass


def send_requests_in_batch(url, requests: List[Dict]):
    pass
