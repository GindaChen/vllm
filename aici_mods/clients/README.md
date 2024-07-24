# Client

## Example Running Command

```bash
python client_backtrack_rest_asyncio.py \
--stream \
--file task_backtrack_1.json \
--output_metric metric.task_backtrack_1.json \ 
--output_metric_stat metric_stat.task_backtrack_1.json \
;
```

## Example JSON file

```json
[{
    "base_prompt_len": 128,
    "backtrack_per_token": 16,
    "backtrack_len": 3,
    "splice_len": 16,
    "max_tokens": 566
},{
    "base_prompt_len": 128,
    "backtrack_per_token": 16,
    "backtrack_len": 3,
    "splice_len": 16,
    "max_tokens": 566
}]
```