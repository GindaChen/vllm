from vllm import LLM, SamplingParams, LLMEngine
from vllm.config import ModelConfig, CacheConfig, DistParallelConfig, SchedulerConfig

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm_engine = LLMEngine(
    model_config=ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format='auto',
        dtype='auto',
        seed=0,
        revision=None,
        tokenizer_revision=None,
        max_model_len=2048,
        quantization=None,
        enforce_eager=False,
        max_context_len_to_capture=2048,
    ),
    cache_config=CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.3,
        swap_space=8,  # Swap Space in GB
        sliding_window=None,
    ),
    parallel_config=DistParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        prefill_pipeline_parallel_size=1,
        prefill_tensor_parallel_size=1,
        worker_use_ray=True,
    ),
    scheduler_config=SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        max_model_len=2048,
        max_paddings=256,
    ),
    placement_group=None, # TODO: Set the placement group by hand?
    log_stats=True,
)
