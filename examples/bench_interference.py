# Start a vLLM server
from vllm import LLM, SamplingParams, AsyncLLMEngine, LLMEngine
from vllm.worker.worker import Worker
from vllm.executor.gpu_executor import GPUExecutor

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.5,
    # disable graph caching
    enforce_eager=True,

)
prompts = [
    "a a a a",
    "a a a a a a a",
]
sampling_params = SamplingParams(temperature=0, top_p=1)

engine: 'LLMEngine' = llm.llm_engine
model_executor = engine.model_executor
assert isinstance(model_executor, GPUExecutor)

worker = model_executor.driver_worker
assert isinstance(worker, Worker)

model_runner = worker.model_runner
model_runner._prepare_prompt
model_runner._prepare_decode

# This just add a few requests into the LLM engine.
llm.generate(prompts, sampling_params, should_run=False)
output_generator = llm._run_engine_generator(use_tqdm=True)

# Now run each step by calling the output_generator
for i, (step_output, finished_outputs) in enumerate(output_generator):
    print(f"Step {i}: {step_output}")
