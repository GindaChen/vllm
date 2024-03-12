# Start a vLLM server
from vllm import LLM, SamplingParams, AsyncLLMEngine

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.5,
)
prompts = [
    "a a a a",
]
sampling_params = SamplingParams(temperature=0, top_p=1)

engine = llm.llm_engine

# This just add a few requests into the LLM engine.
llm.generate(prompts, sampling_params, should_run=False)
output_generator = llm._run_engine_generator(use_tqdm=True)

# Now run each step by calling the output_generator
for i, (step_output, finished_outputs) in enumerate(output_generator):
    print(f"Step {i}: {step_output}")
