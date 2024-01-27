import asyncio
from typing import AsyncIterator

from vllm import AsyncLLMEngine, SamplingParams, LLM, RequestOutput
from vllm.engine.async_llm_engine import AsyncStream


async def main():
    engine_args = LLM(
        model="facebook/opt-125m",
        is_disaggregate=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        enforce_eager=True,
        is_dummy_llm=True,
    ).engine_args

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # start the generation
    result_generators = []
    for request_id, prompt in enumerate(prompts):
        # FIXME: HACK - we assume engine always returns async stream.
        gen: AsyncStream = engine.generate(prompt, sampling_params,
                                           str(request_id))
        result_generators.append(gen)

    # Async get whatever is available from the `result_generators`
    # and print the results: print(f"[User] At user level, received output for request: {request_output = }")
    final_output = []
    while result_generators:
        for gen in result_generators:
            try:
                result: RequestOutput = await gen.__anext__()
                output_text = " ".join([i.text for i in result.outputs])
                text = f"[{result.request_id} ({len(result.outputs)})] {result.prompt} {output_text}"
                print(text)
            except StopAsyncIteration:
                result_generators.remove(gen)
            pass
        pass

    return final_output


if __name__ == '__main__':
    asyncio.run(main())
