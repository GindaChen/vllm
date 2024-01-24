import asyncio
from typing import AsyncIterator

from vllm import AsyncLLMEngine, SamplingParams, LLM, RequestOutput
from vllm.engine.async_llm_engine import AsyncStream


async def main():
    engine_args = LLM(
        model="facebook/opt-125m",
        is_disaggregate=True,
        # tensor_parallel_size=2,
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
        gen: AsyncStream = engine.generate(prompt, sampling_params, str(request_id))
        result_generators.append(gen)

    # Async get whatever is available from the `result_generators`
    # and print the results: print(f"[User] At user level, received output for request: {request_output = }")
    final_output = []
    while result_generators:
        # Run all generators concurrently and wait for the first one to complete
        done, pending = await asyncio.wait(
            [asyncio.create_task(gen.__anext__()) for gen in result_generators],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                result = task.result()
                # Process the result here
                print(f"[User] At user level, received output for request: {result = }")
            except StopAsyncIteration:
                # This generator is exhausted, remove it
                result_generators.remove(task.get_coro().cr_await)

        # Cancel any pending tasks (those that didn't finish first)
        for task in pending:
            task.cancel()

    return final_output


if __name__ == '__main__':
    asyncio.run(main())
