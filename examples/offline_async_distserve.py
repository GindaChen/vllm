import asyncio

from vllm import AsyncLLMEngine, SamplingParams, LLM


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
    example_input = {
        "prompt": "What is LLM?",
        "stream": False,  # assume the non-streaming case
        "temperature": 0.0,
        "request_id": "0",
    }

    # start the generation
    results_generator = engine.generate(
        example_input["prompt"],
        SamplingParams(temperature=example_input["temperature"]),
        example_input["request_id"])

    # get the results
    final_output = []
    async for request_output in results_generator:
        final_output.append(request_output)
        print(
            f"[User] At user level, received output for request: {request_output = }"
        )

    return final_output


if __name__ == '__main__':
    asyncio.run(main())
