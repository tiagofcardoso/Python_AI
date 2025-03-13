from vllm.vllm import LLM, SamplingParams


def test_vllm():
    # Initialize the LLM
    llm = LLM(model="deepseek-r1-distill-llama-8b")

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
        top_p=0.95
    )

    # Test prompt
    prompt = "Explain what is artificial intelligence in one sentence."

    # Generate response
    outputs = llm.generate([prompt], sampling_params)

    # Print result
    for output in outputs:
        print(f"Prompt: {prompt}")
        print(f"Response: {output.outputs[0].text}")
        print(f"Generation time: {output.generation_info['time']:.2f} seconds")


if __name__ == "__main__":
    test_vllm()
