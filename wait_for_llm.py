import argparse
import openai
import time


TEST_MESSAGE = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Are you ready?"
    },
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type = str,
        default = "meta-llama/Llama-3.3-70B-Instruct",
    )

    parser.add_argument(
        "--api_key",
        type = str,
        default = "token-abc123",
    )

    parser.add_argument(
        "--llm_endpoint",
        type = str,
        default = "http://localhost:8000/v1",
    )

    args = parser.parse_args()

    client_kwargs = {
        "api_key": args.api_key,
        "base_url": args.llm_endpoint
    }

    generation_kwargs = {
        "model": args.model_name,
        "max_tokens": 32,
    }

    client = openai.OpenAI(
        **client_kwargs
    )

    is_vllm_ready = False

    while not is_vllm_ready:

        try:  # wait for vLLM to be ready

            response = client.chat.completions.create(
                messages = TEST_MESSAGE,
                **generation_kwargs
            )

            is_vllm_ready = True
            print("vLLM started successfully.")

        except Exception: time.sleep(5)