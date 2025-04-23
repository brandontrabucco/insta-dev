from insta import (
    get_agent_config,
    get_judge_config,
    get_browser_config,
    InstaPipeline
)

import gradio
import argparse

import os


DEFAULT_MAX_ACTIONS = 30
DEFAULT_BEST_OF_N = 4

DEFAULT_MAX_OBS_TOKENS = 2048
DEFAULT_LAST_OBS = 3

DEFAULT_SUCCESS = 0.0


def generate_trajectory(url: str, instruction: str) -> str:

    if instruction is None or len(instruction) == 0:

        return "No task was entered"

    url = (url or "https://duckduckgo.com/")

    has_protocol = (
        url.startswith("http://")
        or url.startswith("https://")
    )

    if not has_protocol:

        url = "https://{}".format(url)

    best_success = None
    best_observations = None
    best_actions = None
    best_judgment = None

    for idx in range(args.best_of_n):

        print("[{i} of {n}] Running Agent\nURL: {url}\nInstruction: {instruction}".format(
            i = idx, n = args.best_of_n,
            url = url, instruction = instruction
        ))

        observations, actions, judgment = pipeline.generate_trajectory(
            url = url, instruction = instruction
        )

        trajectory_valid = (
            observations is not None and 
            len(observations) > 0 and 
            actions is not None and 
            len(actions) > 0 and 
            judgment is not None and 
            len(judgment) > 0
        )

        if not trajectory_valid:
            
            continue

        current_success = (
            judgment.get("success") 
            or DEFAULT_SUCCESS
        )

        trajectory_is_best = (
            best_success is None or 
            current_success > best_success
        )

        if trajectory_is_best:

            best_success, best_observations, best_actions, best_judgment = (
                current_success, observations, actions, judgment
            )

    agent_succeeded = (
        best_success is not None 
        and best_success > 0.5
    )

    if agent_succeeded:

        action_summary = "\n\n".join([
            x["response"]
            for x in best_actions
        ])

        return action_summary
    
    return "Agent did not succeed on this task"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent_model_name",
        type = str,
        default = "./qwen-1.5b-sft"
    )

    parser.add_argument(
        "--agent_client_type",
        type = str,
        default = "vllm"
    )

    parser.add_argument(
        "--agent_llm_endpoint",
        type = str,
        default = None
    )

    parser.add_argument(
        "--agent_api_key",
        type = str,
        default = None
    )

    parser.add_argument(
        "--judge_model_name",
        type = str,
        default = "gpt-4.1-nano"
    )

    parser.add_argument(
        "--judge_client_type",
        type = str,
        default = "openai"
    )

    parser.add_argument(
        "--judge_llm_endpoint",
        type = str,
        default = "https://api.openai.com/v1"
    )

    parser.add_argument(
        "--judge_api_key",
        type = str,
        default = os.environ.get("OPENAI_API_KEY")
    )

    parser.add_argument(
        "--action_parser",
        type = str,
        default = "simplified_json"
    )

    parser.add_argument(
        "--max_obs_tokens",
        type = int,
        help = "Maximum number of tokens per observations",
        default = DEFAULT_MAX_OBS_TOKENS
    )

    parser.add_argument(
        "--last_obs",
        type = int,
        help = "Previous observations in context",
        default = DEFAULT_LAST_OBS
    )

    parser.add_argument(
        "--max_actions",
        type = int,
        help = "Maximum number of actions to take",
        default = DEFAULT_MAX_ACTIONS
    )

    parser.add_argument(
        "--agent_response_key",
        type = str,
        help = "key for response from the agent",
        default = "response",
    )

    parser.add_argument(
        "--playwright_url",
        type = str,
        help = "URL template for Playwright server",
        default = "http://localhost:{port}"
    )

    parser.add_argument(
        "--playwright_port",
        type = int,
        help = "Port for Playwright server",
        default = 3000
    )

    parser.add_argument(
        "--best_of_n",
        type = int,
        help = "Number of tries",
        default = DEFAULT_BEST_OF_N
    )

    args = parser.parse_args()

    agent_client_type = args.agent_client_type

    agent_client_kwargs = {}

    agent_generation_kwargs = {
        "max_tokens": 2048,
        "top_p": 1.0,
        "temperature": 0.5
    }

    if args.agent_client_type == "openai":

        agent_generation_kwargs.update({
            "model": args.agent_model_name
        })

    if args.agent_client_type == "vllm":

        agent_client_kwargs.update({
            "model": args.agent_model_name
        })

    if args.agent_api_key is not None:

        agent_client_kwargs.update({
            "api_key": args.agent_api_key
        })

    if args.agent_llm_endpoint is not None:

        agent_client_kwargs.update({
            "base_url": args.agent_llm_endpoint
        })

    agent_config = get_agent_config(
        client_type = agent_client_type,
        client_kwargs = agent_client_kwargs,
        generation_kwargs = agent_generation_kwargs,
        action_parser = args.action_parser,
        max_obs_tokens = args.max_obs_tokens,
        last_obs = args.last_obs,
    )
    
    judge_client_type = args.judge_client_type

    judge_client_kwargs = {}

    judge_generation_kwargs = {
        "max_tokens": 2048,
        "top_p": 1.0,
        "temperature": 0.5
    }

    if args.judge_client_type == "openai":

        judge_generation_kwargs.update({
            "model": args.judge_model_name
        })

    if args.judge_client_type == "vllm":

        judge_client_kwargs.update({
            "model": args.judge_model_name
        })

    if args.judge_api_key is not None:

        judge_client_kwargs.update({
            "api_key": args.judge_api_key
        })

    if args.judge_llm_endpoint is not None:

        judge_client_kwargs.update({
            "base_url": args.judge_llm_endpoint
        })

    judge_config = get_judge_config(
        client_type = judge_client_type,
        client_kwargs = judge_client_kwargs,
        generation_kwargs = judge_generation_kwargs,
    )

    browser_config = get_browser_config(
        playwright_url = args.playwright_url,
        playwright_port = args.playwright_port
    )

    pipeline = InstaPipeline(
        agent_config = agent_config,
        judge_config = judge_config,
        browser_config = browser_config,
        agent_response_key = args.agent_response_key,
        max_actions = args.max_actions,
    )

    url_textbox = gradio.Textbox(
        label = "URL",
        placeholder = "Enter an initial URL ...",
        value = "https://duckduckgo.com/"
    )

    instruction_textbox = gradio.Textbox(    
        label = "Instruction",
        placeholder = "Enter an instruction ..."
    )

    output_textbox = gradio.Textbox(
        label = "Response",
        placeholder = "Final agent response ..."
    )

    gradio_app = gradio.Interface(
        generate_trajectory,
        inputs = [
            url_textbox,
            instruction_textbox,
        ],
        outputs = [
            output_textbox,
        ],
        title = "LLM Agent App",
    )

    gradio_app.launch()
