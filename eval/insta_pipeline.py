from insta import (
    get_browser_config,
    get_agent_config,
    get_judge_config,
    get_task_proposer_config,
    InstaPipeline
)

from datasets import load_dataset

import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent_model_name",
        type = str,
        default = "./qwen-1.5b-sft"
    )

    parser.add_argument(
        "--agent_api_key",
        type = str,
        default = "token-abc123"
    )

    parser.add_argument(
        "--agent_llm_endpoint",
        type = str,
        default = "http://localhost:8000/v1"
    )

    parser.add_argument(
        "--judge_model_name",
        type = str,
        default = "gpt-4o-mini"
    )

    parser.add_argument(
        "--judge_api_key",
        type = str,
        default = os.environ.get("OPENAI_API_KEY")
    )

    parser.add_argument(
        "--judge_llm_endpoint",
        type = str,
        default = "https://api.openai.com/v1"
    )

    parser.add_argument(
        "--task_proposer_model_name",
        type = str,
        default = "gpt-4o-mini"
    )

    parser.add_argument(
        "--task_proposer_api_key",
        type = str,
        default = os.environ.get("OPENAI_API_KEY")
    )

    parser.add_argument(
        "--task_proposer_llm_endpoint",
        type = str,
        default = "https://api.openai.com/v1"
    )

    parser.add_argument(
        "--dataset",
        type = str,
        default = "data-for-agents/insta-150k-v2",
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default = "train",
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
        "--playwright_workers",
        type = int,
        help = "Number of parallel Playwright servers",
        default = 1
    )

    parser.add_argument(
        "--num_agents",
        type = int,
        help = "Number of parallel agents to run",
        default = 1
    )

    parser.add_argument(
        "--observations_dir",
        type = str,
        help = "Directory to save observations",
        default = "data/observations"
    )

    parser.add_argument(
        "--screenshot_dir",
        type = str,
        help = "Directory to save screenshots",
        default = "data/screenshots"
    )

    parser.add_argument(
        "--actions_dir",
        type = str,
        help = "Directory to save actions",
        default = "data/actions"
    )

    parser.add_argument(
        "--judgments_dir",
        type = str,
        help = "Directory to save judgments",
        default = "data/judgments"
    )

    parser.add_argument(
        "--task_proposals_dir",
        type = str,
        help = "Directory to save task proposals",
        default = "data/task_proposals"
    )

    parser.add_argument(
        "--agent_response_key",
        type = str,
        help = "key for response from the agent",
        default = "response",
    )

    parser.add_argument(
        "--judge_response_key",
        type = str,
        help = "key for response from the judge",
        default = "response",
    )

    parser.add_argument(
        "--max_actions",
        type = int,
        help = "Maximum number of actions to take",
        default = 30
    )

    parser.add_argument(
        "--max_obs_tokens",
        type = int,
        help = "Maximum number of tokens per observation",
        default = 2048
    )

    parser.add_argument(
        "--last_obs",
        type = int,
        help = "Maximum number of observations in context",
        default = 5
    )

    parser.add_argument(
        "--agent_prompt",
        type = str,
        default = "verbose"
    )

    parser.add_argument(
        "--agent_top_p",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 1.0
    )

    parser.add_argument(
        "--agent_top_k",
        type = int,
        help = "Sampling temperature for LLMs",
        default = None
    )

    parser.add_argument(
        "--agent_temperature",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 0.5
    )

    parser.add_argument(
        "--agent_reasoning_effort",
        type = str,
        help = "Set reasoning mode in certain LLMs",
        default = None,
    )

    parser.add_argument(
        "--agent_disable_thinking_chat_template",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs",
    )

    parser.add_argument(
        "--judge_prompt",
        type = str,
        default = "verbose"
    )

    parser.add_argument(
        "--judge_top_p",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 1.0
    )

    parser.add_argument(
        "--judge_top_k",
        type = int,
        help = "Sampling temperature for LLMs",
        default = None
    )

    parser.add_argument(
        "--judge_temperature",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 0.5
    )

    parser.add_argument(
        "--judge_reasoning_effort",
        type = str,
        help = "Set reasoning mode in certain LLMs",
        default = None,
    )

    parser.add_argument(
        "--judge_disable_thinking_chat_template",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs",
    )

    parser.add_argument(
        "--task_proposer_prompt",
        type = str,
        default = "verbose"
    )

    parser.add_argument(
        "--task_proposer_top_p",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 1.0
    )

    parser.add_argument(
        "--task_proposer_top_k",
        type = int,
        help = "Sampling temperature for LLMs",
        default = None
    )

    parser.add_argument(
        "--task_proposer_temperature",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 0.5
    )

    parser.add_argument(
        "--task_proposer_reasoning_effort",
        type = str,
        help = "Set reasoning mode in certain LLMs",
        default = None,
    )

    parser.add_argument(
        "--task_proposer_disable_thinking_chat_template",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs",
    )

    parser.add_argument(
        "--skip_finished",
        action = "store_true",
        help = "Skip finished domains",
        default = False
    )

    parser.add_argument(
        "--prune_observations",
        action = "store_true",
        help = "Prune observation metadata to reduce disk usage",
        default = False
    )

    parser.add_argument(
        "--set_exploration_mode",
        action = "store_true",
        help = "Set the agent to exploration mode",
        default = False
    )

    parser.add_argument(
        "--add_steps_to_agent",
        action = "store_true",
        help = "Add the steps to the instruction",
        default = False
    )

    parser.add_argument(
        "--add_criteria_to_agent",
        action = "store_true",
        help = "Add the success criteria to the instruction",
        default = False
    )

    parser.add_argument(
        "--add_steps_to_judge",
        action = "store_true",
        help = "Add the steps to the instruction",
        default = False
    )

    parser.add_argument(
        "--add_criteria_to_judge",
        action = "store_true",
        help = "Add the success criteria to the instruction",
        default = False
    )

    parser.add_argument(
        "--add_steps_to_task_proposer",
        action = "store_true",
        help = "Add the steps to the instruction",
        default = False
    )

    parser.add_argument(
        "--add_criteria_to_task_proposer",
        action = "store_true",
        help = "Add the success criteria to the instruction",
        default = False
    )

    parser.add_argument(
        "--set_annotate_judge",
        action = "store_true",
        help = "Annotate trajectories with the judge",
        default = False
    )

    parser.add_argument(
        "--set_annotate_task_proposer",
        action = "store_true",
        help = "Annotate trajectories with the task proposer",
        default = False
    )

    parser.add_argument(
        "--seed",
        type = int,
        help = "Seed for the dataset",
        default = 0
    )

    parser.add_argument(
        "--rank",
        type = int,
        help = "Rank of the process",
        default = 0
    )

    parser.add_argument(
        "--world_size",
        type = int,
        help = "Number of processes",
        default = 1
    )

    args = parser.parse_args()

    browser_config = get_browser_config(
        playwright_url = args.playwright_url,
        playwright_port = args.playwright_port
    )

    agent_client_type = "openai"

    agent_client_kwargs = {
        "api_key": args.agent_api_key,
        "base_url": args.agent_llm_endpoint
    }

    agent_generation_kwargs = {
        "model": args.agent_model_name,
        "max_tokens": 1024,
        "top_p": args.agent_top_p,
        "temperature": args.agent_temperature,
        "extra_body": {}
    }

    if args.agent_reasoning_effort:

        agent_generation_kwargs.update({
            "reasoning_effort": 
            args.agent_reasoning_effort
        })

    if args.agent_disable_thinking_chat_template:

        if "chat_template_kwargs" not in agent_generation_kwargs["extra_body"]:

            agent_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        agent_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.agent_top_k is not None:

        agent_generation_kwargs["extra_body"].update({
            "top_k": args.agent_top_k
        })

    agent_config = get_agent_config(
        client_type = agent_client_type,
        client_kwargs = agent_client_kwargs,
        generation_kwargs = agent_generation_kwargs,
        agent_prompt = args.agent_prompt,
        max_obs_tokens = args.max_obs_tokens,
        last_obs = args.last_obs,
        log_errors = True,
    )

    judge_client_type = "openai"

    judge_client_kwargs = {
        "api_key": args.judge_api_key,
        "base_url": args.judge_llm_endpoint
    }

    judge_generation_kwargs = {
        "model": args.judge_model_name,
        "max_tokens": 1024,
        "top_p": args.judge_top_p,
        "temperature": args.judge_temperature,
        "extra_body": {}
    }

    if args.judge_reasoning_effort:

        judge_generation_kwargs.update({
            "reasoning_effort": 
            args.judge_reasoning_effort
        })

    if args.judge_disable_thinking_chat_template:

        if "chat_template_kwargs" not in judge_generation_kwargs["extra_body"]:

            judge_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        judge_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.judge_top_k is not None:

        judge_generation_kwargs["extra_body"].update({
            "top_k": args.judge_top_k
        })

    judge_config = get_judge_config(
        client_type = judge_client_type,
        client_kwargs = judge_client_kwargs,
        generation_kwargs = judge_generation_kwargs,
        log_errors = True,
    )

    task_proposer_client_type = "openai"

    task_proposer_client_kwargs = {
        "api_key": args.task_proposer_api_key,
        "base_url": args.task_proposer_llm_endpoint
    }

    task_proposer_generation_kwargs = {
        "model": args.task_proposer_model_name,
        "max_tokens": 1024,
        "top_p": args.task_proposer_top_p,
        "temperature": args.task_proposer_temperature,
        "extra_body": {}
    }

    if args.task_proposer_reasoning_effort:

        task_proposer_generation_kwargs.update({
            "reasoning_effort": 
            args.task_proposer_reasoning_effort
        })

    if args.task_proposer_disable_thinking_chat_template:

        if "chat_template_kwargs" not in task_proposer_generation_kwargs["extra_body"]:

            task_proposer_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        task_proposer_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.task_proposer_top_k is not None:

        task_proposer_generation_kwargs["extra_body"].update({
            "top_k": args.task_proposer_top_k
        })

    task_proposer_config = get_task_proposer_config(
        client_type = task_proposer_client_type,
        client_kwargs = task_proposer_client_kwargs,
        generation_kwargs = task_proposer_generation_kwargs,
        log_errors = True,
    )

    if not args.set_annotate_judge and \
            not args.set_annotate_task_proposer:

        judge_config = None

    if not args.set_annotate_task_proposer:

        task_proposer_config = None

    pipeline = InstaPipeline(
        browser_config = browser_config,
        agent_config = agent_config,
        judge_config = judge_config,
        task_proposer_config = task_proposer_config,
        seed = args.seed, rank = args.rank,
        world_size = args.world_size,
        observations_dir = args.observations_dir,
        screenshot_dir = args.screenshot_dir,
        actions_dir = args.actions_dir,
        judgments_dir = args.judgments_dir,
        task_proposals_dir = args.task_proposals_dir,
        max_actions = args.max_actions,
        agent_response_key = args.agent_response_key,
        judge_response_key = args.judge_response_key,
        skip_finished = args.skip_finished,
        prune_observations = args.prune_observations,
        add_steps_to_agent = args.add_steps_to_agent,
        add_criteria_to_agent = args.add_criteria_to_agent,
        add_steps_to_judge = args.add_steps_to_judge,
        add_criteria_to_judge = args.add_criteria_to_judge,
        add_steps_to_task_proposer = args.add_steps_to_task_proposer,
        add_criteria_to_task_proposer = args.add_criteria_to_task_proposer,
    )

    dataset = load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    if args.set_exploration_mode:

        dataset = dataset.remove_columns(list({
            "instruction", "task", "steps", "criteria"
        } & set(dataset.column_names)))

    pipeline.launch(
        dataset = dataset,
        num_agents = args.num_agents,
        playwright_workers = args.playwright_workers,
        return_trajectories = False
    )
