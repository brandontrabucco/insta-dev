import argparse
import os


def add_agent_llm_args(parser: argparse.ArgumentParser):
    
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

    return parser


def add_agent_prompt_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--agent_prompt",
        type = str,
        default = "verbose"
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

    return parser


def add_agent_sampling_args(parser: argparse.ArgumentParser):

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

    return parser


def add_judge_llm_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--judge_model_name",
        type = str,
        default = "gpt-4.1-nano"
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

    return parser


def add_judge_name_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "gpt-4.1-nano-judge",
    )

    return parser


def add_judge_prompt_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--judge_prompt",
        type = str,
        default = "verbose"
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

    if not parser.get_default("agent_response_key"):

        parser.add_argument(
            "--agent_response_key",
            type = str,
            help = "key for response from the agent",
            default = "response",
        )

    return parser


def add_judge_sampling_args(parser: argparse.ArgumentParser):

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

    return parser


def add_task_proposer_llm_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--task_proposer_model_name",
        type = str,
        default = "gpt-4.1-nano"
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

    return parser


def add_task_proposer_name_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--task_proposer_name",
        type = str,
        default = "gpt-4.1-nano-task-proposer",
    )

    return parser


def add_task_proposer_prompt_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--task_proposer_prompt",
        type = str,
        default = "verbose"
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

    if not parser.get_default("agent_response_key"):

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

    return parser


def add_task_proposer_sampling_args(parser: argparse.ArgumentParser):

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

    return parser


def add_data_args(parser: argparse.ArgumentParser):

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
        "--input_data_dir",
        type = str,
        help = "Directory to save observations",
        default = "data"
    )

    return parser


def add_parallel_args(parser: argparse.ArgumentParser):

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

    return parser


def add_playwright_args(parser: argparse.ArgumentParser):

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

    return parser


def add_pipeline_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--num_agents",
        type = int,
        help = "Number of parallel agents to run",
        default = 1
    )

    parser.add_argument(
        "--max_actions",
        type = int,
        help = "Maximum number of actions to take",
        default = 30
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

    return parser


def add_annotate_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--skip_finished",
        action = "store_true",
        help = "Whether to skip existing task proposals",
        default = False
    )

    parser.add_argument(
        "--set_exploration_mode",
        action = "store_true",
        help = "Set the agent to exploration mode",
        default = False
    )

    parser.add_argument(
        "--seed",
        type = int,
        help = "Seed for the dataset",
        default = 0
    )

    parser.add_argument(
        "--num_workers",
        type = int,
        help = "Number of agents per machine",
        default = 8
    )

    return parser


def set_annotate_mode(args: argparse.Namespace):

    args.set_annotate_judge = True
    args.set_annotate_task_proposer = True

