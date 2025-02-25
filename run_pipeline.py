from insta import (
    get_agent_config,
    get_judge_config,
    get_browser_config,
    InstaPipeline
)

import argparse
import os


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

    parser.add_argument(
        "--dataset",
        type = str,
        default = "data-for-agents/insta-150k",
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

    os.makedirs(
        args.observations_dir, 
        exist_ok = True
    )

    os.makedirs(
        args.screenshot_dir, 
        exist_ok = True
    )

    os.makedirs(
        args.actions_dir, 
        exist_ok = True
    )

    os.makedirs(
        args.judgments_dir, 
        exist_ok = True
    )

    client_kwargs = {
        "api_key": args.api_key,
        "base_url": args.llm_endpoint
    }

    generation_kwargs = {
        "model": args.model_name,
        "max_tokens": 2048,
        "top_p": 1.0,
        "temperature": 0.0
    }

    agent_config = get_agent_config(
        client_kwargs = client_kwargs,
        generation_kwargs = generation_kwargs
    )

    judge_config = get_judge_config(
        client_kwargs = client_kwargs,
        generation_kwargs = generation_kwargs
    )

    browser_config = get_browser_config(
        playwright_url = args.playwright_url,
        playwright_port = (
            args.playwright_port +
            args.rank % args.playwright_workers
        )
    )

    pipeline = InstaPipeline(
        agent_config = agent_config,
        judge_config = judge_config,
        browser_config = browser_config,
        dataset = args.dataset,
        dataset_split = args.dataset_split,
        observations_dir = args.observations_dir,
        screenshot_dir = args.screenshot_dir,
        actions_dir = args.actions_dir,
        judgments_dir = args.judgments_dir,
        max_actions = args.max_actions,
        skip_finished = args.skip_finished,
        seed = args.seed,
        rank = args.rank,
        world_size = args.world_size
    )

    pipeline.run_pipeline()
