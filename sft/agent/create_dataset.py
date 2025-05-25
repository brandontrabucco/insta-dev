from datasets import load_dataset

from insta.configs.agent_config import (
    AgentConfig,
    get_agent_config,
    DEFAULT_AGENT_CONFIG
)

from insta.agent import BrowserAgent
from functools import partial

from typing import List, Dict, Any

import argparse
import random
import json
import os


def select_valid_samples(
    example_dict: dict = None,
    data_dir: str = "data",
    success_threshold: float = 0.5,
    efficiency_threshold: float = 0.0,
    self_correction_threshold: float = 0.0,
    judge_name: str = "judgments-qwen-235b"
) -> bool:

    observations_dir = os.path.join(
        data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        data_dir,
        "actions"
    )

    judgments_dir = os.path.join(
        data_dir,
        judge_name
    )

    domain = example_dict["domain"]
            
    valid_domain = (
        os.path.exists(
            os.path.join(
                observations_dir,
                "{}.json".format(domain)
            )
        ) and os.path.exists(
            os.path.join(
                actions_dir,
                "{}.json".format(domain)
            )
        ) and os.path.exists(
            os.path.join(
                judgments_dir,
                "{}.json".format(domain)
            )
        )
    )

    if not valid_domain:

        return False
    
    judgments_path = os.path.join(
        judgments_dir,
        "{}.json".format(domain)
    )     

    with open(judgments_path, "r") as file:
        judgments = json.load(file)

    success = judgments["success"]
    efficiency = judgments["efficiency"]
    self_correction = judgments["self_correction"]

    is_success = (
        success is not None and 
        (success_threshold == 0 or success > success_threshold)
    )

    is_efficient = (
        efficiency is not None and 
        (efficiency_threshold == 0 or efficiency > efficiency_threshold)
    )

    is_self_correcting = (
        self_correction is not None and 
        (self_correction_threshold == 0 or self_correction > self_correction_threshold)
    )

    valid_domain = (
        is_success and 
        is_efficient and 
        is_self_correcting
    )

    return valid_domain


def get_prompts(
    observations: List[Dict[str, Any]] = None,
    actions: List[Dict[str, Any]] = None,
    instruction: str = None,
    agent: BrowserAgent = None,
) -> List[Dict[str, str]]:

    last_action = actions.pop()

    prompts = agent.get_prompts(
        observations = [
            x['processed_text'] 
            for x in observations
        ],
        instructions = [
            instruction
        ] * len(observations),
        urls = [
            x['current_url'] 
            for x in observations
        ],
        actions = [
            x['response'] 
            for x in actions
        ],
        last_obs = agent.config.last_obs
    )

    prompts.append({
        "role": "assistant",
        "content": last_action["response"]
    })

    return prompts


def unpack_examples(
    domain: str = None,
    instruction: str = None,
    observations_dir: str = None,
    actions_dir: str = None,
    agent: BrowserAgent = None,
) -> List[List[Dict[str, str]]]:
    
    observations_path = os.path.join(
        observations_dir,
        "{}.json".format(domain)
    )

    actions_path = os.path.join(
        actions_dir,
        "{}.json".format(domain)
    )

    with open(observations_path, "r") as file:
        observations = json.load(file)

    with open(actions_path, "r") as file:
        actions = json.load(file)

    examples = []

    for last_timestep in range(1, len(observations) + 1):

        prompts = get_prompts(
            observations = observations[:last_timestep],
            actions = actions[:last_timestep],
            instruction = instruction,
            agent = agent,
        )

        valid_prompts = all([
            x["content"] is not None
            for x in prompts
        ])

        if not valid_prompts:

            continue

        examples.append(prompts)

    return examples


def process_dataset(
    examples: dict,
    data_dir: str = None,
    agent: BrowserAgent = None,
) -> dict:

    observations_dir = os.path.join(
        data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        data_dir,
        "actions"
    )

    domains = examples["domain"]
    instructions = examples["task"]

    examples = []
    
    for domain, instruction in zip(
        domains,
        instructions
    ):

        examples.extend(unpack_examples(
            domain = domain,
            instruction = instruction,
            observations_dir = observations_dir,
            actions_dir = actions_dir,
            agent = agent,
        ))

    output_dict = {
        "messages": examples
    }

    return output_dict


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type = str,
        default="data-for-agents/insta-150k-v2"
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default="train"
    )

    parser.add_argument(
        "--data_dir",
        type = str,
        default="/data/matrix/projects/rsalakhugroup/btrabucc/insta-150k-v2-qwen3-235b-together"
    )

    parser.add_argument(
        "--last_obs",
        type = int,
        default = 5
    )

    parser.add_argument(
        "--max_obs_length",
        type = int,
        default = 2048
    )

    parser.add_argument(
        "--max_num_samples",
        type = int,
        default = 5000
    )

    parser.add_argument(
        "--dataset_output_dir",
        type = str,
        default="/data/matrix/projects/rsalakhugroup/btrabucc/insta-150k-v2-sft-qwen3-235b-{max_num_samples}x-{success_threshold}s-{judge_name}"
    )

    parser.add_argument(
        "--success_threshold",
        type = float,
        default = 0.5
    )

    parser.add_argument(
        "--efficiency_threshold",
        type = float,
        default = 0.0
    )

    parser.add_argument(
        "--self_correction_threshold",
        type = float,
        default = 0.0
    )

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "judgments-qwen-235b"
    )

    args = parser.parse_args()

    args.dataset_output_dir = (
        args.dataset_output_dir.format(
            **vars(args)
        )
    )

    dataset = load_dataset(
        args.dataset_name,
        split = args.dataset_split
    )

    agent_config = get_agent_config(
        last_obs = args.last_obs,
        max_obs_tokens = args.max_obs_length,
        action_parser = "simplified_json"
    )

    agent: BrowserAgent = BrowserAgent(
        config = agent_config
    )

    # client cannot be pickled
    agent.llm_client = None

    select_valid_samples = partial(
        select_valid_samples,
        data_dir = args.data_dir,
        success_threshold = args.success_threshold,
        efficiency_threshold = args.efficiency_threshold,
        self_correction_threshold = args.self_correction_threshold,
        judge_name = args.judge_name
    )
    
    dataset = dataset.filter(
        select_valid_samples
    )

    if args.max_num_samples is not None:

        dataset = dataset.select(
            random.Random(0).sample(
                list(range(len(dataset))),
                min(len(dataset), args.max_num_samples)
            )
        )

    process_dataset = partial(
        process_dataset,
        data_dir = args.data_dir,
        agent = agent
    )

    dataset = dataset.map(
        process_dataset,
        batched = True,
        remove_columns = dataset.column_names,
        batch_size = 32,
        num_proc = 32,
    )
    
    dataset.save_to_disk(
        args.dataset_output_dir
    )