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


def comparator(x, threshold):

    if x is None:

        return False

    elif threshold == 0:
        
        return True
    
    elif threshold == 1:

        return x == threshold 

    return x > threshold


def select_valid_samples(
    example_dict: dict = None,
    input_data_dir: str = "data",
    success_threshold: float = 0.5,
    efficiency_threshold: float = 0.0,
    self_correction_threshold: float = 0.0,
    judge_names: List[str] = ["qwen3-235b-judge"]
) -> bool:

    for judge_name in judge_names:

        observations_dir = os.path.join(
            input_data_dir,
            "observations"
        )

        actions_dir = os.path.join(
            input_data_dir,
            "actions"
        )

        judgment_dir = os.path.join(
            input_data_dir,
            judge_name
        )

        domain = example_dict.get(
            "website", example_dict.get("domain")
        )

        identifier = example_dict.get(
            "identifier", domain
        )
                
        valid_domain = (
            os.path.exists(
                os.path.join(
                    observations_dir,
                    "{}.json".format(identifier)
                )
            ) and os.path.exists(
                os.path.join(
                    actions_dir,
                    "{}.json".format(identifier)
                )
            ) and os.path.exists(
                os.path.join(
                    judgment_dir,
                    "{}.json".format(identifier)
                )
            )
        )

        if not valid_domain:

            return False

        judgments_path = os.path.join(
            judgment_dir,
            "{}.json".format(identifier)
        )

        with open(judgments_path, "r") as file:

            try: judgments = json.load(file)

            except json.JSONDecodeError:
                
                return False

        success = judgments["success"]
        efficiency = judgments["efficiency"]
        self_correction = judgments["self_correction"]

        is_success = (
            success is not None and 
            comparator(success, success_threshold)
        )

        is_efficient = (
            efficiency is not None and 
            comparator(efficiency, efficiency_threshold)
        )

        is_self_correcting = (
            self_correction is not None and 
            comparator(self_correction, self_correction_threshold)
        )

        valid_domain = (
            (is_success and is_efficient) or
            (is_success and is_self_correcting)
        )

        if not valid_domain:

            return False

    return True


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
    identifier: str = None,
    instruction: str = None,
    observations_dir: str = None,
    actions_dir: str = None,
    agent: BrowserAgent = None,
) -> List[List[Dict[str, str]]]:
    
    observations_path = os.path.join(
        observations_dir,
        "{}.json".format(identifier)
    )

    actions_path = os.path.join(
        actions_dir,
        "{}.json".format(identifier)
    )

    with open(observations_path, "r") as file:

        try: observations = json.load(file)

        except json.JSONDecodeError:
            
            return []

    with open(actions_path, "r") as file:

        try: actions = json.load(file)

        except json.JSONDecodeError:
            
            return []

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
    input_data_dir: str = None,
    agent: BrowserAgent = None,
) -> dict:

    observations_dir = os.path.join(
        input_data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        input_data_dir,
        "actions"
    )

    if "identifier" in examples:

        websites = examples["identifier"]

    elif "website" in examples:

        websites = examples["website"]

    elif "domain" in examples:

        websites = examples["domain"]

    if "instruction" in examples:

        instructions = examples["instruction"]

    elif "task" in examples:

        instructions = examples["task"]

    examples = []
    
    for identifier, instruction in zip(
        websites,
        instructions
    ):

        examples.extend(unpack_examples(
            identifier = identifier,
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
        default="data-for-agents/insta-150k-v3"
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default="train"
    )

    parser.add_argument(
        "--input_data_dir",
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
        "--judge_names",
        type = str,
        default = ["qwen3-235b-judge"],
        nargs = "+"
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
        agent_prompt = "base"
    )

    agent: BrowserAgent = BrowserAgent(
        config = agent_config
    )

    # client cannot be pickled
    agent.llm_client = None

    select_valid_samples = partial(
        select_valid_samples,
        input_data_dir = args.input_data_dir,
        success_threshold = args.success_threshold,
        efficiency_threshold = args.efficiency_threshold,
        self_correction_threshold = args.self_correction_threshold,
        judge_names = args.judge_names
    )
    
    dataset = dataset.filter(
        select_valid_samples,
        num_proc = 32,
        load_from_cache_file = False
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
        input_data_dir = args.input_data_dir,
        agent = agent
    )

    dataset = dataset.map(
        process_dataset,
        batched = True,
        remove_columns = dataset.column_names,
        batch_size = 32,
        num_proc = 32,
        load_from_cache_file = False
    )
    
    dataset.save_to_disk(
        args.dataset_output_dir
    )